"""Start tests and rounds on one GPU."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import analysis
from .client import control
from .gates import check, decision, gap_ms, measured_interval, runner_gate, selected_steps
from .environment import package_deviations
from .launch import REPO, Server, code_hashes, lock, run_head, runtime_environment
from .overlay import overlay_hashes
from .spec import (ATTEMPTS, CONCURRENCIES, CONDITIONS, DECISION_MODEL, MODELS, QUANT, ROUNDS, SCHEMA,
                   read_json, sha, validate_prompts, write_json)
from .telemetry import query_gpu


class RequestFailure(RuntimeError):
    pass


class RunStopped(RuntimeError):
    pass


def child_env():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(REPO)
    env['PYTHONNOUSERSITE'] = '1'
    return env


def child(argv, cpus, log_path):
    command = ['taskset', '-c', ','.join(map(str, cpus)), *map(str, argv)]
    write_json(log_path.with_suffix('.argv.json'), {'argv': command, 'time': time.time()})
    with log_path.open('x') as log:
        subprocess.run(command, env=child_env(), stdin=subprocess.DEVNULL, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def run_control(server, action):
    try:
        return control(server.port, action)
    except Exception as error:
        write_json(server.directory / f'control-{action}-failure.json', {'action': action, 'error': str(error)})
        raise RequestFailure(f'{action} control request failed') from error


def handle_failure(server, error):
    if isinstance(error, RequestFailure):
        try:
            server.request_failed(error)
        except BaseException as cleanup_error:
            server.retain(cleanup_error)
            raise
    else:
        server.retain(error)


def run_client(run, model, condition, concurrency, phase, label, output, topo):
    try:
        child([run.cfg.python, '-s', '-m', 'tailsplit.client', '--manifest', run.prompts / f'{model}.json',
               '--model-path', run.cfg.model_path(model), '--port', run.cfg.port, '--condition', condition,
               '--concurrency', concurrency, '--phase', phase, '--label', label, '--output', output],
              topo['client'], output.with_suffix('.log'))
    except subprocess.CalledProcessError as error:
        write_json(output.with_suffix('.failure.json'), {'phase': phase, 'error': str(error),
                   'exit_code': error.returncode, 'argv': error.cmd})
        raise RequestFailure(f'{phase} client failed; this start test or round is invalid') from error
    return json.loads(output.read_text())


def server_gate(server, model):
    evidence = run_control(server, 'inspect')
    write_json(server.directory / 'worker.json', evidence)
    return evidence, runner_gate(evidence, (server.directory / 'server.log').read_text(), QUANT.get(model))


def start_test(run, model, output, topo):
    """Start both servers of a checkpoint, inspect them (gate V), stop them. No timed request."""
    results = {}
    for mode in ('baseline', 'patched'):
        server = Server(model, mode, output / mode, run.cfg, topo['server'])
        try:
            server.ready()
            evidence, gate = server_gate(server, model)
            results[mode] = {'evidence': evidence, 'gate_V': gate}
            server.completed()
        except BaseException as error:
            handle_failure(server, error)
            raise
    write_json(output / 'start-test.json', results)
    return results


def telemetry_start(run, directory, topo):
    output, stop = directory / 'telemetry.jsonl', directory / 'telemetry.stop'
    log = (directory / 'telemetry.log').open('x')
    argv = ['taskset', '-c', ','.join(map(str, topo['logger'])), str(run.cfg.python), '-s', '-m',
            'tailsplit.telemetry', '--root-pid', str(os.getpid()), '--gpu', str(run.cfg.gpu),
            '--output', str(output), '--stop-file', str(stop)]
    process = subprocess.Popen(argv, env=child_env(), stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 15
    while not output.exists() or not output.stat().st_size:
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError('Telemetry could not record its first sample')
        time.sleep(.1)
    return process, log, stop, output


def measure_round(run, model, round_dir, topo, attempt_fn=None):
    """Up to ATTEMPTS attempts; only a failed quiet check is rerun. If two consecutive attempts
    fail the same quiet interval for the same reasons while a rerun is still left, the run
    stops instead of using it (RunStopped)."""
    attempt_fn = attempt_fn or attempt
    previous = None
    for retry in range(ATTEMPTS):
        result = attempt_fn(run, model, round_dir / f'attempt-{retry + 1}', topo)
        if result.get('gate_V_failed') or result['quiet_gate']['pass']:
            return result
        if previous is not None and retry < ATTEMPTS - 1:
            repeated = analysis.repeated_failure(previous, result)
            if repeated:
                record = {'model': model, 'round': round_dir.name,
                          'attempts': [f'attempt-{retry}', f'attempt-{retry + 1}'],
                          'gate': 'Q', 'repeated_failures': [[label, list(reasons)] for label, reasons in repeated],
                          'rule': 'Two consecutive attempts failed the same quiet interval for the same reasons; '
                                  'the run stops for a person to look instead of using the remaining rerun',
                          'time': time.time()}
                write_json(round_dir / 'STOPPED.json', record)
                raise RunStopped(record)
        previous = result
    return result


def attempt(run, model, directory, topo):
    """One attempt of a round: unpatched server (PLAIN and JSON at c = 1, 8, 32), patched server
    (same, plus one schema-parse request), then the head-alone timings; then the arithmetic."""
    directory.mkdir(parents=True, exist_ok=False)
    telemetry, log, stop, _ = telemetry_start(run, directory, topo)
    captures, evidence, runner_gates = {}, {}, {}
    try:
        for mode in ('baseline', 'patched'):
            server = Server(model, mode, directory / mode, run.cfg, topo['server'])
            try:
                server.ready()
                evidence[mode], runner_gates[mode] = server_gate(server, model)
                if not runner_gates[mode]['pass']:
                    server.completed()   # a failed gate V licenses no measured request
                    write_json(directory / 'gate-V-failed.json', runner_gates)
                    return {'gate_V_failed': True, 'gates': runner_gates}
                for concurrency in CONCURRENCIES:
                    for condition in CONDITIONS:
                        key = analysis.cell_key(condition, concurrency)
                        label = f'{model}-{directory.parent.name}-{directory.name}-{mode}-{key}'
                        run_client(run, model, condition, concurrency, 'warmup', label + '-warm',
                                   server.directory / f'{key}-warmup.json', topo)
                        if mode == 'patched':
                            run_control(server, 'arm')
                        requests = run_client(run, model, condition, concurrency, 'measured', label,
                                              server.directory / f'{key}.json', topo)
                        rows = run_control(server, 'drain')['rows'] if mode == 'patched' else []
                        if mode == 'patched':
                            write_json(server.directory / f'{key}-steps.json', rows)
                        captures[mode, key] = {'requests': requests, 'rows': rows}
                if mode == 'patched':
                    run_client(run, model, 'JSON', 1, 'schema', model + '-schema',
                               server.directory / 'schema.json', topo)
                server.completed()
            except BaseException as error:
                handle_failure(server, error)
                raise
        # Head alone: same card, the server's head shape and dtype, one job per batch size seen
        # in the eligible steps of each PLAIN cell, spaced at that cell's unpatched gap.
        for concurrency in CONCURRENCIES:
            key = analysis.cell_key('PLAIN', concurrency)
            baseline, patched = captures['baseline', key], captures['patched', key]
            try:
                g = gap_ms(baseline['requests'], concurrency)
                rows = selected_steps(patched['rows'], patched['requests'], concurrency)
            except (KeyError, ValueError):
                continue  # no shape or gap is invented when the instrument failed; gate X fails
            for batch in sorted({r['batch_size'] for r in rows}):
                name = f'head-c{concurrency}-b{batch}'
                contract = {'shape': evidence['patched']['head_shape'],
                            'dtype': evidence['patched']['head_dtype'], 'batch': batch, 'gap_ms': g}
                write_json(directory / f'{name}-contract.json', contract)
                run_head(run.cfg, model, directory / f'{name}-contract.json', directory / f'{name}.json',
                         topo['server'])
    finally:
        stop.touch()
        telemetry.wait(timeout=15)
        log.close()
    report = analysis.analyze_attempt(directory, model, topo['sockets'])
    write_json(directory / 'gate-arithmetic.json', report)
    return report


def parse_cpulist(text):
    cpus = set()
    for part in text.strip().split(','):
        if part:
            start, _, end = part.partition('-')
            cpus.update(range(int(start), int(end or start) + 1))
    return cpus


def topology(gpu, lscpu=None, bus_id=None, sys_root=Path('/sys')):
    """Server, head job and coordinator: every CPU of the GPU's socket. Client: two physical
    cores (with their SMT siblings) of the other socket. Telemetry: a third core there."""
    text = lscpu if lscpu is not None else subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET,ONLINE'], text=True)
    cores = {}
    for line in text.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        cpu, core, sock, online = line.split(',')[:4]
        if online == 'Y':
            cores.setdefault((int(sock), int(core)), []).append(int(cpu))
    sockets = sorted({k[0] for k in cores})
    if len(sockets) != 2:
        raise RuntimeError('Expected a machine with two CPU sockets')
    bus = bus_id or subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=pci.bus_id',
                                             '--format=csv,noheader'], text=True).strip()
    bus = bus.lower()
    domain, rest = bus.split(':', 1)
    numa = int((sys_root / 'bus/pci/devices' / f'{int(domain, 16):04x}:{rest}' / 'numa_node').read_text())
    if numa < 0:
        raise RuntimeError(f'GPU {gpu} reports no NUMA node')
    node_cpus = parse_cpulist((sys_root / 'devices/system/node' / f'node{numa}' / 'cpulist').read_text())
    by_socket = {s: sorted(c for (sock, _), v in cores.items() if sock == s for c in v) for s in sockets}
    owners = [s for s in sockets if node_cpus and node_cpus <= set(by_socket[s])]
    if len(owners) != 1:
        raise RuntimeError(f'Cannot place GPU {gpu}\'s NUMA node on one socket')
    server_socket = owners[0]
    client_socket = next(s for s in sockets if s != server_socket)
    client_cores = [v for (sock, _), v in sorted(cores.items()) if sock == client_socket]
    if len(client_cores) < 3:
        raise RuntimeError('Need two physical client cores and one telemetry core on the other socket')
    return {'server': by_socket[server_socket], 'client': [c for group in client_cores[:2] for c in group],
            'logger': client_cores[2], 'sockets': {str(s): by_socket[s] for s in sockets},
            'server_socket': server_socket, 'lscpu': text, 'gpu': gpu, 'pci_bus': bus, 'numa_node': numa}


class Run:
    def __init__(self, cfg, prompts, output, locks):
        self.cfg, self.prompts, self.output, self.locks = cfg, Path(prompts), Path(output), list(locks)


def check_inputs(run):
    for model, (repo, revision) in MODELS.items():
        path = run.cfg.model_path(model)
        if not (path / 'config.json').is_file():
            raise RuntimeError(f'{repo} at revision {revision} not found at {path}')
        manifest = read_json(run.prompts / f'{model}.json')
        if (manifest.get('repo'), manifest.get('revision')) != (repo, revision):
            raise RuntimeError(f'{run.prompts / (model + ".json")} was built for another checkpoint')
        for concurrency in CONCURRENCIES:
            validate_prompts(manifest, concurrency)


def check_start_tests(directory, cache_dir):
    """The start tests must have finished, with this code, these checkpoints and the same compile
    cache directory; returns {model: start-test.json}."""
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    if (manifest.get('step') != 'start-test' or manifest.get('code_sha256') != code_hashes()
            or manifest.get('models') != {k: list(v) for k, v in MODELS.items()}):
        raise RuntimeError('The start tests were made with other code or other checkpoints; rerun them')
    if not (directory / 'DONE').exists() or not all((directory / m / 'start-test.json').exists() for m in MODELS):
        raise RuntimeError('The start test did not finish for every checkpoint; rerun them')
    if Path(manifest.get('cache_dir', '')).resolve() != Path(cache_dir).resolve():
        raise RuntimeError(f"The start tests used the cache directory {manifest.get('cache_dir')}; "
                           f'pass the same --cache-dir or rerun them')
    return {model: read_json(directory / model / 'start-test.json') for model in MODELS}


def execute(run, step, start_tests=None):
    if step == 'rounds':
        check_inputs(run)
        if start_tests is None:
            raise RuntimeError('Rounds need the directory of a finished start test (--start-tests)')
        start_results = check_start_tests(start_tests, run.cfg.cache_dir)
        if read_json(Path(start_tests) / 'manifest.json').get('overlay_sha256') != overlay_hashes(run.cfg.overlay):
            raise RuntimeError('The start tests used another overlay; rerun them')
    runtime_environment(run.cfg, 'baseline')   # environment and overlay checks before any GPU query
    deviations = package_deviations()
    for line in deviations:
        print('package differs from requirements-cu129.lock.txt: ' + line, file=sys.stderr)
    with ExitStack() as stack:
        for path in run.locks:
            stack.enter_context(lock(path))
        run.output.mkdir(parents=True, exist_ok=False)
        stack.enter_context(lock(run.output / 'owner.lock'))
        topo = topology(run.cfg.gpu)
        os.sched_setaffinity(0, topo['server'])
        overlay_dir = Path(run.cfg.overlay)
        write_json(run.output / 'manifest.json', {
            'step': step, 'schema': SCHEMA, 'models': {k: list(v) for k, v in MODELS.items()},
            'topology': topo, 'gpu': run.cfg.gpu, 'gpu_state': query_gpu(run.cfg.gpu), 'port': run.cfg.port,
            'hf_home': str(run.cfg.hf_home), 'cache_dir': str(Path(run.cfg.cache_dir).resolve()),
            'python': run.cfg.python, 'package_deviations': deviations,
            'code_sha256': code_hashes(), 'overlay_sha256': overlay_hashes(overlay_dir),
            'overlay': read_json(overlay_dir / 'overlay.json'),
            'timing_patch': (overlay_dir / 'timing.patch').read_text(),
            'prompts_sha256': {m: sha(run.prompts / f'{m}.json') for m in MODELS} if step == 'rounds' else None,
            'start_tests': str(start_tests) if start_tests else None, 'argv': sys.argv, 'started': time.time()})
        for model in MODELS:
            root = run.output / model
            root.mkdir(exist_ok=False)
            if step == 'start-test':
                start_test(run, model, root, topo)
                continue
            tests = start_results[model]
            if set(tests) != {'baseline', 'patched'} or not all(t['gate_V']['pass'] for t in tests.values()):
                write_json(root / 'not-measured-start-test.json', tests)
                if model == DECISION_MODEL:
                    write_json(run.output / 'decision.json', decision([], []))
                continue
            rounds = []
            for round_id in range(ROUNDS):
                round_dir = root / f'round-{round_id + 1}'
                round_dir.mkdir()
                try:
                    result = measure_round(run, model, round_dir, topo)
                except RunStopped as stop:
                    write_json(root / 'rounds-before-stop.json', rounds)
                    write_json(run.output / 'STOPPED.json', stop.args[0])
                    raise SystemExit(3)   # no decision file, no later checkpoint
                rounds.append(result)
                if result.get('gate_V_failed'):
                    break   # this checkpoint is not measured
            write_json(root / 'rounds.json', rounds)
            if model == DECISION_MODEL:
                plain = [r['cells']['PLAIN-1'] for r in rounds if 'cells' in r]
                grammar = [r['cells']['JSON-1'] for r in rounds if 'cells' in r]
                write_json(run.output / 'decision.json', decision(plain, grammar))
        (run.output / 'DONE').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z') + '\n')
