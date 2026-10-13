"""Synthetic records for the CPU tests: a complete, plausible attempt of one round."""
import json
from pathlib import Path

from tailsplit import spec
from tailsplit.analysis import cell_key

SOCKETS = {'0': [0, 2], '1': [1, 3]}
T0 = 10 ** 12          # CLOCK_MONOTONIC ns of the first request
SECOND = 1_000_000_000


def requests(prefix='r', n=20, gap_ns=10_000_000, concurrency=1, start=0):
    return [{'response_id': f'{prefix}{i}', 'content_times_ns': [start + j * gap_ns for j in range(256)],
             'phase': 'measured', 'concurrency': concurrency, 'started_ns': start, 'ended_ns': start + 255 * gap_ns,
             'usage': {'prompt_tokens': 256, 'completion_tokens': 256}, 'finish_reason': 'length',
             'text': ''} for i in range(n)]


def rows(prefix='r', n=20, period=10., seg=None, mask=True):
    seg = seg or {'f': 7., 'h': 2., 'm': .1, 's1': .2, 's2': .2}
    result = []
    for i in range(n):
        rid = f'{prefix}{i}-1234abcd'
        for step in range(1, 257):
            row = {'req_ids': [rid], 'request_steps': {rid: step}, 'batch_size': 1,
                   'prefill': [step == 1], 'period_ms': period if step < 256 else None,
                   'next_request_steps': {rid: step + 1} if step < 256 else {},
                   'segments_ms': dict(seg)}
            if mask:
                row.update(mask_active=True, mask_request_ids=[rid], mask_rows=1, logits_per_request={rid: 1})
            result.append(row)
    return result


def cell():
    return {'concurrency': 1, 'baseline_requests': requests(), 'patched_requests': requests(), 'rows': rows(),
            'runner_gates': [{'pass': True}, {'pass': True}], 'quiet_gate': {'pass': True},
            'condition': 'PLAIN', 'head_alone': {'1': {'median_ms': 2., 'times_ms': [2.] * 200}},
            'logs': ['', ''], 'schema_probes': [{'text': json.dumps([0] * 400)}]}


def summary(h=.2, m=.01, s=.02, residual=.05):
    return {'gates': {g: {'pass': True} for g in 'VDAXBJTQ'}, 'residual': residual,
            'shares': [{'H': h, 'M': m, 'S': s, 'T': h + m + s} for _ in range(2)]}


def telemetry(n=4):
    return [{'time_ns': i * SECOND, 'cpu': {str(c): {'total': i * 100, 'busy': i * 10} for c in (0, 1)},
             'threads': {f'{c}:1': {'ticks': i * 10, 'cpus': [c]} for c in (0, 1)},
             'owned_pids': [1], 'gpu': {'compute_pids': [1], 'clock_mhz': 2000}} for i in range(n)]


def prompt_manifest(model='qwen25-w4'):
    repo, revision = spec.MODELS[model]
    return {'model': model, 'repo': repo, 'revision': revision, 'prompts': [
        {'messages': [{'role': 'user', 'content': str(i)}], 'prompt_token_ids': [1] * 256}
        for i in range(spec.PROMPT_COUNT)]}


def evidence(model='qwen25-w4'):
    quant = spec.QUANT.get(model)
    if quant:
        method, scheme, _ = quant
        linears = [{'name': f'model.layers.{i // 4}.l{i % 4}', 'module': 'QKVParallelLinear', 'method': method,
                    'scheme': scheme or 'NoneType', 'kernel': 'MarlinLinearKernel'} for i in range(112)]
    else:
        linears = [{'name': f'model.layers.{i // 4}.l{i % 4}', 'module': 'QKVParallelLinear',
                    'method': 'UnquantizedLinearMethod', 'scheme': 'NoneType', 'kernel': 'NoneType'}
                   for i in range(112)]
    head = [{'name': 'lm_head', 'module': 'ParallelLMHead', 'method': 'UnquantizedEmbeddingMethod',
             'scheme': 'NoneType', 'kernel': 'NoneType'}]
    return {'runner_module': 'vllm.v1.worker.gpu.model_runner', 'runner_class': 'GPUModelRunner',
            'head_dtype': 'torch.float16', 'head_element_size': 2, 'head_quant_method': 'UnquantizedEmbeddingMethod',
            'head_shape': [152064, 3584], 'head_contiguous': True, 'quant_methods': linears + head,
            'default_stream': False}


def server_log(model='qwen25-w4'):
    quant = spec.QUANT.get(model)
    return 'INFO Using V2 Model Runner\n' + (f'INFO Using MarlinLinearKernel for {quant[2]}\n' if quant else '')


def micro(start_ns, gap_ms, median_ms=1.17, batch=1):
    gap = round(gap_ms * 1e6)
    preroll = list(range(start_ns, start_ns + spec.PREROLL_NS + gap, gap))
    first = preroll[-1] + gap
    launches = [first + i * gap for i in range(spec.HEAD_CALLS)]
    return {'shape': [152064, 3584], 'dtype': 'torch.float16', 'batch': batch, 'gap_ms': gap_ms, 'gap_ns': gap,
            'times_ms': [median_ms] * spec.HEAD_CALLS, 'median_ms': median_ms,
            'clock': 'clock_gettime(CLOCK_MONOTONIC)', 'preroll_launch_times_ns': preroll,
            'launch_times_ns': launches, 'timed_window_ns': [launches[0], launches[-1] + gap]}


class World:
    """Every raw record of one attempt, consistent with each other.

    Default numbers: unpatched gap g = 6.00 ms (PLAIN) / 6.10 ms (JSON), patched gap and step
    period 6.05 ms, head h = 1.15 ms, so H = 0.19 and the residual is about 0.015."""

    def __init__(self, model='qwen25-w4', h=1.15, m_json=.01, s2=.09, foreign_pid=None, clock_dip=False,
                 head_ms=1.17):
        self.model = model
        self.records, self.rows = {}, {}
        t = T0
        g = {'PLAIN': 6_000_000, 'JSON': 6_100_000}
        gp = 6_050_000
        for mode in ('baseline', 'patched'):
            for c in spec.CONCURRENCIES:
                for condition in spec.CONDITIONS:
                    key = cell_key(condition, c)
                    gap = g[condition] if mode == 'baseline' else gp
                    n = spec.measured_count(c)
                    prefix = f'{mode[0]}{c}{condition[0]}-'
                    recs = requests(prefix, n, gap, c, start=t)
                    if condition == 'JSON':
                        recs = [dict(r, text='') for r in recs]
                    self.records[mode, key] = recs
                    if mode == 'patched':
                        seg = {'f': 4.7, 'h': h, 'm': m_json if condition == 'JSON' else .002, 's1': .02, 's2': s2}
                        self.rows[key] = rows(prefix, n, gp / 1e6, seg)
                    t += 2 * SECOND
        self.micros = {}
        for c in spec.CONCURRENCIES:
            self.micros[c, 1] = micro(t, 6.0, head_ms)
            t = self.micros[c, 1]['timed_window_ns'][1] + 2 * SECOND
        self.end = t
        self.samples = []
        for i, now in enumerate(range(T0 - 2 * SECOND, t + 2 * SECOND, SECOND)):
            sample = {'time_ns': now, 'wall_time': 1.7e9 + i, 'capture_start_boottime_ns': now - 1000, 'clk_tck': 100,
                      'cpu': {str(cpu): {'total': i * 100, 'busy': i} for cpus in SOCKETS.values() for cpu in cpus},
                      'threads': {f'{cpu + 10}:5': {'ticks': i, 'cpus': [cpu]} for cpu in (0, 1)},
                      'owned_pids': [1, 2], 'gpu': {'uuid': 'GPU-test', 'clock_mhz': 2700.,
                                                    'compute_pids': [2], 'power_w': 300.}}
            if foreign_pid is not None:
                sample['gpu']['compute_pids'] = [2, foreign_pid]
            if clock_dip and i % 3 == 0:
                sample['gpu']['clock_mhz'] = 900.
            self.samples.append(sample)

    def probe(self):
        return [{'text': json.dumps(list(range(450))), 'finish_reason': 'stop', 'phase': 'schema',
                 'response_id': 'probe', 'usage': {'completion_tokens': 1500}}]

    def write(self, directory, analyze=True):
        """Write the attempt directory the way the coordinator does."""
        directory = Path(directory)
        for mode in ('baseline', 'patched'):
            (directory / mode).mkdir(parents=True, exist_ok=True)
            spec.write_json(directory / mode / 'worker.json', evidence(self.model))
            (directory / mode / 'server.log').write_text(server_log(self.model))
            for (m, key), recs in self.records.items():
                if m == mode:
                    spec.write_json(directory / mode / f'{key}.json', recs)
                    if mode == 'patched':
                        spec.write_json(directory / mode / f'{key}-steps.json', self.rows[key])
        spec.write_json(directory / 'patched' / 'schema.json', self.probe())
        for (c, b), m in self.micros.items():
            spec.write_json(directory / f'head-c{c}-b{b}.json', m)
        (directory / 'telemetry.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in self.samples))
        if analyze:
            from tailsplit import analysis
            result = analysis.analyze_attempt(directory, self.model, SOCKETS)
            spec.write_json(directory / 'gate-arithmetic.json', result)
            return result


def write_run(root, worlds=None, rounds=5):
    """A rounds directory: worlds = {model: [World per round]} (default: the decision model only)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    spec.write_json(root / 'manifest.json', {'step': 'rounds', 'topology': {'sockets': SOCKETS}, 'gpu': 0,
                                            'overlay': {'installed_identity': {'vllm': spec.VLLM_VERSION}}})
    worlds = worlds or {'qwen25-w4': [World() for _ in range(rounds)]}
    for model, items in worlds.items():
        for i, world in enumerate(items, 1):
            attempts = world if isinstance(world, list) else [world]
            for j, w in enumerate(attempts, 1):
                w.write(root / model / f'round-{i}' / f'attempt-{j}')
    (root / 'DONE').write_text('done\n')
    return root


class CachedAnalysis:
    """Wraps analyze_attempt so identical raw files are analysed once (the tests write many
    copies of the same attempt; one analysis of a six-cell attempt takes seconds)."""

    def __init__(self, function):
        self.function, self.cache = function, {}

    def __call__(self, directory, model, sockets):
        import hashlib
        h = hashlib.sha256(model.encode() + json.dumps(sockets).encode())
        for path in sorted(Path(directory).rglob('*')):
            if path.is_file() and path.name != 'gate-arithmetic.json':
                h.update(str(path.relative_to(directory)).encode())
                h.update(path.read_bytes())
        key = h.hexdigest()
        if key not in self.cache:
            try:
                self.cache[key] = ('ok', self.function(directory, model, sockets))
            except Exception as error:
                self.cache[key] = ('error', error)
        kind, value = self.cache[key]
        if kind == 'error':
            raise value
        return json.loads(json.dumps(value))
