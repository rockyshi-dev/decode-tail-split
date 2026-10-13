"""Everything a round's verdict needs, recomputed from the raw files of one attempt directory.

The coordinator calls analyze_attempt() at the end of every attempt and stores the result as
gate-arithmetic.json; `python -m tailsplit report` calls it again on the same raw files.
"""
import json
from pathlib import Path
import re

from . import gates as G
from .spec import CONCURRENCIES, CONDITIONS, QUANT, read_json

HEAD_FILE = re.compile(r'head-c(\d+)-b(\d+)\.json')


def cell_key(condition, concurrency):
    return f'{condition}-{concurrency}'


def head_files(directory):
    """{concurrency: {batch: path}} of the head-alone results in an attempt directory."""
    result = {}
    for path in Path(directory).iterdir():
        match = HEAD_FILE.fullmatch(path.name)
        if match:
            result.setdefault(int(match[1]), {})[int(match[2])] = path
    return result


def runner_gates(directory, model):
    """Gate V per server mode that was started, from worker.json and server.log."""
    result = {}
    for mode in ('baseline', 'patched'):
        worker = Path(directory) / mode / 'worker.json'
        if worker.exists():
            result[mode] = G.runner_gate(read_json(worker), (Path(directory) / mode / 'server.log').read_text(),
                                         QUANT.get(model))
    return result


def quiet_intervals(samples, sockets, intervals):
    """One labelled quiet check per (label, begin_ns, end_ns); a missing window fails."""
    result = []
    for label, begin, end in intervals:
        q = (G.quiet_gate(samples, sockets, begin, end) if begin is not None and end is not None
             else G.check(False, 'Head timed window evidence missing or invalid'))
        result.append({**q, 'interval': label})
    return result


def analyze_attempt(directory, model, sockets):
    """Raises FileNotFoundError (or ValueError on unreadable data) for an attempt that did not
    finish; returns {'gate_V_failed': True, ...} or {'cells': ..., 'quiet_gate': ...}."""
    directory = Path(directory)
    runner = runner_gates(directory, model)
    if not runner:
        raise FileNotFoundError(f'{directory}: no server was inspected')
    if not all(g['pass'] for g in runner.values()):
        return {'gate_V_failed': True, 'gates': runner}
    if set(runner) != {'baseline', 'patched'}:
        raise FileNotFoundError(f'{directory}: the attempt did not reach both servers')
    captures, intervals = {}, []
    for mode in ('baseline', 'patched'):
        for concurrency in CONCURRENCIES:
            for condition in CONDITIONS:
                key = cell_key(condition, concurrency)
                requests = read_json(directory / mode / f'{key}.json')
                rows = read_json(directory / mode / f'{key}-steps.json') if mode == 'patched' else []
                begin, end = G.measured_interval(requests, concurrency)
                intervals.append((f'{mode}-{key}', begin, end))
                captures[mode, key] = {'requests': requests, 'rows': rows}
    schema_probes = read_json(directory / 'patched' / 'schema.json')
    logs = [(directory / mode / 'server.log').read_text() for mode in ('baseline', 'patched')]
    micros = {}
    for concurrency, files in sorted(head_files(directory).items()):
        micros[concurrency] = {}
        for batch, path in sorted(files.items()):
            micro = read_json(path)
            micros[concurrency][str(batch)] = micro
            try:
                begin, end = G.head_timed_window(micro)
            except (KeyError, TypeError, ValueError):
                begin = end = None
            intervals.append((f'head-c{concurrency}-b{batch}', begin, end))
    samples = [json.loads(line) for line in (directory / 'telemetry.jsonl').read_text().splitlines() if line]
    checks = quiet_intervals(samples, sockets, intervals)
    quiet = {**G.check(bool(checks) and all(q['pass'] for q in checks), 'Round quiet check failed'),
             'intervals': checks}
    cells = {}
    for concurrency in CONCURRENCIES:
        plain_g = None
        for condition in CONDITIONS:
            key = cell_key(condition, concurrency)
            cell = {'condition': condition, 'concurrency': concurrency,
                    'baseline_requests': captures['baseline', key]['requests'],
                    'patched_requests': captures['patched', key]['requests'],
                    'rows': captures['patched', key]['rows'],
                    'runner_gates': list(runner.values()), 'quiet_gate': quiet,
                    'head_alone': micros.get(concurrency, {}), 'logs': logs, 'schema_probes': schema_probes}
            cells[key] = G.condition_gates(cell, plain_g)
            if condition == 'PLAIN':
                plain_g = cells[key].get('g_ms')
    return {'cells': cells, 'quiet_gate': quiet}


def q_failures(result):
    """Failure signature of one attempt: {(interval, reasons)} of its failing quiet intervals."""
    checks = result['quiet_gate'].get('intervals', [])
    if not checks:
        return {('<no intervals>', ('No measured intervals',))}
    return {(q['interval'], tuple(sorted(q['reasons']))) for q in checks if not q['pass']}


def repeated_failure(previous, current):
    """The same quiet interval failed for the same reasons in two consecutive attempts."""
    return sorted(q_failures(previous) & q_failures(current))
