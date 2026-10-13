"""The validity checks (V, D, A, X, B, J, T, Q), the shares, and the decision rule.

Pure arithmetic over recorded data; nothing here starts a process. Missing or inconsistent
evidence fails the check that needs it.
"""
import json
import math
import re
from statistics import median

from .spec import (A_PERIOD, A_SUM, B_SLACK, D_MAX, DISCARD, FUSION_BIG, HEAD_BIG, HEAD_CALLS,
                   OUTPUT_TOKENS, PREROLL_NS, PROMPT_TOKENS, Q_CLOCK, Q_FOREIGN, Q_SLOW, QUANT_LINEARS,
                   RESIDUAL_MAX, ROUNDS, SCHEMA, STOP_TAIL, T_DROP, X_MAX, measured_count)

NEEDED = {'PLAIN': ('V', 'D', 'A', 'X', 'T', 'Q'), 'JSON': ('V', 'D', 'A', 'B', 'J', 'T', 'Q')}

def med(values):
    values = list(values)
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('Missing/nonfinite measurement')
    return median(values)

def check(ok, *reasons):
    return {'pass': bool(ok), 'reasons': [] if ok else list(reasons)}

def runner_gate(evidence, log, quant=None):
    """Gate V. evidence: the worker's inspect result; log: the server log; quant: the spec.QUANT
    entry of a 4-bit model, None for a 16-bit one."""
    ok = (evidence.get('runner_module') == 'vllm.v1.worker.gpu.model_runner'
          and 'Using V2 Model Runner' in log
          and evidence.get('head_dtype') in ('torch.bfloat16', 'torch.float16')
          and evidence.get('head_element_size') == 2
          and evidence.get('head_quant_method') == 'UnquantizedEmbeddingMethod'
          and len(evidence.get('head_shape', [])) == 2)
    methods = evidence.get('quant_methods', [])
    if quant is not None:
        method, scheme, label = quant
        ok &= f'Using MarlinLinearKernel for {label}' in log
        rows = [x for x in methods if x.get('method') == method
                and (scheme is None or x.get('scheme') == scheme)]
        # Every decoder linear on the 4-bit Marlin path; none left in 16 bit.
        ok &= (len(rows) == QUANT_LINEARS and all(x.get('kernel') == 'MarlinLinearKernel' for x in rows)
               and not any(x.get('method') == 'UnquantizedLinearMethod' for x in methods))
    else:
        ok &= not any(x.get('method') not in ('UnquantizedLinearMethod', 'UnquantizedEmbeddingMethod')
                      for x in methods)
    return check(ok, 'V2/Marlin/dense 16-bit head evidence absent or contradictory')

def started_after(key, previous, current):
    """True only if thread `key` ('tid:starttime') provably started after the
    previous sample began. Its start time is in clock ticks since boot (floored,
    so the true start is never earlier); the bound is the previous sample's
    CLOCK_BOOTTIME read before its /proc/stat read. Rows without the boot-time
    fields credit nothing (conservative)."""
    bound, tck = previous.get('capture_start_boottime_ns'), current.get('clk_tck')
    if type(bound) is not int or type(tck) is not int or tck <= 0 or tck != previous.get('clk_tck'):
        return False
    try:
        start_ticks = int(key.rsplit(':', 1)[1])
    except (IndexError, ValueError):
        return False
    return start_ticks * 1_000_000_000 // tck > bound

def head_timed_window(micro):
    """Quiet-check interval of a head-alone job: first timed launch to last timed launch plus
    one gap g, all CLOCK_MONOTONIC ns (the clock of time.perf_counter_ns in every process on
    Linux, as used by the telemetry and the coordinator). Missing or inconsistent evidence
    raises ValueError."""
    if micro.get('clock') != 'clock_gettime(CLOCK_MONOTONIC)':
        raise ValueError('Head launch clock is not CLOCK_MONOTONIC')
    launches, preroll, g = micro['launch_times_ns'], micro['preroll_launch_times_ns'], micro['gap_ns']
    if type(g) is not int or g <= 0 or len(launches) != HEAD_CALLS or not preroll:
        raise ValueError('Head timing evidence incomplete')
    stamps = list(preroll) + list(launches)
    if any(type(t) is not int for t in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError('Head launch times are not strictly increasing integers')
    if launches[0] - preroll[0] < PREROLL_NS:
        raise ValueError('Head pre-roll shorter than 3 s')
    return launches[0], launches[-1] + g

def quiet_gate(samples, sockets, start_ns, end_ns):
    errors = []
    # Include an endpoint before/after the whole measured interval.
    before = [r for r in samples if r['time_ns'] <= start_ns]
    after = [r for r in samples if r['time_ns'] >= end_ns]
    if not before or not after:
        return check(False, 'Telemetry does not bracket the measured interval')
    lo, hi = before[-1]['time_ns'], after[0]['time_ns']
    rows = [r for r in samples if lo <= r['time_ns'] <= hi]
    if len(rows) < 2 or any('error' in r for r in rows):
        return check(False, 'Missing/error telemetry')
    required = {str(c) for cpus in sockets.values() for c in cpus}
    if any(not required <= set(r['cpu']) or not r['threads'] for r in rows):
        return check(False, 'Incomplete per-CPU/thread coverage')
    busy, totals, owned = {}, {}, {}
    for socket in sockets:
        busy[socket] = totals[socket] = owned[socket] = 0
    for a, b in zip(rows, rows[1:]):
        if not 0 < (b['time_ns'] - a['time_ns']) / 1e9 <= 2.5:
            errors.append('Telemetry is not continuous at one-second cadence')
        for socket, cpus in sockets.items():
            for cpu in cpus:
                busy[socket] += b['cpu'][str(cpu)]['busy'] - a['cpu'][str(cpu)]['busy']
                totals[socket] += b['cpu'][str(cpu)]['total'] - a['cpu'][str(cpu)]['total']
        for tid, thread in b['threads'].items():
            cpus = set(thread['cpus'])
            matches = [s for s, ids in sockets.items() if cpus and cpus <= set(ids)]
            if len(matches) != 1:
                errors.append('Owned thread affinity not confined to a declared socket')
                continue
            old = a['threads'].get(tid)
            if old:
                if not old['cpus'] or not set(old['cpus']) <= set(sockets[matches[0]]):
                    errors.append('Owned thread moved across CPU sockets during accounting')
                    continue
                delta = thread['ticks'] - old['ticks']
                if delta < 0:
                    errors.append('Owned thread CPU clock regressed')
                owned[matches[0]] += max(0, delta)
            elif started_after(tid, a, b):
                # Every recorded thread belongs to the owned process tree; one that
                # started after the previous sample began spent all its CPU time
                # inside this sample pair, so all of it is owned.
                owned[matches[0]] += max(0, thread['ticks'])
            # Otherwise (a thread that existed before the previous sample but was
            # not observed in it, or an exited thread) nothing is credited: its
            # CPU time stays in the conservative foreign bound.
    fractions = {}
    for socket in sockets:
        if totals[socket] <= 0 or busy[socket] < 0:
            errors.append('Invalid host CPU clock')
            continue
        fractions[socket] = max(0, busy[socket] - owned[socket]) / totals[socket]
        if fractions[socket] > Q_FOREIGN:
            errors.append(f'Foreign CPU time exceeds {Q_FOREIGN:.0%} on socket {socket}')
    clocks = [r['gpu']['clock_mhz'] for r in rows]
    if any(set(r['gpu']['compute_pids']) - set(r['owned_pids']) for r in rows):
        errors.append('Foreign GPU compute process')
    if min(clocks) <= 0 or sum(x < Q_CLOCK * med(clocks) for x in clocks) / len(clocks) > Q_SLOW:
        errors.append('GPU clock stability failed')
    return {**check(not errors, *sorted(set(errors))), 'foreign_fraction_upper_bound': fractions,
            'median_clock_mhz': med(clocks)}

def valid_schema(text):
    try:
        result = json.loads(text)
    except (TypeError, ValueError):
        return False
    return (isinstance(result, list) and SCHEMA['minItems'] <= len(result) <= SCHEMA['maxItems']
            and all(type(x) is int for x in result))

def measured_requests(requests, concurrency):
    expected = measured_count(concurrency)
    if any(r.get('phase') not in ('measured', 'padding') or r.get('concurrency') != concurrency
           for r in requests):
        raise ValueError('Missing/incorrect request phase or concurrency')
    ids = [r['response_id'] for r in requests]
    if len(ids) != len(set(ids)):
        raise ValueError('Request identities are not unique')
    result = [r for r in requests if r['phase'] == 'measured']
    if len(result) != expected:
        raise ValueError(f'Expected {expected} measured requests at c={concurrency}')
    return result

def measured_interval(requests, concurrency):
    requests = measured_requests(requests, concurrency)
    return min(r['started_ns'] for r in requests), max(r['ended_ns'] for r in requests)

def gap_ms(requests, concurrency=1):
    gaps = []
    for request in measured_requests(requests, concurrency):
        if request['usage'].get('completion_tokens') != OUTPUT_TOKENS:
            continue
        stamps = request['content_times_ns']
        gaps.extend((b-a)/1e6 for a, b in list(zip(stamps, stamps[1:]))[DISCARD:])
    value = med(gaps)
    if value <= 0:
        raise ValueError('Nonpositive client gap')
    return value

def matched_steps(rows, requests, concurrency=1):
    """Verify correspondence against all raw rows, including the padding drain."""
    measured = [r for r in measured_requests(requests, concurrency)
                if r['usage'].get('completion_tokens') == OUTPUT_TOKENS]
    external = {r['response_id'] for r in measured}
    all_ids = {r['response_id'] for r in requests}
    mapping = {}
    for row in rows:
        if (len(row['req_ids']) != row['batch_size'] or len(set(row['req_ids'])) != row['batch_size']
            or set(row['request_steps']) != set(row['req_ids']) or len(row['prefill']) != row['batch_size']):
            raise ValueError('Invalid batch identity')
        for rid in row['req_ids']:
            matches = [x for x in all_ids if rid == x or re.fullmatch(re.escape(x) + r'-[0-9a-f]{8}', rid)]
            if len(matches) != 1:
                raise ValueError('Worker/client request identity mismatch')
            mapping[rid] = matches[0]
    measured_mapping = {rid: x for rid, x in mapping.items() if x in external}
    if len(measured_mapping) != len(external) or set(measured_mapping.values()) != external:
        raise ValueError('Worker did not record every measured request')
    counts = {rid: [] for rid in measured_mapping}
    prefill_rows = {rid: 0 for rid in measured_mapping}
    for row in rows:
        for i, rid in enumerate(row['req_ids']):
            if rid in counts:
                counts[rid].append(row['request_steps'][rid])
                prefill_rows[rid] += bool(row['prefill'][i])
    # Chunked prefill (default 2048 batched tokens) may split one prompt over several
    # rows; the last prefill row samples token 1, then 255 decode rows follow.
    if any(prefill_rows[rid] < 1 or sorted(steps) != list(range(1, OUTPUT_TOKENS + prefill_rows[rid]))
           for rid, steps in counts.items()):
        raise ValueError('Expected one or more prefill rows then 255 decode steps per request')
    return mapping, external

def selected_steps(rows, requests, concurrency=1):
    """Count a GPU batch once; exclude batches containing any padding work.

    Events measure an entire batch, so a mixed measured/padding batch cannot
    be split by request. Its raw rows still prove measured-request coverage.
    """
    mapping, external = matched_steps(rows, requests, concurrency)
    selected = []
    for row in rows:
        if not row['req_ids'] or any(mapping[rid] not in external for rid in row['req_ids']):
            continue
        if any(row['prefill']) or any(n <= DISCARD for n in row['request_steps'].values()):
            continue
        next_steps = row.get('next_request_steps', {})
        if (row.get('period_ms') is None or set(next_steps) != set(row['request_steps'])
            or any(next_steps[rid] != n + 1 for rid, n in row['request_steps'].items())):
            continue  # No period spanning an idle/replacement boundary.
        selected.append(row)
    if not selected:
        raise ValueError('No eligible decode steps')
    return selected

def condition_gates(cell, plain_g=None):
    """Checks and quantities of one cell (condition x concurrency) of one round.

    cell: condition, concurrency, baseline_requests, patched_requests, rows (the patched worker's
    step records), runner_gates, quiet_gate, head_alone ({batch size: head-alone result}), logs,
    schema_probes. plain_g: the unpatched PLAIN gap at the same concurrency (for gate B)."""
    baseline_records, patched_records = cell['baseline_requests'], cell['patched_requests']
    concurrency = cell['concurrency']
    gates = {'V': check(len(cell['runner_gates']) == 2 and all(g['pass'] for g in cell['runner_gates']), 'Runner gate failed'),
             'Q': cell['quiet_gate']}
    result = {'gates': gates, 'condition': cell['condition'], 'concurrency': concurrency}
    try:
        base = measured_requests(baseline_records, concurrency)
        patched = measured_requests(patched_records, concurrency)
        drops = [sum(r['usage'].get('completion_tokens') != OUTPUT_TOKENS for r in arm) for arm in (base, patched)]
        gates['T'] = check(all(drop / measured_count(concurrency) <= T_DROP for drop in drops),
                           f'Token drop rate exceeds {T_DROP:.0%}')
        result['dropped'] = drops
        base = [r for r in base if r['usage'].get('completion_tokens') == OUTPUT_TOKENS]
        patched = [r for r in patched if r['usage'].get('completion_tokens') == OUTPUT_TOKENS]
        g, gp = gap_ms(baseline_records, concurrency), gap_ms(patched_records, concurrency)
        result.update(g_ms=g, gp_ms=gp)
        gates['D'] = check(abs(gp-g)/g <= D_MAX, f'Patched gap differs by more than {D_MAX:.0%}')
        rows = selected_steps(cell['rows'], patched_records, concurrency)
        p = med(r['period_ms'] for r in rows)
        components = {key: med(r['segments_ms'][key] for r in rows) for key in ('f', 'h', 'm', 's1', 's2')}
        if p <= 0 or min(components.values()) < 0:
            raise ValueError('Invalid server times')
        sums = med(sum(r['segments_ms'].values()) for r in rows)
        chunks_ok = all(len(r['content_times_ns']) == OUTPUT_TOKENS
                        and r['usage'].get('prompt_tokens') == PROMPT_TOKENS
                        for arm in (base, patched) for r in arm)
        gates['A'] = check(chunks_ok and abs(p-gp)/gp <= A_PERIOD and sums <= A_SUM*p,
                           'Chunk/prompt count, period agreement, or segment accounting failed')
        result.update(p_ms=p, components_ms=components, sum_median_ms=sums,
                      residual=1-sum(components.values())/p,
                      shares=[{'H': components['h']/d, 'M': components['m']/d,
                               'S': (components['s1']+components['s2'])/d,
                               'T': sum(components[k] for k in ('h','m','s1','s2'))/d}
                              for d in (g,gp)])
        if cell['condition'] == 'PLAIN':
            micro = cell['head_alone']
            reference = med(micro[str(r['batch_size'])]['median_ms'] for r in rows)
            micro_ok = all(len(m['times_ms']) == HEAD_CALLS and med(m['times_ms']) == m['median_ms']
                           and m['median_ms'] > 0 for m in micro.values())
            gates['X'] = check(micro_ok and abs(components['h']-reference)/reference <= X_MAX,
                               f'Head differs from the head-alone timing by more than {X_MAX:.0%}')
            result['head_alone_ms'] = reference
        else:
            gates['B'] = check(plain_g is not None and plain_g > 0 and
                all(s['M'] <= (g-plain_g)/plain_g + B_SLACK for s in result['shares']),
                'Mask exceeds unpatched JSON-minus-PLAIN upper bound')
            probes = cell.get('schema_probes', [])
            mapping, external = matched_steps(cell['rows'], patched_records, concurrency)
            # J checks measured requests even in mixed batches, but never a
            # padding-only step or padding's own mask/finish/usage evidence.
            mask_rows = [(r, {rid for rid in r['req_ids'] if mapping[rid] in external})
                         for r in cell['rows']]
            gates['J'] = check(len(probes) == 1 and all(valid_schema(r['text']) for r in probes)
                and all(r['finish_reason'] == 'length' for arm in (base, patched) for r in arm)
                and all(r['segments_ms']['m'] > 0 and r['mask_rows'] == len(r['mask_request_ids'])
                        and len(set(r['mask_request_ids'])) == len(r['mask_request_ids'])
                        and set(r['mask_request_ids']) <= set(r['req_ids'])
                        and ids <= set(r['mask_request_ids'])
                        and (ids != set(r['req_ids']) or r['mask_active'])
                        and all(r['logits_per_request'][rid] == 1 for rid in ids)
                        for r, ids in mask_rows if ids)
                and not any(re.search(r'(?i)(failed to advance|grammar[^\n]*(?:error|fail))', log)
                            for log in cell['logs']), 'Schema, active mask, finish reason, or grammar log check failed')
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        result['invalid_evidence'] = str(error)
    for gate in NEEDED[cell['condition']]:
        if gate not in gates:
            gates[gate] = check(False, 'Evidence not available')
    return result

def question(rounds, condition):
    """Head question (PLAIN) or fusion question (JSON) over the decision model's rounds at
    concurrency 1: BIG, SMALL, MIXED or NOT MEASURED. A share counts as 'at least x' only if it
    is so with both denominators (g and g'), and as 'below x' only if both are below."""
    needed = NEEDED[condition]
    if len(rounds) != ROUNDS or any(not all(r['gates'].get(g, {}).get('pass') for g in needed) for r in rounds):
        return 'NOT MEASURED'
    threshold = HEAD_BIG if condition == 'PLAIN' else FUSION_BIG
    def values(r):
        return [s['H'] if condition == 'PLAIN' else s['M'] + s['S'] for s in r['shares']]
    if all(all(x >= threshold for x in values(r)) and r['residual'] <= RESIDUAL_MAX for r in rounds):
        return 'BIG'
    if all(all(x < threshold for x in values(r)) for r in rounds):
        return 'SMALL'
    return 'MIXED'


MEANING = {
    'head BIG': 'the output head is worth a faster storage format or kernel on this card',
    'fusion BIG': 'a kernel joining mask and sampler is worth building on this card',
    'STOP': 'the tail is too small on this card for any tail kernel to matter',
    'none': 'no follow-up kernel is indicated; each question is reported with its own answer',
}


def decision(plain_rounds, json_rounds):
    """The decision rule over the decision model's five rounds at concurrency 1."""
    head, fusion = question(plain_rounds, 'PLAIN'), question(json_rounds, 'JSON')
    outcomes = []
    if head == 'BIG':
        outcomes.append('head BIG')
    if fusion == 'BIG':
        outcomes.append('fusion BIG')
    stop = head == fusion == 'SMALL' and all(all(s['T'] < STOP_TAIL for s in r['shares'])
                                              for r in plain_rounds + json_rounds)
    if not outcomes:
        outcomes = ['STOP'] if stop else ['none']
    # Written before measuring: H between 0.10 and 0.25 in every round, fusion SMALL.
    expectation_met = (fusion == 'SMALL' and head in ('BIG', 'SMALL') and
        all(all(.10 <= s['H'] <= .25 for s in r.get('shares', [])) for r in plain_rounds))
    return {'head': head, 'fusion': fusion, 'outcomes': outcomes,
            'meaning': [MEANING[o] for o in outcomes], 'expectation_met': expectation_met}
