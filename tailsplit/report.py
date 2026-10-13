"""`python -m tailsplit report RUN_DIR`: every check, share and the verdict, recomputed from the
raw files of a rounds directory."""
import json
from pathlib import Path
from statistics import median

from . import analysis
from .gates import NEEDED, decision
from .spec import (ATTEMPTS, CONCURRENCIES, DECISION_MODEL, FUSION_BIG, HEAD_BIG, HEAD_BYTE_SHARE, MODELS,
                   RESIDUAL_MAX, ROUNDS, read_json)


def normal(obj):
    return json.loads(json.dumps(obj, allow_nan=False))


QUANTITIES = ('g_ms', 'gp_ms', 'p_ms', 'sum_median_ms', 'residual', 'head_alone_ms', 'dropped')


def projection(cell):
    """What the verdict reads from one cell: the quantities and the pass/fail of each needed check."""
    out = {k: cell[k] for k in QUANTITIES if k in cell}
    out['components_ms'] = cell.get('components_ms')
    out['shares'] = cell.get('shares')
    out['gates'] = {g: bool(cell.get('gates', {}).get(g, {}).get('pass')) for g in NEEDED[cell['condition']]}
    return normal(out)


def same_cells(a, b):
    return all(projection(a['cells'][k]) == projection(b['cells'][k]) for k in a['cells']) and \
        set(a['cells']) == set(b['cells']) and a['quiet_gate']['pass'] == b['quiet_gate']['pass']


def read_round(round_dir, model, sockets, warnings):
    """(used result or None, record) for one round directory."""
    record = {'round': round_dir.name, 'attempts': [], 'used_attempt': None, 'stopped': None}
    if (round_dir / 'STOPPED.json').exists():
        record['stopped'] = read_json(round_dir / 'STOPPED.json')
    attempts = sorted((p for p in round_dir.iterdir() if p.is_dir() and p.name.startswith('attempt-')),
                      key=lambda p: int(p.name.split('-')[1]))
    used = None
    for path in attempts[:ATTEMPTS]:
        try:
            result = analysis.analyze_attempt(path, model, sockets)
        except (FileNotFoundError, ValueError, KeyError) as error:
            record['attempts'].append({'attempt': path.name, 'complete': False, 'error': str(error)})
            continue
        stored_path = path / ('gate-V-failed.json' if result.get('gate_V_failed') else 'gate-arithmetic.json')
        if not stored_path.exists():
            record['attempts'].append({'attempt': path.name, 'complete': False,
                                       'error': f'{stored_path.name} missing: the attempt did not finish'})
            continue
        if not result.get('gate_V_failed') and not same_cells(read_json(stored_path), normal(result)):
            warnings.append(f'{model}/{round_dir.name}/{path.name}: recomputed checks differ from {stored_path.name}')
        quiet = None if result.get('gate_V_failed') else result['quiet_gate']['pass']
        record['attempts'].append({'attempt': path.name, 'complete': True,
                                   'gate_V_failed': bool(result.get('gate_V_failed')), 'quiet_pass': quiet})
        used = (path.name, result)
        if result.get('gate_V_failed') or quiet:
            break
    if record['stopped'] is not None or used is None:
        return None, record
    record['used_attempt'] = used[0]
    return used[1], record


def model_rounds(run_dir, model, sockets, warnings):
    root = run_dir / model
    info = {'model': model, 'repo': MODELS[model][0], 'revision': MODELS[model][1], 'status': None,
            'rounds': [], 'records': []}
    if not root.exists():
        info['status'] = 'not started'
        return info
    if (root / 'not-measured-start-test.json').exists():
        info['status'] = 'not measured: gate V failed in the start test'
        return info
    for i in range(1, ROUNDS + 1):
        round_dir = root / f'round-{i}'
        if not round_dir.exists():
            break
        result, record = read_round(round_dir, model, sockets, warnings)
        info['records'].append(record)
        if result is None:
            break
        if result.get('gate_V_failed'):
            info['status'] = f'not measured: gate V failed in {round_dir.name}'
            break
        info['rounds'].append(result)
    if info['status'] is None:
        info['status'] = 'complete' if len(info['rounds']) == ROUNDS else f'{len(info["rounds"])} of {ROUNDS} rounds'
    return info


def reference_rounds(ref_dir, model):
    """Rounds of a reference run: recomputed from its raw files when they are present, else the
    per-round results it stored in <model>/rounds.json."""
    ref_dir = Path(ref_dir)
    manifest = ref_dir / 'manifest.json'
    if manifest.exists():
        info = model_rounds(ref_dir, model, read_json(manifest)['topology']['sockets'], [])
        if info['rounds']:
            return info['rounds'], 'recomputed from raw files'
    stored = ref_dir / model / 'rounds.json'
    if stored.exists():
        return [r for r in read_json(stored) if 'cells' in r], 'stored rounds.json'
    return [], 'absent'


def rel(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / abs(b) if b else (0. if a == b else float('inf'))


def compare(report, ref_dir):
    """Cell-by-cell comparison of this run with a reference run: same round, cell and denominator."""
    out = {'reference': str(ref_dir), 'models': {}}
    for model, info in report['models'].items():
        ref, source = reference_rounds(ref_dir, model)
        rows = []
        for i, (mine, theirs) in enumerate(zip(info['rounds'], ref), 1):
            for key, cell in mine['cells'].items():
                other = theirs['cells'].get(key)
                if other is None:
                    continue
                a, b = projection(cell), projection(other)
                row = {'round': i, 'cell': key,
                       'checks_equal': a['gates'] == b['gates'],
                       'g_rel': rel(a.get('g_ms'), b.get('g_ms')), 'gp_rel': rel(a.get('gp_ms'), b.get('gp_ms')),
                       'h_rel': rel((a['components_ms'] or {}).get('h'), (b['components_ms'] or {}).get('h'))}
                for name, f in (('H', lambda s: s['H']), ('MS', lambda s: s['M'] + s['S']), ('T', lambda s: s['T'])):
                    row[name] = [f(s) for s in a['shares']] if a['shares'] else None
                    row[name + '_ref'] = [f(s) for s in b['shares']] if b['shares'] else None
                row['R'], row['R_ref'] = a.get('residual'), b.get('residual')
                rows.append(row)
        out['models'][model] = {'source': source, 'rounds_compared': min(len(info['rounds']), len(ref)),
                                'cells': rows}
    ref_rounds, _ = reference_rounds(ref_dir, DECISION_MODEL)
    out['reference_decision'] = decision([r['cells']['PLAIN-1'] for r in ref_rounds],
                                         [r['cells']['JSON-1'] for r in ref_rounds])
    return out


def compare_text(cmp):
    out = ['', f"Comparison with {cmp['reference']} (x/y: over g / over g'; rel: |this - reference| / reference)"]
    for model, info in cmp['models'].items():
        out.append(f"== {model}: {info['rounds_compared']} rounds compared (reference: {info['source']})")
        for r in info['cells']:
            def pair(v):
                return '-' if v is None else '/'.join(f'{100 * x:.1f}%' for x in v)
            out.append(f"  {r['round']} {r['cell']:8} H {pair(r['H'])} vs {pair(r['H_ref'])}  "
                       f"M+S {pair(r['MS'])} vs {pair(r['MS_ref'])}  R {pct(r['R']).strip()} vs {pct(r['R_ref']).strip()}"
                       f"  g rel {fmt(r['g_rel'], 4)}  h rel {fmt(r['h_rel'], 4)}"
                       f"  checks {'same' if r['checks_equal'] else 'DIFFER'}")
    d = cmp['reference_decision']
    out.append(f"Reference decision: head {d['head']}, fusion {d['fusion']}")
    return '\n'.join(out)


def build(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / 'manifest.json')
    if manifest.get('step') != 'rounds':
        raise ValueError(f'{run_dir} is not a rounds directory')
    sockets = manifest['topology']['sockets']
    warnings = []
    models = {m: model_rounds(run_dir, m, sockets, warnings) for m in MODELS}
    rounds = models[DECISION_MODEL]['rounds']
    verdict = decision([r['cells']['PLAIN-1'] for r in rounds], [r['cells']['JSON-1'] for r in rounds])
    if (run_dir / 'decision.json').exists() and \
            {k: v for k, v in read_json(run_dir / 'decision.json').items() if k in ('head', 'fusion')} != \
            {'head': verdict['head'], 'fusion': verdict['fusion']}:
        warnings.append('recomputed decision differs from decision.json')
    notes = []
    if (run_dir / 'STOPPED.json').exists():
        notes.append('the run stopped early: ' + json.dumps(read_json(run_dir / 'STOPPED.json')))
    if not (run_dir / 'DONE').exists():
        notes.append('the run has no DONE marker: it is unfinished or was interrupted')
    return {'run': str(run_dir), 'gpu': manifest.get('gpu'), 'gpu_state': manifest.get('gpu_state'),
            'versions': manifest.get('overlay', {}).get('installed_identity'),
            'package_deviations': manifest.get('package_deviations'), 'models': models,
            'decision_model': DECISION_MODEL, 'decision': verdict, 'warnings': warnings, 'notes': notes}


def pct(x):
    return '   -  ' if x is None else f'{100 * x:5.1f}%'


def fmt(x, digits=3):
    return '-' if x is None else f'{x:.{digits}f}'


def gate_text(cell):
    failed = [g for g in NEEDED[cell['condition']] if not cell['gates'].get(g, {}).get('pass')]
    return 'all pass' if not failed else 'FAIL ' + ','.join(failed)


def share_pair(cell, key):
    shares = cell.get('shares')
    if not shares:
        return '      -       '
    if key == 'MS':
        a, b = (s['M'] + s['S'] for s in shares)
    else:
        a, b = (s[key] for s in shares)
    return f'{pct(a)}/{pct(b)}'


def text(report):
    out = [f"Run: {report['run']}  (GPU {report['gpu']})"]
    v = report.get('versions') or {}
    out.append(f"vLLM {v.get('vllm')}, torch {v.get('torch')}, CUDA {v.get('cuda')}")
    dev = report.get('package_deviations')
    if dev is None:
        out.append('Packages: not recorded in the manifest')
    elif not dev:
        out.append('Packages: as in requirements-cu129.lock.txt')
    else:
        out.append(f'Packages: {len(dev)} differ from requirements-cu129.lock.txt:')
        out.extend('  ' + line for line in dev)
    out.append('Shares are given as x/y: x over the unpatched token gap g, y over the patched gap g\'.')
    for model, info in report['models'].items():
        out.append('')
        out.append(f"== {model}: {info['repo']}@{info['revision'][:8]}  [{info['status']}]"
                   f"  head byte share {pct(HEAD_BYTE_SHARE[model]).strip()}")
        for record in info['records']:
            if record['used_attempt'] is None:
                why = 'stopped' if record['stopped'] else 'no finished attempt'
                out.append(f"  {record['round']}: not used ({why}); attempts: "
                           + '; '.join(f"{a['attempt']} " + ('complete' if a['complete'] else a['error'])
                                       for a in record['attempts']))
        if not info['rounds']:
            continue
        out.append('  round  cell      g ms   g\' ms   p ms   f ms   h ms   m ms  s1 ms  s2 ms        H'
                   '             M+S          T        R      checks')
        for i, (result, record) in enumerate(zip(info['rounds'], info['records']), 1):
            for c in CONCURRENCIES:
                for condition in ('PLAIN', 'JSON'):
                    cell = result['cells'][f'{condition}-{c}']
                    k = cell.get('components_ms') or {}
                    out.append(f"  {i} {record['used_attempt'][-1]:>2}  {condition:5}-{c:<2} "
                               f"{fmt(cell.get('g_ms'), 2):>6} {fmt(cell.get('gp_ms'), 2):>6} "
                               f"{fmt(cell.get('p_ms'), 2):>6} "
                               + ' '.join(f"{fmt(k.get(x)):>6}" for x in ('f', 'h', 'm', 's1', 's2'))
                               + f"  {share_pair(cell, 'H')} {share_pair(cell, 'MS')} {share_pair(cell, 'T')}"
                               f" {pct(cell.get('residual'))}  {gate_text(cell)}")
        plain = [r['cells']['PLAIN-1'] for r in info['rounds'] if r['cells']['PLAIN-1'].get('shares')]
        grammar = [r['cells']['JSON-1'] for r in info['rounds'] if r['cells']['JSON-1'].get('shares')]
        if plain:
            hs = [s['H'] for c in plain for s in c['shares']]
            out.append(f"  c=1 PLAIN over rounds: H {pct(min(hs))}-{pct(max(hs))}, "
                       f"R {pct(min(c['residual'] for c in plain))}-{pct(max(c['residual'] for c in plain))}, "
                       f"median g {median(c['g_ms'] for c in plain):.3f} ms, "
                       f"median h {median(c['components_ms']['h'] for c in plain):.3f} ms, "
                       f"median head alone {median(c['head_alone_ms'] for c in plain if 'head_alone_ms' in c):.3f} ms"
                       if all('head_alone_ms' in c for c in plain) else
                       f"  c=1 PLAIN over rounds: H {pct(min(hs))}-{pct(max(hs))}")
        if grammar:
            ms = [s['M'] + s['S'] for c in grammar for s in c['shares']]
            out.append(f"  c=1 JSON over rounds: M+S {pct(min(ms))}-{pct(max(ms))}, "
                       f"R {pct(min(c['residual'] for c in grammar))}-{pct(max(c['residual'] for c in grammar))}")
    d = report['decision']
    out.append('')
    out.append(f"Decision ({report['decision_model']}, concurrency 1, {ROUNDS} rounds):")
    out.append(f"  head question:   {d['head']}   (BIG: H >= {HEAD_BIG} with both denominators and "
               f"R <= {RESIDUAL_MAX} in every round; SMALL: H < {HEAD_BIG} in every round)")
    out.append(f"  fusion question: {d['fusion']}   (BIG: M+S >= {FUSION_BIG} and R <= {RESIDUAL_MAX} in every round;"
               f" SMALL: M+S < {FUSION_BIG} in every round)")
    for outcome, meaning in zip(d['outcomes'], d['meaning']):
        out.append(f"  -> {outcome}: {meaning}")
    out.append(f"  expectation written before measuring (H 0.10-0.25, fusion SMALL): "
               f"{'met' if d['expectation_met'] else 'not met'}")
    for note in report['notes']:
        out.append('NOTE: ' + note)
    for warning in report['warnings']:
        out.append('WARNING: ' + warning)
    return '\n'.join(out)
