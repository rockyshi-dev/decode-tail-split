"""Command line: python -m tailsplit {check,overlay,prompts,start-test,rounds,report} ... (see --help)."""
import argparse
import json
import os
from pathlib import Path
import sys

from . import spec


def default_hf_home():
    return Path(os.environ.get('HF_HOME') or Path.home() / '.cache' / 'huggingface')


def add_models(p):
    p.add_argument('--hf-home', type=Path, default=default_hf_home(),
                   help='Hugging Face home holding hub/models--*/snapshots/<revision> (default: $HF_HOME, '
                        'else ~/.cache/huggingface)')


def default_cache_dir():
    base = os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache'
    return Path(base) / 'tailsplit' / 'runtime-cache'


def add_run(p):
    add_models(p)
    p.add_argument('--gpu', type=int, required=True, help='GPU index as nvidia-smi numbers it (PCI bus order)')
    p.add_argument('--overlay', type=Path, required=True, help='directory written by the overlay command')
    p.add_argument('--output', type=Path, required=True, help='new directory for this run (must not exist)')
    p.add_argument('--port', type=int, default=8155, help='server port (default 8155)')
    p.add_argument('--cache-dir', type=Path, default=default_cache_dir(),
                   help='compile and kernel caches of the servers; rounds must use the directory their start '
                        'test used (default: $XDG_CACHE_HOME/tailsplit/runtime-cache, else '
                        '~/.cache/tailsplit/runtime-cache)')
    p.add_argument('--lock', type=Path, action='append',
                   help='lock file held for the whole run; repeatable (default: /tmp/tailsplit-gpu<N>.lock)')


def config(a):
    from .launch import Config
    return Config(gpu=a.gpu, hf_home=a.hf_home, cache_dir=a.cache_dir,
                  overlay=a.overlay, port=a.port)


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m tailsplit', description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check', help='check the vLLM installation and that the four checkpoints are present')
    add_models(c)
    o = sub.add_parser('overlay', help='build the timing patch against the installed vLLM runner')
    o.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('prompts', help='build the 256-token prompts of each checkpoint (CPU only)')
    add_models(q)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--model', choices=spec.MODELS, action='append', help='default: all four')
    s = sub.add_parser('start-test', help='start, inspect (gate V) and stop both servers of each checkpoint')
    add_run(s)
    r = sub.add_parser('rounds', help='five measured rounds of each checkpoint, then the decision')
    add_run(r)
    r.add_argument('--prompts', type=Path, required=True, help='directory written by the prompts command')
    r.add_argument('--start-tests', type=Path, required=True, help='output directory of a finished start-test')
    t = sub.add_parser('report', help='recompute every check, share and the verdict from a rounds directory')
    t.add_argument('run', type=Path)
    t.add_argument('--json', type=Path, help='also write the numbers as JSON')
    t.add_argument('--reference', type=Path,
                   help='another run directory to compare with, cell by cell (its raw files are recomputed; '
                        'if it holds only <model>/rounds.json, those stored results are used)')
    a = p.parse_args(argv)

    if a.cmd == 'check':
        from .environment import installed_identity
        print(json.dumps(installed_identity(), indent=2))
        missing = 0
        for model, (repo, revision) in spec.MODELS.items():
            path = spec.snapshot(a.hf_home, model)
            ok = (path / 'config.json').is_file()
            missing += not ok
            print(f"{'ok     ' if ok else 'MISSING'} {repo}@{revision}  {path}")
        from .environment import package_deviations
        deviations = package_deviations()
        for line in deviations:
            print('package differs from requirements-cu129.lock.txt: ' + line)
        if not deviations:
            print('all packages as in requirements-cu129.lock.txt')
        return 1 if missing else 0
    if a.cmd == 'overlay':
        from .overlay import build
        print(build(a.output))
        return 0
    if a.cmd == 'prompts':
        from .prompts import construct
        for model in a.model or spec.MODELS:
            path = construct(model, a.hf_home, a.output)
            print(f'{model}: {spec.PROMPT_COUNT} prompts of {spec.PROMPT_TOKENS} tokens -> {path}')
        return 0
    if a.cmd in ('start-test', 'rounds'):
        from .runner import Run, execute
        run = Run(config(a), getattr(a, 'prompts', None) or a.output, a.output,
                  a.lock or [Path('/tmp') / f'tailsplit-gpu{a.gpu}.lock'])
        execute(run, a.cmd, getattr(a, 'start_tests', None))
        return 0
    if a.cmd == 'report':
        from .report import build, text
        try:
            report = build(a.run)
        except (FileNotFoundError, ValueError) as error:
            print(f'error: {error}', file=sys.stderr)
            return 2
        print(text(report))
        if a.reference:
            from .report import compare, compare_text
            report['comparison'] = compare(report, a.reference)
            print(compare_text(report['comparison']))
        if a.json:
            spec.write_json(a.json, report)
        return 1 if report['warnings'] else 0


if __name__ == '__main__':
    sys.exit(main())
