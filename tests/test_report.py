"""report: verdict recomputed from raw files of a synthetic rounds directory."""
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import shutil
import tempfile
import unittest

from tailsplit import report, spec
from tailsplit.__main__ import main
from tailsplit import analysis
from tests.helpers import CachedAnalysis, World, write_run

_original = analysis.analyze_attempt


def setUpModule():
    analysis.analyze_attempt = CachedAnalysis(_original)


def tearDownModule():
    analysis.analyze_attempt = _original


def run_report(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(['report', *map(str, argv)])
    return code, out.getvalue(), err.getvalue()


class Report(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.base = write_run(Path(cls.tmp) / 'base')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def copy(self, name):
        target = Path(self.tmp) / name
        shutil.copytree(self.base, target)
        return target

    def test_head_big_fusion_small(self):
        code, out, _ = run_report(self.base, '--json', Path(self.tmp) / 'r.json')
        self.assertEqual(code, 0)
        self.assertIn('head question:   BIG', out)
        self.assertIn('fusion question: SMALL', out)
        self.assertIn('expectation written before measuring (H 0.10-0.25, fusion SMALL): met', out)
        self.assertIn('Packages: not recorded in the manifest', out)
        r = spec.read_json(Path(self.tmp) / 'r.json')
        self.assertEqual(r['decision']['outcomes'], ['head BIG'])
        self.assertEqual(r['models']['qwen25-w4']['status'], 'complete')
        self.assertEqual(r['models']['qwen3-bf16']['status'], 'not started')
        cell = r['models']['qwen25-w4']['rounds'][0]['cells']['PLAIN-1']
        self.assertAlmostEqual(cell['shares'][0]['H'], 1.15 / 6.0)
        self.assertAlmostEqual(cell['shares'][1]['H'], 1.15 / 6.05)

    def test_tampered_stored_result_warns(self):
        run = self.copy('tampered')
        path = run / 'qwen25-w4' / 'round-3' / 'attempt-1' / 'gate-arithmetic.json'
        stored = spec.read_json(path)
        stored['cells']['PLAIN-1']['shares'][0]['H'] = .5
        spec.write_json(path, stored)
        code, out, _ = run_report(run)
        self.assertEqual(code, 1)
        self.assertIn('round-3/attempt-1: recomputed checks differ', out)

    def test_extra_checks_in_stored_result_are_ignored(self):
        run = self.copy('extra')
        path = run / 'qwen25-w4' / 'round-1' / 'attempt-1' / 'gate-arithmetic.json'
        stored = spec.read_json(path)
        stored['cells']['PLAIN-1']['gates']['B'] = {'pass': False, 'reasons': ['not applicable']}
        spec.write_json(path, stored)
        self.assertEqual(run_report(run)[0], 0)

    def test_package_deviations_shown(self):
        run = self.copy('packages')
        manifest = spec.read_json(run / 'manifest.json')
        manifest['package_deviations'] = ['xgrammar: installed 0.2.9, lock 0.2.8']
        spec.write_json(run / 'manifest.json', manifest)
        code, out, _ = run_report(run)
        self.assertIn('Packages: 1 differ from requirements-cu129.lock.txt:', out)
        self.assertIn('  xgrammar: installed 0.2.9, lock 0.2.8', out)

    def test_missing_round_is_not_measured(self):
        run = self.copy('four')
        shutil.rmtree(run / 'qwen25-w4' / 'round-5')
        code, out, _ = run_report(run)
        self.assertIn('head question:   NOT MEASURED', out)
        self.assertIn('[4 of 5 rounds]', out)

    def test_unfinished_attempt_is_not_used(self):
        run = self.copy('unfinished')
        (run / 'qwen25-w4' / 'round-2' / 'attempt-1' / 'gate-arithmetic.json').unlink()
        (run / 'DONE').unlink()
        code, out, _ = run_report(run)
        self.assertIn('round-2: not used (no finished attempt)', out)
        self.assertIn('NOTE: the run has no DONE marker', out)
        self.assertIn('NOT MEASURED', out)

    def test_quiet_rerun_uses_the_passing_attempt(self):
        worlds = [World() for _ in range(5)]
        worlds[1] = [World(foreign_pid=99), World()]
        run = write_run(Path(self.tmp) / 'rerun', {'qwen25-w4': worlds})
        r = report.build(run)
        self.assertEqual(r['models']['qwen25-w4']['records'][1]['used_attempt'], 'attempt-2')
        self.assertEqual(r['decision']['head'], 'BIG')
        worlds[1] = [World(clock_dip=True)] * 3
        run = write_run(Path(self.tmp) / 'rerun-fail', {'qwen25-w4': worlds})
        r = report.build(run)
        self.assertEqual(r['models']['qwen25-w4']['records'][1]['used_attempt'], 'attempt-3')
        self.assertEqual(r['decision']['head'], 'NOT MEASURED')

    def test_mixed_and_fusion_big(self):
        run = write_run(Path(self.tmp) / 'mixed', {'qwen25-w4': [World(h=.8)] + [World() for _ in range(4)]})
        r = report.build(run)
        self.assertEqual(r['decision']['head'], 'NOT MEASURED')   # h far from head alone: gate X fails
        run = write_run(Path(self.tmp) / 'mixed2', {'qwen25-w4': [World(h=.8, head_ms=.8)] + [World() for _ in range(4)]})
        r = report.build(run)
        self.assertEqual(r['decision']['head'], 'MIXED')          # H = 0.133 in one round, 0.19 in four
        self.assertEqual(r['decision']['outcomes'], ['none'])
        run = write_run(Path(self.tmp) / 'fusion', {'qwen25-w4': [World(m_json=.25, s2=.5) for _ in range(5)]})
        r = report.build(run)
        # M + S = 0.77 ms of 6.05 ms, but f+h+m+s1+s2 exceeds 1.02 p: accounting fails, not a BIG
        self.assertEqual(r['decision']['fusion'], 'NOT MEASURED')

    def test_stopped_round(self):
        run = self.copy('stopped')
        spec.write_json(run / 'qwen25-w4' / 'round-4' / 'STOPPED.json', {'round': 'round-4'})
        spec.write_json(run / 'STOPPED.json', {'round': 'round-4'})
        code, out, _ = run_report(run)
        self.assertIn('round-4: not used (stopped)', out)
        self.assertIn('NOTE: the run stopped early', out)
        self.assertIn('NOT MEASURED', out)

    def test_not_a_rounds_directory(self):
        code, _, err = run_report(Path(self.tmp) / 'nothing')
        self.assertEqual(code, 2)
        run = Path(self.tmp) / 'start'
        run.mkdir()
        spec.write_json(run / 'manifest.json', {'step': 'start-test'})
        self.assertEqual(run_report(run)[0], 2)

    def test_reference_comparison(self):
        other = write_run(Path(self.tmp) / 'other', {'qwen25-w4': [World(h=1.2) for _ in range(5)]})
        code, out, _ = run_report(self.base, '--reference', other)
        self.assertIn('Comparison with', out)
        self.assertIn('5 rounds compared (reference: recomputed from raw files)', out)
        cmp = report.compare(report.build(self.base), other)
        row = cmp['models']['qwen25-w4']['cells'][0]
        self.assertAlmostEqual(row['h_rel'], .05 / 1.2)
        self.assertTrue(row['checks_equal'])
        self.assertEqual(cmp['reference_decision']['head'], 'BIG')
        # A reference holding only stored per-round results
        stored = Path(self.tmp) / 'stored'
        (stored / 'qwen25-w4').mkdir(parents=True)
        rounds = [spec.read_json(other / 'qwen25-w4' / f'round-{i}' / 'attempt-1' / 'gate-arithmetic.json')
                  for i in range(1, 6)]
        spec.write_json(stored / 'qwen25-w4' / 'rounds.json', rounds)
        cmp = report.compare(report.build(self.base), stored)
        self.assertEqual(cmp['models']['qwen25-w4']['source'], 'stored rounds.json')
        self.assertEqual(cmp['models']['qwen25-w4']['rounds_compared'], 5)


if __name__ == '__main__':
    unittest.main()
