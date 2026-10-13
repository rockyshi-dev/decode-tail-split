"""Timing patch, import hook and GPU-event collector, without vLLM, torch or a GPU."""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tailsplit import instrument, launch, loader, overlay, spec

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
MARKERS = ['begin_step', 'mark("forward_end")', 'mark("head_start")', 'mark("head_end")', 'mark("mask_end")',
           'sampling_hooks', 'set_mask']


class Patch(unittest.TestCase):
    def test_markers_on_fixture(self):
        source = (FIXTURES / 'runner_anchors.txt').read_text()
        patched = overlay.patch_source(source)
        for marker in MARKERS:
            self.assertEqual(patched.count(marker + ('(' if marker in ('begin_step', 'set_mask', 'sampling_hooks') else '')),
                             1, marker)
        lines = patched.splitlines()
        def at(text):
            return next(i for i, line in enumerate(lines) if text in line)
        self.assertLess(at('mark("head_start")'), at('compute_logits'))
        self.assertLess(at('compute_logits'), at('mark("head_end")'))
        self.assertLess(at('apply_grammar_bitmask'), at('mark("mask_end")'))
        self.assertLess(at('begin_step('), at('# Run model.'))
        self.assertLess(at('mark("forward_end")'), at('finish_forward'))
        removed = [line for line in source.splitlines() if line not in lines]
        self.assertEqual(removed, ['            sampler_output = self.sampler(logits, input_batch)'])

    def test_anchor_must_occur_once(self):
        source = (FIXTURES / 'runner_anchors.txt').read_text()
        with self.assertRaisesRegex(RuntimeError, 'exactly once'):
            overlay.patch_source(source.replace('        # Run model.\n', ''))
        with self.assertRaisesRegex(RuntimeError, 'exactly once'):
            overlay.patch_source(source + '        # Run model.\n')

    @unittest.skipUnless(os.environ.get('TAILSPLIT_VLLM_RUNNER'), 'set TAILSPLIT_VLLM_RUNNER to the pinned runner file')
    def test_pinned_runner(self):
        path = Path(os.environ['TAILSPLIT_VLLM_RUNNER'])
        self.assertEqual(spec.sha(path), spec.RUNNER_SHA256)
        source = path.read_text()
        patched = overlay.patch_source(source)
        self.assertEqual(len(patched.splitlines()) - len(source.splitlines()), 9)

    def test_build_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / 'model_runner.py'
            runner.write_text((FIXTURES / 'runner_anchors.txt').read_text())
            identity = {'runner_path': str(runner), 'vllm': spec.VLLM_VERSION}
            with patch.object(spec, 'RUNNER_SHA256', spec.sha(runner)):
                diff = overlay.build(root / 'ov', identity)
                self.assertIn('+        mark("forward_end")', diff)
                hashes = overlay.overlay_hashes(root / 'ov')
                self.assertEqual(set(hashes), set(overlay.FILES))
                (root / 'ov' / 'model_runner.py').write_text('changed')
                with self.assertRaisesRegex(RuntimeError, 'do not match'):
                    overlay.overlay_hashes(root / 'ov')
            with self.assertRaisesRegex(RuntimeError, 'pinned runner'):
                overlay.overlay_hashes(root / 'ov')


class FakeVllm:
    """A tiny package laid out like vllm's worker modules, plus an overlay built for it."""

    def __init__(self, root):
        self.site = Path(root) / 'site'
        gpu = self.site / 'vllm' / 'v1' / 'worker' / 'gpu'
        gpu.mkdir(parents=True)
        for d in (self.site / 'vllm', self.site / 'vllm' / 'v1', gpu.parent, gpu):
            (d / '__init__.py').write_text('')
        self.runner = gpu / 'model_runner.py'
        self.runner.write_text('VALUE = "original"\n')
        (gpu.parent / 'gpu_worker.py').write_text('class Worker:\n    pass\n')
        self.overlay = Path(root) / 'overlay'
        self.overlay.mkdir()
        (self.overlay / 'model_runner.py').write_text('VALUE = "patched"\n')
        (self.overlay / 'timing.patch').write_text('diff\n')
        self.identity = {'prefix': str(root), 'runner_path': str(self.runner.resolve())}
        spec.write_json(self.overlay / 'overlay.json', {
            'commit': spec.VLLM_COMMIT, 'installed_identity': self.identity, 'original_sha256': spec.sha(self.runner),
            'patched_sha256': spec.sha(self.overlay / 'model_runner.py'),
            'diff_sha256': spec.sha(self.overlay / 'timing.patch')})

    def load(self, mode, hashes=None):
        import json
        names = [n for n in sys.modules if n == 'vllm' or n.startswith('vllm.')]
        for n in names:
            del sys.modules[n]
        finder = loader.Finder()
        with patch.object(spec, 'RUNNER_SHA256', spec.sha(self.runner)), \
             patch('tailsplit.environment.installed_identity', return_value=self.identity), \
             patch.dict(os.environ, {'TAILSPLIT_MODE': mode, 'TAILSPLIT_OVERLAY': str(self.overlay),
                                     'TAILSPLIT_OVERLAY_HASHES': json.dumps(
                                         hashes if hashes is not None else
                                         spec.read_json(self.overlay / 'overlay.json') and
                                         overlay.overlay_hashes(self.overlay))}), \
             patch.object(sys, 'path', [str(self.site)] + sys.path), \
             patch.object(sys, 'meta_path', [finder] + sys.meta_path):
            try:
                runner = importlib.import_module('vllm.v1.worker.gpu.model_runner')
                worker = importlib.import_module('vllm.v1.worker.gpu_worker')
                return runner.VALUE, getattr(worker.Worker, 'tailsplit_control', None)
            finally:
                for n in [n for n in sys.modules if n == 'vllm' or n.startswith('vllm.')]:
                    del sys.modules[n]


class Loader(unittest.TestCase):
    def test_baseline_runs_original_and_patched_runs_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeVllm(tmp)
            with patch.object(spec, 'RUNNER_SHA256', spec.sha(fake.runner)):
                hashes = overlay.overlay_hashes(fake.overlay)
            value, control = fake.load('baseline', hashes)
            self.assertEqual(value, 'original')
            self.assertIs(control, instrument.worker_control)
            value, control = fake.load('patched', hashes)
            self.assertEqual(value, 'patched')
            self.assertIs(control, instrument.worker_control)

    def test_changed_overlay_or_runner_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeVllm(tmp)
            with patch.object(spec, 'RUNNER_SHA256', spec.sha(fake.runner)):
                hashes = overlay.overlay_hashes(fake.overlay)
            with self.assertRaisesRegex(RuntimeError, 'recorded at launch'):
                fake.load('patched', dict(hashes, **{'timing.patch': '0' * 64}))
            fake.identity = dict(fake.identity, runner_path='/elsewhere/model_runner.py')
            with self.assertRaisesRegex(RuntimeError, 'not the installation'):
                fake.load('baseline', hashes)

    def test_sitecustomize_is_inert_without_mode(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('TAILSPLIT_')}
        env['PYTHONPATH'] = f'{launch.SITE}{os.pathsep}{launch.REPO}'
        code = 'import sys; print(sum(type(f).__name__ == "Finder" for f in sys.meta_path))'
        out = subprocess.check_output([sys.executable, '-c', code], env=env, text=True)
        self.assertEqual(out.strip(), '0')
        out = subprocess.check_output([sys.executable, '-c', code], env=dict(env, TAILSPLIT_MODE='baseline'), text=True)
        self.assertEqual(out.strip(), '1')


class Collector(unittest.TestCase):
    def fake_torch(self, trace):
        class Event:
            def __init__(self, **kwargs):
                pass
            def record(self):
                trace.append('record')
            def elapsed_time(self, other):
                trace.append('read')
                return 1.
        stream = SimpleNamespace(cuda_stream=0)
        cuda = SimpleNamespace(Event=Event, current_stream=lambda: stream, default_stream=lambda: stream,
                               synchronize=lambda: trace.append('synchronize'))
        return SimpleNamespace(cuda=cuda)

    def batch(self, ids):
        return SimpleNamespace(req_ids=ids, num_reqs=len(ids),
                               is_prefilling_np=SimpleNamespace(tolist=lambda: [False] * len(ids)),
                               num_scheduled_tokens=SimpleNamespace(tolist=lambda: [1] * len(ids)))

    def test_events_read_only_after_idle(self):
        trace = []
        collector = instrument.Collector(self.fake_torch(trace))
        collector.armed = True
        collector.begin(self.batch(['x']), False)
        for name in ('forward_end', 'head_start', 'head_end', 'mask_end', 'params_end', 'draw_end'):
            collector.mark(name)
        self.assertNotIn('synchronize', trace)
        self.assertNotIn('read', trace)
        rows = collector.drain()
        self.assertEqual(trace[7], 'synchronize')
        self.assertEqual(set(rows[0]['segments_ms']), {'f', 'h', 'm', 's1', 's2'})
        self.assertIsNone(rows[0]['period_ms'])
        self.assertFalse(collector.armed)

    def test_period_and_step_counts(self):
        trace = []
        collector = instrument.Collector(self.fake_torch(trace))
        collector.armed = True
        for _ in range(2):
            collector.begin(self.batch(['x', 'y']), False)
            for name in ('forward_end', 'head_start', 'head_end', 'mask_end', 'params_end', 'draw_end'):
                collector.mark(name)
        collector.begin(self.batch(['x']), True)   # dummy runs are not recorded
        rows = collector.drain()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['request_steps'], {'x': 2, 'y': 2})
        self.assertEqual(rows[0]['next_request_steps'], {'x': 2, 'y': 2})
        self.assertEqual(rows[0]['period_ms'], 1.)

    def test_repeated_marker_and_unarmed(self):
        collector = instrument.Collector(self.fake_torch([]))
        collector.begin(self.batch(['x']), False)
        self.assertEqual(collector.rows, [])   # not armed: nothing recorded
        collector.armed = True
        collector.begin(self.batch(['x']), False)
        collector.mark('head_start')
        with self.assertRaisesRegex(RuntimeError, 'Repeated'):
            collector.mark('head_start')

    def test_empty_grammar_object_cannot_pass(self):
        state = SimpleNamespace(current={})
        batch = SimpleNamespace(req_ids=['x'], cu_num_logits_np=[0, 1])
        grammar = SimpleNamespace(structured_output_request_ids=[], grammar_bitmask=SimpleNamespace(shape=(0, 10)))
        with patch.object(instrument, '_collector', state):
            instrument.set_mask(grammar, batch)
        self.assertFalse(state.current['mask_active'])
        grammar.structured_output_request_ids = ['x']
        grammar.grammar_bitmask.shape = (1, 10)
        with patch.object(instrument, '_collector', state):
            instrument.set_mask(grammar, batch)
        self.assertTrue(state.current['mask_active'])

    def test_sampler_hooks_restore_on_error(self):
        class Sampler:
            def apply_sampling_params(self):
                return 1
            def _sample_random(self):
                raise ValueError('draw failed')
        sampler = Sampler()
        marks = []
        with patch.object(instrument, '_collector', SimpleNamespace(current={}, mark=marks.append)):
            with self.assertRaises(ValueError):
                with instrument.sampling_hooks(sampler):
                    self.assertEqual(sampler.apply_sampling_params(), 1)
                    sampler._sample_random()
        self.assertEqual(sampler.__dict__, {})
        self.assertEqual(marks, ['params_end'])

    def test_worker_control_refusals(self):
        with self.assertRaisesRegex(RuntimeError, 'attestation'):
            instrument.worker_control(None, 'arm', 'wrong')
        fake = SimpleNamespace(cuda=None)
        with patch.dict(sys.modules, {'torch': fake}), patch.dict(os.environ, {'TAILSPLIT_MODE': 'baseline'}):
            with self.assertRaisesRegex(RuntimeError, 'baseline'):
                instrument.worker_control(SimpleNamespace(model_runner=None), 'arm',
                                          'engine-paused-wait-clear_cache-false')


if __name__ == '__main__':
    unittest.main()
