"""Coordinator: rerun rule, early stop, topology, file layout of an attempt (with fake servers)."""
from contextlib import ExitStack, nullcontext
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from tailsplit import analysis, launch, runner, spec
from tests.helpers import SOCKETS, World, evidence, prompt_manifest, server_log


def result(*failures, passed=False):
    return {'quiet_gate': {'pass': passed, 'intervals':
            [{'interval': name, 'pass': False, 'reasons': list(reasons)} for name, reasons in failures]
            + [{'interval': 'patched-PLAIN-1', 'pass': True, 'reasons': []}]}}


class RerunRule(unittest.TestCase):
    def run_round(self, results):
        calls = []
        def attempt(run, model, directory, topo):
            calls.append(directory.name)
            return results[len(calls) - 1]
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / 'round-1'
            round_dir.mkdir()
            try:
                out, stopped = runner.measure_round(None, 'qwen25-w4', round_dir, {}, attempt_fn=attempt), None
            except runner.RunStopped as stop:
                out, stopped = None, spec.read_json(round_dir / 'STOPPED.json')
                self.assertEqual(stop.args[0], stopped)
        return calls, out, stopped

    def test_same_interval_same_reasons_stops_after_two(self):
        same = ('head-c1-b1', ['GPU clock stability failed'])
        calls, out, stopped = self.run_round([result(same), result(same), result(passed=True)])
        self.assertEqual(calls, ['attempt-1', 'attempt-2'])
        self.assertEqual(stopped['repeated_failures'], [['head-c1-b1', ['GPU clock stability failed']]])

    def test_different_failures_allow_third_attempt(self):
        for second in (('head-c8-b8', ['GPU clock stability failed']),
                       ('head-c1-b1', ['Foreign CPU time exceeds 2% on socket 1'])):
            final = result(passed=True)
            calls, out, stopped = self.run_round([result(('head-c1-b1', ['GPU clock stability failed'])),
                                                  result(second), final])
            self.assertEqual(calls, ['attempt-1', 'attempt-2', 'attempt-3'])
            self.assertIs(out, final)
            self.assertIsNone(stopped)

    def test_third_attempt_repeating_second_ends_round_without_stop(self):
        a, b = ('head-c1-b1', ['x']), ('head-c8-b8', ['y'])
        last = result(b)
        calls, out, stopped = self.run_round([result(a), result(b), last])
        self.assertEqual(calls, ['attempt-1', 'attempt-2', 'attempt-3'])
        self.assertIs(out, last)          # the round is kept, with its failed quiet check
        self.assertIsNone(stopped)

    def test_pass_and_gate_v_end_round(self):
        self.assertEqual(self.run_round([result(passed=True)])[0], ['attempt-1'])
        self.assertEqual(self.run_round([{'gate_V_failed': True}])[0], ['attempt-1'])


LSCPU = '''# CPU,Core,Socket,Online
0,0,0,Y
1,1,0,Y
2,2,0,Y
3,3,1,Y
4,4,1,Y
5,5,1,Y
6,0,0,Y
7,1,0,Y
8,2,0,Y
9,3,1,Y
10,4,1,Y
11,5,1,Y
'''


class Topology(unittest.TestCase):
    def sysfs(self, root, node, cpus):
        dev = root / 'bus/pci/devices/0000:81:00.0'
        dev.mkdir(parents=True)
        (dev / 'numa_node').write_text(f'{node}\n')
        n = root / f'devices/system/node/node{node}'
        n.mkdir(parents=True)
        (n / 'cpulist').write_text(cpus + '\n')

    def test_gpu_on_second_socket(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.sysfs(Path(tmp), 1, '3-5,9-11')
            topo = runner.topology(4, LSCPU, '00000000:81:00.0', Path(tmp))
        self.assertEqual(topo['server'], [3, 4, 5, 9, 10, 11])
        self.assertEqual(topo['client'], [0, 6, 1, 7])
        self.assertEqual(topo['logger'], [2, 8])
        self.assertEqual(topo['sockets'], {'0': [0, 1, 2, 6, 7, 8], '1': [3, 4, 5, 9, 10, 11]})

    def test_gpu_on_first_socket(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.sysfs(Path(tmp), 0, '0-2,6-8')
            topo = runner.topology(0, LSCPU, '00000000:81:00.0', Path(tmp))
        self.assertEqual(topo['server'], [0, 1, 2, 6, 7, 8])
        self.assertEqual(topo['client'], [3, 9, 4, 10])

    def test_single_socket_or_unknown_node_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.sysfs(Path(tmp), 0, '0-2')
            one = '\n'.join(l for l in LSCPU.splitlines() if not l.endswith(',1,Y'))
            with self.assertRaisesRegex(RuntimeError, 'two CPU sockets'):
                runner.topology(0, one, '00000000:81:00.0', Path(tmp))
        with tempfile.TemporaryDirectory() as tmp:
            self.sysfs(Path(tmp), 0, '0-4')   # spans both sockets
            with self.assertRaisesRegex(RuntimeError, 'one socket'):
                runner.topology(0, LSCPU, '00000000:81:00.0', Path(tmp))


class FakeServer:
    instances = []

    def __init__(self, model, mode, directory, cfg, cpus):
        self.mode, self.directory, self.port = mode, Path(directory), cfg.port
        self.directory.mkdir(parents=True)
        (self.directory / 'server.log').write_text(server_log(model))
        self.closed = False
        FakeServer.instances.append(self)

    def ready(self):
        pass

    def completed(self):
        self.closed = True

    def request_failed(self, error):
        self.closed = True

    def retain(self, error):
        raise AssertionError('retain must not be used here')


class AttemptLayout(unittest.TestCase):
    """attempt() with fake servers, client, control and head job writes the files analysis reads."""

    def test_attempt_files_match_analysis(self):
        world = World()
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            run = SimpleNamespace(cfg=SimpleNamespace(port=8155, gpu=0, python='python',
                                                      model_path=lambda m: Path('/m')), prompts=root)
            state = {}

            def client(run_, model, condition, concurrency, phase, label, output, topo):
                mode = output.parent.name
                key = f'{condition}-{concurrency}'
                data = (world.probe() if phase == 'schema' else
                        world.records[mode, key] if phase == 'measured' else [])
                spec.write_json(output, data)
                state['key'] = key
                return data

            def control(server, action):
                if action == 'inspect':
                    return evidence()
                if action == 'drain':
                    return {'rows': world.rows[state['key']]}
                return {'armed': True}

            def head(cfg, model, contract, output, cpus):
                c = int(output.name.split('-')[1][1:])
                spec.write_json(output, world.micros[c, 1])

            def telemetry(run_, directory, topo):
                (directory / 'telemetry.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in world.samples))
                return Mock(), Mock(), directory / 'telemetry.stop', directory / 'telemetry.jsonl'

            stack.enter_context(patch.object(runner, 'Server', FakeServer))
            stack.enter_context(patch.object(runner, 'run_client', side_effect=client))
            stack.enter_context(patch.object(runner, 'run_control', side_effect=control))
            heads = stack.enter_context(patch.object(runner, 'run_head', side_effect=head))
            stack.enter_context(patch.object(runner, 'telemetry_start', side_effect=telemetry))
            out = runner.attempt(run, 'qwen25-w4', root / 'attempt-1', {'server': [1], 'sockets': SOCKETS})
            self.assertEqual(heads.call_count, 3)
            contract = spec.read_json(root / 'attempt-1' / 'head-c1-b1-contract.json')
            self.assertEqual(contract, {'shape': [152064, 3584], 'dtype': 'torch.float16', 'batch': 1, 'gap_ms': 6.0})
            self.assertEqual(spec.read_json(root / 'attempt-1' / 'gate-arithmetic.json'), json.loads(json.dumps(out)))
            self.assertEqual(out, analysis.analyze_attempt(root / 'attempt-1', 'qwen25-w4', SOCKETS))
            self.assertTrue(out['quiet_gate']['pass'])
            self.assertTrue(all(c['gates'][g]['pass'] for c in out['cells'].values() for g in c['gates']))

    def test_gate_v_failure_sends_no_request(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            run = SimpleNamespace(cfg=SimpleNamespace(port=8155))
            stack.enter_context(patch.object(runner, 'Server', FakeServer))
            client = stack.enter_context(patch.object(runner, 'run_client'))
            stack.enter_context(patch.object(runner, 'run_control', return_value=dict(evidence(), runner_module='v1')))
            stack.enter_context(patch.object(runner, 'telemetry_start',
                                             return_value=(Mock(), Mock(), root / 'stop', root / 't')))
            out = runner.attempt(run, 'qwen25-w4', root / 'a', {'server': [1], 'sockets': SOCKETS})
            self.assertTrue(out['gate_V_failed'])
            client.assert_not_called()
            self.assertTrue((root / 'a' / 'gate-V-failed.json').exists())
            self.assertEqual(analysis.analyze_attempt(root / 'a', 'qwen25-w4', SOCKETS)['gate_V_failed'], True)


class Failures(unittest.TestCase):
    def test_failed_client_is_not_a_dropped_trial(self):
        for phase in ('warmup', 'schema', 'measured'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = SimpleNamespace(prompts=root, cfg=SimpleNamespace(python='python', port=8155,
                                                                        model_path=lambda m: '/m'))
                with patch.object(runner, 'child', side_effect=subprocess.CalledProcessError(1, ['client'])):
                    with self.assertRaisesRegex(runner.RequestFailure, phase):
                        runner.run_client(run, 'qwen25-w4', 'PLAIN', 1, phase, 'test', root / 'out.json', {'client': [0]})
                self.assertEqual(spec.read_json(root / 'out.failure.json')['phase'], phase)

    def test_client_child_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec.write_json(root / 'out.json', [])
            run = SimpleNamespace(prompts=root, cfg=SimpleNamespace(python='/env/python', port=8155,
                                                                    model_path=lambda m: '/m'))
            with patch.object(runner, 'child') as child:
                runner.run_client(run, 'qwen25-w4', 'JSON', 8, 'measured', 'lbl', root / 'out.json', {'client': [0, 1]})
            argv = [str(x) for x in child.call_args.args[0]]
            self.assertEqual(argv[:4], ['/env/python', '-s', '-m', 'tailsplit.client'])
            self.assertIn('--model-path', argv)
            self.assertEqual(child.call_args.args[1], [0, 1])

    def test_failed_inspect_uses_owned_request_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = Mock(directory=root)
            error = HTTPError('http://127.0.0.1/collective_rpc', 500, 'fixture', {}, None)
            run = SimpleNamespace(cfg=SimpleNamespace(port=8155))
            with patch.object(runner, 'Server', return_value=server) as spawn, \
                 patch.object(runner, 'control', side_effect=error):
                with self.assertRaisesRegex(runner.RequestFailure, 'inspect'):
                    runner.start_test(run, 'qwen25-w4', root, {'server': [4]})
            spawn.assert_called_once()
            server.request_failed.assert_called_once()
            server.retain.assert_not_called()

    def test_unexpected_error_retains_server(self):
        server = Mock()
        runner.handle_failure(server, KeyError('bug'))
        server.retain.assert_called_once()
        server.request_failed.assert_not_called()


class Execute(unittest.TestCase):
    def setup_run(self, root):
        cfg = launch.Config(gpu=0, hf_home=root / 'hf', cache_dir=root / 'cache', overlay=root / 'ov')
        (root / 'prompts').mkdir()
        for model in spec.MODELS:
            snap = cfg.model_path(model)
            snap.mkdir(parents=True)
            (snap / 'config.json').write_text('{}')
            spec.write_json(root / 'prompts' / f'{model}.json', prompt_manifest(model))
        starts = root / 'starts'
        for model in spec.MODELS:
            (starts / model).mkdir(parents=True)
            spec.write_json(starts / model / 'start-test.json', {m: {'gate_V': {'pass': True}} for m in ('baseline', 'patched')})
        spec.write_json(starts / 'manifest.json', {'step': 'start-test', 'code_sha256': launch.code_hashes(),
                        'overlay_sha256': {'o': 1}, 'models': {k: list(v) for k, v in spec.MODELS.items()},
                        'cache_dir': str((root / 'cache').resolve())})
        (starts / 'DONE').write_text('done\n')
        return runner.Run(cfg, root / 'prompts', root / 'out', []), starts

    def patches(self, stack, root):
        stack.enter_context(patch.object(runner, 'overlay_hashes', return_value={'o': 1}))
        stack.enter_context(patch.object(runner, 'runtime_environment'))
        stack.enter_context(patch.object(runner, 'topology', return_value={'server': [0], 'sockets': SOCKETS}))
        stack.enter_context(patch.object(runner, 'query_gpu', return_value={}))
        stack.enter_context(patch.object(runner.os, 'sched_setaffinity'))
        stack.enter_context(patch.object(runner, 'package_deviations', return_value=['xgrammar: installed 0.2.9, lock 0.2.8']))
        (root / 'ov').mkdir()
        spec.write_json(root / 'ov' / 'overlay.json', {})
        (root / 'ov' / 'timing.patch').write_text('diff')

    def test_stop_ends_run_without_decision_or_later_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            run, starts = self.setup_run(root)
            self.patches(stack, root)
            record = {'model': 'qwen25-w4', 'round': 'round-2'}
            rounds = stack.enter_context(patch.object(runner, 'measure_round',
                side_effect=[{'cells': {}, 'quiet_gate': {'pass': True}}, runner.RunStopped(record)]))
            with self.assertRaises(SystemExit) as exit_:
                runner.execute(run, 'rounds', starts)
            self.assertEqual(exit_.exception.code, 3)
            self.assertEqual(rounds.call_count, 2)
            self.assertEqual(spec.read_json(run.output / 'STOPPED.json'), record)
            self.assertEqual(len(spec.read_json(run.output / 'qwen25-w4' / 'rounds-before-stop.json')), 1)
            self.assertFalse((run.output / 'decision.json').exists())
            self.assertFalse((run.output / 'DONE').exists())
            self.assertEqual(sorted(x.name for x in run.output.iterdir() if x.is_dir()), ['qwen25-w4'])

    def test_complete_run_writes_decision(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            run, starts = self.setup_run(root)
            self.patches(stack, root)
            from tests.helpers import summary
            cells = {'PLAIN-1': summary(), 'JSON-1': summary()}
            stack.enter_context(patch.object(runner, 'measure_round', return_value={'cells': cells, 'quiet_gate': {'pass': True}}))
            runner.execute(run, 'rounds', starts)
            d = spec.read_json(run.output / 'decision.json')
            self.assertEqual((d['head'], d['fusion']), ('BIG', 'SMALL'))
            self.assertTrue((run.output / 'DONE').exists())
            self.assertEqual(len(spec.read_json(run.output / 'qwen3-bf16' / 'rounds.json')), 5)
            manifest = spec.read_json(run.output / 'manifest.json')
            self.assertEqual(manifest['timing_patch'], 'diff')
            self.assertEqual(set(manifest['prompts_sha256']), set(spec.MODELS))
            self.assertEqual(manifest['package_deviations'], ['xgrammar: installed 0.2.9, lock 0.2.8'])
            self.assertEqual(manifest['cache_dir'], str((root / 'cache').resolve()))

    def test_unfinished_start_test_or_other_cache_refused(self):
        for breakage in ('done', 'model', 'cache'):
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                run, starts = self.setup_run(root)
                self.patches(stack, root)
                rounds = stack.enter_context(patch.object(runner, 'measure_round'))
                if breakage == 'done':
                    (starts / 'DONE').unlink()
                elif breakage == 'model':
                    (starts / 'qwen3-bf16' / 'start-test.json').unlink()
                else:
                    run.cfg.cache_dir = root / 'other-cache'
                with self.assertRaisesRegex(RuntimeError, 'did not finish' if breakage != 'cache' else 'cache directory'):
                    runner.execute(run, 'rounds', starts)
                rounds.assert_not_called()
                self.assertFalse(run.output.exists())
                runner.topology.assert_not_called()

    def test_start_tests_from_other_code_or_overlay_refused(self):
        for field, value in (('code_sha256', {'x': 'y'}), ('overlay_sha256', {'o': 2}), ('step', 'rounds')):
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                run, starts = self.setup_run(root)
                self.patches(stack, root)
                manifest = spec.read_json(starts / 'manifest.json')
                manifest[field] = value
                spec.write_json(starts / 'manifest.json', manifest)
                with self.assertRaisesRegex(RuntimeError, 'rerun them'):
                    runner.execute(run, 'rounds', starts)
                self.assertFalse(run.output.exists())

    def test_wrong_prompts_or_missing_model_refused_before_gpu(self):
        for breakage in ('prompts', 'revision', 'model'):
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                run, starts = self.setup_run(root)
                self.patches(stack, root)
                if breakage == 'prompts':
                    m = prompt_manifest('qwen3-w4')
                    m['prompts'] = m['prompts'][:24]
                    spec.write_json(root / 'prompts' / 'qwen3-w4.json', m)
                elif breakage == 'revision':
                    spec.write_json(root / 'prompts' / 'qwen3-w4.json', dict(prompt_manifest('qwen3-w4'), revision='x'))
                else:
                    (run.cfg.model_path('qwen3-bf16') / 'config.json').unlink()
                with self.assertRaises((RuntimeError, ValueError)):
                    runner.execute(run, 'rounds', starts)
                runner.topology.assert_not_called()

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            run, starts = self.setup_run(root)
            self.patches(stack, root)
            run.output.mkdir()
            (run.output / 'manifest.json').write_text('{"sentinel": true}')
            with self.assertRaises(FileExistsError):
                runner.execute(run, 'start-test')
            self.assertEqual((run.output / 'manifest.json').read_text(), '{"sentinel": true}')

    def test_lock_held_elsewhere_refuses(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            run, starts = self.setup_run(root)
            self.patches(stack, root)
            with open(root / 'gpu.lock', 'w') as held:
                fcntl.flock(held, fcntl.LOCK_EX)   # flock conflicts across open file descriptions
                run.locks = [root / 'gpu.lock']
                with self.assertRaises(BlockingIOError):
                    runner.execute(run, 'start-test')
            self.assertFalse(run.output.exists())


if __name__ == '__main__':
    unittest.main()
