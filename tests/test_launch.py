"""Server environment, argv, ownership and process lifecycle: CPU fixtures and local sockets only."""
from contextlib import ExitStack
import errno
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tailsplit import environment, head_alone, launch, ownership, spec


def state(pids=(), memory=0, uuid='GPU-four'):
    return {'uuid': uuid, 'compute_pids': list(pids), 'memory_mib': memory}


def process_table():
    return {70: {'parent': 1, 'session': 70, 'start': 100, 'state': 'S'},
            71: {'parent': 70, 'session': 70, 'start': 101, 'state': 'S'},
            72: {'parent': 1, 'session': 70, 'start': 102, 'state': 'S'},
            80: {'parent': 1, 'session': 80, 'start': 110, 'state': 'S'}}


def cfg(root, gpu=4):
    return launch.Config(gpu=gpu, hf_home=Path(root) / 'hf', cache_dir=Path(root) / 'cache',
                         overlay=Path(root), port=8155, python='/env/bin/python')


def fake_server(root):
    server = launch.Server.__new__(launch.Server)
    server.directory, server.port, server.gpu = root, 8155, 4
    server.process = Mock(pid=70)
    server.process.poll.return_value = None
    server.expected_uuid = 'GPU-four'
    server.model_path = '/snapshot/pinned'
    server.manifest, server.known = {}, {}
    server.verified, server.closed = False, False
    server.log = Mock()
    return server


class Environment(unittest.TestCase):
    def test_argv_has_only_model_and_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = launch.server_argv(cfg(tmp), 'qwen25-w4', [1, 2])
            self.assertEqual(argv[:3], ['taskset', '-c', '1,2'])
            self.assertEqual(argv[3:7], ['/env/bin/python', '-s', '-m', 'vllm.entrypoints.cli.main'])
            self.assertEqual(argv[7:], ['serve', str(spec.snapshot(Path(tmp) / 'hf', 'qwen25-w4')), '--port', '8155'])

    def test_runtime_environment_per_mode(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            spec.write_json(root / 'overlay.json', {'installed_identity': {}})
            stack.enter_context(patch.dict(os.environ, {'NORMAL_VARIABLE': 'kept', 'PYTHONUSERBASE': '/shadow',
                                                        'PYTHONHOME': '/old', 'PYTHONPATH': '/old/source'}, clear=True))
            stack.enter_context(patch.object(launch, 'installed_identity', return_value={}))
            stack.enter_context(patch.object(launch, 'overlay_hashes', return_value={'a': 'b'}))
            for mode in ('baseline', 'patched', 'head'):
                env = launch.runtime_environment(cfg(root), mode)
                self.assertEqual(env['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '4')
                self.assertEqual(env['NORMAL_VARIABLE'], 'kept')
                self.assertEqual(env['HF_HUB_OFFLINE'], '1')
                self.assertEqual(env['PYTHONPATH'], f'{launch.SITE}{os.pathsep}{launch.REPO}')
                self.assertNotIn('PYTHONUSERBASE', env)
                self.assertNotIn('PYTHONHOME', env)
                self.assertEqual('VLLM_SERVER_DEV_MODE' in env, mode != 'head')
                self.assertEqual(env.get('TAILSPLIT_MODE'), None if mode == 'head' else mode)
                self.assertTrue(env['TRITON_CACHE_DIR'].startswith(str(root / 'cache')))
            self.assertFalse(any(k.startswith('VLLM_') and k != 'VLLM_SERVER_DEV_MODE'
                                 for k in launch.runtime_environment(cfg(root), 'patched')))

    def test_overlay_for_other_installation_refused_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec.write_json(Path(tmp) / 'overlay.json', {'installed_identity': {'prefix': '/other'}})
            with patch.object(launch, 'installed_identity', return_value={'prefix': '/this'}), \
                 patch.dict(os.environ, {}, clear=True), patch.object(launch, 'query_gpu') as query:
                with self.assertRaisesRegex(RuntimeError, 'another installation'):
                    launch.runtime_environment(cfg(tmp), 'patched')
                query.assert_not_called()

    def test_inherited_prefix_denial_precedes_any_check(self):
        for name in ('LD_PRELOAD', 'PYTORCH_NEW_FLAG', 'OMP_NUM_THREADS', 'VLLM_USE_V2_MODEL_RUNNER',
                     'NCCL_NEW_FLAG', 'CUDA_HOME', 'CUDA_VISIBLE_DEVICES', 'CUDA_CACHE_PATH', 'TAILSPLIT_MODE'):
            with self.subTest(name=name), patch.dict(os.environ, {name: ''}, clear=True), \
                 patch.object(launch, 'installed_identity') as identity:
                with self.assertRaisesRegex(RuntimeError, name):
                    launch.runtime_environment(cfg('/unused'), 'baseline')
                identity.assert_not_called()
        launch.check_inherited_environment({'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'})
        for name in ('TORCHDYNAMO_DISABLE', 'TORCH_COMPILE_DISABLE', 'NVIDIA_TF32_OVERRIDE',
                     'CUBLAS_WORKSPACE_CONFIG', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_FORCE_DISABLE_CACHES'):
            with self.assertRaisesRegex(RuntimeError, name):
                launch.check_inherited_environment({name: '1'})
        with self.assertRaises(RuntimeError):
            launch.check_inherited_environment({'CUDA_DEVICE_ORDER': 'FASTEST_FIRST'})

    def test_code_hashes_cover_the_package(self):
        hashes = launch.code_hashes()
        for name in ('runner.py', 'gates.py', 'instrument.py', 'loader.py', '_site/sitecustomize.py'):
            self.assertIn(name, hashes)

    def test_installed_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / spec.RUNNER_FILE
            source.parent.mkdir(parents=True)
            source.write_text('pinned source fixture')
            version = root / 'torch/version.py'
            version.parent.mkdir()
            version.write_text("cuda: str = '12.9'\n")
            for name in ('torch', 'vllm'):
                (root / name / '__init__.py').touch()
            versions = {'vllm': spec.VLLM_VERSION, 'torch': spec.TORCH_VERSION}
            def distribution(name):
                return SimpleNamespace(version=versions[name], locate_file=lambda path: root / path)
            with patch.object(environment.sys, 'prefix', str(root)), \
                 patch.object(environment.sys, 'path', [str(root)]), \
                 patch.object(spec, 'RUNNER_SHA256', spec.sha(source)), \
                 patch.object(environment.importlib.metadata, 'distribution', side_effect=distribution):
                identity = environment.installed_identity()
                self.assertEqual(identity['runner_path'], str(source.resolve()))
                self.assertEqual(identity['cuda'], '12.9')
                shadow = root / 'shadow'
                (shadow / 'torch').mkdir(parents=True)
                (shadow / 'torch/__init__.py').write_text("raise AssertionError('must not execute')")
                with patch.object(environment.sys, 'path', [str(shadow), str(root)]):
                    with self.assertRaisesRegex(RuntimeError, 'shadowed'):
                        environment.installed_identity()
                with patch.object(environment.sys, 'prefix', str(root / 'other-env')):
                    with self.assertRaisesRegex(RuntimeError, 'outside'):
                        environment.installed_identity()
                versions['vllm'] = '0.30.0'
                with self.assertRaisesRegex(RuntimeError, 'Need vLLM'):
                    environment.installed_identity()
                versions['vllm'] = spec.VLLM_VERSION
                version.write_text("cuda: str = '13.0'\n")
                with self.assertRaisesRegex(RuntimeError, '12.9'):
                    environment.installed_identity()
                version.write_text("cuda: str = '12.9'\n")
                source.write_text('modified source')
                with self.assertRaisesRegex(RuntimeError, 'pinned'):
                    environment.installed_identity()


class Ownership(unittest.TestCase):
    def test_card_uuid_and_global_process_mapping(self):
        good = [{'pid': 71, 'uuid': 'GPU-four'}]
        cases = [(state([71]), good, True), (state([]), [], False),
                 (state([80]), [{'pid': 80, 'uuid': 'GPU-four'}], False),
                 (state([71]), good + [{'pid': 72, 'uuid': 'GPU-zero'}], False),
                 (state([71]), good + [{'pid': 80, 'uuid': 'GPU-four'}], False),
                 (state([71], uuid='GPU-zero'), good, False),
                 (state([71]), [], False)]
        for gpu, apps, accepted in cases:
            with self.subTest(gpu=gpu, apps=apps), \
                 patch.object(ownership, 'query_gpu', return_value=gpu) as query, \
                 patch.object(ownership, 'compute_apps', return_value=apps):
                if accepted:
                    self.assertEqual(ownership.gpu_ownership({70, 71, 72}, 'GPU-four', 4)['index'], 4)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'exactly'):
                        ownership.gpu_ownership({70, 71, 72}, 'GPU-four', 4)
                query.assert_called_once_with(4)

    def test_compute_query_includes_uuid_and_all_cards(self):
        with patch.object(ownership.subprocess, 'check_output', return_value='71, GPU-four\n72, GPU-zero\n') as run:
            self.assertEqual(ownership.compute_apps()[1], {'pid': 72, 'uuid': 'GPU-zero'})
        self.assertIn('--query-compute-apps=pid,gpu_uuid', run.call_args.args[0])
        self.assertNotIn('-i', run.call_args.args[0])

    def test_precheck_accepts_time_wait_but_rejects_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
            listener.listen()
            with self.assertRaises(OSError):
                ownership.precheck_port(port)
            with socket.create_connection(('127.0.0.1', port)) as peer:
                accepted, _ = listener.accept()
                accepted.close()
                self.assertEqual(peer.recv(1), b'')
        ownership.precheck_port(port)

    def test_ipv6_unsupported_is_skipped_but_busy_is_not(self):
        for error, accepted in ((errno.EAFNOSUPPORT, True), (errno.EADDRINUSE, False)):
            ipv4 = Mock()
            with patch.object(ownership.socket, 'socket', side_effect=[ipv4, OSError(error, 'fixture')]):
                if accepted:
                    ownership.precheck_port(8155)
                else:
                    with self.assertRaises(OSError):
                        ownership.precheck_port(8155)
            ipv4.close.assert_called_once()

    def test_proc_listener_inode_must_be_in_our_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'net').mkdir()
            (root / '70/fd').mkdir(parents=True)
            (root / '70/fd/5').symlink_to('socket:[555]')
            def net(inode):
                (root / 'net/tcp').write_text('header\n0: 0100007F:1FDB 00000000:0000 0A 0:0 0:0 0 1000 0 ' + inode + '\n')
            net('555')
            self.assertEqual(ownership.listener_ownership(8155, {70}, root), {'555': [70]})
            net('666')
            with self.assertRaisesRegex(RuntimeError, 'Listening socket'):
                ownership.listener_ownership(8155, {70}, root)
            net('555')
            with self.assertRaises(RuntimeError):
                ownership.listener_ownership(8155, {80}, root)

    def test_reparented_session_members_are_owned(self):
        self.assertEqual(set(ownership.owned_processes(70, 70, process_table())), {70, 71, 72})

    def test_waits_for_reparented_worker_then_memory_release(self):
        tables = [{72: process_table()[72]}, {}, {}]
        with patch.object(ownership, 'processes', side_effect=tables), \
             patch.object(ownership, 'query_gpu', side_effect=[state([], 0), state([], 500), state()]), \
             patch.object(ownership.time, 'sleep') as sleep:
            result = ownership.wait_released(70, 70, 'GPU-four', 4)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(result['live_pids'], [])

    def test_release_timeout_fails_closed(self):
        with patch.object(ownership, 'processes', return_value={72: process_table()[72]}), \
             patch.object(ownership, 'query_gpu', return_value=state()), \
             patch.object(ownership.time, 'monotonic', side_effect=[0, 61]):
            with self.assertRaisesRegex(RuntimeError, 'did not release'):
                ownership.wait_released(70, 70, 'GPU-four', 4)


class Lifecycle(unittest.TestCase):
    def test_readiness_requires_ownership_gpu_and_exact_model(self):
        for model, good in (('/snapshot/pinned', True), ('alias', False)):
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                server = fake_server(Path(tmp))
                health = Mock(status=200)
                health.__enter__ = Mock(return_value=health)
                health.__exit__ = Mock(return_value=False)
                model_response = io.BytesIO(json.dumps({'data': [{'id': model}]}).encode())
                urlopen = stack.enter_context(patch.object(launch.urllib.request, 'urlopen',
                                                           side_effect=[health, model_response]))
                stack.enter_context(patch.object(launch, 'owned_processes', return_value=process_table()))
                listener = stack.enter_context(patch.object(launch, 'listener_ownership', return_value={'555': [70]}))
                gpu = stack.enter_context(patch.object(launch, 'gpu_ownership', return_value={'uuid': 'GPU-four'}))
                if good:
                    server.ready()
                    self.assertTrue(server.verified)
                else:
                    with self.assertRaisesRegex(RuntimeError, '/v1/models'):
                        server.ready()
                    self.assertFalse(server.verified)
                listener.assert_called_once()
                self.assertEqual(gpu.call_args.args[1:], ('GPU-four', 4))
                self.assertEqual(urlopen.call_args.args[0], 'http://127.0.0.1:8155/v1/models')

    def test_dead_root_cannot_pass_readiness(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launch, 'listener_ownership') as listener:
            server = fake_server(Path(tmp))
            server.process.poll.return_value = 1
            with self.assertRaisesRegex(RuntimeError, 'exited'):
                server.assert_identity()
            listener.assert_not_called()

    def test_server_manifest_keeps_full_environment_and_sampling_defaults(self):
        for mode in ('baseline', 'patched'):
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                c = cfg(root)
                snap = c.model_path('qwen25-bf16')
                snap.mkdir(parents=True)
                spec.write_json(root / 'overlay.json', {})
                spec.write_json(snap / 'generation_config.json', {'temperature': .7, 'top_p': .8, 'top_k': 20})
                env = {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '4', 'UNLISTED': 'preserved'}
                stack.enter_context(patch.object(launch, 'runtime_environment', return_value=env))
                stack.enter_context(patch.object(launch, 'idle_gpu', return_value=state()))
                probe = stack.enter_context(patch.object(launch, 'precheck_port'))
                spawn = stack.enter_context(patch.object(launch.subprocess, 'Popen', return_value=Mock(pid=70)))
                server = launch.Server('qwen25-bf16', mode, root / 'server', c, [4])
                result = spec.read_json(root / 'server/launch.json')
                self.assertEqual(result['environment'], env)
                self.assertEqual(result['generation_config']['contents']['top_k'], 20)
                self.assertTrue(spawn.call_args.kwargs['start_new_session'])
                self.assertEqual(spawn.call_args.args[0][-2:], ['--port', '8155'])
                probe.assert_called_once_with(8155)
                server.log.close()

    def test_recorded_environments_are_redacted(self):
        secrets = {name: f'value-{i}-of-a-secret' for i, name in enumerate(
            ('OPENROUTER_API_KEY', 'HF_TOKEN', 'aws_secret_access_key', 'DB_PASSWORD', 'GIT_CREDENTIALS',
             'SSH_AUTH_SOCK', 'HTTP_AUTH'))}
        red = launch.redact(dict(secrets, HOME='/home/u', PATH='/bin'))
        self.assertEqual(red['HOME'], '/home/u')
        self.assertTrue(all(red[k] == '<redacted>' for k in secrets))
        for mode in ('server', 'head'):
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                c = cfg(root)
                snap = c.model_path('qwen25-w4')
                snap.mkdir(parents=True)
                spec.write_json(root / 'overlay.json', {})
                spec.write_json(root / 'contract.json', {'shape': [4, 2]})
                spec.write_json(snap / 'generation_config.json', {})
                env = {'CUDA_VISIBLE_DEVICES': '4', 'MY_API_KEY': 'child-secret'}
                stack.enter_context(patch.dict(os.environ, secrets))
                stack.enter_context(patch.object(launch, 'runtime_environment', return_value=env))
                stack.enter_context(patch.object(launch, 'idle_gpu', return_value=state()))
                stack.enter_context(patch.object(launch, 'precheck_port'))
                stack.enter_context(patch.object(launch, 'wait_released', return_value={}))
                process = Mock(pid=70)
                process.wait.return_value = 0
                spawn = stack.enter_context(patch.object(launch.subprocess, 'Popen', return_value=process))
                if mode == 'server':
                    launch.Server('qwen25-w4', 'baseline', root / 'server', c, [4]).log.close()
                    path = root / 'server' / 'launch.json'
                else:
                    launch.run_head(c, 'qwen25-w4', root / 'contract.json', root / 'out.json', [4])
                    path = root / 'out.launch.json'
                text = path.read_text()
                for value in list(secrets.values()) + ['child-secret']:
                    self.assertNotIn(value, text)
                self.assertIn('OPENROUTER_API_KEY', text)
                self.assertEqual(spawn.call_args.kwargs['env']['MY_API_KEY'], 'child-secret')   # child keeps it

    def test_package_deviations(self):
        from tailsplit import environment
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / 'lock.txt'
            lock.write_text('xgrammar==0.2.8\nflashinfer-jit-cache==0.6.18.post1+cu129\nFoo_Bar==1.0\n'
                            'pip==26.2.1\nvllm @ https://example/vllm.whl#sha256=00\n')
            self.assertEqual(environment.locked_versions(lock),
                             {'xgrammar': '0.2.8', 'flashinfer-jit-cache': '0.6.18.post1+cu129',
                              'foo-bar': '1.0', 'pip': '26.2.1'})
            same = {'xgrammar': '0.2.8', 'flashinfer-jit-cache': '0.6.18.post1+cu129', 'foo-bar': '1.0'}
            self.assertEqual(environment.package_deviations(lock, same), [])
            self.assertEqual(environment.package_deviations(lock, dict(same, xgrammar='0.2.9', **{'foo-bar': '1.0'})),
                             ['xgrammar: installed 0.2.9, lock 0.2.8'])
            del same['flashinfer-jit-cache']
            self.assertEqual(environment.package_deviations(lock, same),
                             ['flashinfer-jit-cache: not installed (lock 0.6.18.post1+cu129)'])
        locked = environment.locked_versions()
        self.assertEqual((locked['xgrammar'], locked['flashinfer-python'], locked['flashinfer-jit-cache'],
                          locked['torch']), ('0.2.8', '0.6.18.post1', '0.6.18.post1+cu129', spec.TORCH_VERSION))
        self.assertIn(spec.VLLM_VERSION.replace('+', '%2B'), environment.LOCK.read_text())

    def test_busy_gpu_refused_before_spawn(self):
        with patch.object(launch, 'query_gpu', return_value=state([99], 500)):
            with self.assertRaisesRegex(RuntimeError, 'in use'):
                launch.idle_gpu(4)

    def test_request_failure_cleanup_requires_current_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = fake_server(Path(tmp))
            server.verified = True
            with patch.object(server, 'assert_identity') as verify, patch.object(server, 'completed') as close:
                server.request_failed(ValueError('failed warmup'))
                verify.assert_called_once()
                close.assert_called_once()
            with patch.object(server, 'assert_identity', side_effect=RuntimeError('foreign')), \
                 patch.object(server, 'completed') as close:
                with self.assertRaises(RuntimeError):
                    server.request_failed(ValueError('failed probe'))
                close.assert_not_called()

    def test_completed_waits_for_session_even_if_root_already_exited(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(launch, 'owned_processes', return_value={72: process_table()[72]}), \
             patch.object(launch, 'wait_released', return_value={}) as wait, \
             patch.object(launch.os, 'killpg') as kill:
            server = fake_server(Path(tmp))
            server.process.poll.return_value = 0
            server.completed()
            server.process.wait.assert_called_once_with(timeout=60)
            wait.assert_called_once_with(70, 70, 'GPU-four', 4, {72: 102})
            kill.assert_not_called()
            self.assertTrue(server.closed)

    def test_retain_waits_for_the_server_and_writes_a_note(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launch, 'wait_released') as wait:
            server = fake_server(Path(tmp))
            server.argv = ['vllm']
            server.retain(RuntimeError('unexpected'))
            server.process.wait.assert_called_once_with()
            wait.assert_called_once()
            self.assertIn('unexpected', spec.read_json(Path(tmp) / 'LIVE-SERVER.json')['error'])

    def test_head_manifest_and_release_even_after_failure(self):
        for returncode in (0, 1):
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                c = cfg(root)
                snap = c.model_path('qwen25-w4')
                snap.mkdir(parents=True)
                spec.write_json(root / 'contract.json', {'shape': [4, 2]})
                spec.write_json(snap / 'generation_config.json', {'top_k': 20})
                env = {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '4'}
                stack.enter_context(patch.object(launch, 'runtime_environment', return_value=env))
                stack.enter_context(patch.object(launch, 'idle_gpu', return_value=state()))
                process = Mock(pid=90)
                process.wait.return_value = returncode
                spawn = stack.enter_context(patch.object(launch.subprocess, 'Popen', return_value=process))
                release = stack.enter_context(patch.object(launch, 'wait_released', return_value={}))
                args = (c, 'qwen25-w4', root / 'contract.json', root / 'out.json', [4])
                if returncode:
                    with self.assertRaises(subprocess.CalledProcessError):
                        launch.run_head(*args)
                else:
                    launch.run_head(*args)
                manifest = spec.read_json(root / 'out.launch.json')
                self.assertEqual(manifest['environment'], env)
                self.assertIn('GPU-four', manifest['argv'])
                self.assertIn('tailsplit.head_alone', manifest['argv'])
                self.assertTrue(spawn.call_args.kwargs['start_new_session'])
                release.assert_called_once_with(90, 90, 'GPU-four', 4)

    def test_post_spawn_manifest_failure_still_waits_for_head_release(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            spec.write_json(root / 'contract.json', {'shape': [4, 2]})
            stack.enter_context(patch.object(launch, 'runtime_environment', return_value={}))
            stack.enter_context(patch.object(launch, 'idle_gpu', return_value=state()))
            stack.enter_context(patch.object(launch, 'generation_config', return_value={}))
            process = Mock(pid=90)
            process.wait.return_value = 0
            stack.enter_context(patch.object(launch.subprocess, 'Popen', return_value=process))
            release = stack.enter_context(patch.object(launch, 'wait_released', return_value={}))
            stack.enter_context(patch.object(launch, 'write_json', side_effect=[None, OSError('disk full')]))
            with self.assertRaisesRegex(OSError, 'disk full'):
                launch.run_head(cfg(root), 'qwen25-w4', root / 'contract.json', root / 'out.json', [4])
            process.wait.assert_called_once()
            release.assert_called_once_with(90, 90, 'GPU-four', 4)

    def test_head_rejects_wrong_device_order_before_torch_import(self):
        with patch.dict(os.environ, {'CUDA_DEVICE_ORDER': 'FASTEST_FIRST', 'CUDA_VISIBLE_DEVICES': '4'}):
            with self.assertRaisesRegex(RuntimeError, 'PCI_BUS_ID'):
                head_alone.run([4, 2], 'torch.bfloat16', 1, 1, 'GPU-four', 4)
        with patch.dict(os.environ, {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '3'}):
            with self.assertRaisesRegex(RuntimeError, 'CUDA_VISIBLE_DEVICES=4'):
                head_alone.run([4, 2], 'torch.bfloat16', 1, 1, 'GPU-four', 4)


if __name__ == '__main__':
    unittest.main()
