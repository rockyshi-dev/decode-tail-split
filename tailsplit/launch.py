"""Server and head-alone processes: environment, exact argv, ownership, and release."""
from dataclasses import dataclass
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

from .environment import installed_identity
from .overlay import overlay_hashes
from .ownership import gpu_ownership, listener_ownership, owned_processes, precheck_port, wait_released
from .spec import sha, snapshot, write_json
from .telemetry import query_gpu

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
SITE = PACKAGE / '_site'
# An inherited variable with one of these prefixes would change what is measured.
DENIED_PREFIXES = ('PYTORCH_', 'OMP_', 'VLLM_', 'NCCL_', 'CUDA_', 'TAILSPLIT_', 'TORCH_', 'TORCHDYNAMO_',
                   'TORCHINDUCTOR_', 'TRITON_', 'CUBLAS_', 'CUBLASLT_', 'CUDNN_', 'NVIDIA_TF32_',
                   'FLASHINFER_', 'MKL_', 'OPENBLAS_')


SECRET_NAME = re.compile(r'KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|_AUTH|^AUTH|^SSH_', re.IGNORECASE)


def redact(env):
    """Copy of an environment for the record: values of credential-like variables replaced."""
    return {k: ('<redacted>' if SECRET_NAME.search(k) else v) for k, v in env.items()}


@dataclass
class Config:
    gpu: int
    hf_home: Path
    cache_dir: Path
    overlay: Path
    port: int = 8155
    python: str = sys.executable

    def model_path(self, model):
        return snapshot(self.hf_home, model)


def check_inherited_environment(env):
    """Only CUDA_DEVICE_ORDER=PCI_BUS_ID may be inherited from the denied families."""
    bad = sorted(k for k, value in env.items()
                 if (k == 'LD_PRELOAD' or k.startswith(DENIED_PREFIXES))
                 and not (k == 'CUDA_DEVICE_ORDER' and value == 'PCI_BUS_ID'))
    if bad:
        raise RuntimeError('Unexpected inherited variables that would change the measurement: ' + ', '.join(bad))


def generation_config(cfg, model):
    path = cfg.model_path(model) / 'generation_config.json'
    return {'path': str(path), 'sha256': sha(path), 'contents': json.loads(path.read_text())}


def code_hashes():
    """sha256 of every Python file of this package (binds a start test to the rounds)."""
    return {str(p.relative_to(PACKAGE)): sha(p) for p in sorted(PACKAGE.rglob('*.py'))
            if '__pycache__' not in p.parts}


def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + '\n')
        handle.flush()
    except Exception:
        handle.close()
        raise
    return handle


def runtime_environment(cfg, mode):
    """Environment of a server ('baseline' or 'patched') or of the head-alone job ('head')."""
    check_inherited_environment(os.environ)
    identity = installed_identity()
    if json.loads((Path(cfg.overlay) / 'overlay.json').read_text()).get('installed_identity') != identity:
        raise RuntimeError('The overlay was built for another installation; rebuild it')
    env = os.environ.copy()
    for key in ('PYTHONUSERBASE', 'PYTHONHOME'):
        env.pop(key, None)
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(cfg.gpu), HF_HOME=str(cfg.hf_home),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=f'{SITE}{os.pathsep}{REPO}', PYTHONNOUSERSITE='1',
               TAILSPLIT_MODE=mode, TAILSPLIT_OVERLAY=str(cfg.overlay), VLLM_SERVER_DEV_MODE='1',
               TAILSPLIT_OVERLAY_HASHES=json.dumps(overlay_hashes(cfg.overlay), sort_keys=True),
               XDG_CACHE_HOME=str(cache / 'xdg'), XDG_CONFIG_HOME=str(cache / 'config'),
               TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'),
               CUDA_CACHE_PATH=str(cache / 'cuda'), FLASHINFER_WORKSPACE_BASE=str(cache / 'flashinfer'))
    if mode == 'head':
        for key in ('TAILSPLIT_MODE', 'TAILSPLIT_OVERLAY', 'TAILSPLIT_OVERLAY_HASHES', 'VLLM_SERVER_DEV_MODE'):
            env.pop(key)
    return env


def server_argv(cfg, model, cpus):
    """No server flag beyond the model and the port."""
    return ['taskset', '-c', ','.join(map(str, cpus)), str(cfg.python), '-s', '-m', 'vllm.entrypoints.cli.main',
            'serve', str(cfg.model_path(model)), '--port', str(cfg.port)]


def idle_gpu(gpu):
    state = query_gpu(gpu)
    if state['compute_pids'] or state['memory_mib'] >= 100:
        raise RuntimeError(f'GPU {gpu} is in use; nothing was started and nothing was touched')
    return state


class Server:
    def __init__(self, model, mode, directory, cfg, cpus):
        self.directory, self.port, self.gpu = Path(directory), cfg.port, cfg.gpu
        self.directory.mkdir(parents=True, exist_ok=False)
        env = runtime_environment(cfg, mode)
        state = idle_gpu(cfg.gpu)
        self.expected_uuid = state['uuid']
        self.model_path = str(cfg.model_path(model))
        self.verified = False
        self.closed = False
        self.known = {}
        defaults = generation_config(cfg, model)
        precheck_port(cfg.port)
        self.argv = server_argv(cfg, model, cpus)
        self.manifest = {'argv': self.argv, 'time': time.time(), 'idle_gpu': state,
            'mode': mode, 'model': model, 'model_path': self.model_path,
            'versions': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            'environment': redact(env), 'inherited_environment': redact(os.environ),
            'added_environment': {'VLLM_SERVER_DEV_MODE': '1: registers /pause, /resume and /collective_rpc, '
                                  'used only while the server is idle'},
            'generation_config': defaults, 'code_sha256': code_hashes(),
            'overlay': json.loads((Path(cfg.overlay) / 'overlay.json').read_text())}
        write_json(self.directory / 'launch.json', self.manifest)
        self.log = (self.directory / 'server.log').open('x')
        self.process = subprocess.Popen(self.argv, env=env, stdin=subprocess.DEVNULL, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        try:
            self.manifest.update(pid=self.process.pid, session=self.process.pid)
            write_json(self.directory / 'launch.json', self.manifest)
            (self.directory / 'server.pid').write_text(str(self.process.pid) + '\n')
        except BaseException as error:
            self.retain(error)
            raise

    def ready(self):
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('Server exited; see server.log')
            healthy = False
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health', timeout=2) as response:
                    healthy = response.status == 200
            except OSError:
                pass
            if healthy:
                # An ownership failure is not retried as a slow start.
                self.assert_identity()
                self.verified = True
                return
            time.sleep(1)
        raise RuntimeError('Server did not become healthy within 20 minutes; left running for inspection')

    def assert_identity(self):
        if self.process.poll() is not None:
            raise RuntimeError('Server exited before ownership verification')
        owned = owned_processes(self.process.pid, self.process.pid)
        self.known.update({pid: row['start'] for pid, row in owned.items()})
        listeners = listener_ownership(self.port, owned)
        gpu = gpu_ownership(owned, self.expected_uuid, self.gpu)
        with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/v1/models', timeout=5) as response:
            models = json.load(response)
        if not models.get('data') or models['data'][0].get('id') != self.model_path:
            raise RuntimeError('/v1/models does not report the pinned model path')
        if self.process.poll() is not None:
            raise RuntimeError('Server exited during ownership verification')
        self.manifest['readiness'] = {'processes': owned, 'listeners': listeners,
                                      'gpu': gpu, 'models': models, 'time': time.time()}
        write_json(self.directory / 'launch.json', self.manifest)

    def completed(self):
        """Normal end: stop the server's session and wait until the GPU is released."""
        if self.closed:
            return
        self.known.update({p: row['start'] for p, row in
                           owned_processes(self.process.pid, self.process.pid).items()})
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
        self.process.wait(timeout=60)
        release = wait_released(self.process.pid, self.process.pid, self.expected_uuid, self.gpu, self.known)
        write_json(self.directory / 'released.json', release)
        self.closed = True
        self.log.close()

    def request_failed(self, error):
        """A failed request on a server that is still ours: record it and stop the server."""
        write_json(self.directory / 'request-failed.json', {'error': str(error),
                   'action': 'This start test or round fails; no request is dropped or retried'})
        if not self.verified:
            raise RuntimeError('A request was sent before readiness was verified')
        self.assert_identity()
        self.completed()

    def retain(self, error):
        """Any other failure: the server is left running for inspection; wait until it exits."""
        if self.closed:
            return
        try:
            write_json(self.directory / 'LIVE-SERVER.json', {'error': str(error), 'pid': self.process.pid,
                'argv': self.argv, 'coordinator_pid': os.getpid(),
                'action': 'Inspect the server, then stop it (kill its process group); the coordinator '
                          'keeps its locks until the server and its GPU memory are gone'})
        finally:
            self.process.wait()
            wait_released(self.process.pid, self.process.pid, self.expected_uuid, self.gpu, self.known)
            self.log.close()


def run_head(cfg, model, contract, output, cpus):
    """One head-alone job in its own session; returns after the GPU is released."""
    env = runtime_environment(cfg, 'head')
    state = idle_gpu(cfg.gpu)
    argv = ['taskset', '-c', ','.join(map(str, cpus)), str(cfg.python), '-s', '-m', 'tailsplit.head_alone',
            '--contract', str(contract), '--output', str(output), '--gpu', str(cfg.gpu),
            '--gpu-uuid', state['uuid']]
    manifest = {'argv': argv, 'environment': redact(env), 'inherited_environment': redact(os.environ),
                'idle_gpu': state, 'model_path': str(cfg.model_path(model)),
                'generation_config': generation_config(cfg, model),
                'sampling_defaults_applied': False, 'contract': json.loads(Path(contract).read_text()),
                'code_sha256': code_hashes(), 'time': time.time()}
    path = output.with_suffix('.launch.json')
    write_json(path, manifest)
    with output.with_suffix('.log').open('x') as log:
        process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            manifest.update(pid=process.pid, session=process.pid)
            write_json(path, manifest)
        finally:
            # A provenance write can fail after spawn; still wait for the job and its GPU memory.
            returncode = process.wait()
            release = wait_released(process.pid, process.pid, state['uuid'], cfg.gpu)
        write_json(output.with_suffix('.released.json'), release)
    if returncode:
        raise subprocess.CalledProcessError(returncode, argv)
