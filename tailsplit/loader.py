"""Import hook used by the servers this tool starts (through tailsplit/_site/sitecustomize.py).

In both modes it checks that the runner vLLM imports is the pinned file and adds one idle-only
control method to vLLM's GPU worker. Only in patched mode does it execute the timing overlay in
place of the installed runner; a baseline server computes with unmodified vLLM.
"""
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys

RUNNER = 'vllm.v1.worker.gpu.model_runner'
WORKER = 'vllm.v1.worker.gpu_worker'
MODE = 'TAILSPLIT_MODE'


def check_and_load_runner(module, root, mode, expected_hashes, identity_fn=None):
    """Verify the imported runner and the overlay; in patched mode return the overlay path."""
    from . import spec
    from .environment import installed_identity
    from .overlay import overlay_hashes
    manifest = json.loads((root / 'overlay.json').read_text())
    identity = (identity_fn or installed_identity)()
    if (manifest.get('installed_identity') != identity
            or Path(module.__file__).resolve() != Path(identity['runner_path'])):
        raise RuntimeError('Imported runner is not the installation the overlay was built from')
    if overlay_hashes(root) != expected_hashes:
        raise RuntimeError('Overlay differs from the one recorded at launch')
    if spec.sha(module.__file__) != spec.RUNNER_SHA256:
        raise RuntimeError('Installed runner differs from the pinned source')
    if mode == 'patched':
        overlay = root / 'model_runner.py'
        if spec.sha(overlay) != manifest['patched_sha256']:
            raise RuntimeError('Timing overlay changed')
        return overlay
    return None


class Loader(importlib.abc.Loader):
    def __init__(self, original, name):
        self.original, self.name = original, name

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        if self.name == RUNNER:
            mode = os.environ[MODE]
            overlay = check_and_load_runner(module, Path(os.environ['TAILSPLIT_OVERLAY']), mode,
                                            json.loads(os.environ['TAILSPLIT_OVERLAY_HASHES']))
            if overlay is not None:
                exec(compile(overlay.read_bytes(), str(overlay), 'exec'), module.__dict__)
                return
        self.original.exec_module(module)
        if self.name == WORKER:
            from .instrument import worker_control
            module.Worker.tailsplit_control = worker_control


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (RUNNER, WORKER):
            return None
        found = importlib.machinery.PathFinder.find_spec(fullname, path)
        if found is None or found.loader is None:
            raise ImportError(fullname)
        found.loader = Loader(found.loader, fullname)
        return found


def install():
    if os.environ.get(MODE) in ('baseline', 'patched') and not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())
