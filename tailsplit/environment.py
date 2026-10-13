"""Identity of the vLLM installation, checked without importing torch or vLLM."""
import ast
import importlib.machinery
import importlib.metadata
from pathlib import Path
import sys

from . import spec


def installed_identity():
    """vLLM 0.30.0+cu129, torch 2.13.0+cu129 (CUDA 12.9), both installed in this interpreter's
    environment and not shadowed by another copy on sys.path, and the model runner source
    byte-identical to the pinned one (the timing patch is built against it)."""
    prefix = Path(sys.prefix).resolve()
    vllm = importlib.metadata.distribution('vllm')
    torch = importlib.metadata.distribution('torch')
    if vllm.version != spec.VLLM_VERSION or torch.version != spec.TORCH_VERSION:
        raise RuntimeError(f'Need vLLM {spec.VLLM_VERSION} and torch {spec.TORCH_VERSION}; '
                           f'found {vllm.version} and {torch.version}')
    source = Path(vllm.locate_file(spec.RUNNER_FILE)).resolve()
    torch_source = Path(torch.locate_file('torch/version.py')).resolve()
    if not all(p.is_relative_to(prefix) for p in (source, torch_source)):
        raise RuntimeError(f'vLLM or torch files are outside this interpreter\'s environment {prefix}')
    # Metadata can be right while a bare package earlier on sys.path supplies the code.
    # Resolve without executing either package.
    for name, distribution in (('vllm', vllm), ('torch', torch)):
        expected = Path(distribution.locate_file(f'{name}/__init__.py')).resolve()
        resolved = importlib.machinery.PathFinder.find_spec(name, sys.path)
        if (not expected.is_relative_to(prefix) or not expected.is_file()
                or resolved is None or resolved.origin is None
                or Path(resolved.origin).resolve() != expected):
            raise RuntimeError(f'{name} import is shadowed by a copy outside its installation')
    cuda = None
    for node in ast.parse(torch_source.read_text()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'cuda':
            cuda = ast.literal_eval(node.value)
    if cuda != spec.TORCH_CUDA:
        raise RuntimeError(f'Need the CUDA {spec.TORCH_CUDA} torch build; found {cuda}')
    if spec.sha(source) != spec.RUNNER_SHA256:
        raise RuntimeError(f'{source} differs from the pinned vLLM v0.30.0 source')
    return {'prefix': str(prefix), 'vllm': vllm.version, 'torch': torch.version,
            'cuda': cuda, 'runner_path': str(source), 'runner_sha256': spec.sha(source)}


LOCK = Path(__file__).resolve().parents[1] / 'requirements-cu129.lock.txt'
# Packaging tools do not take part in the measurement.
IGNORED = {'pip', 'setuptools', 'wheel'}


def normalize(name):
    return name.lower().replace('_', '-').replace('.', '-')


def locked_versions(lock=LOCK):
    """{normalized name: version} of the lock file's `name==version` lines (URL lines, such as
    the vLLM wheel, are checked by installed_identity instead)."""
    result = {}
    for line in Path(lock).read_text().splitlines():
        line = line.strip()
        if line.startswith('#') or ' @ ' in line:
            continue
        if '==' in line:
            name, version = line.split('==', 1)
            result[normalize(name.strip())] = version.strip()
    return result


def package_deviations(lock=LOCK, installed=None):
    """One line per package whose installed version differs from the lock file, or is missing.
    xgrammar sets the cost of the mask; the FlashInfer packages decide whether the sampler's
    kernels are prebuilt; so a deviation here can change m, s1 and s2."""
    if installed is None:
        installed = {}
        for dist in importlib.metadata.distributions():
            name = dist.metadata['Name']
            if name:
                installed[normalize(name)] = dist.version
    lines = []
    for name, version in sorted(locked_versions(lock).items()):
        if name in IGNORED:
            continue
        have = installed.get(name)
        if have is None:
            lines.append(f'{name}: not installed (lock {version})')
        elif have != version:
            lines.append(f'{name}: installed {have}, lock {version}')
    return lines
