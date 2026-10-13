"""The timing patch: a patched copy of vLLM's V2 model runner, loaded in place of the installed
file only by a server started in patched mode. The installed environment is never edited.

Building it never imports vLLM or torch.
"""
import ast
import difflib
import json
from pathlib import Path

from . import spec
from .environment import installed_identity

FILES = ('model_runner.py', 'timing.patch', 'overlay.json')


def patch_source(source):
    """Insert the six timing markers into the pinned runner source. Two functions change
    (execute_model and sample); no computation changes. Every anchor must occur exactly once."""
    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError('Patch anchor not found exactly once: ' + old[:100])
        source = source.replace(old, new)

    replace_once('        # Run model.\n',
                 '        from tailsplit.instrument import begin_step, mark\n'
                 '        begin_step(self, input_batch, dummy_run)\n'
                 '        # Run model.\n')
    replace_once('        self.kv_connector.finish_forward()\n',
                 '        mark("forward_end")\n'
                 '        self.kv_connector.finish_forward()\n')
    replace_once('            logits = self.model.compute_logits(sample_hidden_states)\n',
                 '            from tailsplit.instrument import mark, sampling_hooks, set_mask\n'
                 '            mark("head_start")\n'
                 '            logits = self.model.compute_logits(sample_hidden_states)\n'
                 '            mark("head_end")\n')
    replace_once('        sampler_output: SamplerOutput | None\n',
                 '        mark("mask_end")\n'
                 '        sampler_output: SamplerOutput | None\n')
    replace_once('            sampler_output = self.sampler(logits, input_batch)\n',
                 '            with sampling_hooks(self.sampler):\n'
                 '                sampler_output = self.sampler(logits, input_batch)\n'
                 '            set_mask(grammar_output, input_batch)\n')
    ast.parse(source)
    return source


def build(output, identity=None):
    """Write model_runner.py, timing.patch and overlay.json into `output` (created if missing)."""
    output = Path(output)
    identity = identity or installed_identity()
    path = Path(identity['runner_path'])
    source = path.read_text()
    patched = patch_source(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'model_runner.py').write_text(patched)
    diff = ''.join(difflib.unified_diff(source.splitlines(True), patched.splitlines(True),
                                        fromfile='a/' + spec.RUNNER_FILE, tofile='b/' + spec.RUNNER_FILE))
    (output / 'timing.patch').write_text(diff)
    spec.write_json(output / 'overlay.json', {
        'commit': spec.VLLM_COMMIT, 'source': spec.RUNNER_FILE, 'installed_identity': identity,
        'original_sha256': spec.sha(path), 'patched_sha256': spec.sha(output / 'model_runner.py'),
        'diff_sha256': spec.sha(output / 'timing.patch')})
    return diff


def overlay_hashes(path):
    """sha256 of the three overlay files, after checking they match their own manifest."""
    path = Path(path)
    manifest = json.loads((path / 'overlay.json').read_text())
    if manifest.get('commit') != spec.VLLM_COMMIT or manifest.get('original_sha256') != spec.RUNNER_SHA256:
        raise RuntimeError('Overlay is not built from the pinned runner source')
    result = {name: spec.sha(path / name) for name in FILES}
    if (result['model_runner.py'] != manifest.get('patched_sha256')
            or result['timing.patch'] != manifest.get('diff_sha256')):
        raise RuntimeError('Overlay files do not match overlay.json')
    return result
