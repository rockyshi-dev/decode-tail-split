"""GPU timestamps inside the patched runner, read only while the engine is paused.

No torch import at module load. Seven events per step delimit five segments: forward (f),
head (h), mask (m), logits processing (s1) and draw (s2); the last four share their
boundaries. Recording an event never waits for the GPU. Temporary instance hooks on the
sampler place the s1/s2 boundary after apply_sampling_params and the last marker after the draw.
"""
from contextlib import contextmanager
import os
import threading

_collector = None

class Collector:
    def __init__(self, torch):
        self.torch = torch
        self.rows = []
        self.current = None
        self.armed = False
        self.thread = threading.get_ident()
        self.counts = {}
        self.capacity = 25000  # Stop recording with an error, never silently evict.

    def begin(self, batch, dummy):
        self.current = None
        if not self.armed or dummy:
            return
        if threading.get_ident() != self.thread:
            raise RuntimeError('Collector left its worker thread')
        if len(self.rows) >= self.capacity:
            raise RuntimeError('Timestamp buffer full; condition is unusable')
        stream = self.torch.cuda.current_stream()
        ids = list(batch.req_ids)
        steps = {}
        for rid in ids:
            self.counts[rid] = self.counts.get(rid, 0) + 1
            steps[rid] = self.counts[rid]
        self.current = {'req_ids': ids, 'request_steps': steps,
                        'prefill': batch.is_prefilling_np.tolist(),
                        'scheduled_tokens': batch.num_scheduled_tokens.tolist(),
                        'batch_size': batch.num_reqs,
                        'stream': int(stream.cuda_stream),
                        'default_stream': stream == self.torch.cuda.default_stream(),
                        'events': {}, 'mask_active': False}
        self.rows.append(self.current)
        self.mark('forward_start')

    def mark(self, name):
        if self.current is None:
            return
        if name in self.current['events']:
            raise RuntimeError('Repeated timestamp boundary: ' + name)
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        self.current['events'][name] = event

    def drain(self):
        if threading.get_ident() != self.thread:
            raise RuntimeError('Drain is not on the worker thread')
        # Reachable only through worker_control, after /pause?mode=wait.
        self.torch.cuda.synchronize()
        rows = []
        names = [('f', 'forward_start', 'forward_end'),
                 ('h', 'head_start', 'head_end'), ('m', 'head_end', 'mask_end'),
                 ('s1', 'mask_end', 'params_end'), ('s2', 'params_end', 'draw_end')]
        for i, row in enumerate(self.rows):
            result = {k: v for k, v in row.items() if k != 'events'}
            events = row['events']
            result['segments_ms'] = {key: events[a].elapsed_time(events[b]) for key, a, b in names}
            result['period_ms'] = None
            if i + 1 < len(self.rows):
                result['period_ms'] = events['forward_start'].elapsed_time(self.rows[i+1]['events']['forward_start'])
                result['next_request_steps'] = self.rows[i+1]['request_steps']
            rows.append(result)
        self.rows, self.current, self.counts = [], None, {}
        self.armed = False
        return rows

def begin_step(runner, batch, dummy):
    if _collector is not None:
        _collector.begin(batch, dummy)

def mark(name):
    if _collector is not None:
        _collector.mark(name)

def set_mask(grammar_output, batch):
    if _collector is not None and _collector.current is not None:
        ids = list(grammar_output.structured_output_request_ids) if grammar_output is not None else []
        num_masks = int(grammar_output.grammar_bitmask.shape[0]) if grammar_output is not None else 0
        positions = {rid:int(batch.cu_num_logits_np[i+1]-batch.cu_num_logits_np[i])
                     for i,rid in enumerate(batch.req_ids)}
        # The pinned worker returns early on an empty ID list. Without speculative decoding
        # each request must map to exactly one mask row. Record the actual coverage after the
        # worker returned; the presence of a grammar object proves nothing.
        _collector.current.update(mask_request_ids=ids, mask_rows=num_masks,
                                  logits_per_request=positions,
                                  mask_active=(bool(ids) and len(ids)==len(set(ids))
                                    and set(ids)==set(batch.req_ids)
                                    and all(n==1 for n in positions.values())
                                    and num_masks==sum(positions.values())))

@contextmanager
def sampling_hooks(sampler):
    if _collector is None or _collector.current is None:
        yield
        return
    original_params, original_draw = sampler.apply_sampling_params, sampler._sample_random
    # Restore instance state exactly, including whether attributes were inherited.
    sentinel = object()
    saved = {name: sampler.__dict__.get(name, sentinel) for name in ('apply_sampling_params', '_sample_random')}
    def params(*args, **kwargs):
        result = original_params(*args, **kwargs)
        mark('params_end')
        return result
    def draw(*args, **kwargs):
        result = original_draw(*args, **kwargs)
        mark('draw_end')
        return result
    sampler.apply_sampling_params, sampler._sample_random = params, draw
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is sentinel:
                delattr(sampler, name)
            else:
                setattr(sampler, name, value)

def worker_control(worker, action, idle_attestation):
    """Invoked on the worker through /collective_rpc, after the engine is drained.

    The controller always pauses with mode=wait and clear_cache=false first. The attestation
    string rejects accidental raw RPC calls; it is not an access boundary.
    """
    global _collector
    if idle_attestation != 'engine-paused-wait-clear_cache-false':
        raise RuntimeError('Refuse event read without engine idle attestation')
    import torch
    runner = worker.model_runner
    if action == 'inspect':
        model = runner.model
        head = model.lm_head
        quant = []
        for name, module in model.named_modules():
            method = getattr(module, 'quant_method', None)
            scheme = getattr(module, 'scheme', None)
            # Compressed-tensors keeps the kernel on the layer's scheme; GPTQ on its method.
            kernel = getattr(scheme, 'kernel', None) or getattr(method, 'kernel', None)
            if method is not None:
                quant.append({'name': name, 'module': type(module).__name__, 'method': type(method).__name__,
                              'scheme': type(scheme).__name__, 'kernel': type(kernel).__name__})
        return {'runner_module': type(runner).__module__, 'runner_class': type(runner).__name__,
                'head_shape': list(head.weight.shape), 'head_dtype': str(head.weight.dtype),
                'head_element_size': head.weight.element_size(),
                'head_quant_method': type(head.quant_method).__name__,
                'head_contiguous': head.weight.is_contiguous(), 'quant_methods': quant,
                'worker_pid': os.getpid(), 'worker_thread': threading.get_ident(),
                'default_stream': torch.cuda.current_stream() == torch.cuda.default_stream()}
    if os.environ.get('TAILSPLIT_MODE') != 'patched':
        raise RuntimeError('No event collection in baseline')
    if action == 'arm':
        if _collector is not None and _collector.armed:
            raise RuntimeError('Refuse to discard undrained events')
        _collector = Collector(torch)
        _collector.armed = True
        return {'armed': True, 'worker_pid': os.getpid()}
    if action == 'drain' and _collector is not None:
        return {'rows': _collector.drain()}
    raise ValueError('Unknown action: ' + action)
