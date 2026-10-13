"""Head alone: the output-head multiplication with the server's shape and dtype, in its own
process on the same GPU, one call every g (the unpatched server's median token gap): at least
3 s of untimed calls at that spacing, then 200 timed calls (python -m tailsplit.head_alone)."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import time
from .ownership import gpu_ownership, owned_processes
from .spec import HEAD_CALLS, PREROLL_NS, write_json

CLOCK = 'clock_gettime(CLOCK_MONOTONIC)'

def gap_ns_of(gap_ms):
    return round(gap_ms * 1_000_000)

def paced(call, gap_ns, count=None, min_span_ns=None, after_ns=None,
          clock=time.perf_counter_ns, sleep=time.sleep):
    """Launch call(i) every gap_ns (measured from the previous launch), either
    `count` times or until the next launch would be >= min_span_ns after the
    first one. Returns the launch times in CLOCK_MONOTONIC ns."""
    if (count is None) == (min_span_ns is None):
        raise ValueError('Give exactly one of count / min_span_ns')
    launched, previous = [], after_ns
    while True:
        if previous is not None:
            delay = previous + gap_ns - clock()
            if delay > 0:
                sleep(delay / 1e9)
        previous = clock()
        launched.append(previous)
        call(len(launched) - 1)
        if count is not None and len(launched) >= count:
            return launched
        if min_span_ns is not None and previous + gap_ns - launched[0] >= min_span_ns:
            return launched

def run(shape, dtype, batch, gap_ms, expected_uuid, gpu):
    if (os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
        or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu)
        or os.getsid(0) != os.getpid()):
        raise RuntimeError(f'Head timing requires its own session and CUDA_DEVICE_ORDER=PCI_BUS_ID, '
                           f'CUDA_VISIBLE_DEVICES={gpu}')
    # Launch stamps must share the clock of the telemetry and the coordinator.
    if time.get_clock_info('perf_counter').implementation != CLOCK:
        raise RuntimeError('perf_counter is not CLOCK_MONOTONIC; head window cannot be aligned')
    import torch
    if dtype not in ('torch.bfloat16','torch.float16') or gap_ms <= 0 or batch < 1:
        raise ValueError('Invalid head microbenchmark contract')
    torch_dtype = getattr(torch,dtype.split('.')[1])
    gap = gap_ns_of(gap_ms)
    with torch.inference_mode():
        weights = torch.randn(shape,device='cuda',dtype=torch_dtype)
        hidden = torch.randn((batch,shape[1]),device='cuda',dtype=torch_dtype)
        torch.cuda.synchronize()
        ownership = gpu_ownership(owned_processes(os.getpid(), os.getsid(0)), expected_uuid, gpu)
        # One synchronized allocator/library warm-up call.
        torch.nn.functional.linear(hidden,weights)
        torch.cuda.synchronize()
        # Pre-roll: untimed calls at spacing g for >= 3 s, so the telemetry samples
        # bracketing the timed window see the warm clock.
        preroll = paced(lambda i: torch.nn.functional.linear(hidden,weights), gap,
                        min_span_ns=PREROLL_NS)
        torch.cuda.synchronize()
        pairs = [(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                 for _ in range(HEAD_CALLS)]
        def timed(i):
            start, end = pairs[i]
            start.record()
            torch.nn.functional.linear(hidden,weights)
            end.record()
        launched = paced(timed, gap, count=HEAD_CALLS, after_ns=preroll[-1])
        torch.cuda.synchronize()
        times = [a.elapsed_time(b) for a,b in pairs]
    return {'shape':shape,'dtype':dtype,'batch':batch,'gap_ms':gap_ms,'gap_ns':gap,
            'times_ms':times,'median_ms':median(times),'clock':CLOCK,
            'preroll_launch_times_ns':preroll,'launch_times_ns':launched,
            'timed_window_ns':[launched[0], launched[-1] + gap],
            'gpu_ownership': ownership}

def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m tailsplit.head_alone')
    p.add_argument('--contract', type=Path, required=True, help='JSON with shape, dtype, batch, gap_ms')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--gpu-uuid', required=True)
    a = p.parse_args(argv)
    contract = json.loads(a.contract.read_text())
    write_json(a.output, run(**contract, expected_uuid=a.gpu_uuid, gpu=a.gpu))


if __name__ == '__main__':
    main()
