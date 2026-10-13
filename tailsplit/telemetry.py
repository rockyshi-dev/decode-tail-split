"""Once-a-second recorder: GPU clock, power, temperature, utilisation and compute processes;
per-CPU busy time; CPU time and affinity of every thread of the coordinator's process tree.
Runs as its own process (python -m tailsplit.telemetry). Never touches CUDA."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import time

def query_gpu(gpu):
    output = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'], text=True, timeout=5)
    fields = next(csv.reader(io.StringIO(output), skipinitialspace=True))
    apps = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=5)
    return dict(zip(('uuid', 'clock_mhz', 'power_w', 'temperature_c', 'utilization_pct', 'memory_mib'),
                    [fields[0]] + [float(x) for x in fields[1:]]),
                compute_pids=[int(p) for p in apps.split()])

def proc_fields(path):
    return path.read_text().rsplit(')', 1)[1].split()

CLK_TCK = os.sysconf('SC_CLK_TCK')

def capture(root, gpu):
    cpu = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields = line.split()
        if fields[0].startswith('cpu') and fields[0][3:].isdigit():
            ticks = list(map(int, fields[1:9]))
            cpu[fields[0][3:]] = {'total': sum(ticks), 'busy': sum(ticks) - ticks[3] - ticks[4]}
    # Read after /proc/stat, so a thread whose /proc start time
    # (ticks since boot, CLOCK_BOOTTIME) exceeds it started after this sample's CPU
    # counters were read, and all of its ticks fall in the next interval.
    boot = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    processes = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                fields = proc_fields(path / 'stat')
                processes[int(path.name)] = int(fields[1])
            except (FileNotFoundError, ProcessLookupError):
                pass
    tree = {root}
    while True:
        more = {pid for pid, parent in processes.items() if parent in tree}
        if more <= tree:
            break
        tree |= more
    if root not in processes:
        raise RuntimeError('Coordinator missing')
    threads = {}
    for pid in tree:
        try:
            for path in (Path('/proc') / str(pid) / 'task').iterdir():
                try:
                    fields = proc_fields(path / 'stat')
                    threads[f'{path.name}:{fields[19]}'] = {
                        'ticks': int(fields[11]) + int(fields[12]),
                        'cpus': sorted(os.sched_getaffinity(int(path.name)))}
                except (FileNotFoundError, ProcessLookupError):
                    pass
        except FileNotFoundError:
            pass
    return {'time_ns': time.perf_counter_ns(), 'wall_time': time.time(),
            'capture_start_boottime_ns': boot, 'clk_tck': CLK_TCK,
            'cpu': cpu, 'threads': threads, 'owned_pids': sorted(tree), 'gpu': query_gpu(gpu)}

def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m tailsplit.telemetry')
    p.add_argument('--root-pid', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stop-file', type=Path, required=True)
    a = p.parse_args(argv)
    deadline = time.monotonic()
    with a.output.open('x') as out:
        while True:
            stop = a.stop_file.exists()
            try:
                row = capture(a.root_pid, a.gpu)
            except Exception as error:
                row = {'time_ns': time.perf_counter_ns(), 'error': repr(error)}
            out.write(json.dumps(row) + '\n')
            out.flush()
            if stop:
                break
            deadline += 1
            time.sleep(max(0, deadline - time.monotonic()))

if __name__ == '__main__':
    main()
