"""Quiet check on recorded telemetry (tests/fixtures/head_jobs*): why the head-alone check covers
only the timed calls, credits threads that started after the previous sample, and is preceded
by a 3 s pre-roll."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from tailsplit import gates, head_alone, spec, telemetry
from tailsplit.analysis import quiet_intervals

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
NAMES = ('head-c1-b1', 'head-c8-b8', 'head-c32-b32')


def load():
    meta = json.loads((FIXTURES / 'head_jobs.json').read_text())
    with gzip.open(FIXTURES / 'head_jobs_telemetry.jsonl.gz', 'rt') as f:
        rows = [json.loads(line) for line in f]
    return meta, rows


def window(head):
    return head['launch_times_ns'][0], head['launch_times_ns'][-1] + head['gap_ns']


class Recorded(unittest.TestCase):
    def setUp(self):
        self.meta, self.rows = load()
        self.sockets = self.meta['sockets']

    def with_boot_fields(self):
        """These samples predate the start-time fields. On a machine that never suspended,
        CLOCK_BOOTTIME equals CLOCK_MONOTONIC; checked here through each job's /proc start time.
        The sample stamp (end of capture) is later than the capture start, so this credits less."""
        offset = self.rows[0]['wall_time'] - self.rows[0]['time_ns'] / 1e9
        for name, head in self.meta['heads'].items():
            starts = {int(k.split(':')[1]) for r in self.rows for k in r['threads'] if k.split(':')[0] == str(head['pid'])}
            self.assertEqual(len(starts), 1, name)
            self.assertLess(abs(starts.pop() / 100 - (head['launch_wall_time'] - offset)), .5, name)
        return [{**r, 'capture_start_boottime_ns': r['time_ns'], 'clk_tck': 100} for r in self.rows]

    def test_whole_process_window_charges_startup_threads_as_foreign(self):
        for name, head in self.meta['heads'].items():
            i, j = head['whole_process_samples']
            q = gates.quiet_gate(self.rows, self.sockets, self.rows[i]['time_ns'], self.rows[j]['time_ns'])
            self.assertFalse(q['pass'])
            self.assertIn('Foreign CPU time exceeds 2% on socket 1', q['reasons'])
            self.assertEqual(q['foreign_fraction_upper_bound']['1'], head['whole_process_foreign_socket1'])

    def test_crediting_alone_fixes_foreign_cpu(self):
        rows = self.with_boot_fields()
        for name, head in self.meta['heads'].items():
            i, j = head['whole_process_samples']
            q = gates.quiet_gate(rows, self.sockets, rows[i]['time_ns'], rows[j]['time_ns'])
            self.assertLess(q['foreign_fraction_upper_bound']['1'], .02, name)

    def test_timed_window_passes_foreign_cpu_but_needs_preroll_for_clock(self):
        rows = self.with_boot_fields()
        for name in NAMES:
            q = gates.quiet_gate(rows, self.sockets, *window(self.meta['heads'][name]))
            self.assertLess(q['foreign_fraction_upper_bound']['1'], .02, name)
        # Without a pre-roll the sample before the window is the idle CUDA-start sample.
        for name in ('head-c1-b1', 'head-c8-b8'):
            q = gates.quiet_gate(rows, self.sockets, *window(self.meta['heads'][name]))
            self.assertEqual(q['reasons'], ['GPU clock stability failed'])
        self.assertTrue(gates.quiet_gate(rows, self.sockets, *window(self.meta['heads']['head-c32-b32']))['pass'])

    def test_labelled_intervals_and_missing_window(self):
        checks = quiet_intervals(self.rows, self.sockets,
                                 [('head-c32-b32', *window(self.meta['heads']['head-c32-b32'])), ('head-x', None, None)])
        self.assertTrue(checks[0]['pass'])
        self.assertEqual(checks[0]['interval'], 'head-c32-b32')
        self.assertFalse(checks[1]['pass'])


class ClockDomain(unittest.TestCase):
    def test_perf_counter_is_system_wide_monotonic(self):
        self.assertEqual(time.get_clock_info('perf_counter').implementation, head_alone.CLOCK)
        before = time.perf_counter_ns()
        child = int(subprocess.check_output([sys.executable, '-c', 'import time;print(time.perf_counter_ns())'], text=True))
        after = time.perf_counter_ns()
        self.assertTrue(before < child < after)

    def test_telemetry_records_boot_clock_before_thread_read(self):
        with patch.object(telemetry, 'query_gpu', return_value={'clock_mhz': 1, 'compute_pids': []}):
            before = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
            row = telemetry.capture(os.getpid(), 0)
        self.assertEqual(row['clk_tck'], os.sysconf('SC_CLK_TCK'))
        self.assertGreaterEqual(row['capture_start_boottime_ns'], before)
        self.assertIn(os.getpid(), row['owned_pids'])
        self.assertTrue(any(gates.started_after(k, {'capture_start_boottime_ns': 0, 'clk_tck': row['clk_tck']}, row)
                            for k in row['threads']))


class FakeClock:
    def __init__(self, cost_ns=200_000):
        self.now, self.cost = 10 ** 12, cost_ns

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)

    def call(self, i):
        self.now += self.cost


class PreRoll(unittest.TestCase):
    def test_preroll_at_least_3s_at_gap_spacing(self):
        for gap_ms in (5.97, 7.3, 40.0, 3500.0):
            c = FakeClock()
            g = head_alone.gap_ns_of(gap_ms)
            pre = head_alone.paced(c.call, g, min_span_ns=spec.PREROLL_NS, clock=c.clock, sleep=c.sleep)
            timed = head_alone.paced(c.call, g, count=200, after_ns=pre[-1], clock=c.clock, sleep=c.sleep)
            self.assertGreaterEqual(timed[0] - pre[0], spec.PREROLL_NS)
            self.assertLess(timed[0] - pre[0], spec.PREROLL_NS + g)
            stamps = pre + timed
            self.assertTrue(all(b - a == g for a, b in zip(stamps, stamps[1:])))
            self.assertEqual(len(timed), 200)
            micro = {'clock': head_alone.CLOCK, 'gap_ns': g, 'preroll_launch_times_ns': pre, 'launch_times_ns': timed}
            self.assertEqual(gates.head_timed_window(micro), (timed[0], timed[-1] + g))

    def test_paced_arguments(self):
        with self.assertRaises(ValueError):
            head_alone.paced(lambda i: None, 1, count=1, min_span_ns=1)


if __name__ == '__main__':
    unittest.main()
