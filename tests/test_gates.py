"""Checks, shares and decision rule on synthetic records."""
import copy
import json
import unittest

from tailsplit import gates, spec
from tests.helpers import cell, requests, rows, summary, telemetry


class Shares(unittest.TestCase):
    def test_gate_arithmetic(self):
        result = gates.condition_gates(cell())
        self.assertTrue(all(result['gates'][g]['pass'] for g in 'VDAXTQ'))
        self.assertEqual(set(result['gates']), set(gates.NEEDED['PLAIN']))
        self.assertAlmostEqual(result['shares'][0]['H'], .2)
        self.assertAlmostEqual(result['shares'][0]['S'], .04)
        self.assertAlmostEqual(result['residual'], .05)
        # 20 requests x (255 periods - 16 discarded steps) usable steps
        self.assertEqual(len(gates.selected_steps(rows(), requests())), 20 * 239)

    def test_both_denominators(self):
        c = cell()
        for r in c['patched_requests']:
            r['content_times_ns'] = [int(t * 1.02) for t in r['content_times_ns']]
        for row in c['rows']:
            if row['period_ms'] is not None:
                row['period_ms'] = 10.2
        result = gates.condition_gates(c)
        self.assertAlmostEqual(result['g_ms'], 10)
        self.assertAlmostEqual(result['gp_ms'], 10.2, places=5)
        self.assertAlmostEqual(result['shares'][0]['H'], .2)
        self.assertAlmostEqual(result['shares'][1]['H'], 2 / 10.2, places=5)

    def test_accounting_is_falsifiable(self):
        for mutate in ('chunks', 'period', 'sum', 'identity', 'steps', 'prompt'):
            c = cell()
            if mutate == 'chunks':
                c['patched_requests'][0]['content_times_ns'].pop()
            elif mutate == 'period':
                for row in c['rows']:
                    row['period_ms'] = 20
            elif mutate == 'sum':
                for row in c['rows']:
                    row['segments_ms']['f'] = 20
            elif mutate == 'identity':
                c['rows'][0]['req_ids'] = ['other']
            elif mutate == 'steps':
                c['rows'].pop()
            else:
                c['patched_requests'][0]['usage']['prompt_tokens'] = 255
            with self.subTest(mutate=mutate):
                self.assertFalse(gates.condition_gates(c)['gates']['A']['pass'])

    def test_token_and_distortion_failure(self):
        c = cell()
        c['baseline_requests'][0]['usage']['completion_tokens'] = 255
        self.assertFalse(gates.condition_gates(c)['gates']['T']['pass'])
        c = cell()
        for r in c['patched_requests']:
            r['content_times_ns'] = [int(t * 1.04) for t in r['content_times_ns']]
        self.assertFalse(gates.condition_gates(c)['gates']['D']['pass'])

    def test_head_crosscheck_failure(self):
        c = cell()
        c['head_alone']['1'] = {'median_ms': 1, 'times_ms': [1] * 200}
        self.assertFalse(gates.condition_gates(c)['gates']['X']['pass'])
        c = cell()
        c['head_alone'] = {}
        self.assertFalse(gates.condition_gates(c)['gates']['X']['pass'])

    def test_mask_gates(self):
        c = cell()
        c['condition'] = 'JSON'
        result = gates.condition_gates(c, 10)
        self.assertTrue(result['gates']['J']['pass'] and result['gates']['B']['pass'])
        self.assertEqual(set(result['gates']), set(gates.NEEDED['JSON']))
        self.assertFalse(gates.condition_gates(c, 11)['gates']['B']['pass'])
        c['rows'][0]['mask_active'] = False
        self.assertFalse(gates.condition_gates(c, 10)['gates']['J']['pass'])
        for mutate in ('probe', 'finish', 'log', 'm0', 'empty'):
            c = cell()
            c['condition'] = 'JSON'
            if mutate == 'probe':
                c['schema_probes'][0]['text'] = '[true]'
            elif mutate == 'finish':
                c['patched_requests'][3]['finish_reason'] = 'stop'
            elif mutate == 'log':
                c['logs'][1] = 'ERROR xgrammar: failed to advance the matcher'
            elif mutate == 'm0':
                c['rows'][100]['segments_ms']['m'] = 0
            else:
                c['rows'][0]['mask_request_ids'] = []
            with self.subTest(mutate=mutate):
                self.assertFalse(gates.condition_gates(c, 10)['gates']['J']['pass'])

    def test_valid_schema(self):
        self.assertTrue(gates.valid_schema(json.dumps([1] * 400)))
        self.assertFalse(gates.valid_schema(json.dumps([1] * 399)))
        self.assertFalse(gates.valid_schema(json.dumps([1] * 601)))
        self.assertFalse(gates.valid_schema('[1, 2'))
        self.assertFalse(gates.valid_schema(json.dumps([1.5] * 400)))

    def test_empty_measurements_fail_closed(self):
        c = cell()
        c['patched_requests'] = []
        result = gates.condition_gates(c)
        self.assertFalse(result['gates']['A']['pass'])
        self.assertFalse(result['gates']['D']['pass'])
        self.assertIn('invalid_evidence', result)

    def test_chunked_prefill_rows_are_accepted_once_per_request(self):
        r = rows()
        rid = 'r0-1234abcd'
        first = next(x for x in r if x['req_ids'] == [rid])
        extra = copy.deepcopy(first)
        for x in r:
            if x['req_ids'] == [rid]:
                x['request_steps'] = {rid: x['request_steps'][rid] + 1}
        r.insert(r.index(first), extra)
        gates.matched_steps(r, requests())
        extra['prefill'] = [False]
        with self.assertRaisesRegex(ValueError, '255 decode steps'):
            gates.matched_steps(r, requests())


class Padding(unittest.TestCase):
    def test_padding_excluded_from_gaps_tokens_mask_and_interval(self):
        c = cell()
        for arm in ('baseline_requests', 'patched_requests'):
            for r in c[arm]:
                r.update(started_ns=100, ended_ns=200)
            c[arm].append({'response_id': 'pad', 'phase': 'padding', 'concurrency': 1, 'started_ns': 0, 'ended_ns': 9999})
        c['rows'].append({'req_ids': ['pad'], 'request_steps': {'pad': 1}, 'batch_size': 1, 'prefill': [True]})
        self.assertEqual(gates.gap_ms(c['baseline_requests']), 10)
        self.assertEqual(gates.measured_interval(c['baseline_requests'], 1), (100, 200))
        result = gates.condition_gates(c)
        self.assertTrue(all(result['gates'][g]['pass'] for g in 'VDAXTQ'), result)
        c['condition'] = 'JSON'
        self.assertTrue(gates.condition_gates(c, 10)['gates']['J']['pass'])

    def test_mixed_batch_not_used_as_measured_gpu_timing(self):
        records = requests()
        records.append({'response_id': 'pad', 'phase': 'padding', 'concurrency': 1})
        raw = rows()
        mixed = raw[17]
        mixed['req_ids'].append('pad')
        mixed['request_steps']['pad'] = 1
        mixed['prefill'].append(False)
        mixed['batch_size'] = 2
        mixed['segments_ms'] = {}
        selected = gates.selected_steps(raw, records)
        self.assertEqual(len(selected), 20 * 239 - 1)
        self.assertNotIn(mixed, selected)
        raw.pop(17)
        with self.assertRaisesRegex(ValueError, '255 decode steps'):
            gates.selected_steps(raw, records)

    def test_unknown_worker_identity_and_unlabelled_client_fail(self):
        for mutate in ('unknown', 'phase', 'duplicate', 'count', 'concurrency'):
            c = cell()
            if mutate == 'unknown':
                c['rows'][0]['req_ids'] = ['unknown']
            elif mutate == 'phase':
                del c['patched_requests'][0]['phase']
            elif mutate == 'duplicate':
                c['patched_requests'].append(c['patched_requests'][0])
            elif mutate == 'count':
                c['patched_requests'].pop()
            else:
                c['patched_requests'][0]['concurrency'] = 32
            with self.subTest(mutate=mutate):
                self.assertFalse(gates.condition_gates(c)['gates']['A']['pass'])

    def test_c32_counts_and_token_drop_denominator(self):
        c = cell()
        c['concurrency'] = 32
        for arm in ('baseline_requests', 'patched_requests'):
            c[arm] = [dict(copy.deepcopy(c[arm][i % 20]), response_id=f'r{i}', concurrency=32) for i in range(64)]
            c[arm].append({'response_id': 'pad', 'phase': 'padding', 'concurrency': 32})
        c['rows'] = rows(n=64)
        self.assertTrue(gates.condition_gates(c)['gates']['A']['pass'])
        self.assertEqual(len(gates.selected_steps(c['rows'], c['patched_requests'], 32)), 64 * 239)
        for arm in ('baseline_requests', 'patched_requests'):
            short = c[arm][0]
            short['usage']['completion_tokens'] = 255
            short['content_times_ns'].pop()
            short['finish_reason'] = 'stop'
        c['rows'] = [r for r in c['rows'] if r['request_steps'].get('r0-1234abcd') != 256]
        result = gates.condition_gates(c)
        self.assertTrue(all(result['gates'][g]['pass'] for g in 'VDAXTQ'), result)   # 1 of 64 dropped
        self.assertEqual(len(gates.selected_steps(c['rows'], c['patched_requests'], 32)), 63 * 239)
        c['condition'] = 'JSON'
        self.assertTrue(gates.condition_gates(c, 10)['gates']['J']['pass'])
        c['baseline_requests'][1]['usage']['completion_tokens'] = 255   # 2 of 64 > 2%
        self.assertFalse(gates.condition_gates(c)['gates']['T']['pass'])


class RunnerGate(unittest.TestCase):
    def test_runner_gate_per_quantization_method(self):
        def evidence(method, scheme, kernel, n=112):
            linears = [{'name': f'model.layers.{i // 4}.l{i % 4}', 'module': 'QKVParallelLinear',
                        'method': method, 'scheme': scheme, 'kernel': kernel} for i in range(n)]
            head = [{'name': 'lm_head', 'module': 'ParallelLMHead', 'method': 'UnquantizedEmbeddingMethod',
                     'scheme': 'NoneType', 'kernel': 'NoneType'}]
            return {'runner_module': 'vllm.v1.worker.gpu.model_runner', 'head_dtype': 'torch.float16',
                    'head_element_size': 2, 'head_quant_method': 'UnquantizedEmbeddingMethod',
                    'head_shape': [152064, 3584], 'quant_methods': linears + head}
        log = 'Using V2 Model Runner\nUsing MarlinLinearKernel for AutoGPTQLinearMethod\n'
        gptq = spec.QUANT['qwen25-w4']
        ok = evidence('AutoGPTQLinearMethod', 'NoneType', 'MarlinLinearKernel')
        self.assertTrue(gates.runner_gate(ok, log, gptq)['pass'])
        self.assertFalse(gates.runner_gate(evidence('AutoGPTQLinearMethod', 'NoneType', 'MarlinLinearKernel', 111),
                                           log, gptq)['pass'])
        mixed = copy.deepcopy(ok)
        mixed['quant_methods'].append({'name': 'x', 'module': 'RowParallelLinear', 'method': 'UnquantizedLinearMethod',
                                       'scheme': 'NoneType', 'kernel': 'NoneType'})
        self.assertFalse(gates.runner_gate(mixed, log, gptq)['pass'])
        self.assertFalse(gates.runner_gate(evidence('AutoGPTQLinearMethod', 'NoneType', 'ExllamaLinearKernel'),
                                           log, gptq)['pass'])
        self.assertFalse(gates.runner_gate(ok, 'Using V2 Model Runner\n', gptq)['pass'])
        quantized_head = dict(ok, head_quant_method='GPTQEmbeddingMethod')
        self.assertFalse(gates.runner_gate(quantized_head, log, gptq)['pass'])
        v1 = dict(ok, runner_module='vllm.v1.worker.gpu_model_runner')
        self.assertFalse(gates.runner_gate(v1, log, gptq)['pass'])
        ct_log = 'Using V2 Model Runner\nUsing MarlinLinearKernel for CompressedTensorsWNA16\n'
        ct = spec.QUANT['qwen3-w4']
        self.assertTrue(gates.runner_gate(evidence('CompressedTensorsLinearMethod', 'CompressedTensorsWNA16',
                                                   'MarlinLinearKernel'), ct_log, ct)['pass'])
        dense = evidence('UnquantizedLinearMethod', 'NoneType', 'NoneType')
        self.assertTrue(gates.runner_gate(dense, 'Using V2 Model Runner\n', None)['pass'])
        self.assertFalse(gates.runner_gate(ok, 'Using V2 Model Runner\n', None)['pass'])


class Decision(unittest.TestCase):
    def test_dual_denominators(self):
        rounds = [summary() for _ in range(5)]
        self.assertEqual(gates.question(rounds, 'PLAIN'), 'BIG')
        rounds[0]['shares'][1]['H'] = .149
        self.assertEqual(gates.question(rounds, 'PLAIN'), 'MIXED')
        small = [summary(h=.1) for _ in range(5)]
        small[0]['shares'][1]['H'] = .151
        self.assertEqual(gates.question(small, 'PLAIN'), 'MIXED')

    def test_thresholds_are_inclusive_where_written(self):
        self.assertEqual(gates.question([summary(h=.15, residual=.15)] * 5, 'PLAIN'), 'BIG')
        self.assertEqual(gates.question([summary(m=.05, s=.05, residual=.15)] * 5, 'JSON'), 'BIG')
        self.assertEqual(gates.question([summary(m=.05, s=.0499)] * 5, 'JSON'), 'SMALL')

    def test_residual_and_missing_round(self):
        self.assertEqual(gates.question([summary(residual=.16)] * 5, 'PLAIN'), 'MIXED')
        self.assertEqual(gates.question([summary()] * 4, 'PLAIN'), 'NOT MEASURED')
        r = [summary() for _ in range(5)]
        r[2]['gates']['X']['pass'] = False
        self.assertEqual(gates.question(r, 'PLAIN'), 'NOT MEASURED')
        # X is not needed by the fusion question, B is not needed by the head question
        self.assertEqual(gates.question(r, 'JSON'), 'SMALL')
        r = [summary() for _ in range(5)]
        del r[1]['gates']['B']
        self.assertEqual(gates.question(r, 'JSON'), 'NOT MEASURED')
        self.assertEqual(gates.question(r, 'PLAIN'), 'BIG')

    def test_two_questions_and_stop(self):
        r = gates.decision([summary()] * 5, [summary(m=.05, s=.06)] * 5)
        self.assertEqual((r['head'], r['fusion']), ('BIG', 'BIG'))
        self.assertEqual(r['outcomes'], ['head BIG', 'fusion BIG'])
        small = [summary(h=.02, m=.01, s=.01)] * 5
        self.assertEqual(gates.decision(small, small)['outcomes'], ['STOP'])
        self.assertFalse(gates.decision(small, small)['expectation_met'])
        tail = [summary(h=.08, m=.01, s=.02)] * 5   # SMALL and SMALL but T = 0.11
        self.assertEqual(gates.decision(tail, tail)['outcomes'], ['none'])
        self.assertEqual(gates.decision([], small)['head'], 'NOT MEASURED')
        self.assertEqual(gates.decision([], small)['outcomes'], ['none'])

    def test_expectation(self):
        self.assertTrue(gates.decision([summary(h=.19)] * 5, [summary(m=.0, s=.02)] * 5)['expectation_met'])
        self.assertFalse(gates.decision([summary(h=.26)] * 5, [summary(m=.0, s=.02)] * 5)['expectation_met'])
        self.assertFalse(gates.decision([summary(h=.19)] * 5, [summary(m=.05, s=.06)] * 5)['expectation_met'])


class Quiet(unittest.TestCase):
    def test_quiet_gate_coverage_and_foreign(self):
        args = ({'0': [0], '1': [1]}, 500_000_000, 2_500_000_000)
        self.assertTrue(gates.quiet_gate(telemetry(), *args)['pass'])
        for kind in ('cpu', 'thread', 'foreign', 'clock', 'migration', 'bracket', 'error', 'busy'):
            s = telemetry()
            if kind == 'cpu':
                s[1]['cpu'].pop('1')
            elif kind == 'thread':
                s[1]['threads'] = {}
            elif kind == 'foreign':
                s[1]['gpu']['compute_pids'].append(2)
            elif kind == 'migration':
                s[1]['threads']['0:1']['cpus'] = [1]
            elif kind == 'bracket':
                s = s[1:]
            elif kind == 'error':
                s[2] = {'time_ns': s[2]['time_ns'], 'error': 'nvidia-smi failed'}
            elif kind == 'busy':
                for i, row in enumerate(s):
                    row['cpu']['1']['busy'] = i * 13   # 3% of socket 1 not owned
            else:
                s[1]['gpu']['clock_mhz'] = 1000
            self.assertFalse(gates.quiet_gate(s, *args)['pass'], kind)

    def test_started_after_boundaries(self):
        a = {'capture_start_boottime_ns': 1_000_000_000, 'clk_tck': 100}
        self.assertFalse(gates.started_after('5:100', a, a))   # equal is not after
        self.assertTrue(gates.started_after('5:101', a, a))
        self.assertFalse(gates.started_after('5:101', a, {**a, 'clk_tck': 1000}))
        self.assertFalse(gates.started_after('5', a, a))

    def crediting_rows(self, start_ticks, boot=True):
        result = []
        for i in range(3):
            threads = {'0:1': {'ticks': 0, 'cpus': [0]}, '1:1': {'ticks': 0, 'cpus': [1]}}
            if i == 2:
                threads[f'9:{start_ticks}'] = {'ticks': 40, 'cpus': [1]}
            row = {'time_ns': i * 1_000_000_000, 'owned_pids': [1, 9],
                   'cpu': {'0': {'total': i * 100, 'busy': 0}, '1': {'total': i * 100, 'busy': i * 20}},
                   'threads': threads, 'gpu': {'compute_pids': [], 'clock_mhz': 2000}}
            if boot:
                row.update(capture_start_boottime_ns=i * 1_000_000_000 - 1000, clk_tck=100)
            result.append(row)
        return result

    def test_new_thread_started_after_previous_sample_is_owned(self):
        q = gates.quiet_gate(self.crediting_rows(150), {'0': [0], '1': [1]}, 0, 2_000_000_000)
        self.assertEqual(q['foreign_fraction_upper_bound']['1'], 0)

    def test_unknown_start_stays_foreign(self):
        for r in (self.crediting_rows(50), self.crediting_rows(150, boot=False)):
            q = gates.quiet_gate(r, {'0': [0], '1': [1]}, 0, 2_000_000_000)
            self.assertAlmostEqual(q['foreign_fraction_upper_bound']['1'], .2)
            self.assertFalse(q['pass'])


class HeadWindow(unittest.TestCase):
    def micro(self):
        from tests.helpers import micro
        return micro(10 ** 12, 6.0)

    def test_window_bounds(self):
        m = self.micro()
        self.assertEqual(gates.head_timed_window(m), (m['launch_times_ns'][0], m['launch_times_ns'][-1] + m['gap_ns']))

    def test_window_fails_closed(self):
        bad = [lambda m: m.update(clock='clock_gettime(CLOCK_REALTIME)'),
               lambda m: m.update(launch_times_ns=m['launch_times_ns'][:199]),
               lambda m: m.update(preroll_launch_times_ns=m['preroll_launch_times_ns'][-10:]),
               lambda m: m.update(preroll_launch_times_ns=[]),
               lambda m: m.update(gap_ns=0),
               lambda m: m['launch_times_ns'].__setitem__(5, m['launch_times_ns'][4])]
        for mutate in bad:
            m = self.micro()
            mutate(m)
            with self.assertRaises(ValueError):
                gates.head_timed_window(m)
        m = self.micro()
        m.pop('launch_times_ns')
        with self.assertRaises(KeyError):
            gates.head_timed_window(m)


if __name__ == '__main__':
    unittest.main()
