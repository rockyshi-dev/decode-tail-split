"""Client: SSE parsing, request bodies, closed loop with padding. In-memory transport only."""
import concurrent.futures
import io
import json
import queue
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tailsplit import client, prompts, spec
from tests.helpers import prompt_manifest


def wire(data):
    return (''.join('data: ' + json.dumps(d) + '\r\n\r\n' for d in data) + 'data: [DONE]\n\n').encode()


class Bodies(unittest.TestCase):
    def test_default_requests(self):
        for condition in spec.CONDITIONS:
            body = spec.request_body('/model', [], condition)
            self.assertFalse({'seed', 'temperature', 'top_p', 'top_k', 'min_tokens', 'repetition_penalty'} & set(body))
            self.assertEqual(body['max_tokens'], 256)
            self.assertEqual(body['stream_options'], {'include_usage': True})
            self.assertEqual('ignore_eos' in body, condition == 'PLAIN')
            self.assertEqual('structured_outputs' in body, condition == 'JSON')
        self.assertEqual(spec.SCHEMA['minItems'], 400)

    def test_counts(self):
        self.assertEqual([spec.warmup_count(c) for c in spec.CONCURRENCIES], [4, 8, 32])
        self.assertEqual([spec.measured_count(c) for c in spec.CONCURRENCIES], [20, 20, 64])
        self.assertEqual(spec.PROMPT_COUNT, 96)
        for value in (0, 2, True):
            with self.assertRaises(ValueError):
                spec.validate_concurrency(value)


class Stream(unittest.TestCase):
    def test_sse_boundaries_and_usage(self):
        data = [{'id': 'x', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]},
                {'id': 'x', 'choices': [{'index': 0, 'delta': {'content': '1'}, 'finish_reason': None}]},
                {'id': 'x', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]},
                {'id': 'x', 'choices': [], 'usage': {'completion_tokens': 1}}]
        result = client.summarize(client.read_sse(io.BytesIO(wire(data))))
        self.assertEqual(len(result['content_times_ns']), 1)
        self.assertEqual((result['text'], result['finish_reason'], result['response_id']), ('1', 'length', 'x'))
        with self.assertRaisesRegex(RuntimeError, 'DONE'):
            client.read_sse(io.BytesIO(b''))

    def test_empty_token_text_counts_but_role_chunk_does_not(self):
        data = [{'id': 'x', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]},
                {'id': 'x', 'choices': [{'index': 0, 'delta': {'content': '1'}, 'finish_reason': None}]},
                {'id': 'x', 'choices': [{'index': 0, 'delta': {'content': ''}, 'finish_reason': None}]},
                {'id': 'x', 'choices': [{'index': 0, 'delta': {'content': '2'}, 'finish_reason': 'length'}]},
                {'id': 'x', 'choices': [], 'usage': {'completion_tokens': 3}}]
        result = client.summarize(client.read_sse(io.BytesIO(wire(data))))
        self.assertEqual(len(result['content_times_ns']), 3)
        self.assertEqual(result['text'], '12')

    def test_server_error_and_inconsistent_stream_fail(self):
        with self.assertRaises(RuntimeError):
            client.read_sse(io.BytesIO(wire([{'error': {'message': 'boom'}}])))
        two_ids = [{'id': 'x', 'choices': [{'index': 0, 'delta': {'content': '1'}, 'finish_reason': 'length'}]},
                   {'id': 'y', 'choices': [], 'usage': {}}]
        with self.assertRaises(RuntimeError):
            client.summarize(client.read_sse(io.BytesIO(wire(two_ids))))

    def test_idle_pause_does_not_clear_cache(self):
        calls = []
        def post(port, route, body=None):
            calls.append((route, body))
            return {'results': [{'rows': []}]} if route == '/collective_rpc' else {}
        with patch.object(client, 'post', post):
            self.assertEqual(client.control(1, 'drain'), {'rows': []})
        self.assertEqual([r for r, _ in calls], ['/pause?mode=wait&clear_cache=false', '/collective_rpc', '/resume'])
        self.assertEqual(calls[1][1]['method'], 'tailsplit_control')

    def test_resume_even_if_rpc_fails(self):
        calls = []
        def post(port, route, body=None):
            calls.append(route)
            if route == '/collective_rpc':
                raise OSError('rpc failed')
            return {}
        with patch.object(client, 'post', post), self.assertRaises(OSError):
            client.control(1, 'arm')
        self.assertEqual(calls[-1], '/resume')

    def test_dispatch_callback_follows_http_body_write(self):
        calls = []
        connection = SimpleNamespace(
            request=lambda *args: calls.append('body'),
            getresponse=lambda: (calls.append('response') or SimpleNamespace(status=200)),
            close=lambda: calls.append('close'))
        with patch.object(client.http.client, 'HTTPConnection', return_value=connection), \
             patch.object(client, 'read_sse', return_value=[]), patch.object(client, 'summarize', return_value={}):
            client.one(0, {}, on_dispatched=lambda: calls.append('dispatched'))
        self.assertEqual(calls, ['body', 'dispatched', 'response', 'close'])

    def test_fresh_connection_per_request(self):
        connections = [Mock(), Mock()]
        for conn in connections:
            conn.getresponse.return_value.status = 200
        with patch.object(client.http.client, 'HTTPConnection', side_effect=connections) as make, \
             patch.object(client, 'read_sse', return_value=[]), patch.object(client, 'summarize', return_value={}):
            client.one(8155, {})
            client.one(8155, {})
        self.assertEqual(make.call_count, 2)
        for conn in connections:
            conn.request.assert_called_once()
            conn.close.assert_called_once()


class ClosedLoop(unittest.TestCase):
    def test_malformed_prompts_fail_before_send(self):
        for mutate in ('short', 'tokens'):
            m = prompt_manifest()
            if mutate == 'short':
                m['prompts'] = m['prompts'][:24]
            else:
                m['prompts'][-1]['prompt_token_ids'].pop()
            with patch.object(client, 'one') as send:
                with self.assertRaises(ValueError):
                    client.batch(0, '/m', m, 'PLAIN', 32, False, 'fixture')
                send.assert_not_called()

    def test_warmup_and_schema_have_no_padding(self):
        for c in spec.CONCURRENCIES:
            for phase in ('warmup', 'schema') if c == 1 else ('warmup',):
                with patch.object(client, 'one', side_effect=lambda port, body: {'request': body}):
                    result = client.batch(0, '/m', prompt_manifest(), 'JSON', c, phase == 'warmup', 'fixture',
                                          4096 if phase == 'schema' else 256)
                self.assertEqual(len(result), 1 if phase == 'schema' else spec.warmup_count(c))
                self.assertEqual({r['phase'] for r in result}, {phase})
                self.assertEqual({r['request']['model'] for r in result}, {'/m'})
                self.assertEqual({r['request']['max_tokens'] for r in result}, {4096 if phase == 'schema' else 256})
        with self.assertRaises(ValueError):
            client.batch(0, '/m', prompt_manifest(), 'JSON', 8, False, 'fixture', 4096)

    def test_closed_loop_refills_through_last_measured_completion(self):
        for c in spec.CONCURRENCIES:
            with self.subTest(concurrency=c):
                arrivals, events, guard = queue.Queue(), [], threading.Lock()

                def send(port, body, on_dispatched=None):
                    if on_dispatched:
                        on_dispatched()
                    event = threading.Event()
                    with guard:
                        events.append(event)
                    arrivals.put((body, event))
                    if not event.wait(10):
                        raise RuntimeError('Test controller did not release stub')
                    return {'request': body, 'response_id': body['request_id']}

                def receive():
                    return arrivals.get(timeout=5)

                with patch.object(client, 'one', side_effect=send), \
                     concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(client.batch, 0, '/m', prompt_manifest(), 'JSON', c, False, 'fixture')
                    try:
                        pending = [receive() for _ in range(c)]
                        count = spec.measured_count(c)
                        for _ in range(count - c):
                            pending.pop(-1 if c > 1 else 0)[1].set()
                            pending.append(receive())
                        if c > 1:
                            pending.pop()[1].set()
                            pad = receive()
                            self.assertIn('-padding-', pad[0]['request_id'])
                            pad[1].set()
                            pending.append(receive())
                        while True:
                            measured = [i for i, (b, _) in enumerate(pending) if '-measured-' in b['request_id']]
                            if not measured:
                                break
                            pending.pop(measured[-1])[1].set()
                            if len(measured) > 1:
                                replacement = receive()
                                self.assertIn('-padding-', replacement[0]['request_id'])
                                pending.append(replacement)
                        for _, event in pending:
                            event.set()
                        for _ in range(100):
                            if future.done():
                                break
                            try:
                                body, event = arrivals.get(timeout=.05)
                            except queue.Empty:
                                continue
                            self.assertIn('-padding-', body['request_id'])
                            event.set()
                        result = future.result(timeout=5)
                        self.assertTrue(arrivals.empty())
                    finally:
                        for event in events:
                            event.set()
                measured = [r for r in result if r['phase'] == 'measured']
                padding = [r for r in result if r['phase'] == 'padding']
                self.assertEqual(len(measured), count)
                self.assertEqual(len({r['response_id'] for r in result}), len(result))
                self.assertEqual({r['concurrency'] for r in result}, {c})
                self.assertEqual({r['slot'] for r in result}, set(range(c)))
                self.assertEqual({r['request_index'] for r in measured}, set(range(count)))
                self.assertEqual(bool(padding), c > 1)
                self.assertTrue(all(r['request']['structured_outputs']['json'] == spec.SCHEMA for r in result))
                self.assertTrue(all(r['prompt_index'] >= spec.warmup_count(c) for r in result))

    def test_transport_error_stops_refills_and_propagates(self):
        with patch.object(client, 'one', side_effect=RuntimeError('fixture failure')) as send:
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                client.batch(0, '/m', prompt_manifest(), 'PLAIN', 8, False, 'fixture')
        self.assertLessEqual(send.call_count, 8)

    def test_padding_waits_for_actual_last_measured_dispatch(self):
        final_entered, release_final, padding_entered = threading.Event(), threading.Event(), threading.Event()
        def send(port, body, on_dispatched=None):
            if body['request_id'].endswith('-measured-000019'):
                final_entered.set()
                if not release_final.wait(5):
                    raise RuntimeError('Dispatch fixture timeout')
            if '-padding-' in body['request_id']:
                padding_entered.set()
            if on_dispatched:
                on_dispatched()
            return {'request': body, 'response_id': body['request_id']}
        with patch.object(client, 'one', side_effect=send), \
             concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(client.batch, 0, '/m', prompt_manifest(), 'PLAIN', 8, False, 'dispatch')
            try:
                self.assertTrue(final_entered.wait(5))
                self.assertFalse(padding_entered.wait(.1))
            finally:
                release_final.set()
            result = future.result(timeout=5)
        self.assertEqual(sum(r['phase'] == 'measured' for r in result), 20)

    def test_cli_writes_records_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec.write_json(root / 'm.json', prompt_manifest())
            argv = ['--manifest', str(root / 'm.json'), '--model-path', '/m', '--port', '1', '--condition', 'PLAIN',
                    '--concurrency', '1', '--label', 'x', '--phase', 'warmup', '--output', str(root / 'out.json')]
            with patch.object(client, 'one', side_effect=lambda port, body: {'request': body}):
                client.main(argv)
            self.assertEqual(len(spec.read_json(root / 'out.json')), 4)
            with patch.object(client, 'one', side_effect=RuntimeError('refused')), self.assertRaises(RuntimeError):
                client.main(argv[:-1] + [str(root / 'bad.json')])
            self.assertIn('refused', spec.read_json(root / 'bad.client-failure.json')['error'])


class Prompts(unittest.TestCase):
    def test_constructs_96_offline_prompts(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1] * (200 + messages[0]['content'].count(' number'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = spec.snapshot(root, 'qwen3-bf16')
            snap.mkdir(parents=True)
            (snap / 'tokenizer_config.json').write_text('{}')
            path = prompts.construct('qwen3-bf16', root, root / 'out', tokenizer=Tokenizer())
            result = spec.read_json(path)
        self.assertEqual(len(result['prompts']), 96)
        self.assertEqual(len({p['messages'][0]['content'] for p in result['prompts']}), 96)
        self.assertEqual(result['revision'], spec.MODELS['qwen3-bf16'][1])
        self.assertIn('tokenizer_config.json', result['tokenizer_files'])
        self.assertTrue(all(p['messages'][0]['content'].count(' number') == 56 for p in result['prompts']))
        for c in spec.CONCURRENCIES:
            spec.validate_prompts(result, c)

    def test_unreachable_length_fails(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1] * 1000
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                prompts.construct('qwen3-bf16', tmp, tmp, tokenizer=Tokenizer())


if __name__ == '__main__':
    unittest.main()
