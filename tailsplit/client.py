"""Streaming chat client: records the arrival time of every streamed token.

Run by the coordinator as a separate process pinned to its own CPUs
(python -m tailsplit.client). No request is sent by importing this module."""
import argparse
import concurrent.futures
import http.client
import json
import threading
import time
import urllib.request
from pathlib import Path
from .spec import (OUTPUT_TOKENS, PROBE_TOKENS, measured_count, request_body, validate_prompts,
                   warmup_count, write_json)

def post(port, route, body=None):
    data = json.dumps(body or {}).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}{route}', data=data,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)

def control(port, action):
    # The server rejects new generation and drains in-flight work before RPC.
    post(port, '/pause?mode=wait&clear_cache=false')
    try:
        result = post(port, '/collective_rpc', {'method': 'tailsplit_control',
                    'args': [action, 'engine-paused-wait-clear_cache-false']})['results']
        if len(result) != 1:
            raise RuntimeError('Expected exactly one worker')
        return result[0]
    finally:
        post(port, '/resume')

def read_sse(response, now=time.perf_counter_ns):
    events, data = [], []
    done = False
    while True:
        line = response.readline()
        arrival = now()
        if not line:
            break
        line = line.decode('utf-8').rstrip('\r\n')
        if line.startswith('data:'):
            data.append(line[5:].lstrip())
        elif line == '' and data:
            payload = '\n'.join(data)
            data = []
            if payload == '[DONE]':
                done = True
                break
            event = json.loads(payload)
            if 'error' in event:
                raise RuntimeError(event['error'])
            events.append({'time_ns': arrival, 'data': event})
    if not done:
        raise RuntimeError('SSE ended without [DONE]')
    return events

def summarize(events):
    chunks, text, ids, finishes, usage = [], [], set(), [], None
    for event in events:
        data = event['data']
        if data.get('id'):
            ids.add(data['id'])
        if data.get('usage') is not None:
            usage = data['usage']
        for choice in data.get('choices', []):
            if choice.get('index') != 0:
                raise RuntimeError('Unexpected multi-choice response')
            delta = choice.get('delta', {})
            content = delta.get('content')
            # One streamed delta per generated token. Without a parser, vLLM v0.30.0
            # sends content='' for a token whose text is empty (special token, partial
            # UTF-8; chat_completion/serving.py:642-666), so '' counts as a token chunk.
            # The first chunk carries only the role (serving.py:542) and is not a token.
            if isinstance(content, str) and delta.get('role') is None:
                chunks.append(event['time_ns'])
                text.append(content)
            if choice.get('finish_reason') is not None:
                finishes.append(choice['finish_reason'])
    if len(ids) != 1 or len(finishes) != 1 or usage is None:
        raise RuntimeError('Incomplete or inconsistent SSE identity/finish/usage')
    return {'response_id': ids.pop(), 'content_times_ns': chunks,
            'text': ''.join(text), 'finish_reason': finishes[0], 'usage': usage}

def one(port, body, on_dispatched=None):
    # Always fresh, including warmups and padding: never reuse a connection
    # idle for >1 s (the serving default closes idle sockets after 5 s).
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1800)
    started = time.perf_counter_ns()
    try:
        conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
        if on_dispatched is not None:
            on_dispatched()
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {response.read().decode()}')
        events = read_sse(response)
        return {'request': body, 'started_ns': started, 'ended_ns': time.perf_counter_ns(),
                'events': events, **summarize(events)}
    finally:
        conn.close()

def batch(port, model_path, manifest, condition, concurrency, warmup, label, max_tokens=OUTPUT_TOKENS):
    """Closed loop with `concurrency` requests in flight. Measured phase: warm-up prompts are
    skipped; once every measured request has been sent, finished slots are refilled with
    padding requests (same shape, labelled 'padding') until the last measured request ends."""
    prompts = validate_prompts(manifest, concurrency)
    if (max_tokens not in (OUTPUT_TOKENS, PROBE_TOKENS)
            or (max_tokens == PROBE_TOKENS and (warmup or concurrency != 1))):
        raise ValueError(f'The schema probe is one non-warm-up request with max_tokens={PROBE_TOKENS}')
    phase = 'schema' if max_tokens == PROBE_TOKENS else 'warmup' if warmup else 'measured'
    count = 1 if phase == 'schema' else warmup_count(concurrency) if warmup else measured_count(concurrency)
    offset = 0 if warmup else warmup_count(concurrency)
    # One persistent worker per closed-loop slot. A completion reserves its
    # replacement under the same lock: no wave barriers or final measured drain.
    # The initial barrier fills all slots before any response can trigger refill.
    barrier = threading.Barrier(concurrency)
    lock = threading.Lock()
    all_dispatched = threading.Event()
    dispatched = 0
    next_index, remaining, failure = concurrency, count, None
    records = []

    def mark_dispatched():
        nonlocal dispatched
        with lock:
            dispatched += 1
            if dispatched == count:
                all_dispatched.set()

    def worker(slot):
        nonlocal next_index, remaining, failure
        index = slot
        try:
            barrier.wait()
            while True:
                role = phase if index < count else 'padding'
                prompt_index = offset + index % count
                body = request_body(model_path, prompts[prompt_index]['messages'], condition, max_tokens)
                body['request_id'] = f'dts-{label}-{role}-{index:06d}'
                if role == 'padding':
                    # Reserving the final measured index is not proof that its
                    # HTTP body has been sent; padding starts only afterwards.
                    all_dispatched.wait()
                    with lock:
                        if failure is not None or remaining == 0:
                            return
                record = (one(port, body, on_dispatched=mark_dispatched)
                          if role == 'measured' else one(port, body))
                record.update(phase=role, request_index=index, prompt_index=prompt_index,
                              slot=slot, concurrency=concurrency)
                with lock:
                    records.append(record)
                    if role == phase:
                        remaining -= 1
                    if failure is not None or remaining == 0 or (phase != 'measured' and next_index >= count):
                        return
                    index = next_index
                    next_index += 1
        except BaseException as error:
            with lock:
                if failure is None:
                    failure = error
                all_dispatched.set()  # Wake padding slots on a dispatch failure.
            barrier.abort()
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, slot) for slot in range(concurrency)]
        concurrent.futures.wait(futures)
        if failure is not None:
            raise failure
        for future in futures:
            future.result()
    # Padding already in flight completes normally, before the idle-only RPC.
    return sorted(records, key=lambda r: r['request_index'])

def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m tailsplit.client')
    p.add_argument('--manifest', type=Path, required=True, help='constructed prompts of the model')
    p.add_argument('--model-path', required=True, help='model path the server was started with')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--condition', choices=('PLAIN', 'JSON'), required=True)
    p.add_argument('--concurrency', type=int, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--phase', choices=('warmup', 'measured', 'schema'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    manifest = json.loads(a.manifest.read_text())
    try:
        result = batch(a.port, a.model_path, manifest, a.condition, a.concurrency, a.phase == 'warmup',
                       a.label, PROBE_TOKENS if a.phase == 'schema' else OUTPUT_TOKENS)
    except Exception as error:
        write_json(a.output.with_suffix('.client-failure.json'), {'phase': a.phase, 'error': str(error),
                   'action': 'The start test or round fails; the request is not dropped or retried'})
        raise
    write_json(a.output, result)


if __name__ == '__main__':
    main()
