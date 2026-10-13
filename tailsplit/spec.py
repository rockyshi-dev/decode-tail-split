"""Fixed constants of the measurement. Safe to import without CUDA, torch or vLLM."""
import hashlib
import json
from pathlib import Path

VLLM_COMMIT = 'ced6857afa0ea7b2e3f0846a62e1394e90f15607'   # tag v0.30.0
VLLM_VERSION = '0.30.0+cu129'   # the CUDA 12.9 build of v0.30.0 (runs on 550-series drivers)
TORCH_VERSION = '2.13.0+cu129'
TORCH_CUDA = '12.9'
RUNNER_FILE = 'vllm/v1/worker/gpu/model_runner.py'
RUNNER_SHA256 = '174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce'

# name: (Hugging Face repository, revision)
MODELS = {
    'qwen25-w4': ('Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', 'e9c932ac1893a49ae0fc497ad6e1e86e2e39af20'),
    'qwen25-bf16': ('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28'),
    'qwen3-w4': ('RedHatAI/Qwen3-1.7B-quantized.w4a16', 'f8467681051842a5be10a8478d8cfdc5facb0af3'),
    'qwen3-bf16': ('Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
}
DECISION_MODEL = 'qwen25-w4'
# Share of the bytes one decode step reads that is the 16-bit output head (weights only).
HEAD_BYTE_SHARE = {'qwen25-w4': .243, 'qwen25-bf16': .077, 'qwen3-w4': .450, 'qwen3-bf16': .181}

# Quantized-linear evidence per 4-bit model: (quant method class, scheme class or None, label in
# vLLM's "Using <kernel> for <label>" log line). Both models have 28 decoder layers x
# (qkv, o, gate_up, down) = 112 quantized linears.
QUANT = {
    'qwen25-w4': ('AutoGPTQLinearMethod', None, 'AutoGPTQLinearMethod'),
    'qwen3-w4': ('CompressedTensorsLinearMethod', 'CompressedTensorsWNA16', 'CompressedTensorsWNA16'),
}
QUANT_LINEARS = 112

SCHEMA = {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 400, 'maxItems': 600}
CONCURRENCIES = (1, 8, 32)
CONDITIONS = ('PLAIN', 'JSON')
ROUNDS = 5
ATTEMPTS = 3            # a round is rerun at most twice, and only for the quiet check
DISCARD = 16            # first token gaps / steps of each request not used
PROMPT_TOKENS = 256
OUTPUT_TOKENS = 256
PROBE_TOKENS = 4096     # the one schema-parse request per round
HEAD_CALLS = 200
PREROLL_NS = 3_000_000_000

# Thresholds of the checks and of the decision rule.
D_MAX = .03             # |g' - g| / g
A_PERIOD = .03          # |p - g'| / g'
A_SUM = 1.02            # median(f+h+m+s1+s2) <= A_SUM * p
X_MAX = .25             # |h - head alone| / head alone
B_SLACK = .02           # M <= (g_JSON - g_PLAIN) / g_PLAIN + B_SLACK
T_DROP = .02            # dropped requests / measured requests
Q_FOREIGN = .02         # foreign CPU time per socket
Q_CLOCK = .9            # a sample is slow below Q_CLOCK x median clock ...
Q_SLOW = .10            # ... and more than Q_SLOW of samples may not be slow
HEAD_BIG = .15
FUSION_BIG = .10
RESIDUAL_MAX = .15
STOP_TAIL = .10


def validate_concurrency(concurrency):
    if type(concurrency) is not int or concurrency not in CONCURRENCIES:
        raise ValueError('Concurrency is not one of %s' % (CONCURRENCIES,))


def warmup_count(concurrency):
    validate_concurrency(concurrency)
    return max(4, concurrency)


def measured_count(concurrency):
    validate_concurrency(concurrency)
    return max(20, 2 * concurrency)


PROMPT_COUNT = max(warmup_count(c) + measured_count(c) for c in CONCURRENCIES)


def validate_prompts(manifest, concurrency):
    required = warmup_count(concurrency) + measured_count(concurrency)
    prompts = manifest['prompts']
    if len(prompts) < required:
        raise ValueError(f'{required} constructed prompts are needed at c={concurrency}')
    if any(len(p['prompt_token_ids']) != PROMPT_TOKENS or not p['messages'] for p in prompts[:required]):
        raise ValueError(f'Constructed prompts must contain {PROMPT_TOKENS} template tokens and messages')
    return prompts


def snapshot(hf_home, model):
    repo, revision = MODELS[model]
    return Path(hf_home) / 'hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def request_body(model_path, messages, condition, max_tokens=OUTPUT_TOKENS):
    """No sampling parameters: the server applies the model's own generation defaults."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    body = {'model': str(model_path), 'messages': messages, 'max_tokens': max_tokens,
            'stream': True, 'stream_options': {'include_usage': True}}
    if condition == 'PLAIN':
        body['ignore_eos'] = True
    else:
        body['structured_outputs'] = {'json': SCHEMA}
    return body
