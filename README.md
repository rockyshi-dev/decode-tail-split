# decode-tail-split

Measures how much of the time between two output tokens in vLLM v0.30.0, on one GPU, is spent in
the three pieces of work that end every decode step:

1. **head**: multiplying the last hidden state by the vocabulary matrix (for Qwen2.5-7B,
   152,064 x 3,584 numbers, 1.09 GB in 16-bit) to get one score per token;
2. **mask**: when the output must follow a JSON schema, setting the scores of forbidden tokens to
   minus infinity;
3. **sampler**: logits processing (penalties, temperature, top-k/top-p) and the draw of the token.

It does this for a 4-bit checkpoint (4-bit layers, 16-bit head) and its 16-bit original, where
the head is a much larger share of the bytes a step reads in the 4-bit model (24.3% against 7.7%
for Qwen2.5-7B), and answers two questions for the 4-bit Qwen2.5-7B at one request in flight:
is the head worth a faster format or kernel on this card, and is a kernel that joins mask and
sampler worth building?

## What it measures

**Checkpoints** (served by local path, at these revisions):

| Name | Checkpoint | Role |
|---|---|---|
| `qwen25-w4` | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` @ `e9c932ac` | decision model: GPTQ 4-bit, group 128, no activation reordering (Marlin kernels), fp16, head not quantized |
| `qwen25-bf16` | `Qwen/Qwen2.5-7B-Instruct` @ `a09a3545` | its 16-bit original |
| `qwen3-w4` | `RedHatAI/Qwen3-1.7B-quantized.w4a16` @ `f8467681` | small pair, where the head's byte share is largest (45.0%); reported only |
| `qwen3-bf16` | `Qwen/Qwen3-1.7B` @ `70d244cc` | (18.1%); reported only |

**Server.** `vllm serve <model path> --port <port>` and nothing else, so each model's own
generation defaults apply (Qwen2.5-7B: temperature 0.7, top-p 0.8, top-k 20, repetition penalty
1.05; Qwen3-1.7B: 0.6, 0.95, 20) and the default V2 model runner with asynchronous scheduling
runs. The one addition, identical for every server, is the environment variable
`VLLM_SERVER_DEV_MODE=1`: it registers `/pause`, `/resume` and `/collective_rpc`, which are used
only while the server is idle to read the timing data.

**Requests.** Streaming chat requests with no sampling parameters, `max_tokens` 256, usage
reported; prompts of exactly 256 template tokens (built offline from each model's tokenizer).

- **PLAIN** adds `ignore_eos: true`, so every request produces 256 tokens.
- **JSON** adds a structured-output schema (an array of 400 to 600 integers), whose shortest valid
  output is longer than 256 tokens, so the grammar is active on every step until `max_tokens`.

Each condition runs at 1, 8 and 32 requests in flight (closed loop). Per condition and
concurrency c: max(4, c) warm-up requests, then max(20, 2c) measured requests (20, 20, 64). Once
every measured request has been sent, finished slots are refilled with padding requests of the
same shape until the last measured request ends, so exactly c stay in flight; padding is labelled
and excluded from every number. The first 16 token gaps and steps of each request are discarded.
The decision uses concurrency 1; 8 and 32 are reported.

**Timing.** A patched copy of vLLM's V2 model runner (built from the installed file by
`python -m tailsplit overlay`, loaded only by servers started in patched mode; the installed
environment is never edited) records CUDA timing events at six points of every step: before and
after the forward pass; before the head (`compute_logits`), after it, after the grammar mask;
and, through temporary hooks on the sampler, between logits processing and the draw and after
the draw. Recording an event never waits for the GPU; the events are read only after all
requests of a condition have finished and the engine is paused. The difference between two
markers is the GPU time of that segment: per step, f (forward), h (head), m (mask), s1 (logits
processing), s2 (draw), and the step period p (forward start to the next step's forward start).

**Two denominators.** g is the median gap between streamed tokens on an *unpatched* server, g'
the same on the patched server. Shares: **H = h / g, M = m / g, S = (s1 + s2) / g**, each also
over g'; T = H + M + S. A share counts as "at least x" only if it is so over both g and g', and as
"below x" only if it is below over both.

**Residual** R = 1 - (f + h + m + s1 + s2) / p, from medians: the part of a step in which the GPU
is on none of the segments, waiting for the CPU. If R is large, the CPU sets the pace and making
the head faster would shorten nothing.

**Head alone.** In a separate process on the same GPU, the head multiplication with the server's
shape and dtype, one call every g milliseconds: at least 3 s of untimed calls at that spacing,
then 200 timed calls; median.

**A round** is: unpatched server (PLAIN and JSON at c = 1, then 8, then 32), patched server (the
same, plus one JSON request with `max_tokens` 4096 whose output must parse under the schema),
then the head-alone timings. Five rounds per checkpoint, the four checkpoints one after another,
all on one GPU.

**Placement and monitoring.** Servers, head-alone job and coordinator run on all CPUs of the
GPU's socket; the client on two physical cores (with their SMT siblings) of the other socket; a
once-a-second recorder on a third core there logs GPU clock, power, temperature, utilisation and
compute processes, per-CPU busy time, and the CPU time and affinity of every thread the run owns.

## The checks

A cell (condition x concurrency) of a round is used only if the checks its question needs pass.

| Check | Passes when | Needed by |
|---|---|---|
| V runner | the log shows the V2 runner; the head is a dense 16-bit matrix; for a 4-bit model all 112 decoder linears use the expected quantization method on the Marlin kernel (`Using MarlinLinearKernel for ...` in the log), for a 16-bit model none is quantized | both |
| D distortion | \|g' - g\| / g <= 3% | both |
| A accounting | every measured request streamed 256 tokens for 256 prompt tokens; every measured request is found in the worker's step records (one or more prefill steps, then 255 decode steps); median p within 3% of g'; median of f + h + m + s1 + s2 at most 1.02 p | both |
| X head cross-check | (PLAIN) h within 25% of the head-alone time | head |
| B mask bound | (JSON) M <= (g_JSON - g_PLAIN) / g_PLAIN + 0.02, both g from the unpatched server | fusion |
| J mask active | (JSON) every measured request ends with `finish_reason: length`; no grammar failure in the server logs; m > 0 and the mask covers every measured request on every step; the 4096-token request parses (400-600 integers) | fusion |
| T tokens | requests with fewer than 256 output tokens are dropped; at most 2% dropped | both |
| Q quiet | in every serving interval (first measured send to last measured finish) and every head-alone timed window: no foreign compute process on the GPU; CPU time not used by the run's own threads at most 2% on each socket; GPU clock below 90% of its median in at most 10% of samples | both |

Only Q is rerun: a round is tried at most three times. If two consecutive attempts fail the same
quiet interval for the same reasons while a rerun is still left, the run stops (exit code 3,
`STOPPED.json`) for a person to look at the machine. A failed V means that checkpoint is not
measured. Thread CPU time is credited to the run only for threads seen in two samples, or
started (by their `/proc` start time) after the previous sample began; everything else counts as
foreign.

## Decision rule

Decision model `qwen25-w4`, concurrency 1, five rounds. A question is NOT MEASURED if a check it
needs fails in any round.

| Question | BIG | SMALL | MIXED |
|---|---|---|---|
| head (PLAIN) | in every round H >= 0.15 and R <= 0.15 | in every round H < 0.15 | anything else, including H >= 0.15 with R > 0.15 (large but not removable: the CPU sets the pace) |
| fusion (JSON) | in every round M + S >= 0.10 and R <= 0.15 | in every round M + S < 0.10 | anything else |

| Answers | Meaning |
|---|---|
| head BIG | the head's storage format and kernel are worth work on this card |
| fusion BIG | a kernel joining mask and sampler is worth building on this card |
| head SMALL, fusion SMALL, and T < 0.10 in both conditions in every round | the tail is too small on this card for any tail kernel to matter |
| any other combination without a BIG | no follow-up kernel; each question is reported with its own answer (a NOT MEASURED question is open, not "no") |

Why these thresholds: a head stored in 4 bits reads a quarter of the bytes, so it can remove at
most three quarters of H; the smallest H for which that reaches a 10% gain in time per token is
0.133, rounded up to 0.15. A fused mask-and-sample kernel can remove at most M + S, hence 0.10.
R <= 0.15 means the GPU is busy on the measured segments for at least 85% of each step.

Written before measuring: on the 4-bit 7B model H is between 0.10 and 0.25 and M + S is well
below 0.10; the report says whether that held.

## Requirements

- Linux, x86_64, two CPU sockets, an NVIDIA GPU with `nvidia-smi`, `taskset` and `lscpu`. The
  reference GPU is an RTX 4090 (24 GB).
- One Python environment with `vllm==0.30.0+cu129` (the CUDA 12.9 build of v0.30.0, commit
  `ced6857a`; it runs on 550-series drivers) and its `torch==2.13.0+cu129`. The tool checks the
  versions and that `vllm/v1/worker/gpu/model_runner.py` is byte-identical to the released file
  (sha256 `174c93db...`), because the timing patch is built against it. `transformers` (comes with
  vLLM) builds the prompts.
- `requirements-cu129.lock.txt` pins every package of the reference environment, with the vLLM
  release wheel by URL and sha256. Two packages are replaced after the main install, both without
  dependencies: PyPI's `torchcodec` 0.17.0 links CUDA 13, so its CPU build is used (vLLM imports it
  at start; no request here decodes audio or video), and FlashInfer's prebuilt kernels for CUDA
  12.9 are added, so the sampler's FlashInfer kernels are not compiled on first use:

  ```bash
  pip install --extra-index-url https://download.pytorch.org/whl/cu129 \
      $(grep -v -e '^torchcodec==' -e '^flashinfer-jit-cache==' requirements-cu129.lock.txt)
  pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torchcodec==0.17.0+cpu
  pip install --no-deps --extra-index-url https://flashinfer.ai/whl/cu129 flashinfer-jit-cache==0.6.18.post1+cu129
  ```

  The mask time depends on `xgrammar` and the sampler time on the FlashInfer packages, so their
  versions matter. `check`, `start-test` and `rounds` print one line per package that differs from
  the lock file; the run records the lines in `manifest.json` and the report prints them.
- The four checkpoints in a Hugging Face cache, e.g.
  `HF_HOME=/models/hf huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --revision e9c932ac1893a49ae0fc497ad6e1e86e2e39af20`
  and likewise for the other three (revisions in `tailsplit/spec.py`); about 27 GB in total.
  Servers run with `HF_HUB_OFFLINE=1`.
- An otherwise idle GPU and quiet CPUs. A GPU in use stops the run before anything starts.

## Usage

Run every command from the repository root with the vLLM environment's Python.

```bash
# installation and checkpoints (prints what was found)
python -m tailsplit check --hf-home /models/hf

# the timing patch, built from the installed runner (writes model_runner.py, timing.patch, overlay.json)
python -m tailsplit overlay --output work/overlay

# 96 prompts of 256 tokens per checkpoint (CPU only)
python -m tailsplit prompts --hf-home /models/hf --output work/prompts

# start test: start, inspect (check V) and stop both servers of each checkpoint; no timed request
setsid python -m tailsplit start-test --gpu 4 --hf-home /models/hf --overlay work/overlay \
    --cache-dir work/runtime-cache --lock /var/lock/gpu4.lock \
    --output runs/start-1 > start-1.log 2>&1 < /dev/null &

# five rounds per checkpoint, then the decision; refuses a start test that did not finish for every
# checkpoint, or that was made with other code, another overlay or another cache directory
setsid python -m tailsplit rounds --gpu 4 --hf-home /models/hf --overlay work/overlay \
    --cache-dir work/runtime-cache --lock /var/lock/gpu4.lock \
    --prompts work/prompts --start-tests runs/start-1 --output runs/rounds-1 > rounds-1.log 2>&1 < /dev/null &

# every check, share and the verdict, recomputed from the raw files
python -m tailsplit report runs/rounds-1 --json runs/rounds-1/report.json
# the same, compared cell by cell with another run
python -m tailsplit report runs/rounds-2 --reference runs/rounds-1
```

Options of `start-test` and `rounds`: `--gpu` (nvidia-smi index, PCI bus order; required),
`--port` (default 8155), `--hf-home` (default `$HF_HOME`, else `~/.cache/huggingface`),
`--cache-dir` (the servers' compile, Triton and FlashInfer caches; default
`$XDG_CACHE_HOME/tailsplit/runtime-cache`, else `~/.cache/tailsplit/runtime-cache`, the same for
every run; rounds must use the directory their start test used, so the rounds start from the cache
state the start test left; to compare with an earlier run, use the cache directory it used),
`--lock FILE` (held for the whole run, repeatable; default `/tmp/tailsplit-gpu<N>.lock`). The GPU
is idle between servers, so on a machine where other jobs schedule work on the same GPU, pass
every lock file they take for it, or another job can start in such a gap and stop the run. Inherited variables that would change the measurement (`VLLM_*`, `CUDA_*` other
than `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `PYTORCH_*`, `TORCH*`, `TRITON_*`, `NCCL_*`, `OMP_*`, `MKL_*`,
`CUBLAS*`, `CUDNN_*`, `FLASHINFER_*`, `NVIDIA_TF32_*`, `LD_PRELOAD`, ...) stop the run.

Start the long commands under `setsid` with output redirected, so a closed terminal cannot stop
them. On a failed request or control call on a verified server, the server is stopped and the run
ends. On any other failure the server is left running for inspection (`LIVE-SERVER.json` in its
directory) and the run waits until someone stops it; it never kills a server it cannot account for.

Wall time is dominated by server starts (two per attempt, each with compilation unless the cache
directory already holds it) and by the requests at concurrency 1; expect the start test to take
on the order of an hour and the rounds several hours on an RTX 4090.

## Output

`--output` is a new directory:

- `manifest.json`: step, checkpoints and revisions, CPU topology (`lscpu`, sockets, GPU NUMA
  node and PCI address), GPU state, versions and installation identity, sha256 of every source
  file of this tool, the overlay's hashes and the full text of the timing patch, the prompt
  files' hashes, the cache directory, the packages that differ from
  `requirements-cu129.lock.txt`, the command line.
- `<model>/round-N/attempt-K/`:
  - `baseline/` and `patched/`: `launch.json` (argv, complete environment, versions, generation
    defaults, readiness evidence: listening socket, GPU processes by UUID, served model path),
    `server.log`, `worker.json` (runner class, head shape/dtype/method, every quantized linear and
    its kernel, CUDA stream), for each cell `<COND>-<c>-warmup.json` and `<COND>-<c>.json`
    (every request: body, send and end time, every stream event with its arrival time, token
    arrival times, finish reason, usage, phase, slot), `released.json`;
  - `patched/<COND>-<c>-steps.json`: one record per engine step (request ids and per-request step
    numbers, prefill flags, batch size, f/h/m/s1/s2 in ms, period, mask coverage);
    `patched/schema.json`: the 4096-token request;
  - `head-c<c>-b<b>.json` (200 timed calls, launch times, pre-roll), with its contract, launch
    manifest and log;
  - `telemetry.jsonl`: the once-a-second samples;
  - `gate-arithmetic.json`: every check and number of the attempt (or `gate-V-failed.json`).
- `<model>/rounds.json`, `decision.json` (decision model), `DONE` when the run finished;
  `STOPPED.json` when it stopped early.
- `start-test` writes `<model>/start-test.json` instead of rounds.

`launch.json` records the complete environment, with the values of variables whose names contain
KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL or AUTH, or start with SSH_, replaced by `<redacted>`
(case-insensitive). It still records paths, user and host-specific variables and GPU UUIDs, so
read a run directory before sharing it.

## How to read the report

`python -m tailsplit report` recomputes every check from the raw files of each attempt (not from
`gate-arithmetic.json`), picks for each round the attempt the run used (the first that passed Q
or failed V, else the last), and prints per checkpoint and round one line per cell: g, g', p, the
five segment medians in ms, H, M + S and T over g and over g', R, and the checks that failed.
Below each checkpoint: the c = 1 range over rounds of H and R, the median g, h and head-alone
time. Then the two answers and their meaning, and whether the written expectation held. It warns
(exit code 1) where a stored result differs from the recomputed one and exits with code 2 when
the directory is not a rounds directory. A missing, unfinished or stopped round makes the
questions NOT MEASURED.

How the numbers relate: the head-alone time is the time the head takes with nothing else
running; h is the same work inside a real step, so X compares them. A large h with a large R
means the head is slow but the CPU, not the GPU, sets the pace. Comparing H with the head's
share of the bytes a step reads (printed per checkpoint) shows whether the step is bound by
memory bandwidth: an H far below the byte share means it is not.

`--reference DIR` compares two runs cell by cell (same round, cell and denominator): shares,
residual, relative differences of g and h, and whether the same checks passed. A reference
directory may also hold only `<model>/rounds.json` files with stored per-round results.

## Tests

```bash
python -m unittest
# also check the patch against the released runner file
TAILSPLIT_VLLM_RUNNER=/path/to/vllm/v1/worker/gpu/model_runner.py python -m unittest tests.test_overlay
```

The tests use synthetic records, fake servers, a stand-in `vllm` package, and one recorded
telemetry file; they need neither a GPU, torch, vLLM nor network access.

## Reference results

TODO: filled in after the run on an RTX 4090.

## License

MIT
