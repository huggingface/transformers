## Branch state
- The TPU test enablement was merged to main in #49207. This branch (`tpu-tests`) is rebased on main
  and only adds the runner (`utils/tpu_ci/`), the report-dir fixes in `utils/get_test_reports.py`
  and these notes.
- The `ctc_loss` skip and the same-size `interpolate` skip in `get_rel_pos` were reverted on main, now
  that both gaps are fixed in torch_tpu. The only TPU gate left in the tests reports FlexAttention as
  unavailable (35 skips).

## Backend probes
`utils/tpu_ci/backend_gaps.py` on `torch_tpu 0.1.2.dev20261005100409`:
- **Now pass:** `attention_precision`, `boolean_mask_indexing`, `ctc_loss`, `interpolate_same_size`,
  `multinomial_seeding`.
- **Still present:** `avg_pool1d_gradient`, `cholesky`, `compile` (inductor), `compile_storage_offset_guard`,
  `device_index`, `device_sharing`, `differentiable_attention_mask`, `fully_masked_attention_row`,
  `visible_devices`, `weight_tying`.

## Fast test run
Fast tests only (`RUN_SLOW` off), all 27 `IMPORTANT_MODELS`, 8 × TPU v6e, 2026-10-05 15:14 → 17:19 UTC:

| Passed | Failed | Skipped | Errors | Pass rate over attempted |
|---|---|---|---|---|
| 3831 | 313 | 4130 | 0 | 92.4% |

On 2026-09-23, with the older build, the fast run had 3810 passed, 304 failed and 4070 skipped (92.6%).

- Every model produced a report, and every failure is attributed to a cause. The main causes are float32 precision (115),
  Cholesky (69), the differentiable `attn_mask` (37), the subprocess that can't open the device (37) and the
  ignored device index (25).
- 10 old failures are gone. The backend fixes cleared the padding-free and sliding-window tests,
  the seeded T5 generation and two compiled-generation tests, and two borderline precision checks passed this time.
- 19 failures are new:
  - 10 tests that the removed gates used to skip now run and hit two backward ops TPU doesn't implement: `_weight_norm_interface_backward` (wav2vec2, 4)
    and `upsample_linear1d_backward` (got_ocr2, 6). Both are the drafts in `NEW_TORCH_TPU_BUGS.md`,
    which aren't filed yet.
  - 4 more T5 tests fail on the differentiable `attn_mask` gap.
  - 3 CLIP SDPA comparisons are just over tolerance (a mean relative difference of about 1e-6).
  - One wav2vec2 batching check and Whisper's `test_longform_generate_multi_batch` differ.

## Slow test run
`RUN_SLOW=1`, same machine and build, 2026-10-05 17:24 → 2026-10-06 05:29 UTC:

| Passed | Failed | Skipped | Errors | Pass rate over attempted |
|---|---|---|---|---|
| 4240 | 429 | 3605 | 0 | 90.8% |

On 2026-09-24 the slow run had 4224 passed, 402 failed and 3558 skipped (91.3%).

- Every model produced a report and no pytest process crashed. Every failure is attributed to a cause.
- Compared with the 2026-09-24 slow run, 11 failures are gone and 38 are new. 17 of the new ones are the new
  fast-run failures. These 21 show up only in the slow run:
  - 9 integration tests in qwen2_5_vl, qwen2_5_omni and smolvlm now load with `device_map="auto"`, which leaves the
    model on CPU. That makes 18 tests with this problem in total, up from 6. accelerate finds no TPU to map to.
  - 6 wav2vec2 integration tests used to be skipped by the CTC gate. 5 of them give values that differ from the
    recorded references, and `test_phoneme_recognition` needs `phonemizer`.
  - 3 more borderline CLIP SDPA checks, plus three numeric differences: T5 summarization, one Whisper long-form
    transcription and one Whisper encoder SDPA check.
- Integration tests still need TPU reference values. 26 have no TPU entry in their `Expectations`, and 27
  give text or logits that differ from the values recorded on other hardware.
- In Whisper, 5 tests now hit the one-hour per-test timeout, up from 3. The two new ones are
  `test_whisper_longform_multi_batch` and `test_whisper_longform_multi_batch_prev_cond`. The whisper
  file took 8 h 02 min, 5 h of it waiting on those timeouts.
- Carried over from the 2026-09-24 run: `_upsample_bicubic2d_aa` is not implemented (2). The dynamo backend rejects `mode=` (1).
  Mistral-Small-24B runs out of device memory under `cpu_offload` (1). The Whisper speculative-decoding tests have a float16 bug of
  their own (2). The CSM checkpoint is gated (4).

## The float32 precision failures: cause found
**torch_tpu sets `torch.get_float32_matmul_precision()` to `"medium"` by default. PyTorch's default is `"highest"`.**
So every float32 matmul, Linear and attention runs at about bfloat16 accuracy unless the caller asks for
more. That is why the precision failures barely moved (113 → 115) even though SDPA now honours the setting,
which is what the `attention_precision` probe checks.

Mean relative error of a float32 matmul on this build:

| Precision | TPU | CPU, for reference |
|---|---|---|
| `medium` (the TPU default) | 2.4e-3 | |
| `high` | 1.3e-5 | |
| `highest` | 7e-8 | 2e-7 |

At `highest`, Linear, SDPA and eager attention also give float32-level error.

I re-ran the fast run's 115 precision failures with `torch.set_float32_matmul_precision("highest")`. 108 pass and 7 still fail:
- 4 CLIP fp16/fp32 SDPA checks;
- 2 Gemma3n fp32 SDPA checks;
- 1 GPT-OSS bf16 grouped-inference check.

The uploaded numbers use the backend default: nothing in the tests or the runner sets the precision.
There are two ways to change that:
- **Set `"highest"` when `torch_device == "tpu"`**, for example in `testing_utils`. The tests would then run at the
  float32 accuracy they were written for, the same as CUDA, where TF32 is off by default.
- **Report it upstream** as a default that differs from PyTorch's. It's arguably intentional for performance,
  but it surprises any code that assumes float32 means float32.

## Uploaded
Fast run in `2026-10-05/ci_results_run_models_gpu/`, slow run in `2026-10-06/ci_results_run_models_gpu/`:
- `model_results.json`, the file the dashboard reads. I checked that the Hub copy matches the local file.
- `model_results.md`, the generated summary.
- `tpu_report.md`, a table attributing each failure to its cause. To respect the no-private-repo rule, it names gaps by their
  probe names, not by torch_tpu issue numbers.

The script that attributes failures is still a one-off. It wasn't on this VM, so I recovered it from the previous
session's transcript into the scratchpad. Moving its rules into `make_model_results.py` is still pending.

## Running the suite

### Environment

A fresh TPU machine's `.venv` only has `torch`, `torch_tpu` and `libtpu`. From the repo root:

```bash
source .venv/bin/activate
uv pip install -e ".[testing]" slack_sdk
# Pin torch, or the resolver silently replaces the TPU-compatible build.
uv pip install --index-url https://download.pytorch.org/whl/cpu torchvision torchaudio \
    "torch==$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
uv pip install librosa av timm sentencepiece protobuf num2words
getent hosts localhost   # must resolve, or pytest dies before collecting anything
```

For the slow tests, also `torchcodec`. It needs FFmpeg's shared libraries and `libpython`, which a fresh
VM image doesn't have. The PyPI wheels are CUDA builds, so take the CPU one:

```bash
apt-get install -y ffmpeg libpython3.13
uv pip install --index-url https://download.pytorch.org/whl/cpu torchcodec \
    "torch==$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
```

Leave out `pyctcdecode` (it pins numpy < 2) and `phonemizer` (it needs espeak; one wav2vec2 integration test
fails without it). Nothing else may hold a TPU chip while the suite runs, because only one process at a time can open a chip.

### Fast tests

```bash
source .venv/bin/activate
bash utils/tpu_ci/run_tpu_tests.sh
```

This runs every `IMPORTANT_MODELS` directory in its own pytest process and writes `reports/`,
`model_results.json` and `model_results.md`. On a TPU v6e with 8 chips it took **2 h 05 min**
(2026-10-05). The slowest models are `vit` (23 min) and `whisper` (18 min). Most others take 2–5 min.

### Slow tests

```bash
source .venv/bin/activate
RUN_SLOW=1 bash utils/tpu_ci/run_tpu_tests.sh
# optionally with a throwaway hub cache for the downloaded checkpoints:
RUN_SLOW=1 TMP_CACHE=/mnt/cache/tmp bash utils/tpu_ci/run_tpu_tests.sh
```

This runs the fast tests too. It took **12 h 05 min** on the same machine (2026-10-05/06), with the
checkpoints already in the shared hub cache (`HF_HUB_CACHE`). `whisper` alone takes 8 h, 5 h of it in
five long-form tests that hit the per-test timeout. Next are `mistral3` (34 min, a 24B checkpoint streamed
through `cpu_offload`), `vit` (22 min) and `qwen2_5_vl` (18 min). Everything except whisper takes about
3.8 h.

A single test that runs longer than `TEST_TIMEOUT` seconds (default 3600) is failed so the run moves
on. The slowest tests that do finish take about 22 minutes.

Both steps write to `reports/` and `model_results.*` in the repo root. Copy them somewhere before starting
the next run.

## Uploading

Results (what the dashboard reads), from the reports of a finished run:

```bash
python utils/tpu_ci/make_model_results.py reports/ --summary model_results.md --upload \
    --repo-id hf-gcp-tpu-internal/transformers_daily_ci --date "$(date -u +%F)"
```

The upload takes the file name from `--output` (default `model_results.json`). Leave it alone, or the dashboard won't find the file.

Or run and upload in one go:

```bash
UPLOAD=1 RESULTS_REPO_ID=hf-gcp-tpu-internal/transformers_daily_ci bash utils/tpu_ci/run_tpu_tests.sh
```

Summaries, next to the results:

```bash
DAY="$(date -u +%F)"
hf upload hf-gcp-tpu-internal/transformers_daily_ci model_results.md \
    "$DAY/ci_results_run_models_gpu/model_results.md" --repo-type dataset
hf upload hf-gcp-tpu-internal/transformers_daily_ci tpu_report.md \
    "$DAY/ci_results_run_models_gpu/tpu_report.md" --repo-type dataset
```

`--date` defaults to today (UTC). The fast run goes under the day it ran. The slow run starts the same day and
ends the next, so it goes under the day it ended, which keeps it from overwriting the fast run. Use the same day
for the summaries. `tpu_report.md` is written by the one-off attribution script, which is not in the repo.
