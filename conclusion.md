## Branch state
- This branch (`tpu-tests`) is rebased on main `78633ef9ff` (2026-10-12). It only adds the runner
  (`utils/tpu_ci/`), the report-dir fixes in `utils/get_test_reports.py` and these notes.
- Main now has the TPU test fixes from #49425: `conftest.py` requests `"highest"` float32 matmul
  precision, tests that re-run themselves in a subprocess run in-process on devices that lock to one
  process, TPU uses the CUDA SDPA tolerances, and several integration tests have TPU expectations.
  It also has #49426 (Whisper generation optimised for TPU) and #49502 (TP loading on TPU).
- The only TPU gate left in the tests reports FlexAttention as unavailable (35 skips).

## Backend probes
`utils/tpu_ci/backend_gaps.py` on `torch_tpu 0.2.1.dev20261012152517`, `libtpu 0.0.50`:
- **Now pass:** `attention_precision`, `boolean_mask_indexing`, `ctc_loss`, `differentiable_attention_mask`,
  `fully_masked_attention_row`, `interpolate_same_size`, `multinomial_seeding`. The new ones since the
  2026-10-05 build are `differentiable_attention_mask` and `fully_masked_attention_row`.
- **Still present:** `avg_pool1d_gradient`, `cholesky`, `compile` (inductor), `compile_storage_offset_guard`,
  `device_index`, `device_sharing`, `visible_devices`, `weight_tying`.
- torch_tpu still defaults `torch.get_float32_matmul_precision()` to `"medium"`. The suite now overrides it.

## Fast test run
Fast tests only (`RUN_SLOW` off), all 27 `IMPORTANT_MODELS`, TPU v6e-4 (4 chips; the earlier runs used 8),
2026-10-12 22:15 → 23:47 UTC:

| Passed | Failed | Skipped | Errors | Pass rate over attempted |
|---|---|---|---|---|
| 4034 | 124 | 4161 | 0 | 97.0% |

On 2026-10-05 the fast run had 3831 passed, 313 failed and 4130 skipped (92.4%).

- Every model produced a report, and every failure is attributed to a cause.
- 189 old failures are gone and none are new. By their cause on 2026-10-05:
  - float32 precision: 114, cleared by `"highest"` in `conftest.py`;
  - subprocess tests: 37, which now run in-process;
  - differentiable `attn_mask`: 32;
  - `_weight_norm_interface_backward`: 4 wav2vec2 tests;
  - Whisper's `test_longform_generate_multi_batch` and GPT-2's cross-attention padding test.
- 5 more T5 tests that failed on the `attn_mask` gap now get further and fail on `device_index` (2) and
  `weight_tying` (3).
- What's left: Cholesky (69), the ignored device index (27), distributed init outside a launcher (10),
  `upsample_linear1d_backward` (6), weight tying (6), `avg_pool1d` gradient (3), `unique(dim=)` (2) and one
  GPT-OSS bf16 grouped-inference check, which also failed at `"highest"` on 2026-10-05.

## Slow test run
`RUN_SLOW=1`, same machine and build, 2026-10-12 23:48 → 2026-10-13 04:02 UTC:

| Passed | Failed | Skipped | Errors | Pass rate over attempted |
|---|---|---|---|---|
| 4518 | 165 | 3636 | 0 | 96.5% |

On 2026-10-06 the slow run had 4240 passed, 429 failed and 3605 skipped (90.8%).

- Every model produced a report, no pytest process crashed, and no test hit the one-hour timeout.
  Every failure is attributed to a cause.
- 264 old failures are gone and none are new. Besides the fast-run causes above, 25 integration tests whose
  output differed and 16 that had no TPU reference now pass. 4 of the 5 Whisper tests that used to time out
  pass.
- All 124 fast-run failures also fail here. These 41 show up only in the slow run:
  - 18 `device_map="auto"` loads leave the model on CPU, in llama, llava, qwen2_5_omni, qwen2_5_vl, smolvlm
    and vit. accelerate finds no TPU to map to.
  - 13 integration tests give text or logits that differ from the reference. **9 of them got TPU expectations
    in #49425 on 2026-10-11, and this build doesn't reproduce them:** gemma3 (1), gemma3n (4), internvl (3)
    and llama (1). The differences are a few words of generated text, or float16 logits. Either the values
    were recorded on another build, or the output isn't stable from run to run. This hasn't been investigated.
    The other 4:
    - Mistral3's multi-image generation, which ran out of device memory on 2026-10-06;
    - Table Transformer's detection;
    - two Whisper long-form transcriptions, one of which used to time out and is now one punctuation mark off.
  - Environment: the CSM checkpoint is gated (4), and `test_phoneme_recognition` needs `phonemizer` (1).
  - Carried over: `_upsample_bicubic2d_aa` is not implemented (2), the Whisper speculative-decoding tests have a
    float16 bug of their own (2), and one InternVL test has no TPU entry in its `Expectations` (1).
- Whisper took 63 min, against 8 h 02 min on 2026-10-06.

## torch_tpu bugs
- One failure cause isn't in `tpu-test/torch_tpu_bugs/TORCH_TPU_ISSUES.md` yet: in a process started with
  `torch.multiprocessing.spawn`, `torch._C._get_accelerator()` raises because the torchrun-only variables
  are missing (10 FSDP2/TP tests). It is drafted in `NEW_TORCH_TPU_BUGS.md`, but not reduced to a standalone
  repro or filed yet.
- `TORCH_TPU_ISSUES.md` still lists three issues as open that this build seems to have fixed:
  - the fully masked attention row and the differentiable `attn_mask`: both probes pass;
  - `_weight_norm_interface_backward`: its wav2vec2 failures are gone, but `check_weight_norm_backward.sh`
    wasn't re-run.

## Uploaded
Fast run in `2026-10-12/ci_results_run_models_gpu/`, slow run in `2026-10-13/ci_results_run_models_gpu/`:
- `model_results.json`, the file the dashboard reads;
- `model_results.md`, the generated summary;
- `tpu_report.md`, a table attributing each failure to its cause. It names gaps by their probe names, not
  by torch_tpu issue numbers, to respect the no-private-repo rule.

I re-downloaded all six files and they match the local copies.

The attribution script is still a one-off and not in the repo. It is saved with the run outputs in
`~/Dev/tpu_runs_2026-10-12/report/`: `make_report.py`, the two meta files, and helpers to compare two runs and
count skips. Moving its rules into `make_model_results.py` is still pending.

## Running the suite

### Environment

From the repo root, on a TPU VM with Python 3.13 available through uv. The torch_tpu and libtpu wheels are
in `~/Dev/tpu-test/torch_tpu_packages/<version>/`:

```bash
W=~/Dev/tpu-test/torch_tpu_packages/0.2.1.dev20261012152517
CPU=https://download.pytorch.org/whl/cpu PYPI=https://pypi.org/simple
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install --index-url $CPU --extra-index-url $PYPI --index-strategy unsafe-best-match \
    "torch==2.11.0" $W/torch_tpu-*.whl $W/libtpu-*.whl
# Pin torch in every later install, or the resolver silently replaces the TPU-compatible build.
uv pip install -e ".[testing]" slack_sdk "torch==2.11.0+cpu" \
    --extra-index-url $CPU --index-strategy unsafe-best-match
uv pip install --index-url $CPU --extra-index-url $PYPI --index-strategy unsafe-best-match \
    torchvision torchaudio torchcodec "torch==2.11.0+cpu"
uv pip install librosa av timm sentencepiece protobuf num2words "torch==2.11.0+cpu" \
    --extra-index-url $CPU --index-strategy unsafe-best-match
getent hosts localhost   # must resolve, or pytest dies before collecting anything
```

`torchcodec` (slow tests) needs FFmpeg's shared libraries (`sudo apt-get install -y ffmpeg`; Ubuntu 22.04's
FFmpeg 4.4 works) and `libpython3.13`. apt on 22.04 doesn't have `libpython3.13`, but uv's Python ships it.

Leave out `pyctcdecode` (it pins numpy < 2) and `phonemizer` (it needs espeak; one wav2vec2 integration test
fails without it).

Nothing else may hold a TPU chip while the suite runs, because only one process at a time can open a chip.
Check with `sudo fuser /dev/vfio/[0-9]*`; the model sweeps in `~/Dev/tpu-test` use the same chips.
torch_tpu loads on `import torch`, so set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for any non-TPU check while the
TPU is busy.

### Fast tests

```bash
source .venv/bin/activate
bash utils/tpu_ci/run_tpu_tests.sh
```

This runs every `IMPORTANT_MODELS` directory in its own pytest process and writes `reports/`,
`model_results.json` and `model_results.md`. On a TPU v6e-4 it took **1 h 32 min** (2026-10-12).

For runs this long, start it detached so it outlives the shell, e.g.
`setsid nohup bash utils/tpu_ci/run_tpu_tests.sh > fast_run.log 2>&1 < /dev/null &`. Activate the venv
first: the script calls `python3`, and without the venv that is the system Python, which has no
transformers.

### Slow tests

```bash
source .venv/bin/activate
RUN_SLOW=1 bash utils/tpu_ci/run_tpu_tests.sh
# optionally with a throwaway hub cache for the downloaded checkpoints:
RUN_SLOW=1 TMP_CACHE=/mnt/cache/tmp bash utils/tpu_ci/run_tpu_tests.sh
```

This runs the fast tests too. It took **4 h 14 min** on the v6e-4 (2026-10-12/13), downloading every
checkpoint into an empty hub cache along the way. The slowest models are `whisper` (63 min), `mistral3`
(28 min), `vit` (23 min) and `qwen2_5_vl` (17 min).

The hub cache here is `~/.cache/huggingface`, a symlink to `/mnt/disks/data/hf_cache`. The two runs together
grew it from 5 GB to 248 GB. A model sweep once filled that disk, so check it before starting.

A single test that runs longer than `TEST_TIMEOUT` seconds (default 3600) is failed so the run moves on.
None did on 2026-10-13.

Both steps write to `reports/` and `model_results.*` in the repo root. Copy them somewhere before starting
the next run.

## Uploading

Results (what the dashboard reads), from the reports of a finished run:

```bash
python utils/tpu_ci/make_model_results.py reports/ --summary model_results.md --upload \
    --repo-id hf-gcp-tpu-internal/transformers_daily_ci --date "$(date -u +%F)"
```

The upload takes the file name from `--output` (default `model_results.json`). A path in another directory is
fine, but keep the file name, or the dashboard won't find the file.

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

`--date` defaults to today (UTC). The fast run goes under the day it started. The slow run usually ends the
next day, so it goes under the day it ended, which keeps it from overwriting the fast run. Use the same day
for the summaries. `tpu_report.md` is written by the one-off attribution script, with
`python make_report.py meta.json tpu_report.md`. Copy the previous meta file and update its text.
