# TODO: TPU test failures

Based on the slow run of 2026-10-13 (165 failures, see `conclusion.md`). Each item gives the number of
failures it would remove.

## transformers changes

- [ ] **Skip the spawn-based FSDP2 and TP tests on single-process devices** (10: 6 FSDP2 tests in llama, 4 TP
  tests in qwen2).
  - The tests start one rank per chip with `mp.spawn`, but the pytest process already holds the chips, so
    the ranks can't open them (`device_sharing`). Running in-process, as #49425 does for subprocess tests,
    isn't possible with several ranks.
  - Add a `_skip_if_single_process_device` next to `_skip_if_mps` (`tests/test_fsdp_mixin.py:610`) and in
    `tests/test_tensor_parallel_mixin.py`, keyed on `SINGLE_PROCESS_DEVICES`
    (`src/transformers/testing_utils.py:3114`).
  - Real FSDP/TP coverage on TPU would need a separate job launched with torchrun, outside pytest.
- [ ] **Fix the Whisper speculative-decoding test bug** (2: `test_speculative_decoding_distil`,
  `test_speculative_decoding_non_distil`).
  - `tests/models/whisper/test_modeling_whisper.py:2251` picks float16 only on CUDA/XPU, otherwise float32.
    Line 2266 then always casts the inputs to `torch.float16`.
  - Cast the inputs to `dtype` instead. This is a general fix, not a TPU one.

## Elsewhere

- [ ] **torch_tpu:** reduce `NEW_TORCH_TPU_BUGS.md` (the `mp.spawn` accelerator query) to a standalone repro
  and file it.
- [ ] **torch_tpu:** update the stale statuses in `tpu-test/torch_tpu_bugs/TORCH_TPU_ISSUES.md`.
  - The fully masked attention row and the differentiable `attn_mask` are listed as open, but both probes
    pass on `0.2.1.dev20261012152517`.
  - The `_weight_norm_interface_backward` failures are gone too. Re-run `check_weight_norm_backward.sh` to
    confirm.
- [ ] **accelerate: `device_map="auto"` leaves the model on CPU** (18). `get_max_memory` doesn't enumerate
  torch_tpu devices. Handed over in `~/Dev/ACCELERATE_TPU_DEVICE_MAP_HANDOVER.md`.
- [ ] **Runner environment:**
  - request access to the gated https://huggingface.co/sesame/csm-1b for the token that runs the suite (4);
  - install `phonemizer` plus `espeak-ng` (1: wav2vec2 `test_phoneme_recognition`), and add them to the
    setup in `conclusion.md`.
- [ ] **Runner tooling (this branch):** move the attribution rules from the one-off `make_report.py`
  (`~/Dev/tpu_runs_2026-10-12/report/`) into `utils/tpu_ci/make_model_results.py`, so the runner writes
  `tpu_report.md` itself.

## Deferred or waiting on others

- **Integration expectations (14).**
  - 13 outputs differ from their reference; 9 of them got TPU values in #49425 that this build doesn't
    reproduce.
  - InternVL's `test_qwen2_small_model_integration_batched_generate` has no TPU entry.
  - Deferred until the suite runs on the v6e-8 it will run on long term.
- **Backend gaps, nothing to do in transformers (115).**
  - Cholesky (69), fixed upstream by torch_tpu#4158;
  - the ignored device index (27);
  - tied weights coming apart on the move (6);
  - `upsample_linear1d_backward` (6), `avg_pool1d` with no gradient (3), `unique(dim=)` (2) and
    `_upsample_bicubic2d_aa` (2).
- **GPT-OSS bf16 grouped-inference check (1).** A borderline numeric failure that also failed at `"highest"`
  precision. Not worth a tolerance bump without understanding it.
