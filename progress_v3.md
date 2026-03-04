# Progress V3 Log

## 2026-03-04 - LW-DETR tiny checkpoint-copy experiment (HF AutoModel + HF Jobs)

### Goal
- Test whether class-logit calibration issues are caused by initialization by **directly copying original LW-DETR tiny COCO checkpoint weights** into `AutoModelForObjectDetection` and running a short 5-sample training run.
- Check original LW-DETR initialization/loading semantics to validate the assumption "pretrained base + random heads".

### 1) Original implementation initialization semantics (verified)

#### Model constructor init (`/Users/nielsrogge/Documents/python_projecten/LW-DETR/models/lwdetr.py`)
- `class_embed.bias` is initialized to focal prior bias (`prior_prob=0.01`).
- `bbox_embed` last layer is zero-initialized.
- For two-stage mode, encoder heads are created as `copy.deepcopy(self.class_embed)` and `copy.deepcopy(self.bbox_embed)`.

#### Training-time checkpoint loading (`/Users/nielsrogge/Documents/python_projecten/LW-DETR/main.py`)
- If `--pretrain_weights` is provided, weights are loaded with `strict=False`.
- Optional `--pretrain_keys_modify_to_load` remaps selected keys (including class heads) via `util/obj365_to_coco_model.py`.

#### Tiny COCO training script (`/Users/nielsrogge/Documents/python_projecten/LW-DETR/scripts/lwdetr_tiny_coco_train.sh`)
- Uses both:
  - `--pretrained_encoder pretrain_weights/caev2_tiny_300e_objects365.pth`
  - `--pretrain_weights pretrain_weights/LWDETR_tiny_30e_objects365.pth`
- Explicitly includes class-head keys in `--pretrain_keys_modify_to_load`.

**Conclusion on assumption:** for the official tiny COCO recipe, heads are not simply left random after startup; they are loaded/remapped from pretraining checkpoints.

---

### 2) Script updates made

Updated:
- `examples/pytorch/object-detection/investigate_lw_detr_compare.py`

Added:
- `--hf-load-from-original-checkpoint`
- `--hf-checkpoint-model-name`

New behavior:
- Builds HF model via `AutoModelForObjectDetection.from_config(...)`.
- Converts and loads original `.pth` weights using:
  - `transformers.models.lw_detr.convert_lw_detr_to_hf.get_model_config`
  - `...get_checkpoint_state_dict`
  - `...convert_original_checkpoint_state_dict`
- Reports explicit load summary (missing/unexpected keys) for this converted load.

Compatibility fix:
- Added fairscale checkpoint-wrapper stub in script for original-repo imports in light investigation environments.

---

### 3) HF Jobs run (GPU)

Job:
- [69a842f1de6eb6ba6d52ccb9](https://huggingface.co/jobs/nielsr/69a842f1de6eb6ba6d52ccb9)

Hardware:
- `a10g-small`

Command:
```bash
hf jobs uv run -d \
  --flavor a10g-small \
  --python 3.10 \
  --timeout 2h \
  --namespace nielsr \
  --with "transformers@git+https://github.com/NielsRogge/transformers.git@investigate_lw_detr" \
  examples/pytorch/object-detection/investigate_lw_detr_compare.py \
  --num-samples 5 \
  --train-steps 20 \
  --batch-size 1 \
  --image-size 640 \
  --dataset-id nielsr/tray-cart-detection \
  --model-id AnnaZhang/lwdetr_tiny_60e_coco \
  --original-repo-url https://github.com/NielsRogge/LW-DETR.git \
  --original-repo-commit d5e6e6c4add2d24dafb965ced8b50163c50b9788 \
  --checkpoint-repo-id xbsu/LW-DETR \
  --checkpoint-filename pretrain_weights/LWDETR_tiny_60e_coco.pth \
  --hf-load-from-original-checkpoint \
  --hf-checkpoint-model-name lwdetr_tiny_60e_coco
```

---

### 4) Key results

#### Weight copy into HF AutoModel
- `hf_model_source`: `original_checkpoint_converted`
- `hf_loading_summary_for_comparison_model`:
  - `missing_keys_count: 0`
  - `unexpected_keys_count: 0`

This confirms the original `.pth` weights were successfully copied into HF model parameters for the comparison model.

#### 5-sample / 20-step training summary
- HF loss: `15.6408 -> 10.4442` (min `8.4762`)
- Original loss: `15.6030 -> 7.9873` (min `7.1217`)

#### Post-train calibration
- HF @ threshold 0.5: `avg_num_predictions = 0.0`
- Original @ threshold 0.5: `avg_num_predictions = 1.0`

HF still lags original on confidence calibration in this short run, even when initialized from copied original checkpoint weights.

---

### 5) Interpretation for initialization hypothesis

- Copying original tiny COCO checkpoint weights into HF `AutoModelForObjectDetection` **did not eliminate** the short-run class-confidence gap on this 5-sample setup.
- This weakens the hypothesis that the issue is purely from initial weight mismatch alone.
- More likely contributors remain in training-path behavior/details beyond the initial checkpoint copy.

### 6) Noted caveat

- In this HF Jobs run, `pretrain_forward_parity.max_abs_diff_logits` was high (`3.5664`), unlike earlier local parity checks in this investigation that were much tighter.
- So this specific run should be interpreted primarily as a checkpoint-copy training signal, not as a clean full-parity baseline.

---

## 2026-03-04 - Deeper trace: from forward parity to gradient-path mismatch

### 7) HF Jobs ablations to isolate numerical/runtime confounders

Ran the following additional Jobs on `a10g-small`:

- Eager attention:
  - [69a844ce73aa33c77fb22a05](https://huggingface.co/jobs/nielsr/69a844ce73aa33c77fb22a05)
- Eager + `USE_HUB_KERNELS=0`:
  - [69a845ad73aa33c77fb22a0f](https://huggingface.co/jobs/nielsr/69a845ad73aa33c77fb22a0f)
- Eager + `--disable-tf32` (20 steps):
  - [69a8466773aa33c77fb22a11](https://huggingface.co/jobs/nielsr/69a8466773aa33c77fb22a11)
- Eager + `--disable-tf32` + loss parity check (1 step):
  - [69a8471c73aa33c77fb22a1f](https://huggingface.co/jobs/nielsr/69a8471c73aa33c77fb22a1f)
- Eager + `--disable-tf32` + gradient parity check (1 step):
  - [69a8487273aa33c77fb22a2b](https://huggingface.co/jobs/nielsr/69a8487273aa33c77fb22a2b)

#### Findings

- Pretrain forward mismatch on CUDA is mostly TF32-related:
  - Eager (TF32 default): `max_abs_diff_logits = 3.7783`, `boxes = 0.5613`.
  - Eager + `USE_HUB_KERNELS=0`: same (`3.7783`, `0.5613`).
  - Eager + `--disable-tf32`: drops to `0.000493` logits and `4.97e-05` boxes.
- Therefore, large eval-forward parity drift in earlier GPU runs was numerical precision (TF32), not Hub kernels.

### 8) Even with tight forward parity, calibration gap remains

From Job `69a84667` (TF32 disabled, 5 samples, 20 steps):

- `pretrain_forward_parity` is tight (`0.000493` logits).
- Training still diverges in confidence:
  - HF `threshold_0.5.avg_num_predictions = 0.0`
  - Original `threshold_0.5.avg_num_predictions = 2.0`

This rules out "only checkpoint copy / only forward numerical mismatch" as the main explanation.

### 9) Loss implementation parity is exact

From Job `69a8471c`:

- `loss_impl_parity_on_original_outputs.abs_diff_total_loss = 0.0`
- All loss keys (`loss_ce`, `loss_bbox`, `loss_giou`, aux/enc variants) have zero absolute difference.

So HF loss computation itself matches the original objective on identical original-model outputs.

### 10) Gradient parity exposes the real mismatch

From Job `69a84872` (GPU, TF32 disabled, 1 step):

- `train_forward_parity` remains small (`max_abs_diff_logits = 0.0077`, `boxes = 0.0158`).
- But first-batch gradients diverge strongly:
  - `hf_global_grad_norm = 746.94`
  - `original_global_grad_norm = 118.82`
  - `shared_max_abs_diff = 101.23`
  - `shared_mean_abs_diff = 4.18`
- Conversion/mapping sanity check passes:
  - `control_weight_max_abs_diff = 0.0`
  - `control_weight_mean_abs_diff = 0.0`

Interpretation: forward and loss are close, but backward graph/gradient flow differs materially.

### 11) Code-level trace-down: likely gradient leak at two-stage encoder->decoder bridge

Original LW-DETR detaches selected encoder proposals before feeding decoder initial refs:

- `/Users/nielsrogge/Documents/python_projecten/LW-DETR/models/transformer.py`
  - lines `248-252`: `refpoint_embed_gidx_undetach` then `refpoint_embed_gidx = ...detach()`

HF LW-DETR currently computes decoder reference points from **undetached** proposals:

- `/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/lw_detr/modular_lw_detr.py`
  - lines `1372-1377` build both detached and undetached variants
  - line `1393` uses `topk_coords_logits_undetach` in `reference_points = refine_bboxes(...)`
- `/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/lw_detr/modeling_lw_detr.py`
  - analogous logic at lines `1443-1448` and `1464`

This is exactly a forward-close / backward-different failure mode.

### 12) Targeted A/B validation (runtime hook, no source edit)

I ran a local A/B gradient test by monkey-patching HF decoder forward to `detach()` `reference_points` just before decoder execution.

- Baseline (no hook):
  - `hf_global_grad_norm = 1409.04`
  - `original_global_grad_norm = 319.06`
  - `shared_max_abs_diff = 220.79`
  - `shared_mean_abs_diff = 7.93`
- Patched (`reference_points.detach()` hook):
  - `hf_global_grad_norm = 205.29`
  - `original_global_grad_norm = 319.06`
  - `shared_max_abs_diff = 38.02`
  - `shared_mean_abs_diff = 1.40`

This is a major reduction in gradient mismatch and strongly supports the detach-bridge issue as a primary contributor.

Additional short CPU A/B (5 samples, 20 steps) with same hook:

- Baseline HF @ `threshold_0.5`: `avg_num_predictions = 0.0`
- Hooked HF @ `threshold_0.5`: `avg_num_predictions = 1.0`

(Original CPU run in this local setup happened to stay at `0.0` for this tiny run, so this is directional evidence only.)

### 13) Current root-cause assessment

- Most likely primary cause of calibration mismatch:
  - wrong detach semantics at two-stage encoder->decoder reference-point handoff in HF LW-DETR.
- Confirmed *not* primary causes:
  - checkpoint conversion/loading mismatch,
  - loss implementation mismatch,
  - Hub kernels,
  - attention backend alone,
  - pure eval-forward mismatch (once TF32 disabled).

### 14) Potential investigation routes (complete list + priority)

1. **Fix and verify detach semantics at two-stage bridge (P0)**
- Change HF to feed decoder refs from detached top-k coords (original behavior) while keeping encoder outputs for loss unchanged.
- Re-run gradient parity + 20-step calibration check on GPU.

2. **Check remaining backbone gradient deltas after detach fix (P1)**
- Even after runtime detach hook, residual gradient diffs remain.
- Investigate top residual params (mostly ViT layernorm/gamma and projector norms).

3. **Audit decoder reference handling and `intermediate_reference_points` semantics (P1)**
- Ensure shape/time-step behavior matches original for both `lite_refpoint_refine=True/False` paths.

4. **Verify matcher + loss normalization under grouped queries (P1)**
- Confirm Hungarian matching/group handling and normalization (`num_boxes`, aux + enc loss weighting) exactly match original for grouped training.

5. **Compare optimizer parameter grouping and schedules vs original recipe (P1)**
- Original training uses custom lr decays (`lr_encoder`, layer/component decay).
- Current parity script uses plain AdamW single LR; this can affect confidence dynamics.

6. **Validate pixel mask / padding / valid-ratio path parity (P2)**
- Confirm no hidden divergence from mask usage in HF vs original list-input path.

7. **Freeze precision/runtime settings for parity runs (P2)**
- Keep TF32 disabled and fixed attention backend when validating architecture-level parity.

8. **Re-check data pipeline parity (fast vs slow processor, box transforms) (P2)**
- Warning indicates fast processor default; verify this does not alter tiny-run calibration behavior.

9. **Inspect potential duplicated/aliased parameter mapping effects in conversion (P2)**
- `shared_grad_param_count` > `original_grad_param_count` after name conversion suggests mapping inflation worth sanity-checking.

10. **Extension path parity (MSDeformAttn CUDA vs fallback) (P3)**
- Investigation runs force fallback path for portability; official training may use extension path.
- Likely secondary, but can be rechecked once P0 is fixed.

11. **Evaluate class-head calibration diagnostics beyond threshold count (P3)**
- Add ECE/Brier per-class diagnostics to quantify calibration improvements after structural fixes.

12. **Longer-run parity once P0/P1 are fixed (P3)**
- Repeat 60e-like schedule subset with original optimizer grouping to verify the fix persists beyond short runs.

---

## 2026-03-04 - HF Jobs validation after fix commit `d447074a09`

### 15) Jobs launched from pushed commit

Used the exact pushed revision:

- `transformers@git+https://github.com/NielsRogge/transformers.git@d447074a09`

Ran two comparative jobs on `a10g-small`:

1. Strict-parity runtime (eager + TF32 disabled + gradient parity):
- Job: [69a84dc873aa33c77fb22a6d](https://huggingface.co/jobs/nielsr/69a84dc873aa33c77fb22a6d)
- Flags: `--hf-attn-implementation eager --disable-tf32 --compute-gradient-parity`

2. Default runtime (auto attention, TF32 default):
- Job: [69a84dc8de6eb6ba6d52ccd1](https://huggingface.co/jobs/nielsr/69a84dc8de6eb6ba6d52ccd1)
- Flags: default (no `--disable-tf32`, no explicit attn backend)

### 16) Strict-parity result: training parity essentially restored

From job `69a84dc873aa33c77fb22a6d`:

- Pretrain forward parity:
  - `max_abs_diff_logits = 4.93e-04`
  - `max_abs_diff_boxes = 4.97e-05`
- Train-forward parity:
  - `max_abs_diff_logits = 0.00771`
  - `max_abs_diff_boxes = 0.0158`
- Loss implementation parity on original outputs:
  - `abs_diff_total_loss = 0.0`
- Gradient parity (first batch):
  - `hf_global_grad_norm = 118.64`
  - `original_global_grad_norm = 118.82`
  - `shared_max_abs_diff = 0.193`
  - `shared_mean_abs_diff = 0.00248`
  - `control_weight_max_abs_diff = 0.0`

Training/calibration on 5 samples, 20 steps:

- HF loss: `15.6617 -> 8.0103` (min `7.4970`)
- Original loss: `15.6617 -> 7.9985` (min `7.2140`)
- Post-train @ threshold 0.5:
  - HF: `avg_num_predictions = 0.2`, `max_score = 0.5648`
  - Original: `avg_num_predictions = 0.2`, `max_score = 0.5165`

Interpretation: with the fix and strict runtime controls, the calibration/training parity gap is resolved on this parity harness.

### 17) Default runtime result: divergence still present (same TF32 pattern)

From job `69a84dc8de6eb6ba6d52ccd1`:

- Pretrain forward parity still large:
  - `max_abs_diff_logits = 3.5664`
  - `max_abs_diff_boxes = 0.5614`
- Train-forward parity also large:
  - `max_abs_diff_logits = 5.1922`
  - `max_abs_diff_boxes = 0.9730`
- Loss implementation parity remains exact (`abs_diff_total_loss = 0.0`).

Training/calibration on 5 samples, 20 steps:

- HF loss: `15.6408 -> 8.4755`
- Original loss: `15.6030 -> 7.6867`
- Post-train @ threshold 0.5:
  - HF: `avg_num_predictions = 0.0`
  - Original: `avg_num_predictions = 2.0`

Interpretation: this is consistent with prior TF32-driven runtime divergence; the model-graph bug is fixed, but default CUDA numerical mode can still mask parity.

### 18) Conclusion after HF Jobs validation

- The pushed fix in `d447074a09` resolves the previously traced training-time calibration/parity issue when running in strict parity mode.
- For reliable parity benchmarking between HF and original LW-DETR, keep:
  - `--disable-tf32`
  - a fixed attention backend (e.g. `--hf-attn-implementation eager`)
- Remaining non-parity under default runtime appears to be numerical/runtime-level, not the original gradient-flow bug.

---

## 2026-03-04 - Full tray-cart training on HF Jobs (HF vs original side-by-side)

### 19) New full-training script added

Created:
- `/Users/nielsrogge/Documents/python_projecten/transformers/examples/pytorch/object-detection/train_lw_detr_tray_cart_full_compare.py`

What it does:
- Runs **full training** on `nielsr/tray-cart-detection` (`train=100`, `val=13`, `test=12`) for 60 epochs.
- Trains **HF LW-DETR** and **original LW-DETR** side-by-side in one run from the same starting checkpoint (`xbsu/LW-DETR/pretrain_weights/LWDETR_tiny_60e_coco.pth`).
- Mirrors original optimizer/scheduler settings as closely as possible:
  - AdamW
  - original-like param-group logic (`lr_encoder`, `lr_vit_layer_decay`, `lr_component_decay`)
  - StepLR with `lr_drop=60`
  - gradient clipping `0.1`
- Evaluates both models periodically and reports:
  - `map`, `map_50`, `map_75`
  - score calibration stats (`threshold=0.0` and `threshold=0.5`)

Note:
- This script uses a shared HF `AutoImageProcessor` pipeline for both models for controlled comparison, not the original repository's random resize/crop augmentation stack.

### 20) HF Job run

Job:
- [69a87a7773aa33c77fb22c1f](https://huggingface.co/jobs/nielsr/69a87a7773aa33c77fb22c1f)

Hardware:
- `a10g-small`

Command:
```bash
hf jobs uv run -d \
  --flavor a10g-small \
  --python 3.10 \
  --timeout 6h \
  --namespace nielsr \
  --with "transformers@git+https://github.com/NielsRogge/transformers.git@investigate_lw_detr" \
  examples/pytorch/object-detection/train_lw_detr_tray_cart_full_compare.py \
  --dataset-id nielsr/tray-cart-detection \
  --model-id AnnaZhang/lwdetr_tiny_60e_coco \
  --checkpoint-repo-id xbsu/LW-DETR \
  --checkpoint-filename pretrain_weights/LWDETR_tiny_60e_coco.pth \
  --checkpoint-model-name lwdetr_tiny_60e_coco \
  --epochs 60 \
  --batch-size 4 \
  --num-workers 4 \
  --eval-every 5 \
  --image-size 640 \
  --lr 1e-4 \
  --lr-encoder 1.5e-4 \
  --weight-decay 1e-4 \
  --lr-drop 60 \
  --lr-vit-layer-decay 0.8 \
  --lr-component-decay 0.7 \
  --max-grad-norm 0.1 \
  --hf-init-source original_checkpoint \
  --hf-attn-implementation eager \
  --disable-tf32
```

### 21) Results (full 60 epochs)

Runtime:
- `835.25s` (~13m55s)

Final validation:
- HF: `map=0.2614`, `map_50=0.5382`, `map_75=0.2024`
- Original: `map=0.2598`, `map_50=0.5377`, `map_75=0.1966`

Best validation `map_50`:
- HF: `0.5543`
- Original: `0.5522`

Final test:
- HF: `map=0.4071`, `map_50=0.7583`
- Original: `map=0.4259`, `map_50=0.7836`

Calibration (`threshold=0.5`, validation):
- Initial (both): `avg_num_predictions=0.0769`
- Final HF: `44.08`
- Final Original: `45.38`

Training loss (epoch 1 first batch -> epoch 60 last batch):
- HF: `17.6881 -> 2.9522`
- Original: `17.6881 -> 2.9447`

Epoch-60 validation parity snapshot:
- `map_50` delta HF-original: `+0.0004`
- `predictions@0.5` delta HF-original: `-1.31`

### 22) Interpretation vs original

- On this full-dataset training run, HF and original are now very close on validation quality and confidence calibration.
- The severe under-confidence pattern from earlier short-run diagnostics (HF near-zero detections at threshold 0.5 after training) is **not present** in this full run.
- This supports that the training-time detach fix resolves the main calibration/parity issue under this controlled setup.

---

## 2026-03-04 - Longer Trainer runs with early stopping (HF model, fixed)

### 23) Is previous result the best possible?

Short answer: **not guaranteed**.

I ran additional longer HF-only experiments with `Trainer` + `EarlyStoppingCallback` to push performance further and test whether we can beat the prior full-run result.

### 24) New script (Trainer + early stopping)

Created:
- `/Users/nielsrogge/Documents/python_projecten/transformers/examples/pytorch/object-detection/train_lw_detr_tray_cart_hf_trainer_early_stop.py`

Key features:
- Uses HF `Trainer` API.
- Early stopping support (`EarlyStoppingCallback`) with configurable patience.
- Supports selecting best checkpoint on either `eval_map_50` or `eval_map`.
- Loads LW-DETR tiny from converted original checkpoint (`xbsu/LW-DETR/...LWDETR_tiny_60e_coco.pth`).
- Outputs `FINAL_SUMMARY_JSON` for easy job-level comparison.

### 25) Debug/fix cycle on HF Jobs

Initial long jobs failed due evaluation-time target size shape edge cases (`orig_size` occasionally malformed in eval collation path).

Failed jobs (for traceability):
- [69a884c173aa33c77fb22c92](https://huggingface.co/jobs/nielsr/69a884c173aa33c77fb22c92)
- [69a884c173aa33c77fb22c90](https://huggingface.co/jobs/nielsr/69a884c173aa33c77fb22c90)
- [69a8859873aa33c77fb22c9c](https://huggingface.co/jobs/nielsr/69a8859873aa33c77fb22c9c)
- [69a88598de6eb6ba6d52cd36](https://huggingface.co/jobs/nielsr/69a88598de6eb6ba6d52cd36)

Fixes applied in script:
- Hardened target size extraction for metrics (`orig_size` fallback handling).
- Made bbox-to-absolute conversion robust to non-standard size tensor shapes.

Quick validation job after fixes:
- [69a88682de6eb6ba6d52cd39](https://huggingface.co/jobs/nielsr/69a88682de6eb6ba6d52cd39)
- Completed successfully (no eval crash).

### 26) Longer runs (successful)

All runs on `a10g-small`, fixed branch (`investigate_lw_detr`), `num_train_epochs=300`, early stopping patience `3`.

#### A) No augmentation, best by `eval_map_50`
- Job: [69a88713de6eb6ba6d52cd3b](https://huggingface.co/jobs/nielsr/69a88713de6eb6ba6d52cd3b)
- `best_metric (eval_map_50)`: `0.5638`
- Final validation: `map=0.2832`, `map_50=0.5638`
- Final test: `map=0.3961`, `map_50=0.7540`
- Stopped at: `epoch=16`, `global_step=400`

#### B) With augmentation, best by `eval_map_50`
- Job: [69a88713de6eb6ba6d52cd3d](https://huggingface.co/jobs/nielsr/69a88713de6eb6ba6d52cd3d)
- `best_metric (eval_map_50)`: `0.5413`
- Final validation: `map=0.2514`, `map_50=0.5413`
- Final test: `map=0.3695`, `map_50=0.7124`
- Stopped at: `epoch=16`, `global_step=400`

#### C) No augmentation, best by `eval_map`
- Job: [69a8882673aa33c77fb22cb8](https://huggingface.co/jobs/nielsr/69a8882673aa33c77fb22cb8)
- `best_metric (eval_map)`: `0.2836`
- Final validation: `map=0.2836`, `map_50=0.5478`
- Final test: `map=0.3776`, `map_50=0.7357`
- Stopped at: `epoch=16`, `global_step=400`

### 27) Comparison to prior full run (section 21)

Prior full side-by-side HF run (`69a87a7773aa33c77fb22c1f`):
- Validation HF `map_50`: `0.5382`
- Test HF `map_50`: `0.7583`
- Test HF `map`: `0.4071`

Best new early-stopping run (A):
- Validation `map_50`: **`0.5638`** (improved by `+0.0256`)
- Test `map_50`: `0.7540` (slightly lower by `-0.0043`)
- Test `map`: `0.3961` (lower by `-0.0110`)

### 28) Current conclusion on “best possible”

- If objective is **validation `map_50`**, best current run is **A** (`0.5638`).
- If objective is **held-out test `map_50`/`map`**, the earlier full run (`69a87a...`) remains slightly better on this dataset split.
- So this is **not a strict global optimum yet**; we improved one axis (val `map_50`) but not all axes simultaneously.
