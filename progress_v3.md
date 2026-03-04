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
