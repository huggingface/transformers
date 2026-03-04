# Progress V2 Log

## 2026-03-04 - LW-DETR fine-tuning investigation (HF vs original)

### Goal
- Check whether LW-DETR can be properly fine-tuned on `nielsr/tray-cart-detection`.
- Verify:
  - weight initialization behavior,
  - bounding box format expectations,
  - short training behavior versus the original LW-DETR implementation.

### Current setup
- Transformers repo branch: `investigate_lw_detr`
- Original repo reference:
  - local path: `/Users/nielsrogge/Documents/python_projecten/LW-DETR`
  - commit: `d5e6e6c4add2d24dafb965ced8b50163c50b9788`
- Added investigation script:
  - `examples/pytorch/object-detection/investigate_lw_detr_compare.py`

### What the script does
- Loads first `N` samples (default `N=5`) from `nielsr/tray-cart-detection`.
- Validates raw bbox format from dataset annotations.
- Builds HF image-processor labels and validates processed bbox range.
- Verifies HF weight-loading behavior in two modes:
  - default checkpoint label space,
  - custom label remap with `ignore_mismatched_sizes=True`.
- Runs short side-by-side training loops (HF model vs original LW-DETR model/criterion) on the same batches.
- Reports confidence/calibration stats at thresholds `0.0` and `0.5` before/after training.

### Status
- [x] Investigation script created.
- [x] Local smoke test completed.
- [x] HF Jobs GPU run launched.
- [x] HF Jobs runs completed and analyzed.
- [x] Final conclusions added.

### Local smoke tests
- Ran local script via `uv run --with "transformers @ file:///Users/nielsrogge/Documents/python_projecten/transformers" ...`.
- Key result:
  - With HF `pixel_mask` enabled, HF LW-DETR failed at step 2 with NaN boxes (`generalized_box_iou` input became NaN).
  - With HF `pixel_mask` disabled (closer to original list-input behavior), both HF and original ran multiple steps without NaN.

### HF Jobs runs (GPU, `a10g-small`)
- Run with mask-enabled behavior (before parity switch):
  - Job: [69a8003438de7adae6c02a55](https://huggingface.co/jobs/nielsr/69a8003438de7adae6c02a55)
  - Result: completed with captured error summary
  - HF failed at step 2 (`nan` boxes in loss path), original did not fail.
- Final parity-mode run (mask disabled for HF):
  - Job: [69a800f8e5509825c7f78e5e](https://huggingface.co/jobs/nielsr/69a800f8e5509825c7f78e5e)
  - Result: completed 20/20 training steps for both implementations.

### Final run command (GPU)
```bash
hf jobs uv run \
  --flavor a10g-small \
  --python 3.10 \
  --timeout 2h \
  --namespace nielsr \
  --with "git+https://github.com/nielsrogge/transformers.git@investigate_lw_detr" \
  -d examples/pytorch/object-detection/investigate_lw_detr_compare.py \
  --num-samples 5 \
  --train-steps 20 \
  --batch-size 1 \
  --image-size 640 \
  --dataset-id nielsr/tray-cart-detection \
  --model-id AnnaZhang/lwdetr_small_60e_coco \
  --original-repo-url https://github.com/NielsRogge/LW-DETR.git \
  --original-repo-commit d5e6e6c4add2d24dafb965ced8b50163c50b9788
```

### Verification results

#### 1) Weights initialization
- HF default load (`num_labels=91`): no mismatched keys.
- HF custom remap (`num_labels=2`, `ignore_mismatched_sizes=True`):
  - `28` mismatched keys reinitialized.
  - Reinitialized keys include:
    - `class_embed.{weight,bias}`
    - `model.enc_out_class_embed.[0..12].{weight,bias}`
- Interpretation: class heads are reinitialized exactly as expected when adapting to dataset label space.

#### 2) Bounding box format
- Raw HF dataset (`nielsr/tray-cart-detection`) checks on first 5 samples:
  - `total_boxes=137`
  - `invalid_count=0`
  - Boxes are valid COCO-style absolute `[x, y, w, h]`.
- After HF image processor conversion:
  - `num_boxes=137`
  - normalized boxes in `[0.0151, 0.9458]`
- Interpretation: dataset bbox format and processor conversion are valid and consistent with LW-DETR training expectations.

#### 3) HF vs original short training comparison (5 samples, 20 steps)
- Run mode: parity mode (`hf_use_pixel_mask=false`).
- HF loss:
  - first: `12.6417`
  - last: `7.4092`
  - min: `7.0995`
- Original loss:
  - first: `12.6706`
  - last: `5.2410`
  - min: `5.2410`
- Both optimize (loss decreases), original improves faster in this setup.

#### 4) Confidence calibration comparison
- Pre-train (both):
  - threshold `0.5`: avg predictions `0.0`
- After 20 steps:
  - HF:
    - threshold `0.0`: avg preds `100.0`, avg mean score `0.2672`
    - threshold `0.5`: avg preds `1.0`
  - Original:
    - threshold `0.0`: avg preds `100.0`, avg mean score `0.3008`
    - threshold `0.5`: avg preds `9.8`
- Interpretation: calibration improves for both, but remains much weaker in HF than original under matched tiny-run conditions.

### Conclusions
- LW-DETR can optimize on this dataset in the `investigate_lw_detr` branch when run in the parity mode used above.
- Bounding-box formatting looks correct.
- Initial weight-loading behavior looked correct in that first pass; tiny-model follow-up below found an encoder-class-head init mismatch for custom label remap.
- A concrete discrepancy remains:
  - HF path with `pixel_mask` enabled produced NaN loss/boxes at step 2 in this setup, while original did not.
  - In stable parity mode (`hf_use_pixel_mask=false`), HF trains but still lags original confidence calibration (`0.5` threshold behavior).
- Practical takeaway for this dataset:
  - bounding-box learning is possible,
  - confidence calibration is still weaker on HF than original in short-run fine-tuning.

---

## 2026-03-04 - Further discrepancy investigation (tiny checkpoint)

### Scope of this pass
- Focused comparison target:
  - script: `examples/pytorch/object-detection/investigate_lw_detr_compare.py`
  - original training code: `/Users/nielsrogge/Documents/python_projecten/LW-DETR`
- Tiny checkpoints used:
  - Original `.pth`: [xbsu/LW-DETR/pretrain_weights/LWDETR_tiny_60e_coco.pth](https://huggingface.co/xbsu/LW-DETR/resolve/main/pretrain_weights/LWDETR_tiny_60e_coco.pth)
  - HF model id: `AnnaZhang/lwdetr_tiny_60e_coco`
- Transformers branch used in runs: `investigate_lw_detr` (local + GitHub branch install in HF Jobs).

### Script updates made
- Switched tiny defaults:
  - `--model-id AnnaZhang/lwdetr_tiny_60e_coco`
  - `--checkpoint-filename pretrain_weights/LWDETR_tiny_60e_coco.pth`
- Original model init now reads architecture args from checkpoint `args` and applies them before construction.
- Prevented original init from trying to load external encoder pretrain paths from checkpoint args:
  - force `pretrained_encoder=None`, `pretrain_weights=None`, `resume=""`.
- Added `pretrain_forward_parity` reporting.
- Added `custom_head_initialization` diagnostics under `hf_initialization_summary` to inspect class-head reinit behavior for custom labels.

### Verified bbox format against original source
- Original repository data flow:
  - `datasets/coco.py` converts COCO `xywh -> xyxy` (`boxes[:, 2:] += boxes[:, :2]`).
  - `datasets/transforms.py::Normalize` converts `xyxy -> cxcywh`, then normalizes by image size.
- Conclusion: original training loss sees normalized `cxcywh` boxes.
- Dataset checks in current runs remain valid (`invalid_count=0`, normalized range inside `[0, 1]`).

### Weight initialization findings (important)
- HF custom label remap still reports expected 28 mismatched keys (`class_embed` + `enc_out_class_embed[0..12]`).
- New diagnostic showed a discrepancy vs original initialization semantics:
  - Main `class_embed.bias` is correctly set to focal prior (`-4.5951`).
  - `enc_out_class_embed[*].bias` stays at `0.0` (not focal prior).
  - `enc_out_class_embed[*]` weights are not clones of `class_embed` (and not clones of each other).
- Why this differs from original:
  - In original `models/lwdetr.py`, `class_embed.bias` is set to focal prior first, then `enc_out_class_embed` is built via `copy.deepcopy(self.class_embed)`, so encoder class heads inherit that initialization.
- This is now a prime candidate for confidence-calibration differences during custom-label fine-tuning.

### Local tiny smoke (CPU, 2 samples, 3 steps)
- `hf_use_pixel_mask=false`:
  - pretrain parity close (`max_abs_diff_logits=0.0023`, `boxes=0.00047`)
  - both HF + original train for all 3 steps.
- `hf_use_pixel_mask=true`:
  - parity diverges strongly (`logits~5.90`, `boxes~0.91`)
  - HF fails at step 2 with NaN boxes (`generalized_box_iou` path), original stays stable.

### HF Jobs (GPU, a10g-small, 5 samples, 20 steps)
- First submissions failed due CLI package spec formatting (`--with transformers @ ...` split at `@`):
  - [69a8042be5509825c7f78e84](https://huggingface.co/jobs/nielsr/69a8042be5509825c7f78e84)
  - [69a8042fe5509825c7f78e86](https://huggingface.co/jobs/nielsr/69a8042fe5509825c7f78e86)
- Corrected submissions (`--with "transformers@git+https://github.com/NielsRogge/transformers.git@investigate_lw_detr"`):
  - no-mask: [69a804cfe5509825c7f78e88](https://huggingface.co/jobs/nielsr/69a804cfe5509825c7f78e88)
  - mask: [69a804d438de7adae6c02a61](https://huggingface.co/jobs/nielsr/69a804d438de7adae6c02a61)

### GPU results summary (tiny)
- No-mask run (`69a804cfe5509825c7f78e88`):
  - `device=cuda`
  - completed `20/20` steps for both models.
  - HF loss: `15.6408 -> 10.4454` (min `9.0095`)
  - Original loss: `15.6030 -> 8.3721` (min `7.0524`)
  - Calibration @0.5 post-train:
    - HF avg preds: `0.0`
    - Original avg preds: `2.0` (max score `0.5567`)
- Mask run (`69a804d438de7adae6c02a61`):
  - HF fails after first step with NaN boxes (same pattern as local).
  - Original remains stable for the executed step.

### Current conclusion after tiny pass
- Bounding box format handling is aligned with original training expectations.
- There is a real class-head initialization discrepancy for custom-label remap:
  - `enc_out_class_embed` init in HF does not match original semantics (prior bias + cloned head).
- Pixel-mask path still causes HF instability (NaN boxes) in this setup.
- In stable no-mask parity mode, HF learns boxes but confidence calibration remains weaker than original.

### 2026-03-04 - Progress: copied original two-stage head initialization into HF LW-DETR

- Implemented an explicit post-init synchronization step in `LwDetrForObjectDetection` to match original LW-DETR semantics for two-stage heads.
- New method `_init_two_stage_heads_like_original()` now copies:
  - `class_embed` -> each `model.enc_out_class_embed[*]`
  - `bbox_embed` -> each `model.enc_out_bbox_embed[*]`
- This is executed after `post_init()` so it also applies when custom-label remapping (`ignore_mismatched_sizes=True`) causes fresh head initialization.
- Rationale: original LW-DETR constructs `enc_out_*` as deep copies of already-initialized main heads; HF previously left these encoder heads independently initialized after remap.

Planned next check:
- Re-run the tiny 5-sample comparison with this change to verify whether post-train confidence calibration at threshold 0.5 moves closer to original behavior.
