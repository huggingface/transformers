# Progress Log

## 2026-03-04 - LW-DETR tiny parity investigation

### Task
Verify LW-DETR parity between the original implementation (`/Users/nielsrogge/Documents/python_projecten/LW-DETR`) and Transformers for:
- image preprocessing
- forward outputs (eval + train)
- loss computation
- postprocessing

Checkpoint under test:
- `/Users/nielsrogge/Downloads/LWDETR_tiny_60e_coco.pth`

### Progress
- [x] Revisited and extended `src/transformers/models/lw_detr/convert_lw_detr_to_hf.py` to support direct parity checks with the original repo.
- [x] Added optional CLI args:
  - `--original_repo_path`
  - `--strict_original_parity`
- [x] Added parity diagnostics in conversion flow:
  - preprocessing parity (`pixel_values`)
  - eval-mode model parity (`logits`, `pred_boxes`)
  - postprocess parity using identical logits/boxes (isolates postprocess implementation)
  - train-mode model parity (`logits`, `pred_boxes`)
  - train loss parity from model outputs
  - isolated loss-implementation parity by running HF loss on original model outputs
- [x] Ran parity check for `lwdetr_tiny_60e_coco` against original repo.
- [x] Investigated train-path mismatch root cause.
- [x] Fixed train-time self-attention reshape in HF LW-DETR (`modular_lw_detr.py`) to use post-split shape in group-DETR mode.
- [x] Fixed conversion configs to use `dropout=0.0` (original LW-DETR training uses no dropout).
- [x] Regenerated modular outputs with `make fix-repo`.
- [x] Re-ran parity checks after fixes.
- [x] Ran ablation to verify whether disabling dropout *alone* fixes train mismatch.
- [x] Verified single-batch overfitting behavior after fixes.

### Findings (current)
- Preprocessing parity: PASS (exact match).
- Eval forward parity: PASS (`max_abs_diff(logits)=0.00016987`, `max_abs_diff(pred_boxes)=0.00018728`).
- Postprocessing parity (same logits/boxes): PASS (exact match).
- Train forward parity: PASS (`max_abs_diff(logits)=0.00049925`, `max_abs_diff(pred_boxes)=0.00041944`).
- Train total loss parity from each model output: PASS (`abs_diff=0.00005150`).
- Loss implementation parity (HF loss on original model outputs): PASS (weighted loss terms match original).
- Single-batch overfitting: PASS (model can memorize one example).

### Root cause summary
- Converted HF configs were using `dropout=0.1`, while original LW-DETR training config uses `dropout=0.0`.
- Train-time group-DETR self-attention in HF used a reshape based on pre-split `(batch, seq_len)` instead of post-split shape.

### Ablation: does disabling dropout alone fix it?
No.

On the same dummy batch with the tiny checkpoint:
- `dropout=0.0` + **buggy** attention reshape (dropout-only fix): still mismatched
  - `max_abs_diff(logits)=5.00910091`
  - `max_abs_diff(pred_boxes)=0.88600707`
  - `abs_diff(loss)=0.48869514`
- `dropout=0.1` + fixed reshape (reshape-only fix): still mismatched
  - `max_abs_diff(logits)=5.74030876`
  - `max_abs_diff(pred_boxes)=1.29158878`
  - `abs_diff(loss)=6.39123344`
- `dropout=0.0` + fixed reshape (both fixes): aligned
  - `max_abs_diff(logits)=0.00049925`
  - `max_abs_diff(pred_boxes)=0.00041944`
  - `abs_diff(loss)=0.00005150`

### Overfit check (single batch)
Setup:
- Model: converted `lwdetr_tiny_60e_coco`
- Batch: one image + one GT box/class (`category_id=0`)
- Optimizer: `AdamW(lr=1e-3, weight_decay=0.0)`

Observed:
- At `256x256` input size for speed, training loss dropped from `21.5808` to `3.7007` over 400 steps (min `3.5348`).
- On the same single sample after training:
  - best predicted IoU vs GT = `0.9783`
  - score for GT class at that query = `0.9647`
- At `640x640` input size, loss dropped from `22.5128` to `9.4690` in 60 steps (min `9.3038`), showing the same direction.

Interpretation:
- The model now shows clear single-batch memorization dynamics, consistent with training being fixed.

### Reproduction command
```bash
PYTHONPATH=src python src/transformers/models/lw_detr/convert_lw_detr_to_hf.py \
  --model_name lwdetr_tiny_60e_coco \
  --checkpoint_path /Users/nielsrogge/Downloads/LWDETR_tiny_60e_coco.pth \
  --pytorch_dump_folder_path /tmp/lwdetr_tiny_60e_coco_converted \
  --original_repo_path /Users/nielsrogge/Documents/python_projecten/LW-DETR
```

### Conclusion so far
For `LWDETR_tiny_60e_coco`, conversion + preprocessing + model forward (eval/train) + loss + postprocessing are now aligned with the original implementation on the dummy batch.

## 2026-03-04 - LW-DETR fine-tuning investigation on tray-cart dataset

### Task
Investigate why LW-DETR fine-tuning (using this Transformers branch with
`/Users/nielsrogge/Documents/python_projecten/rf-detr-fine-tuning/rt_detr_transformers/train.py`)
does not learn well on `nielsr/tray-cart-detection`.

### Scope correction
- Initially started reproducing with RT-DETR by mistake.
- Corrected to LW-DETR (`AnnaZhang/lwdetr_small_60e_coco`) after user clarification.

### Reproductions and findings
- Reproduced LW-DETR overfit instability locally on the same script and dataset.
- Overfit run (`image_size=640`, single-batch mode, 20 epochs, `lr=1e-3`, `max_grad_norm=0.1`) showed only weak improvement:
  - best `eval_map=0.0168` (epoch 16)
  - `eval_loss` moved roughly from `29.24` to low `20s`, but predictions remained poor.
- Found a concrete LW-DETR runtime bug at smaller image sizes:
  - With `image_size=256`, training crashes in
    `src/transformers/models/lw_detr/modeling_lw_detr.py` at:
    - `group_topk_proposals = torch.topk(..., topk=self.num_queries, ...)`
  - Error: `RuntimeError: selected index k out of range`
  - Cause: LW-DETR uses `num_feature_levels=1`; for small spatial grids the number of encoder proposals can be `< num_queries` (300), so `topk` is invalid.
- On Mac/MPS, LW-DETR training also requires
  `PYTORCH_ENABLE_MPS_FALLBACK=1` due missing
  `aten::grid_sampler_2d_backward` on MPS.

### Checks for likely culprits
- Fast vs slow image processor:
  - `AutoImageProcessor(..., use_fast=True)` vs `use_fast=False`
  - Label boxes/class labels matched (max box diff ~`3e-8`), so preprocessing mismatch is not the main issue.
- Dropout ablation:
  - Hub checkpoints (`AnnaZhang/lwdetr_*_60e_coco`) have `dropout=0.1`.
  - Forcing `dropout=0.0` in config did not materially fix overfit behavior in manual one-sample loops.
- Trainer vs model/loss isolation:
  - Manual optimization loops (outside Trainer) on one sample also failed to produce robust localization/classification.
  - This indicates the issue is not only a Trainer orchestration artifact.

### HF vs original implementation sanity check
- Ran side-by-side short optimization (same sample, same checkpoint source `xbsu/LW-DETR`):
  - HF LW-DETR and original LW-DETR both showed similarly unstable/weak one-sample optimization dynamics.
  - This suggests the poor fine-tuning behavior on this setup is not a clear HF-only regression from conversion/modeling mismatch.

### Current conclusion
- LW-DETR still does not reliably overfit this tray-cart setup in practice, even after the earlier parity fixes.
- Confirmed hard bug:
  - top-k proposal selection can crash for smaller image sizes (`k > num_encoder_tokens`).
- For the no-learning behavior at `640`, current evidence points to broader LW-DETR fine-tuning dynamics on this setup (also observable in original code path), not a simple preprocessing mismatch or a single obvious HF-only bug.
