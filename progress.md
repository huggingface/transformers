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

### Findings (current)
- Preprocessing parity: PASS (exact match).
- Eval forward parity: PASS (`max_abs_diff(logits)=0.00016987`, `max_abs_diff(pred_boxes)=0.00018728`).
- Postprocessing parity (same logits/boxes): PASS (exact match).
- Train forward parity: PASS (`max_abs_diff(logits)=0.00049925`, `max_abs_diff(pred_boxes)=0.00041944`).
- Train total loss parity from each model output: PASS (`abs_diff=0.00005150`).
- Loss implementation parity (HF loss on original model outputs): PASS (weighted loss terms match original).

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
