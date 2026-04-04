# Multi-Label Text Classification (Transformers Example)

This example demonstrates a **multi-label** text classifier using BCEWithLogitsLoss
(`problem_type="multi_label_classification"`). It reports **F1 (micro/macro)**,
**Hamming loss**, and **Subset accuracy**, and can tune the decision threshold on
the validation set with an F1–threshold curve.

## Quick start (fast eval-only)
```bash
python -u examples/pytorch/text-classification/run_multilabel_classification.py \
  --model_name_or_path prajjwal1/bert-tiny \
  --dataset_name go_emotions --text_column text \
  --output_dir ./mlc_out \
  --do_eval --tune_thresholds --plot_threshold_curve \
  --max_eval_samples 300 --per_device_eval_batch_size 64
