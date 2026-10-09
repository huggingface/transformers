# Schema extraction

Fine-tune GLiNER2 with the standard `Trainer` loop. `run_gliner2.py` is a small `Trainer` subclass plus one schedule callback. It sets the optimizer groups, the sampler, the learning-rate lambdas, and the boundary step schedules.

## Rows

Each example is `text`, `schema`, and `labels`. A string label matches every occurrence of that string. `{"text", "start", "end"}` selects one occurrence.

```json
{"text": "Ada Lovelace wrote notes.", "schema": {"entities": ["person"]}, "labels": {"entities": {"person": ["Ada Lovelace"]}}}
```

A legacy `{"input", "output"}` JSONL file is converted once. `true_label` becomes the classification label. `entity_descriptions` and `json_descriptions` stay on the schema. Structure values, choice fields, `record_metadata`, and relation arguments (including names other than `head` and `tail`) become labels.

## Run

```bash
python examples/pytorch/schema-extraction/run_gliner2.py \
  --model_name_or_path fastino/gliner2-base-v1 \
  --train_file train.jsonl \
  --output_dir /tmp/gliner2 \
  --do_train \
  --weight_decay 0.01 \
  --scheduler_type linear
```

`scheduler_type` is `linear`, `cosine`, `cosine_restarts`, or `constant`. These are local `LambdaLR` functions. `--weight_decay 0.01` matches gliner2 and is applied to every parameter, including bias and LayerNorm.

Span training shuffles with `seed + epoch`. Boundary training groups rows by `len(text.split())` inside a window of `batch_size * 50`, leaves a partial batch last, and drops that partial when the dataset is longer than the batch. Workers are 0 on macOS. The load-time shuffle uses `random.seed(seed)` before the sampler.

AdamW uses `betas=(0.9, 0.999)` and `eps=1e-8`. Parameters whose names contain `encoder` use `--encoder_lr` (`1e-5`); other parameters use `--task_lr` (`5e-4`). CUDA uses fused AdamW. Other devices use foreach.

## LoRA

`--use_lora` builds a PEFT `LoraConfig` with `bias="none"`. Targets are encoder `Linear` modules whose leaf name contains `query`, `key`, `value`, or `dense`, plus `span_rep`, `classifier`, `count_embed`, `count_pred`, `boundary_head`, `record_decoder`, and `relation_scorer` when those modules exist. Aliases `all_task_heads`, `classification_head`, `extractive_head`, `relation_head`, and `record_head` expand to the same modules.

`get_peft_model(..., autocast_adapter_dtype=False)` leaves adapter weights in the base dtype. The script then casts LoRA A and B to that dtype. `--lora_use_dora` maps to `use_dora`. LoRA is one optimizer group at `task_lr`.

Evaluation builds batches while the model is in eval mode and gold injection is 0. Gold injection holds `1.0` for `gold_injection_hold_frac` (`0.15`) of training, then decays linearly to `0.25`. Consistency warmup and soft-IoU annealing are applied at `on_step_begin`.
