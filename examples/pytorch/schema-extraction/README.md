# Schema extraction

Fine-tune GLiNER2 with the standard `Trainer` loop. `run_gliner2.py` loads JSONL with `datasets.load_dataset`, builds batches in `SchemaCollator`, and uses separate encoder and task learning rates.

## Rows

Each example is `text`, `schema`, and `labels`. A string label matches every occurrence of that string. `{"text", "start", "end"}` selects one occurrence.

```json
{"text": "Ada Lovelace wrote notes.", "schema": {"entities": ["person"]}, "labels": {"entities": {"person": ["Ada Lovelace"]}}}
```

## Run

```bash
python examples/pytorch/schema-extraction/run_gliner2.py \
  --model_name_or_path fastino/gliner2.5-base-v1 \
  --train_file train.jsonl \
  --output_dir /tmp/gliner2 \
  --do_train \
  --weight_decay 0.01 \
  --lr_scheduler_type linear
```

`--weight_decay 0.01` is applied to every parameter, including bias and LayerNorm. The learning-rate schedule is the Trainer schedule selected by `--lr_scheduler_type`.

AdamW uses `betas=(0.9, 0.999)` and `eps=1e-8`. Parameters whose names contain `encoder` use `--encoder_lr` (`1e-5`); other parameters use `--task_lr` (`5e-4`). CUDA uses fused AdamW. Other devices use foreach.

Boundary checkpoints can leave the record and relation heads unused for a span-only batch. Train those models with `--ddp_find_unused_parameters True`.

## LoRA

`--use_lora` builds a PEFT `LoraConfig` with `bias="none"`. `--lora_target_modules` is a comma-separated list passed through to `target_modules` unchanged.

`get_peft_model(..., autocast_adapter_dtype=False)` leaves adapter weights in the base dtype. The script then casts LoRA A and B to that dtype. `--lora_use_dora` maps to `use_dora`. LoRA is one optimizer group at `task_lr`.
