# Distributed training reference

This doc outlines distributed training options.

## Choosing a strategy

| Scenario | Strategy | Setup |
|---|---|---|
| Multi-GPU, model fits on one GPU | DDP (default) | Just launch with `accelerate launch` |
| Multi-GPU, model doesn't fit | FSDP or DeepSpeed ZeRO-2/3 | See sections below |
| Very large model (100B+), CPU offload needed | DeepSpeed ZeRO-3 | Config JSON + `deepspeed=` arg |
| Models with fast intra-node links (NVLink) | Tensor parallelism | `tp_plan="auto"` |
| Very long sequences (context > 32k) | Sequence parallelism | `parallelism_config` with `sp_size` |
| CPU only | DDP on CPU | `use_cpu=True` + `mpirun` |

Trainer uses Accelerate under the hood — no code changes needed for DDP. Just change the launch command.

---

## Multi-GPU DDP (model fits on one GPU)

No TrainingArguments changes. Launch with:

```bash
accelerate launch --num_processes 4 train.py
# or
torchrun --nproc_per_node=4 train.py
```

To use specific GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 train.py
CUDA_DEVICE_ORDER=PCI_BUS_ID  # match nvidia-smi GPU numbering
```

---

## FSDP (model doesn't fit on one GPU)

FSDP shards parameters, gradients, and optimizer states across GPUs.

**TrainingArguments:**
```python
TrainingArguments(
    fsdp="full_shard",
    fsdp_config="fsdp_config.yaml",
)
```

**fsdp_config.yaml:**
```yaml
fsdp_sharding_strategy: 1                          # 1=FULL_SHARD, 2=SHARD_GRAD_OP
fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer  # match your model's layer class
fsdp_offload_params: false
fsdp_state_dict_type: SHARDED_STATE_DICT           # use for mid-training checkpoints
```

**Save final model** (convert to full state dict first):
```python
trainer.train()
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model(output_dir)
```

Gotchas:
- Mid-training checkpoints MUST use `SHARDED_STATE_DICT` — gathering full state dict across nodes causes NCCL timeouts
- `fsdp_transformer_layer_cls_to_wrap` must match the actual layer class in your model (inspect the model to find it)

**Launch:**
```bash
accelerate launch --num_processes 8 --config_file fsdp_config.yaml train.py
```

---

## DeepSpeed ZeRO

Use DeepSpeed when you need CPU/NVMe offloading or prefer ZeRO's optimizer. ZeRO-2 is the recommended starting point. ZeRO-3 for models that don't fit across GPUs even with ZeRO-2.

**ZeRO stages:**
- **ZeRO-2** — shards optimizer states + gradients; lower communication overhead than ZeRO-3
- **ZeRO-3** — also shards parameters; extreme memory savings; higher communication cost

**TrainingArguments:**
```python
# For ZeRO-3: create TrainingArguments BEFORE loading the model —
# if the model is already on each GPU, ZeRO-3 can't shard its parameters
args = TrainingArguments(
    deepspeed="ds_config_zero2.json",
    load_best_model_at_end=True,
    per_device_train_batch_size=4,
    ...
)
model = AutoModelForCausalLM.from_pretrained(...)   # load AFTER args for ZeRO-3
```

**Starter ZeRO-2 config (`ds_config_zero2.json`):**
```json
{
  "bf16": { "enabled": "auto" },
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "allgather_bucket_size": 2e8,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_clipping": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto"
}
```

**Starter ZeRO-3 config** — add to the above:
```json
{
  "zero_optimization": {
    "stage": 3,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

**Save final model:**
```python
trainer.train()
trainer.save_model("./best-model")   # Trainer consolidates sharded weights
```

Gotchas:
- Use `"auto"` for batch size / accumulation / precision in the config — keeps it in sync with TrainingArguments. If they disagree, training proceeds silently with the wrong values
- DeepSpeed checkpoint format is sharded — can't load with `from_pretrained` directly; always save via `trainer.save_model()`
- CPU offload (`offload_optimizer`, `offload_param`) reduces GPU memory but slows training; set `pin_memory: true` to mitigate

---

## Tensor parallelism

Splits weight matrices column/row-wise across GPUs. Best on hardware with fast intra-node links (NVLink). Only models that define `base_model_tp_plan` in their config support it.

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    tp_plan="auto",    # auto-shard layers across all available GPUs
)
trainer = Trainer(model=model, args=args, ...)   # Trainer auto-detects tp_plan
```

Gotcha: Don't use `device_map` with `tp_plan="auto"` — they conflict at weight-loading level.

---

## Sequence parallelism (very long sequences)

Splits sequences across GPUs using DeepSpeed Ulysses. Each GPU processes a chunk of the sequence. Requires Accelerate ≥ 1.12.0.

```python
from accelerate.utils import ParallelismConfig, DeepSpeedSequenceParallelConfig

sp_size = 4  # number of GPUs per sequence

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
    pad_to_multiple_of=sp_size,   # sequences must be padded to a multiple of sp_size
)

parallelism_config = ParallelismConfig(
    sp_backend="deepspeed",
    sp_size=sp_size,
    dp_replicate_size=1,
    sp_handler=DeepSpeedSequenceParallelConfig(
        sp_seq_length_is_variable=True,
        sp_attn_implementation="flash_attention_2",  # sdpa can attend incorrectly across samples
    ),
)

args = TrainingArguments(
    deepspeed="ds_config.json",
    parallelism_config=parallelism_config,
    ...
)
```

Gotchas:
- Number of attention heads must be divisible by `sp_size`
- For 8 GPUs with `sp_size=4`, effective `dp_world_size=2`; adjust `dp_replicate_size` accordingly

---

## CPU training

```python
TrainingArguments(use_cpu=True, bf16=True)   # Intel CPUs support bf16 via PyTorch AMP
```

Multi-socket / multi-node via MPI:
```bash
# Single node, 2 sockets (OMP_NUM_THREADS = cores_per_socket - 1)
OMP_NUM_THREADS=23 mpirun -n 2 python train.py

# Multi-node (hostfile: one IP per line)
mpirun -f hostfile -n 4 -ppn 2 python train.py
```

---
