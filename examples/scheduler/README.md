# GreedyLR: Adaptive Learning Rate Scheduler

GreedyLR monitors training metrics and adaptively adjusts the learning rate -- increasing when improving, decreasing when plateauing. It works for both pre-training and fine-tuning.

## Paper

> **GreedyLR: A Novel Adaptive Learning Rate Scheduler**
>
> Despite significant advances in optimizers for training, most research works use common scheduler choices like Cosine or exponential decay. In this paper, we study GreedyLR, a novel scheduler that adaptively adjusts the learning rate during training based on the current loss. To validate the effectiveness of our proposed scheduler, we conduct experiments on several NLP, CV, and LLM tasks with up to 7B parameters, including both fine-tuning and pretraining experiments. The results show that our approach outperforms several state-of-the-art schedulers in terms of accuracy, speed, and convergence.

arXiv: https://arxiv.org/abs/2512.14527

## How It Works

```
+-------------------------------------------------------------+
|                    GreedyLR Decision Flow                    |
|                                                              |
|   Metrics Improving?  --Yes-->  Increase LR (/ factor)       |
|         |                              |                     |
|         No                        Enter Warmup               |
|         |                              |                     |
|         v                              v                     |
|   Metrics Plateau?   --Yes-->  Decrease LR (* factor)        |
|         |                              |                     |
|         No                        Enter Cooldown             |
|         |                              |                     |
|         v                              v                     |
|   Continue Training              Continue Training           |
+-------------------------------------------------------------+
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patience` | 10 | Steps to wait before adjusting LR |
| `factor` | 0.95 | Multiplicative factor for LR adjustment |
| `min_lr` | 1e-3 | Minimum learning rate bound |
| `smooth` | False | Apply streaming average to metrics |

## Quick Start

### Fine-tuning LLaMA 3.2 1B on simpleCoT

Fine-tuning runs on a single GPU. This example uses [w601sxs/simpleCoT](https://huggingface.co/datasets/w601sxs/simpleCoT), a chain-of-thought reasoning dataset:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

raw = load_dataset("w601sxs/simpleCoT", split="train[:5000]")
eval_raw = load_dataset("w601sxs/simpleCoT", split="train[5000:5500]")

def format_and_tokenize(examples):
    texts = [
        f"Question: {s}\nReasoning: {r}\nAnswer: {t}"
        for s, r, t in zip(examples["source"], examples["rationale"], examples["target"])
    ]
    return tokenizer(texts, truncation=True, max_length=512, padding=False)

train_ds = raw.map(format_and_tokenize, batched=True, remove_columns=raw.column_names)
eval_ds = eval_raw.map(format_and_tokenize, batched=True, remove_columns=eval_raw.column_names)

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    lr_scheduler_type="greedy",
    lr_scheduler_kwargs={"patience": 2, "factor": 0.9, "min_lr": 1e-7},
    max_steps=2000,
    eval_strategy="steps",
    eval_steps=100,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)
trainer.train()
```

### Pre-training LLaMA 3.2 1B on RedPajama (multi-GPU)

Pre-training a 1B+ parameter model requires multiple GPUs. This example uses a subset of [RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) with DeepSpeed ZeRO-3 across 4 GPUs. The dataset should be pre-tokenized with 2048-token sequences.

Create a DeepSpeed config (`ds_config.json`):

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

Training script (`pretrain_greedy.py`):

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk

config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

tokenized_datasets = load_from_disk("./datasets/redpajama")

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=2e-4,
    weight_decay=0.1,
    max_grad_norm=1.0,
    lr_scheduler_type="greedy",
    lr_scheduler_kwargs={"patience": 2, "factor": 0.95, "min_lr": 1e-5},
    max_steps=2000,
    eval_strategy="steps",
    eval_steps=500,
    bf16=True,
    gradient_checkpointing=True,
    dataloader_drop_last=True,
    deepspeed="ds_config.json",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)
trainer.train()
```

Launch with `torchrun`:

```bash
torchrun --nproc_per_node=4 pretrain_greedy.py
```

### Standalone Usage (without Trainer)

```python
import torch
from transformers import GreedyLR

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
scheduler = GreedyLR(
    optimizer,
    mode="min",
    factor=0.99,
    patience=10,
    min_lr=1e-5,
    smooth=True,
    window_size=50,
)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    scheduler.step(val_loss)
```

## Comparison with Other Schedulers

| Feature | Cosine | ReduceLROnPlateau | GreedyLR |
|---------|--------|-------------------|----------|
| Adaptive to metrics | No | Yes (decrease only) | Yes (increase and decrease) |
| LR increase | No | No | Yes |
| Warmup after increase | No | No | Yes |
| Metric smoothing | No | No | Yes (optional) |
| Auto-reset | No | No | Yes |

## Using with Other Models

GreedyLR is model-agnostic. The `run_greedy.py` script pre-trains or fine-tunes a causal language model on WikiText-2:

```bash
# Pre-train with GreedyLR (default)
python examples/scheduler/run_greedy.py

# Compare with cosine scheduler
python examples/scheduler/run_greedy.py --lr_scheduler_type cosine

# Use a different model
python examples/scheduler/run_greedy.py --model_name_or_path Qwen/Qwen3-0.6B

# Fine-tune from pretrained weights
python examples/scheduler/run_greedy.py --finetune
```

## Citation

```bibtex
@article{greedylr2025,
  title={GreedyLR: A Novel Adaptive Learning Rate Scheduler},
  author={Subramanian, Shreyas and Krishnamoorthy, Bala and Murthy, Pranav},
  journal={arXiv preprint arXiv:2512.14527},
  year={2025}
}
```
