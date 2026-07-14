<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Optimizers and schedulers

An optimizer updates model weights during training. The scheduler wraps the optimizer and adjusts the learning rate each training step. [`Trainer`] creates both when it calls [`~Trainer.create_optimizer_and_scheduler`].

```md
                                    ┌────────────┐         ┌──────────────┐
                                    │ Optimizer  │         │  Scheduler   │
                                    │ (adamw_torch_fused)◄─│  (linear)    │
                                    │            │         │              │
                                    │ param_groups         |              |
                                    │  └ lr       ◄────────┤              |
                                    │  └ weight_decay      │              │
                                    └──────┬─────┘         └──────────────┘
                                           │                      
  ┌──── EACH TRAINING STEP ───────────────────────────────────────────┐
  │                                        │                          │
  │   model(batch)                         │                          │
  │       │                                │                          │
  │       ▼                                │                          │
  │     loss ──► loss.backward() ──► param.grad                       │
  │                                        │                          │
  │                          ┌─────────────┘                          │
  │                          ▼                                        │
  │              optimizer.step()                                     │
  │                          │                                        │
  │                          ▼                                        │
  │                   param.data updated                              │
  │                          │                                        │
  │                          ▼                                        │
  │              lr_scheduler.step()  ──► recalculates lr             │
  │                          │            writes to optimizer         │
  │                          ▼            .param_groups['lr']         │
  │              model.zero_grad()                                    │
  │                                                                   │
  └───────────────────────────────────────────────────────────────────┘
```

Configure optimizer and scheduler behavior, like [`~TrainingArguments.lr_scheduler_type`] and [`~TrainingArguments.optim`], in [`TrainingArguments`]. The defaults (`adamw_torch` optimizer and `linear` warmup scheduler) are a good starting point for most fine-tuning runs.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    ...,
    # Optimizer
    optim="adamw_torch",          # or "adamw_torch_fused", "adafactor", "sgd", etc.
    learning_rate=2e-5,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    # Scheduler
    lr_scheduler_type="cosine",   # "linear", "cosine", "constant_with_warmup", etc.
    warmup_steps=500,
    lr_scheduler_kwargs={"num_cycles": 3},  # scheduler-specific extras
)
```

## Metric-based schedulers

Some schedulers adapt to training dynamics instead of following a fixed schedule.

[GreedyLR](https://huggingface.co/papers/2512.14527) updates the learning rate from evaluation results. It raises the learning rate by dividing it by `factor` when the metric keeps improving, and lowers the learning rate by multiplying it by `factor` when the metric doesn't improve. When the learning rate stops at `min_lr` and doesn't improve after `reset_start` steps, [`GreedyLR`] resets to its initial state and starts a new cycle.

[`GreedyLR`] requires evaluation during training. Set `eval_strategy` to `"steps"` or `"epoch"`.

```diff
args = TrainingArguments(
+   lr_scheduler_type="greedy",
+   lr_scheduler_kwargs={"patience": 10, "factor": 0.95, "min_lr": 1e-5},
+   eval_strategy="steps",
+   eval_steps=200,
    ...  # remaining args from the TrainingArguments intro config
)
```

> [!TIP]
> The default `mode="min"` works for loss. If you're tracking a metric where a higher value is better, like accuracy, pass `"mode": "max"` in `lr_scheduler_kwargs`.

See the [`GreedyLR`] class for the full list of configurable parameters.

## Optimizer integrations

Transformers integrates third-party optimizers for specialized training scenarios.

| Optimizer | Install | `optim="value"` | Description |
|---|---|---|---|
| APOLLO | `apollo-torch` | `apollo_adamw` | Memory-efficient full-param via random projections; rank-1 sufficient |
| FlashOptim | `flashoptim` | `flash_adamw`, `flash_adam`, `flash_sgd`, `flash_sgdw`, `flash_lion` | Reduces optimizer memory with low-precision master weights |
| GrokAdamW | `grokadamw` | `grokadamw` | Targets delayed generalization (grokking) |
| LOMO / AdaLomo | `lomo-optim` | `lomo` / `adalomo` | Fuses gradient + update step for low-memory full-param fine-tuning |
| Schedule Free | `schedulefree` | `schedule_free_adamw`, `schedule_free_radam`, `schedule_free_sgd` | Eliminates LR annealing; pair with `lr_scheduler_type="constant"` |
| GaLore | `galore-torch` | `galore_adamw`, `galore_adafactor`, `galore_adamw_8bit` | Full-parameter learning via gradient low-rank projection |
| StableAdamW | `torch-optimi` | `stable_adamw` | AdamW + AdaFactor update clipping; no gradient clipping needed |

<hfoptions id="optimizer">
<hfoption id="APOLLO">

```bash
pip install apollo-torch
```

[Approximated Gradient Scaling for Memory Efficient LLM Optimization (APOLLO)](https://huggingface.co/papers/2412.05270) is a memory-efficient optimizer for full-parameter learning during pretraining and fine-tuning. It matches AdamW performance with SGD-like memory cost by using cheap random projections instead of SVD. For extreme memory savings, use APOLLO-Mini, a rank-1 variant.

Use the `optim_target_modules` parameter to specify which layers to train.

```diff
args = TrainingArguments(
+   optim="apollo_adamw",
+   optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    ...  # remaining args from the TrainingArguments intro config
)
```

Pass additional hyperparameters through `optim_args`.

> [!TIP]
> Set `scale` to `n/r`, where `n` is the original space dimension and `r` is the low-rank space dimension. Adjusting the learning rate while keeping `scale` at its default achieves a similar effect.

| parameter | description | APOLLO | APOLLO-Mini |
|---|---|---|---|
| rank | rank of the auxiliary sub-space for gradient scaling | 256 | 1 |
| scale_type | how scaling factors are applied | `channel` (per-channel scaling) | `tensor` (per-tensor scaling) |
| scale | adjusts gradient updates to stabilize training | 1.0 | 128 |
| update_proj_gap | steps before updating projection matrices | 200 | 200 |
| proj | projection type | `random` | `random` |

Enable APOLLO-Mini with a rank-1 configuration.

```py
args = TrainingArguments(
    optim="apollo_adamw",
    optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    optim_args="proj=random,rank=1,scale=128.0,scale_type=tensor,update_proj_gap=200",
    ...  # remaining args from the TrainingArguments intro config
)
```

</hfoption>
<hfoption id="FlashOptim">

```bash
pip install flashoptim
```

[FlashOptim](https://huggingface.co/papers/2602.23349) reduces optimizer memory by storing master weights in lower precision. It supports AdamW, Adam, SGD, SGDW, and Lion variants.

> [!TIP]
> FlashOptim requires bf16 or fp16 model weights. It automatically disables `master_weight_bits` and warns if your model uses fp32.

```diff
args = TrainingArguments(
+   optim="flash_adamw",
+   bf16=True,
    ...  # remaining args from the TrainingArguments intro config
)
```

`master_weight_bits` controls the precision of the optimizer's master weight copy. By default, it stores the master copy in 24 bits. Set it to `"None"` to remove the master copy entirely for maximum memory savings at the cost of a slightly higher loss.

```diff
args = TrainingArguments(
+   optim="flash_adamw",
+   optim_args="master_weight_bits=None",
+   bf16=True,
    ...  # remaining args from the TrainingArguments intro config
)
```

</hfoption>
<hfoption id="GrokAdamW">

```bash
pip install grokadamw
```

[GrokAdamW](https://github.com/cognitivecomputations/grokadamw) targets *grokking*, where models exhibit delayed generalization due to slow-varying gradients.

```diff
args = TrainingArguments(
+   optim="grokadamw",
    ...  # remaining args from the TrainingArguments intro config
)
```

</hfoption>
<hfoption id="LOMO">

```bash
pip install lomo-optim
```

[Low-Memory Optimization (LOMO)](https://github.com/OpenLMLab/LOMO) includes two optimizers for low-memory full-parameter finetuning, [LOMO](https://huggingface.co/papers/2306.09782) and [AdaLomo](https://hf.co/papers/2310.10195). Both fuse gradient computation and parameter updates into one step. AdaLomo adds an adaptive per-parameter learning rate, similar to Adam.

> [!TIP]
> AdaLomo works best without `grad_norm`, improving performance and throughput.

```diff
args = TrainingArguments(
+   optim="adalomo",
    learning_rate=2e-6,
    ...  # remaining args from the TrainingArguments intro config
)
```

</hfoption>
<hfoption id="Schedule Free">

```bash
pip install schedulefree
```

[Schedule Free optimizer (SFO)](https://hf.co/papers/2405.15682) replaces momentum with a combination of averaging and interpolation, completely removing the need to anneal the learning rate.

SFO supports the RAdam (`schedule_free_radam`), AdamW (`schedule_free_adamw`), and SGD (`schedule_free_sgd`) optimizers. The RAdam scheduler doesn't require `warmup_steps`.

Pair SFO with `lr_scheduler_type="constant"`. Other scheduler types work but affect SFO's intended behavior.

```diff
args = TrainingArguments(
+   optim="schedule_free_radam",
+   lr_scheduler_type="constant",
    learning_rate=2e-6,
    ...  # remaining args from the TrainingArguments intro config
)
```

</hfoption>
<hfoption id="StableAdamW">

```bash
pip install torch-optimi
```

[StableAdamW](https://huggingface.co/papers/2304.13013) ports AdaFactor's update clipping into AdamW, removing the need for gradient clipping. Otherwise, it's a drop-in replacement for AdamW.

> [!TIP]
> If you're training with large batch sizes or still observing loss spikes, try setting `beta_2` between 0.95 and 0.99.

```diff
args = TrainingArguments(
+   optim="stable_adamw",
    learning_rate=2e-6,
    ...  # remaining args from the TrainingArguments intro config
)
```
</hfoption>
<hfoption id="GaLore">

```bash
pip install galore-torch trl
```

[Gradient Low-Rank Projection (GaLore)](https://hf.co/papers/2403.03507) reduces memory for training LLMs. Unlike low-rank adaptation methods like [LoRA](https://hf.co/papers/2106.09685), GaLore preserves *full-parameter* learning.

Set `optim` in [`trl.SFTConfig`] to a GaLore optimizer (`"galore_adamw"`, `"galore_adafactor"`, or `"galore_adamw_8bit"`). Specify target modules with `optim_target_modules` and GaLore-specific parameters (`rank`, `update_proj_gap`, `scale`) through `optim_args`.

```py
from trl import SFTConfig

args = SFTConfig(
    output_dir="./galore",
    max_steps=100,
    optim="galore_adamw",
    optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    optim_args="rank=64, update_proj_gap=100, scale=0.10",
)
```

Append `_layerwise` to the optimizer name for layerwise optimization (`"galore_adamw_layerwise"`). Only linear layers targeted by GaLore use low-rank decomposition. All other layers are optimized normally.

```py
from trl import SFTConfig, SFTTrainer

args = SFTConfig(
    output_dir="./galore",
    max_steps=100,
    optim="galore_adamw_layerwise",
    optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    optim_args="rank=64, update_proj_gap=100, scale=0.10",
)
```

Layerwise mode is experimental. It only runs on a [single GPU](https://github.com/jiaweizzhao/GaLore?tab=readme-ov-file#train-7b-model-with-a-single-gpu-with-24gb-memory), doesn't support DistributedDataParallel (DDP), and gradient clipping and DeepSpeed may not work.

</hfoption>
</hfoptions>

## Customizing optimizer and scheduler

Create a custom optimizer and scheduler to use an optimizer not yet integrated, adjust per-layer learning rates, or apply custom logic.

### Pass a class and kwargs

[`~Trainer.optimizer_cls_and_kwargs`] accepts a custom optimizer class while delegating parameter grouping and device placement to [`Trainer`].

[`Trainer`] defers building the optimizer until [`~Trainer.create_optimizer`] runs, so the model is already on the correct device.

```py
import torch

trainer = Trainer(
    ...
    optimizer_cls_and_kwargs=(
        torch.optim.SGD,
        {"momentum": 0.9, "nesterov": True}
    ),
)
```

### Pass prebuilt instances

Pass a predefined optimizer and scheduler to [`~Trainer.optimizers`]. [`Trainer`] skips [`~Trainer.create_optimizer`] and [`~Trainer.create_scheduler`] when prebuilt instances are provided. If you don't pass a scheduler, [`Trainer`] automatically creates one.

> [!WARNING]
> Build the optimizer after placing your model on the correct device. Parameters are resolved at construction time, before `Trainer` moves the model. In distributed training, mismatched devices can silently cause incorrect behavior.

```py
import torch
from transformers import Trainer, get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=500, num_training_steps=10_000
)

trainer = Trainer(
    ...
    optimizers=(optimizer, scheduler),
)
```

Prebuilt instances bypass [`~Trainer.create_optimizer`] and [`~Trainer.create_scheduler`], so you need to specify your own parameter groups.

### Override optimizer and scheduler methods

Subclass [`~Trainer.create_optimizer`] and [`~Trainer.create_scheduler`] for full control. Both methods run *during* [`~Trainer.train`].

Override [`~Trainer.create_scheduler`] to use a scheduler like [OneCycleLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) that isn't available in [`SchedulerType`].

For each method, make sure to assign to `self` and return it.

```py
import torch
from transformers import Trainer

class MyTrainer(Trainer):

    def create_scheduler(self, num_training_steps, optimizer=None):
        optimizer = optimizer or self.optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.1,
            total_steps=num_training_steps,
        )
        return self.lr_scheduler
```

You don't need to override [`~Trainer.create_optimizer`] if the default optimizer works. Extending a method with `super()` is easier than replacing it entirely. For example, add an extra parameter group while keeping everything else the same.

```py
class MyTrainer(Trainer):
    def create_optimizer(self, model=None):
        super().create_optimizer(model)  # builds the default two param groups
        # add extra param group
        self.optimizer.add_param_group({
            "params": self.model.classifier.parameters(),
            "lr": self.args.learning_rate * 10,
        })
        return self.optimizer
```
