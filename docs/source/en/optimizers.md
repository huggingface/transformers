<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
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

## Optimizer update transforms

Some training methods transform the update proposed by an optimizer instead of
replacing the optimizer itself. These methods can compose with the optimizer
selected by [`TrainingArguments`].

### Subspace Tuning (SST)

[Subspace Tuning (SST)](https://github.com/Arnav0400/SST) projects each applied
optimizer update outside the principal left and right singular subspaces of
selected weight matrices. [`SST`] is an optimizer wrapper: it takes an existing
optimizer, calls its `step()`, and projects the resulting parameter update.
The inner optimizer receives the original gradients and keeps its own parameter
groups, momentum, variance, and weight decay. SST is not a separate AdamW variant
and does not require an external package.

Wrap any compatible PyTorch optimizer in a custom training loop:

```py
import torch
from transformers import SST

inner_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
optimizer = SST(
    inner_optimizer,
    model=model,
    rank_fraction=0.99,
    target_modules=[r".*self_attn.*", r".*mlp.*"],
)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

`rank_fraction` controls the fraction of principal singular directions protected
in each selected matrix. By default, the basis stays fixed at initialization.
Without `target_modules`, SST selects the inner optimizer's
trainable matrices with both dimensions at least two. Use `"all-linear"` to select
linear-layer weights, or pass module suffixes or full-match regular expressions.

Set `refresh_every` to a positive integer to re-anchor the protected singular
triples to the current weights after that many successful, projected optimizer
updates. For example:

```py
optimizer = SST(inner_optimizer, model=model, rank_fraction=0.99, refresh_every=100)
# Alternatively, refresh explicitly at a chosen boundary:
optimizer.refresh()
```

A refresh does not change parameters, gradients, or inner optimizer state.
It takes effect for subsequent updates and restarts the interval. Accumulation
and AMP-overflow skips do not advance the clock. Refresh protects the latest
anchor rather than retaining a guarantee against the original initialization,
and adds SVD and communication costs. `initialize()` only adds new targets;
it does not refresh an existing basis.

With [`Trainer`], pass the wrapper class and its arguments as `optimizer_wrapper`.
Keep `TrainingArguments.optim` or `optimizer_cls_and_kwargs` set to the inner
optimizer you want to use.

```py
from transformers import SST, Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="sst-output", optim="adamw_torch"),
    train_dataset=train_dataset,
    optimizer_wrapper=(
        SST,
        {
            "rank_fraction": 0.99,
            "target_modules": [r".*self_attn.*", r".*mlp.*"],
            "refresh_every": 100,
        },
    ),
)
trainer.train()
```

To use a custom inner optimizer without subclassing [`Trainer`], combine the
class-plus-keyword-arguments entry points:

```py
import torch
from transformers import SST, Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="sst-output"),
    train_dataset=train_dataset,
    optimizer_cls_and_kwargs=(torch.optim.RMSprop, {"lr": 2e-5, "momentum": 0.9}),
    optimizer_wrapper=(SST, {"rank_fraction": 0.99}),
)
trainer.train()
```

Trainer attaches SST after distributed preparation and before scheduler
construction. This lets the wrapper bind to the actual optimizer parameters
after FSDP flattening or sharding and run after DeepSpeed's engine update. Do not
construct a prebuilt SST wrapper before these backends prepare the model.

SST supports ordinary parameters, DDP, FSDP1 (original or flat parameters),
FSDP2/DTensor, and DeepSpeed ZeRO stages 1–3. The inner optimizer must itself
support the selected backend and parameter layout. For example, an optimizer
that only accepts two-dimensional parameters cannot operate on FSDP1's
one-dimensional flat parameters. Transformers' [`Adafactor`] is supported with
DDP, but its factored buffers and RMS checkpoint metadata are not
FSDP-compatible; use the FSDP-compatible `torch.optim.Adafactor` instead.

Projection and refresh are collective: all participating ranks must call
`step()` and any manual `refresh()` in the same order with the same schedule.
Initialization and refresh compute a basis on one source and synchronize it
within the parameter's owning process groups, including hybrid mesh axes.
Independent submeshes do not share their bases. FSDP1 gathers one wrapping
unit at a time; FSDP2 and ZeRO gather one selected matrix at a time. These
gathers, SVDs, and CPU-resident bases add communication, compute, and memory
costs. SST does not support
XLA, SageMaker model parallelism, or differentiable optimizer steps.

For FSDP2 CPU offload, use a process group that also supports CPU collectives,
such as `TrainingArguments(ddp_backend="cpu:gloo,cuda:nccl", ...)`. An NCCL-only
group cannot reduce CPU DTensor gradients during the inner optimizer's gradient
clipping. SST itself moves offloaded matrices to the mesh device for projection
and writes them back to their original CPU shards.

Trainer checkpoints preserve the inner optimizer, active basis, update count,
and refresh schedule. Resume normally; SST restores the basis without an
extra SVD, then continues refreshing at the saved boundaries:

```py
trainer.train(resume_from_checkpoint=checkpoint_dir)
```

Ordinary optimizer checkpoints include the basis under the `"sst"` key. FSDP
and DeepSpeed additionally save `sst_state.pt` beside their native optimizer
checkpoint because their checkpoint converters only retain engine-owned state.
Targets owned by a subgroup or submesh use `sst_state_rank_<rank>.pt` instead.
These checkpoints require the same distributed rank and mesh topology on resume,
so independent submeshes cannot accidentally share a fixed basis. Resuming
without the saved basis raises an error. The checkpoint's rank and refresh
schedule take precedence over constructor arguments. Version-1 SST checkpoints
remain loadable as fixed-basis checkpoints with a fresh refresh clock.

> [!WARNING]
> Exact projection requires fp32 model parameters or fp32 optimizer master
> weights. bf16/fp16 autocast with fp32 optimizer parameters is supported.
> Direct projection of bf16/fp16 optimizer parameters introduces rounding error.
> DeepSpeed projects its fp32 master parameters and writes the rounded result
> back to the low-precision model shards.

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
