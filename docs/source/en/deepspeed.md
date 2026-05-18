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

# DeepSpeed ZeRO

[DeepSpeed](https://www.deepspeed.ai/) ZeRO (Zero Redundancy Optimizer) eliminates memory redundancy across distributed training by sharding optimizer states, gradients, and parameters across GPUs. ZeRO has three stages, each sharding more state than the last. DeepSpeed also supports offloading to CPU or NVMe memory for further savings. Every additional stage and offload level reduces peak memory, at the cost of more inter-GPU communication.

```text
                  params        grads       opt states
                ┌──────────┐ ┌──────────┐ ┌──────────┐
ZeRO-1          │██████████│ │██████████│ │███░░░░░░░│  GPU 0
                │██████████│ │██████████│ │░░░███░░░░│  GPU 1
                │██████████│ │██████████│ │░░░░░░████│  GPU 2
                └──────────┘ └──────────┘ └──────────┘
                ┌──────────┐ ┌──────────┐ ┌──────────┐
ZeRO-2          │██████████│ │███░░░░░░░│ │███░░░░░░░│  GPU 0
                │██████████│ │░░░███░░░░│ │░░░███░░░░│  GPU 1
                │██████████│ │░░░░░░████│ │░░░░░░████│  GPU 2
                └──────────┘ └──────────┘ └──────────┘
                ┌──────────┐ ┌──────────┐ ┌──────────┐
ZeRO-3          │███░░░░░░░│ │███░░░░░░░│ │███░░░░░░░│  GPU 0
                │░░░███░░░░│ │░░░███░░░░│ │░░░███░░░░│  GPU 1
                │░░░░░░████│ │░░░░░░████│ │░░░░░░████│  GPU 2
                └──────────┘ └──────────┘ └──────────┘
  █ resident    ░ held on another GPU
```

ZeRO-2 shards gradients and optimizer states with lower communication overhead than ZeRO-3. Use ZeRO-3 only when your model doesn't fit across GPUs with ZeRO-2.

## Installation

Install DeepSpeed from PyPI, or install Transformers with the `deepspeed` extra.

```shell
pip install deepspeed
pip install transformers[deepspeed]
```

If you run into CUDA-related install errors, check the [DeepSpeed CUDA](./debugging#deepspeed-cuda) docs. [Installing from source](https://www.deepspeed.ai/tutorials/advanced-install/#install-deepspeed-from-source) is the more reliable option because it matches your exact hardware and includes features not yet available in the PyPI release.

## Configure

[`Trainer`] integrates DeepSpeed through the [TrainingArguments.deepspeed](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.deepspeed) argument, which accepts a JSON config file. Alternatively, use an [Accelerate config file](./accelerate#accelerate-config-file) instead of [`TrainingArguments`].

<hfoptions id="launch">
<hfoption id="TrainingArguments">

Use `"auto"` in your config for values you want DeepSpeed to fill from [`TrainingArguments`]. If you want to explicitly specify a value, make sure you use the *same* value for both the DeepSpeed argument and [`TrainingArguments`].

> [!NOTE]
> See the [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/) reference for a complete list of DeepSpeed config options.

```json
"train_micro_batch_size_per_gpu": "auto",  // ← per_device_train_batch_size in TrainingArguments
"gradient_accumulation_steps": "auto",     // ← gradient_accumulation_steps in TrainingArguments
"optimizer.params.lr": "auto",             // ← learning_rate in TrainingArguments
"fp16.enabled": "auto",                    // ← fp16 flag in TrainingArguments
```

Pass the config to the `deepspeed` argument.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    deepspeed="path/to/deepspeed_config.json",
    ...
)
```

```cli
# DeepSpeed launcher
deepspeed --num_gpus 4 train.py

# torchrun
torchrun --nproc_per_node 4 train.py

# Accelerate
accelerate launch --num_processes 4 train.py
```

</hfoption>
<hfoption id="Accelerate config file">

> [!NOTE]
> Accelerate ignores the [`deepspeed`] argument in [`TrainingArguments`].

Run the [accelerate config](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-config) command and answer questions about your hardware and training setup to create a `default_config.yaml` file in your cache.

```yaml
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: path/to/ds_config.json
machine_rank: 0
num_machines: 1
num_processes: 4
```

Run [accelerate launch](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-launch) with a [`Trainer`]-based script.

```shell
accelerate launch --config_file deepspeed_config.yaml train.py
```

</hfoption>
</hfoptions>

## ZeRO stages

Select a ZeRO stage config to use as a starting point.

<hfoptions id="zero">
<hfoption id="ZeRO-1">

```json
{
    "bf16": { "enabled": "auto" },
    "zero_optimization": { "stage": 1 },
    "gradient_clipping": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto"
}
```

</hfoption>
<hfoption id="ZeRO-2">

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

</hfoption>
<hfoption id="ZeRO-3">

> [!WARNING]
> ZeRO-3 shards parameters during initialization. You must instantiate [`TrainingArguments`] before loading your model — if the model is already on each GPU before DeepSpeed is configured, no memory is saved.

```json
{
    "bf16": { "enabled": "auto" },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_gather_16bit_weights_on_model_save": true,
        "offload_optimizer": { "device": "cpu", "pin_memory": true },  // optional offloading
        "offload_param":     { "device": "cpu", "pin_memory": true }  // optional offloading
    },
    "gradient_clipping": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto"
}
```

</hfoption>
</hfoptions>

The following fields are important for customizing training.

- `zero_optimization` sets the ZeRO stage.

    ```json
    { "zero_optimization": { "stage": 3 } }
    ```

- Set the batch size and gradient accumulation arguments to `"auto"`. If you manually set these to values that disagree with [`TrainingArguments`], training continues silently with the wrong values.

    ```json
    {
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto"
    }
    ```

- `bf16` sets the training precision. Set it to `"auto"` so it mirrors the `bf16` flag in [`TrainingArguments`].

    ```json
    { "bf16": { "enabled": "auto" } }
    ```

- `stage3_gather_16bit_weights_on_model_save` performs an all-gather across all GPUs before saving, reconstructing the full tensors from their shards. This is a ZeRO-3 argument.

    ```json
    {
        "zero_optimization": {
            "stage": 3,
            "stage3_gather_16bit_weights_on_model_save": true,
        }
    }
    ```

- Set `overlap_comm` to `true` to hide all-reduce latency behind the backward pass. `allgather_bucket_size` and `reduce_bucket_size` trade communication speed for GPU memory. Lower values use less memory but slow communication.

    ```json
    {
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": true,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": true
        }
    }
    ```

- `offload_optimizer` offloads the optimizer to CPU memory. To save even more memory, also offload model parameters with `offload_param` (ZeRO-3 only). Set `pin_memory` to `true` to speed up CPU-GPU transfers, but this locks RAM that is unavailable to other processes.

    ```json
    {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": { "device": "cpu", "pin_memory": true },
            "offload_param":     { "device": "cpu", "pin_memory": true }
        }
    }
    ```

- `optimizer` and `scheduler` default to the optimizer and scheduler configured in [`TrainingArguments`]. Set to `"auto"` so DeepSpeed reads the values from [`TrainingArguments`] unless you need a DeepSpeed-native optimizer like LAMB.

    ```json
    {
        "optimizer": {
            "type": "AdamW",
            "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": { "total_num_steps": "auto", "warmup_min_lr": "auto", "warmup_max_lr": "auto", "warmup_num_steps": "auto" }
        }
    }
    ```

    If you're offloading the optimizer, set `zero_force_ds_cpu_optimizer` to `false` to use DeepSpeed's CPU Adam optimizer.

    ```json
    {
        "zero_force_ds_cpu_optimizer": false
    }
    ```

## Checkpoints

DeepSpeed saves checkpoints in a sharded format that can't be loaded directly with [`~PreTrainedModel.from_pretrained`]. Set [`~TrainingArguments.load_best_model_at_end`] to `True` to have Trainer track and reload the best checkpoint at the end of training.

```py
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    deepspeed="ds_config_zero3.json",
    load_best_model_at_end=True,
    ...
)
# after training, save a normal transformers checkpoint
trainer.save_model("./best-model")
```

Setting `save_only_model=True` skips saving the full optimizer state, which means you can't reload the best model at the end of training. Also set `stage3_gather_16bit_weights_on_model_save: true` to reconstruct full weights from their shards. This is required for saving a consolidated 16-bit model artifact or 16-bit state dict with ZeRO-3. Transformers raises an error when `save_only_model=True` is combined with `load_best_model_at_end=True`.

> [!TIP]
> For resuming across different parallelism configurations, see DeepSpeed's [Universal Checkpointing](https://www.deepspeed.ai/tutorials/universal-checkpointing) guide.

## Next steps

- Read the [Zero Redundancy Optimizer](https://nanotron-ultrascale-playbook.static.hf.space/index.html#zero_redundancy_optimizer_(zero)) chapter from The Ultra-Scale Playbook to learn more about how ZeRO works.
- Read the ZeRO papers: [Memory Optimizations Toward Training Trillion Parameter Models](https://hf.co/papers/1910.02054), [Democratizing Billion-Scale Model Training](https://hf.co/papers/2101.06840), and [Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://hf.co/papers/2104.07857).
