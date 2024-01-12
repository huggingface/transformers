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

# DeepSpeed

[DeepSpeed](https://www.deepspeed.ai/) is a PyTorch optimization library that makes distributed training memory-efficient and fast. At it's core is the [Zero Redundancy Optimizer (ZeRO)](https://hf.co/papers/1910.02054) which enables training large models at scale. ZeRO works in several stages:

* ZeRO-1, optimizer state partioning across GPUs
* ZeRO-2, gradient partitioning across GPUs
* ZeRO-3, parameteter partitioning across GPUs

In GPU-limited environments, ZeRO also enables offloading optimizer memory and computation from the GPU to the CPU to fit and train really large models on a single GPU. DeepSpeed is integrated with the Transformers [`Trainer`] class for all ZeRO stages and offloading. All you need to do is provide a config file or you can use a provided template. For inference, Transformers support ZeRO-3 and offloading since it allows loading huge models.

This guide will walk you through how to deploy DeepSpeed training, the features you can enable, how to setup the config files for different ZeRO stages, offloading, and using DeepSpeed without the [`Trainer`].

## Installation

DeepSpeed is available to install from PyPI or Transformers (for more detailed installation options, take a look at the DeepSpeed [installation details](https://www.deepspeed.ai/tutorials/advanced-install/) or the GitHub [README](https://github.com/microsoft/deepspeed#installation)).

<hfoptions id="install">
<hfoption id="PyPI">

```bash
pip install deepspeed
```

</hfoption>
<hfoption id="Transformers">

```bash
pip install transformers[deepspeed]
```

</hfoption>
</hfoptions>

<Tip>

If you're having difficulties installing DeepSpeed, check the [DeepSpeed CUDA installation](../debugging#deepspeed-cuda-installation) guide.

</Tip>

## DeepSpeed configuration file

DeepSpeed works with the [`Trainer`] class by way of a config file containing all the parameters for configuring how you want setup your training run. When you execute your training script, DeepSpeed logs the configuration it received from [`Trainer`] to the console so you can see exactly what configuration was used.

<Tip>

Find a complete list of DeepSpeed configuration options on the [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/) reference. You can also find more practical examples of various DeepSpeed configuration examples on the [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples) repository or the main [DeepSpeed](https://github.com/microsoft/DeepSpeed) repository. To quickly find specific examples, you can:

```bash
git clone https://github.com/microsoft/DeepSpeedExamples
cd DeepSpeedExamples
find . -name '*json'
# find examples with the Lamb optimizer
grep -i Lamb $(find . -name '*json')
```

</Tip>

The DeepSpeed configuration file is passed as a path to a JSON file if you're training from the command line interface or as a nested `dict` object if you're using the [`Trainer`] in a notebook setting.

<hfoptions id="pass-config">
<hfoption id="path to file">

```py
TrainingArguments(..., deepspeed="path/to/deepspeed_config.json")
```

</hfoption>
<hfoption id="nested dict">

```py
ds_config_dict = dict(scheduler=scheduler_params, optimizer=optimizer_params)
args = TrainingArguments(..., deepspeed=ds_config_dict)
trainer = Trainer(model, args, ...)
```

</hfoption>
</hfoptions>

### DeepSpeed and Trainer parameters

There are three types of configuration parameters:

1. Some of the configuration parameters are shared by [`Trainer`] and DeepSpeed, and it can be difficult to identify errors when there are conflicting definitions. To make it easier, these shared configuration parameters are configured from the [`Trainer`] command line arguments.

2. Some configuration parameters that are automatically derived from the model configuration so you don't need to manually adjust these values. The [`Trainer`] uses a configuration value `auto` to determine set the most correct or efficient value. You could set your own configuration parameters explicitly, but you must take care to ensure the [`Trainer`] arguments and DeepSpeed configuration parameters agree. Mismatches may cause the training to fail in very difficult to detect ways!

3. Some configuration parameters specific to DeepSpeed only which need to be manually set based on your training needs.

You could also modify the DeepSpeed configuration and edit [`TrainingArguments`] from it:

1. Create or load a DeepSpeed configuration to used as the main configuration
2. Create a [`TrainingArguments`] object based on these DeepSpeed configuration values

Some values, such as `scheduler.params.total_num_steps` are calculated by the [`Trainer`] during training.

### ZeRO configuration

There are three configurations, each corresponding to a different ZeRO stage. Stage 1 is not as interesting for scalability, and this guide focuses on stages 2 and 3. The `zero_optimization` configuration contains all the options for what to enable and how to configure them. For a more detailed explanation of each parameter, take a look at the [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/) reference.

<Tip warning={true}>
DeepSpeed doesn’t validate parameter names and any typos fallback on the parameter's default setting. You can watch the DeepSpeed engine startup log messages to see what values it is going to use.

</Tip>

The following configurations must be setup with DeepSpeed because the [`Trainer`] doesn't provide equivalent command line arguments.

<hfoptions id="zero-config">
<hfoption id="ZeRO-1">

ZeRO-1 shards the optimizer states across GPUs, and you can expect a tiny speed up. The ZeRO-1 config can be setup like this:

```yml
{
    "zero_optimization": {
        "stage": 1
    }
}
```

</hfoption>
<hfoption id="ZeRO-2">

ZeRO-2 shards the optimizer and gradients across GPUs. This stage is primarily used for training since it's features are not relevant to inference. Some important parameters to configure for better performance include:

* `offload_optimizer` should be enabled to reduce GPU memory usage.
* `overlap_comm` when set to `true` trades off increased GPU memory usage to lower allreduce latency. This feature uses 4.5x the `allgather_bucket_size` and `reduce_bucket_size` values. In this example, they're set to `5e8` which means it requires 9GB of GPU memory. If your GPU memory is 8GB or less, you should reduce `overlap_comm` to lower the memory requirements and prevent an out-of-memory (OOM) error.
* `allgather_bucket_size` and `reduce_bucket_size` trade off available GPU memory for communication speed. The smaller their values, the slower communication is and the more GPU memory is available. You can balance, for example, whether a bigger batch size is more important than a slightly slower training time.
* `round_robin_gradients` is available in DeepSpeed 0.4.4 for CPU offloading. It parallelizes gradient copying to CPU memory among ranks by fine-grained gradient partitioning. Performance benefit grows with gradient accumulation steps (more copying between optimizer steps) or GPU count (increased parallelism).

```yml
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
        "round_robin_gradients": true
    }
}
```

</hfoption>
<hfoption id="ZeRO-3">

ZeRO-3 shards the optimizer, gradient, and parameters across GPUs. Unlike ZeRO-2, ZeRO-3 can also be used for inference, in addition to training, because it allows large models to be loaded on multiple GPUs. Some important parameters to configure include:

* `device: "cpu"` can help if you're running out of GPU memory and if you have free CPU memory available. This allows offloading model parameters to the CPU.
* `pin_memory: true` can improve throughput, but less memory becomes available for other processes because the pinned memory is reserved for the specific process that requested it and it's typically accessed much faster than normal CPU memory.
* `stage3_max_live_parameters` is the upper limit on how many full parameters you want to keep on the GPU at any given time. Reduce this value if you encounter an OOM error.
* `stage3_max_reuse_distance` is a value for determining when a parameter is used again in the future, and it helps decide whether to throw the parameter away or to keep it. If the parameter is going to be reused (if the value is less than `stage3_max_reuse_distance`), then it is kept to reduce communication overhead. This is super helpful when activation checkpointing is enabled and you want to keep the parameter in the forward recompute until the backward pass. But reduce this value if you encounter an OOM error.
* `stage3_gather_16bit_weights_on_model_save` consolidates fp16 weights when a model is saved. For large models and multiple GPUs, this is an expensive in terms of memory and speed. You should enable it if you're planning on resuming training.
* `sub_group_size` controls which parameters are updated during the optimizer step. Parameters are grouped into buckets of `sub_group_size` and each bucket is updated one at a time. When used with NVMe offload, `sub_group_size` determines when model states are moved in and out of CPU memory from during the optimization step. This prevents running out of CPU memory for extremely large models. `sub_group_size` can be left to its default value if you aren't using NVMe offload, but you may want to change it if you:

    1. Run into an OOM error during the optimizer step. In this case, reduce `sub_group_size` to reduce memory usage of the temporary buffers.
    2. The optimizer step is taking a really long time. In this case, increase `sub_group_size` to improve bandwidth utilization as a result of increased data buffers.

* `reduce_bucket_size`, `stage3_prefetch_bucket_size`, and `stage3_param_persistence_threshold` are dependent on a model's hidden size. It is recommended to set these values to `auto` and allow the [`Trainer`] to automatically assign the values.

```yml
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

</hfoption>
</hfoptions>

### NVMe configuration

[ZeRO-Infinity](https://hf.co/papers/2104.07857) allows offloading model states to the CPU and/or NVMe to save even more memory. Smart partitioning and tiling algorithms allow each GPU to send and receive very small amounts of data during offloading such that a modern NVMe can fit an even larger total memory pool than is available to your training process. ZeRO-Infinity requires ZeRO-3.

Depending on the CPU and/or NVMe memory available, you can offload both the [optimizer states](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading) and [parameters](https://www.deepspeed.ai/docs/config-json/#parameter-offloading), just one of them, or none. You should also make sure the `nvme_path` is pointing to an NVMe device, because while it still works with a normal hard drive or solid state drive, it'll be significantly slower. With a modern NVMe, you can expect peak transfer speeds of ~3.5GB/s for read and ~3GB/s for write operations. Lastly, [run a benchmark](https://github.com/microsoft/DeepSpeed/issues/998) on your training setup to determine the optimal `aio` configuration.

The example ZeRO-3/Infinity configuration file below sets most of the parameter values to `auto`, but you could also manually add these values.

```yml
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

## DeepSpeed features

There are a number of important parameters to specify in the DeepSpeed configuration file which are briefly described in this section.

### Activation/gradient checkpointing

Activation and gradient checkpointing trades speed for more GPU memory which allows you to overcome scenarios where your GPU is out of memory or to increase your batch size for better performance. To enable this feature:

1. For a Hugging Face model, set `model.gradient_checkpointing_enable()` or `--gradient_checkpointing` in the [`Trainer`].
2. For a non-Hugging Face model, use the DeepSpeed [Activation Checkpointing API](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html). You could also replace the Transformers modeling code and replace `torch.utils.checkpoint` with the DeepSpeed API. This approach is more flexible because you can offload the forward activations to the CPU memory instead of recalculating them.

### Optimizer and scheduler

DeepSpeed and Transformers optimizer and scheduler can be mixed and matched as long as you don't enable `offload_optimizer`. When `offload_optimizer` is enabled, you could use a non-DeepSpeede optimizer (except for LAMB) as long as it has both a CPU and GPU implementation.

<hfoptions id="opt-sched">
<hfoption id="optimizer">

DeepSpeed offers several [optimizers](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters) (Adam, AdamW, OneBitAdam, and LAMB) but you can also import other optimizers from PyTorch.

</hfoption>
<hfoption id="scheduler">


</hfoption>
<hfoptions>

### Precision

### Batch size

### Gradient accumulation

### Gradient clipping

### NCCL collectives

### Apex

## Deployment
