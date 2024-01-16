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

<Tip warning={true}>

The optimizer and scheduler parameters for the config file can be set from the command line to avoid hard to find errors. For example, if the learning rate is set to a different value in another place you can override it from the command line. Aside from the optimizer and scheduler parameters, you'll need to ensure your [`Trainer`] command line arguments match the DeepSpeed configuration.

</Tip>

<hfoptions id="opt-sched">
<hfoption id="optimizer">

DeepSpeed offers several [optimizers](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters) (Adam, AdamW, OneBitAdam, and LAMB) but you can also import other optimizers from PyTorch. If you don't configure the optimizer in the config, the [`Trainer`] automatically selects AdamW and either uses the supplied values or the default values for the following parameters from the command line: `lr`, `adam_beta1`, `adam_beta2`, `adam_epsilon`, `weight_decay`.

You can set the parameters to `"auto"` or manually input your own desired values.

```yaml
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": "auto",
         "betas": "auto",
         "eps": "auto",
         "weight_decay": "auto"
       }
   }
}
```

You can also use an unsupported optimizer by adding the following to the top level configuration.

```yaml
{
   "zero_allow_untested_optimizer": true
}
```

From DeepSpeed==0.8.3 on, if you want to use offload, you'll also need to the following to the top level configuration because offload works best with DeepSpeed's CPU Adam optimizer.

```yaml
{
   "zero_force_ds_cpu_optimizer": false
}
```

</hfoption>
<hfoption id="scheduler">

DeepSpeed supports the LRRangeTest, OneCycle, WarmupLR and WarmupDecayLR learning rate [schedulers](https://www.deepspeed.ai/docs/config-json/#scheduler-parameters).

Transformers and DeepSpeed provide two of the same schedulers:

* WarmupLR is the same as `--lr_scheduler_type constant_with_warmup` in Transformers
* WarmupDecayLR is the same as  `--lr_scheduler_type linear` in Transformers (this is the default scheduler used in Transformers)

If you don't configure the scheduler in the config, the [`Trainer`] automatically selects WarmupDecayLR and either uses the supplied values or the default values for the following parameters from the command line: `warmup_min_lr`, `warmup_max_lr`, `warmup_num_steps`, `total_num_steps` (automatically calculated during run time if `max_steps` is not provided).

You can set the parameters to `"auto"` or manually input your own desired values.

```yaml
{
   "scheduler": {
         "type": "WarmupDecayLR",
         "params": {
             "total_num_steps": "auto",
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     }
}
```

</hfoption>
</hfoptions>

### Precision

Deepspeed supports fp32, fp16, and bf16 mixed precision.

<hfoptions id="precision">
<hfoption id="fp32">

If your model doesn't work well with mixed precision, for example if it wasn't pretrained in mixed precision, you may encounter overflow or underflow issues which can cause NaN loss. For these cases, you should use full fp32 precision by explicitly disabling the default fp16 mode.

```yaml
{
    "fp16": {
        "enabled": false
    }
}
```

For Ampere GPUs and PyTorch > 1.7, it automatically switches to the more efficient [tf32](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices) format for some operations but the results are still in fp32. You can control it from the [`Trainer`] by setting `--tf32` to enable it, and `--tf32 0` or `--no_tf32` to disable it.

</hfoption>
<hfoption id="fp16">

To configure PyTorch AMP-like fp16 mixed precision reduces memory usage and accelerates training speed. [`Trainer`] automatically enables or disables fp16 based on the value of `args.fp16_backend`, and the rest of the config can be set by you. fp16 is enabled from the command line when the following arguments are passed: `--fp16`, `--fp16_backend amp` or `--fp16_full_eval`.

```yaml
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

For additional DeepSpeed fp16 training options, take a look at the [FP16 Training Options](https://www.deepspeed.ai/docs/config-json/#fp16-training-options) reference.

To configure Apex-like fp16 mixed precision, setup the config as shown below with `"auto"` or your own values. [`Trainer`] automatically configure `amp` based on the values of `args.fp16_backend` and `args.fp16_opt_level`. It can also be enabled from the command line when the following arguments are passed: `--fp16`, `--fp16_backend apex` or `--fp16_opt_level 01`.

```yaml
{
    "amp": {
        "enabled": "auto",
        "opt_level": "auto"
    }
}
```

</hfoption>
<hfoption id="bf16">

To use bf16, you'll need at least DeepSpeed==0.6.0. bf16 has the same dynamic range as fp32 and doesn’t require loss scaling. However, if you use [gradient accumulation](#gradient-accumulation) with bf16, gradients are accumulated in bf16 which may not be desired because this format's low precision can lead to lossy accumulation.

bf16 can be setup in the config file or enabled from the command line when the following arguments are passed: `--bf16` or `--bf16_full_eval`.

```yaml
{
    "bf16": {
        "enabled": "auto"
    }
}
```

</hfoption>
</hfoptions>

### Batch size

The batch size can be auto-configured or explicitly set. If you choose to use the `"auto"` option, [`Trainer`] sets `train_micro_batch_size_per_gpu` to the value of args.`per_device_train_batch_size` and `train_batch_size` to `args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps`.

```yaml
{
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto"
}
```

### Gradient accumulation

Gradient accumulation can be auto-configured or explicitly set. If you choose to use the `"auto"` option, [`Trainer`] sets it to the value of `args.gradient_accumulation_steps`.

```yaml
{
    "gradient_accumulation_steps": "auto"
}

```

### Gradient clipping

Gradient clipping can be auto-configured or explicitly set. If you choose to use the `"auto"` option, [`Trainer`] sets it to the value of `args.max_grad_norm`.

```yaml
{
    "gradient_clipping": "auto"
}
```

### Communication data type

For communication collectives like reduction, gathering and scattering operations, a separate data type is used.

All gather and scatter operations are performed in the same data type the data is in. For example, if you're training with bf16, the data is also gathered in bf16 because gathering is a non-lossy operation.

Reduce operations are lossy, for example when gradients are averaged across multiple GPUs. When the communication is done in fp16 or bf16, it is more likely to be lossy because adding multiple numbers in low precision isn't exact. This is especially the case with bf16 which has a lower precision than fp16. For this reason, fp16 is the default for reduction operations because the loss is minimal when averaging gradients.

You can choose the communication data type by setting the `communication_data_type` parameter in the config file. For example, choosing fp32 adds a small amount of overhead but ensures the reduction operation is accumulated in fp32 and when it is ready, it is downcasted to whichever half-precision dtype you're training in.

```yaml
{
    "communication_data_type": "fp32"
}
```

## Deployment

DeepSpeed can be deployed by different launchers such as [torchrun](https://pytorch.org/docs/stable/elastic/run.html), the `deepspeed` launcher, or [Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch). To deploy, add `--deepspeed ds_config.json` to the [`Trainer`] command line. It’s recommended to use DeepSpeed’s [`add_config_arguments`](https://deepspeed.readthedocs.io/en/latest/initialize.html#argument-parsing) utility to add any necessary command line arguments to your code.

This guide will show you how to deploy DeepSpeed with the `deepspeed` launcher for different training setups. You can check out this [post](https://github.com/huggingface/transformers/issues/8771#issuecomment-759248400) for more practical usage examples.


<hfoptions id="deploy">
<hfoption id="multi-GPU">

To deploy DeepSpeed on multiple GPUs, add the `--num_gpus` parameter. If you want to use all available GPUs, you don't need to add `--num_gpus`. The example below uses 2 GPUs.

```bash
deepspeed --num_gpus=2 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

</hfoption>
<hfoption id="single-GPU">

To deploy DeepSpeed on a single GPU, add the `--num_gpus` parameter. It isn't necessary to explicitly set this value if you only have 1 GPU because DeepSpeed deploys all GPUs it can see on a given node.

```bash
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero2.json \
--model_name_or_path t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

DeepSpeed is still useful with just 1 GPU because you can:

1. Offload some computations and memory to the CPU to make more GPU resources available to your model to use a larger batch size or fit a very large model that normally won't fit.
2. Minimze memory fragmentation with it's smart GPU memory management system which also allows you to fit bigger models and data batches.

<Tip>

Set the `allgather_bucket_size` and `reduce_bucket_size` values to 2e8 in the [ZeRO-2](#zero-configuration) configuration file to get better performance on a single GPU.

</Tip>

</hfoption>
</hfoptions>

### Multi-node deployment

A node is one or more GPUs for running a workload. A more powerful setup is a multi-node setup which can be launched with the `deepspeed` launcher. For this guide, let's assume there are two nodes with 8 GPUs each. The first node can be accessed `ssh hostname1` and the second node with `ssh hostname2`. Both nodes must be able to communicate with each other locally over ssh without a password.

By default, DeepSpeed expects your multi-node environment to use a shared storage. If this is not the case and each node can only see the local filesystem, you need to adjust the config file to include a [`checkpoint`](https://www.deepspeed.ai/docs/config-json/#checkpoint-options) to allow loading without access to a shared filesystem:

```yaml
{
  "checkpoint": {
    "use_node_local_storage": true
  }
}
```

You could also use the [`Trainer`]'s `--save_on_each_node` argument to automatically add the above `checkpoint` to your config.

<hfoptions id="multinode">
<hfoption id="torchrun">

For [torchrun](https://pytorch.org/docs/stable/elastic/run.html), you have to ssh to each node and run the following command on both of them. The launcher waits until both nodes are synchronized before launching the training.

```bash
python -m torch.run --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=hostname1 \
--master_port=9901 your_program.py <normal cl args> --deepspeed ds_config.json
```

</hfoption>
<hfoption id="deepspeed">

For the `deepspeed` launcher, start by creating a `hostfile`.

```bash
hostname1 slots=8
hostname2 slots=8
```

Then you can launch the training with the following command. The `deepspeed` launcher automatically launches the command on both nodes at once.

```bash
deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 \
your_program.py <normal cl args> --deepspeed ds_config.json
```

Check out the [Resource Configuration (multi-node)](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) guide for more details about configuring multi-node compute resources.

</hfoption>
</hfoptions>

### SLURM

In a SLURM environment, you'll need to adapt your SLURM script to your specific SLURM environment. An example SLURM script may look like:

```bash
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'
```

Then you can schedule your multi-node deployment with the following command which launches training simultaneously on all nodes.

```bash
sbatch launch.slurm
```

### Notebook

The `deepspeed` launcher doesn't support deployment from a notebook so you'll need to emulate the distributed environment. However, this only works for 1 GPU. If you want to use more than 1 GPU, you must use a multi-process environment for DeepSpeed to work. This means you have to use the `deepspeed` launcher which can't be emulated as shown here.

```py
# DeepSpeed requires a distributed environment even when only one process is used.
# This emulates a launcher in the notebook
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# Now proceed as normal, plus pass the DeepSpeed config file
training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
trainer = Trainer(...)
trainer.train()
```

If you want to create the config file on the fly in the notebook in the current directory, you could have a dedicated cell.

```py
%%bash
cat <<'EOT' > ds_config_zero3.json
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
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT
```

If the training script is in a file and not in a notebook cell, you can launch `deepspeed` normally from the shell in a notebook cell. For example, to launch `run_translation.py`:

```py
!git clone https://github.com/huggingface/transformers
!cd transformers; deepspeed examples/pytorch/translation/run_translation.py ...
```

You could also use `%%bash` magic and write multi-line code to run the shell program, but you won't be able to view the logs until training is complete. With `%%bash` magic, you don't need to emulate a distributed environment.

```py
%%bash

git clone https://github.com/huggingface/transformers
cd transformers
deepspeed examples/pytorch/translation/run_translation.py ...
```
