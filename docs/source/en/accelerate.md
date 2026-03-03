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

# Accelerate

[Accelerate](https://hf.co/docs/accelerate/index) provides a unified interface for distributed training backends like [FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) or [DeepSpeed](https://www.deepspeed.ai/). It detects your environment (number of GPUs, distributed backend, mixed precision, etc.) and automatically configures training, whether you're on 1 GPU with DDP or 8 GPUs with FSDP.

Accelerate wraps the model in the appropriate distributed wrapper, moves it to the correct device, and creates a compatible optimizer. During training, Accelerate uses its own [`~accelerate.Accelerator.backward`] method to handle gradient scaling for mixed precision. [`Trainer`] calls the appropriate Accelerate APIs and delegates all distributed mechanics to Accelerate.

Configure Accelerate for [`Trainer`] with either an Accelerate config file or [`TrainingArguments`].

## Accelerate config file

Run the [accelerate config](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-config) command and answer questions about your hardware and training setup. This creates a `default_config.yaml` file in your cache. The example below is for FSDP.

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_use_orig_params: true
mixed_precision: bf16
num_machines: 1
num_processes: 4
```

Run [accelerate launch](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-launch) with a [`Trainer`]-based script, and Accelerate reads the config file to set up training. The [fsdp_config](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.fsdp_config) and [deepspeed](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.deepspeed) args are unnecessary because the Accelerate config file covers the same settings.

```cli
accelerate launch train.py
```

## TrainingArguments

Pass a backend-specific config to [`TrainingArguments`]. The [`~Trainer.create_accelerator_and_postprocess`] method reads the settings and configures training.

<hfoptions id="backend">
<hfoption id="FSDP">

Pass a JSON config file or dict to [`fsdp_config`]. See [FSDP](./fsdp) for a full guide and config reference.

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    fsdp="full_shard auto_wrap",
    fsdp_config="path/to/fsdp.json",
)
```

</hfoption>
<hfoption id="DeepSpeed">

Pass a JSON config file or dict to [`deepspeed`]. See [DeepSpeed](./deepspeed) for a full guide and config reference.

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    deepspeed="path/to/ds_config.json",
)
```

</hfoption>
<hfoption id="DDP">

DDP is configured directly through [`TrainingArguments`] fields. See [DDP](./ddp) for details.

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    ddp_backend="nccl",
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=25,
    ddp_timeout=1800,
)
```

</hfoption>
</hfoptions>

## Accelerate training settings

The [accelerator_config](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.accelerator_config) accepts settings that don't have dedicated top-level arguments. For example, set `non_blocking=True` together with [`~TrainingArguments.dataloader_pin_memory`] to overlap data transfer with compute for higher GPU throughput.

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    dataloader_pin_memory=True,
    accelerator_config={
        "non_blocking": True,
    },
)
```

## Next steps

- See [DDP](./ddp) for data-parallel training when your model fits on one GPU.
- See [FSDP](./fsdp) for sharding parameters, gradients, and optimizer states across GPUs.
- See [DeepSpeed](./deepspeed) for ZeRO optimization and offloading.
