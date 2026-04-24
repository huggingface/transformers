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

# FSDP2

[Fully Sharded Data Parallel (FSDP2)](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) shards the model, gradients, and optimizer states across GPUs. Before computation, each GPU gathers a complete set of parameters from all shards, then frees them afterward. Sharding lets you train models larger than a single GPU's memory, at the cost of more communication than [DDP](./ddp). Use FSDP when your model or optimizer states don't fit on a single GPU.

```text
                      ┌─────────────────┐
                      │  training data  │
                      └────────┬────────┘
            ┌──────────────────┼──────────────────┐
            │ shard 0          │ shard 1          │ shard 2
            ▼                  ▼                  ▼
     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
     │  param      │    │  param      │    │  param      │
     │  shard 0    │    │  shard 1    │    │  shard 2    │
     │  GPU 0      │    │  GPU 1      │    │  GPU 2      │
     └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
            │                  │                  │
            └──────── all-gather (params) ────────┘
                               │
                    full params on each GPU
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
         forward             forward             forward
            │                  │                  │
            └───── reduce-scatter (grads) ────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
     grad shard 0       grad shard 1       grad shard 2
     optim shard 0      optim shard 1      optim shard 2
        step               step               step
```

## Sharding strategies

FSDP2 controls sharding with [fsdp_config](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.fsdp_config). Set `fsdp=True` to enable FSDP, and set `reshard_after_forward` in the FSDP config to choose the memory and throughput tradeoff.

| `reshard_after_forward` | behavior |
|---|---|
| `true` | reshard parameters after the forward pass to save more memory |
| `false` | keep parameters gathered between forward and backward to avoid the re-all-gather, at the cost of higher peak memory |

Use `auto_wrap_policy` to enable wrapping. Without wrapping, the entire model is one FSDP unit and you lose the memory benefit of sharding.

## Configure FSDP

These fields control how FSDP2 wraps, shards, and loads the model.

- `reshard_after_forward` determines whether to reshard parameters after the forward pass. The default `true` saves more memory. Set it to `false` to keep parameters gathered between the forward and backward passes and reduce communication at the cost of higher peak memory.

- `cpu_offload` offloads parameters and gradients to CPU when they aren't in use to save GPU memory.

- `auto_wrap_policy` determines how modules are wrapped into FSDP units. Use `"TRANSFORMER_BASED_WRAP"` for transformer layers, `"SIZE_BASED_WRAP"` for modules above a parameter-count threshold, or `"NO_WRAP"` to disable auto wrapping.

- `transformer_layer_cls_to_wrap` defines the transformer layer to wrap into an FSDP unit when `auto_wrap_policy` is `"TRANSFORMER_BASED_WRAP"`. Each unit manages its own gather and scatter ops. Only the current unit's parameters are gathered during the forward pass. The previous units' parameters are released to save memory.

  Wrapping only the top-level model yields no GPU memory savings. Wrapping every individual `Linear` layer makes inter-unit communication very expensive. Leave this field empty and FSDP reads the value from the model definition.

- `min_num_params` sets the minimum number of parameters per module for size-based wrapping. It is only used when `auto_wrap_policy` is `"SIZE_BASED_WRAP"`.

- `state_dict_type` controls the checkpoint format. Use `"FULL_STATE_DICT"` for a single Transformers-compatible checkpoint or `"SHARDED_STATE_DICT"` for one checkpoint file per rank.

- `cpu_ram_efficient_loading` loads the checkpoint from disk on rank 0 only. Other GPUs initialize an empty model and receive the weights by broadcast, avoiding multiple processes loading a large model into CPU RAM.

- `activation_checkpointing` recomputes activations during the backward pass instead of storing them. Use this instead of [gradient checkpointing](./grad_checkpointing) in [`TrainingArguments`]. Setting both raises an error.

Configure FSDP training with either an [Accelerate config file](./accelerate#accelerate-config-file) or an FSDP config file passed to [fsdp_config](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.fsdp_config).

<hfoptions id="launch">
<hfoption id="Accelerate config file">

Run the [accelerate config](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-config) command and answer questions about your hardware and training setup. This creates a `default_config.yaml` file in your cache.

Run [accelerate launch](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-launch) with a [`Trainer`]-based script. The [`fsdp_config`] is unnecessary because the Accelerate config file covers the same settings.

```cli
accelerate launch train.py
```

</hfoption>
<hfoption id="FSDP config file">

```json
{
  "version": 2,
  "reshard_after_forward": true,
  "cpu_offload": false,
  "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
  "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
  "state_dict_type": "FULL_STATE_DICT",
  "cpu_ram_efficient_loading": true,
  "activation_checkpointing": true
}
```

Set `fsdp=True` and pass the FSDP config file to [`fsdp_config`].

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    fsdp=True,
    fsdp_config="path/to/fsdp.json",
)
```

</hfoption>
</hfoptions>

## Next steps

- See [DDP](./ddp) for data-parallel training when your model fits on one GPU.
- See [DeepSpeed](./deepspeed) for ZeRO optimization and NVMe offloading.
- Read the [FSDP chapter](https://nanotron-ultrascale-playbook.static.hf.space/index.html#zero-3:_adding_parameter_partitioning_(fsdp)) from The Ultra-Scale Playbook for more information about how FSDP works.
