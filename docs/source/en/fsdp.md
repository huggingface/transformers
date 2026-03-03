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

# FSDP

[Fully Sharded Data Parallel (FSDP)](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) shards the model, gradients, and optimizer states across GPUs. Before computation, each GPU gathers a complete set of parameters from all shards, then frees them afterward. Sharding lets you train models larger than a single GPU's memory, at the cost of more communication than [DDP](./ddp). Use FSDP when your model or optimizer states don't fit on a single GPU.

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

Pass one of the sharding strategies below to [fsdp](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.fsdp).

| strategy | description |
|---|---|
| `full_shard` | shard parameters, gradients, and optimizer states |
| `shard_grad_op` | shard gradients and optimizer states |
| `no_shard` | DDP |
| `hybrid_shard` | full shard within a node, replicate across nodes |
| `hybrid_shard_zero2` | shard gradients and optimizer states within a node, replicate across nodes |
| `offload` | CPU offload (combine with `full_shard` or `shard_grad_op`) |

Always combine a sharding strategy with `auto_wrap` to enable the auto-wrapping policy like `fsdp="full_shard auto_wrap"`. Without `auto_wrap`, the entire model is one FSDP unit and you lose the memory benefit of sharding.

## Configure FSDP

These fields control how FSDP wraps and loads the model.

- `transformer_layer_cls_to_wrap` defines the transformer layer to wrap into an FSDP unit. Each unit manages its own gather and scatter ops. Only the current unit's parameters are gathered during the forward pass. The previous units' parameters are released to save memory.

  Wrapping only the top-level model yields no GPU memory savings. Wrapping every individual `Linear` layer makes inter-unit communication very expensive. Leave this field empty and FSDP reads the value from the model definition.

- `backward_prefetch` determines when to start the all-gather for the next FSDP unit during the backward pass. The default `"backward_pre"` prefetches before the current unit's backward to overlap communication with compute.

- `forward_prefetch` prefetches the next FSDP unit during the forward pass, improving throughput at the cost of higher peak memory.

- `limit_all_gathers` adds a CPU synchronization point to prevent too many simultaneous all-gathers, reducing peak memory at the cost of slightly lower throughput.

- `cpu_ram_efficient_loading` loads the checkpoint from disk on rank 0 only. Other GPUs initialize an empty model and receive the weights by broadcast, avoiding multiple processes loading a large model into CPU RAM. Use with `sync_module_states` to broadcast the parameters from rank 0 to other processes.

- `sync_module_states` broadcasts rank 0's parameters to all other ranks after wrapping. Required when `cpu_ram_efficient_loading` is enabled. Without it, non-rank-0 processes train on uninitialized weights.

- `use_orig_params` preserves the original parameter structure, allowing non-uniform `requires_grad` within an FSDP unit. Required for parameter-efficient fine-tuning (PEFT/LoRA) where only adapter layers are trainable.

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

Pass an FSDP config file to [`fsdp_config`]. All fields are optional except for the sharding strategy in `fsdp`.

```json
{
  "version": 1,
  "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
  "backward_prefetch": "backward_pre",
  "forward_prefetch": false,
  "limit_all_gathers": true,
  "use_orig_params": true,
  "sync_module_states": true,
  "cpu_ram_efficient_loading": true,
  "activation_checkpointing": true
}
```

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    fsdp="full_shard auto_wrap",
    fsdp_config="path/to/fsdp.json",
)
```

</hfoption>
</hfoptions>

## Next steps

- See [DDP](./ddp) for data-parallel training when your model fits on one GPU.
- See [DeepSpeed](./deepspeed) for ZeRO optimization and NVMe offloading.
- Read the [FSDP chapter](https://nanotron-ultrascale-playbook.static.hf.space/index.html#zero-3:_adding_parameter_partitioning_(fsdp)) from The Ultra-Scale Playbook for more information about how FSDP works.
