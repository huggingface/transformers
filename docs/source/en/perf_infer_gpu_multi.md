<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Tensor parallelism

[Tensor parallelism](./perf_train_gpu_many#tensor-parallelism) slices a model layer into pieces so multiple hardware accelerators work on it simultaneously. This lets you run models that exceed a single GPU's memory capacity and achieve higher throughput. You'll need fast intra-node communication because GPUs exchange partial results at each layer.

The list below shows models with native tensor parallelism support. Open a GitHub issue or pull request to add support for a model.

<details>
<summary>Show supported models</summary>

* [Cohere](./model_doc/cohere) and [Cohere 2](./model_doc/cohere2)
* [Gemma](./model_doc/gemma) and [Gemma 2](./model_doc/gemma2)
* [GLM](./model_doc/glm)
* [Granite](./model_doc/granite)
* [Llama](./model_doc/llama)
* [Mistral](./model_doc/mistral)
* [Mixtral](./model_doc/mixtral)
* [OLMo](./model_doc/olmo) and [OLMo2](./model_doc/olmo2)
* [Phi](./model_doc/phi) and [Phi-3](./model_doc/phi3)
* [Qwen2](./model_doc/qwen2), [Qwen2Moe](./model_doc/qwen2_moe), and [Qwen2-VL](./model_doc/qwen2_5_vl)
* [Starcoder2](./model_doc/starcoder2)

</details>

This guide covers enabling tensor parallelism in Transformers and the available partitioning strategies.

## Partitioning a model

Transformers enables tensor parallelism when a model has a `tp_plan`. Choose from two partitioning methods.

- Set `tp_plan="auto"` for an automatic plan based on the model's predefined configuration.
- Define and pass a manual `tp_plan`.

<hfoptions id="tp_plan">
<hfoption id="auto plan">

```py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct" # better to visualize all the possible strategies
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct" , dtype=torch.bfloat16, tp_plan="auto")
print(model._tp_plan)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
prompt = "Can I help"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# distributed run
outputs = model(inputs)
```

Launch the inference script with [torchrun](https://pytorch.org/docs/stable/elastic/run.html). Use 4 processes per GPU.

```bash
torchrun --nproc-per-node 4 demo.py
```

</hfoption>
<hfoption id="manual plan">

Define a tensor parallel plan for each layer in `tp_plan`. Pass it to [`~PreTrainedModel.from_pretrained`]. The example below uses column and row partitioning. See the [Partitioning strategies](#partitioning-strategies) section for other supported strategies.

Manual partitioning requires a deep understanding of model architecture and strategy interactions. Poor partitioning choices create slow models that fail or produce incorrect results. The [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism) explains partitioning strategies in detail.

```py
from transformers import AutoModelForCausalLM

tp_plan = {
    "model.layers.*.self_attn.q_proj": "colwise",
    "model.layers.*.self_attn.k_proj": "colwise",
    "model.layers.*.self_attn.v_proj": "colwise",
    "model.layers.*.self_attn.o_proj": "rowwise",
    ...
}

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", dtype="auto", tp_plan=tp_plan)
print(model.tp_plan)
```

</hfoption>
</hfoptions>

## Partitioning strategies

The [`ParallelInterface`] class defines all partitioning strategies. It maps a string to the strategy implementation. You don't need to interact with this class directly since you set strategies with `tp_plan` in [`~PreTrainedModel.from_pretrained`]. It's useful for checking available strategies.

```py
class ParallelInterface(MutableMapping):
    """
    Dict-like object keeping track of allowed attention functions. You can easily add a new attention function
    with a call to `register()`. If a model needs to locally overwrite an existing attention function, say `sdpa`,
    it needs to declare a new instance of this class inside the `modeling_<model>.py`, and declare it on that instance.
    """
    _global_mapping = {
        "colwise": ColwiseParallel(),
        "rowwise": RowwiseParallel(),
        "colwise_rep": ColwiseParallel(output_layouts=Replicate()),
        "rowwise_rep": RowwiseParallel(input_layouts=Replicate()),
        "local_colwise": ColwiseParallel(use_dtensor=False),
        "local_rowwise": RowwiseParallel(use_dtensor=False),
        "local": IsolatedParallel(),
        "gather": GatherParallel(),
        "local_packed_rowwise": PackedRowwiseParallel(use_dtensor=False),
        "sequence_parallel": SequenceParallel(),
        "replicate": ReplicateParallel(),
    }
```

The table below describes each strategy.

| Strategy | Description |
|---|---|
| `ColwiseParallel` | Partitions weights and biases column-wise. |
| `RowwiseParallel` | Partitions weights and biases row-wise. Supports `nn.Embedding` modules partitioning. |
| `SequenceParallel` | Sequence parallel implementation to support `LayerNorm` and `Dropout` layers. Supports Python implementation of [RMSNorm](https://github.com/facebookresearch/llama/blob/main/llama/model.py#L34). |
| `PackedColwiseParallel` | A variant of `ColwiseParallel` that supports packed weights (for example, packing `up_proj` and `gate_proj` together). Refer to the [code](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py#L79-#L108) for more details. |
| `PackedRowwiseParallel` | A variant of `RowwiseParallel` that supports packed weights (refer to the [code](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py#L79-#L108) for more details). |
| `GatherParallel` | Gathers module outputs across devices. |
| `IsolatedParallel` | Isolates a module from other devices. Used for Experts in Mixture-of-Experts (MoE) layers. |
| `ReplicateParallel` | Replicates modules across all devices. Prevents `torch.distributed` APIs from breaking due to a partially sharded model. |

### Packed strategies

Weight packing combines multiple linear layers into a single, larger layer. The `PackedColwiseParallel` and `PackedRowwiseParallel` strategies shard packed weights correctly. Basic `ColwiseParallel` or `RowwiseParallel` strategies shard packed weights incorrectly.

The example below packs `up_proj` and `gate_proj` into a single `gate_up_proj` module and requires the `PackedRowwiseParallel` strategy to shard `gate_up_proj`.

```python
class Llama4TextExperts(nn.Module):
    ...
    self.gate_up_proj = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size, 2 * self.expert_dim))
```

Use batch matrix multiplication in the `forward` pass to compute the output of the `gate_up_proj` module.

```python
def forward(self, hidden_states):
    ...
    gate_up = torch.bmm(hidden_states, self.gate_up_proj) # Compute the output of the gate_up_proj module
    gate, up = gate_up.chunk(2, dim=-1) # Split the output into gate and up
```

> [!TIP]
> See [this comment](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py#L79-#L108) for a visual representation of why `Packed*` needs to be used.

### Local strategies

Local strategies (`local_colwise`, `local_rowwise`, `local_packed_rowwise`) don't use [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html) because it lacks support for some operations like [torch.chunk](https://docs.pytorch.org/docs/stable/generated/torch.chunk.html). Instead, local strategies use the basic [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html) and perform distributed logic manually.

<!--
Readd this when I get the exact error message
> [!TIP]
> If you are using a custom partitioning strategy, and it's not working with `... is not supported` error, try using the `local*` strategies to see if they work better.
-->

## Custom partitioning strategies

Inherit from [TensorParallelLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py) to create a custom partitioning strategy. Implement `partition_tensor`, `_prepare_input_fn` and `_prepare_output_fn`.

Register the strategy in the `ParallelInterface` mapping so the dispatching logic finds it when specified in `tp_plan`.

The example below shows how to implement `ColwiseParallel` with this workflow.

1. Inherit from `TensorParallelLayer`. In the `__init__` method, define `input_layouts` and `output_layouts` to describe how the input and output tensors should be placed on devices. The `desired_input_layouts` attribute is used to specify *how* the input should be placed on devices.

    ```python
    class ColwiseParallel(TensorParallelLayer):
        def __init__(
            self,
            *,
            input_layouts: Optional[Placement] = None, # The input layout coming from the previous layer
            output_layouts: Optional[Placement] = None, # The output layout we want to achieve
            use_local_output: bool = True, # Whether to use local output or not
            use_dtensor=True, # Whether to use DTensor or not
        ):
            self.input_layouts = (input_layouts or Replicate(),) # The input sharding coming from the previous layer
            self.output_layouts = (output_layouts or Shard(-1),) # Desired output sharding
            self.desired_input_layouts = (Replicate(),) # Desired input sharding, inputs should be replicated across GPUs
            self.use_local_output = use_local_output
            self.use_dtensor = use_dtensor
    ```

2. Implement the `partition_tensor`, `_prepare_input_fn`, and `_prepare_output_fn` methods.

    The `partition_tensor` method partitions the tensor and fills `empty_param` with the partitioned tensor. Use the utility function `get_tensor_shard` to help you get the correct shard of the original parameter for a given rank and `get_packed_weights` to help with packed weights.

    ```python
    def partition_tensor(
        self,
        param, # Full tensor of the parameter
        empty_param, # Empty tensor of the parameter, will be filled with the partitioned tensor
        param_type, # Type of the parameter, `bias` or `weight`
        param_casting_dtype, # The type to cast the parameter to
        to_contiguous, # Whether to convert the tensor to a contiguous memory layout
        rank, # The rank of the current device
        device_mesh, # The device mesh
    ) -> nn.Parameter: # Return the partitioned parameter
        ...
    ```

    The `_prepare_input_fn` and `_prepare_output_fn` methods are used in the [pre-forward](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_pre_hook.html) and [forward](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) hooks. They redistribute the inputs and outputs to the desired layout as specified in the `__init__`.

    ```python
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        ...
        # Do some custom logic, cast to DTensor etc.
        ...
        return inputs.redistribute(placements=desired_input_layouts, device_mesh=device_mesh)
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        ...
        # Do some custom logic, cast to DTensor etc.
        ...
        return outputs.redistribute(placements=output_layouts, device_mesh=device_mesh)
    ```

3. Register the strategy to [`ParallelInterface`] to enable it for use with `tp_plan`.

    ```python
    from transformers.integrations.tensor_parallel import ParallelInterface

    ParallelInterface.register_strategy("colwise_custom", ColwiseParallel)
    tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise_custom",
        ...
    }
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, tp_plan=tp_plan)
    ```

## Benchmarks

Tensor parallelism significantly speeds up inference, especially for large batch sizes or long sequences.

This chart shows the expected speedup for a single forward pass on [Llama](./model_doc/llama) with a sequence length of 512.

<div style="text-align: center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Meta-Llama-3-8B-Instruct%2C%20seqlen%20%3D%20512%2C%20python%2C%20w_%20compile.png">
</div>

## Design implementation

Transformers implements tensor parallelism in a framework-agnostic way. It relies on [DeviceMesh](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html) and [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html) from [torch.distributed](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) to provide a simple, extensible interface.

### DeviceMesh

`DeviceMesh` creates a multi-dimensional grid of devices that communicate together. Different parallelization strategies require different communication patterns. Create a `DeviceMesh` with multiple sub-meshes to handle these patterns.

```python
from torch.distributed.device_mesh import init_device_mesh

# Create a 1D mesh of 4 GPUs
device_mesh = init_device_mesh("cuda", (4,), mesh_dim_names=["tp"])
```

Most `torch.distributed` parallelization strategies apply to the mesh itself or its sub-mesh. The mesh automatically handles communication patterns.

### DTensor

`DTensor` (Distributed Tensor) handles distributed logic on top of usual tensor operations. Most model weights in tensor parallelism are stored as `DTensor`s.

The `placement` attribute tells PyTorch how to place a tensor on devices in `DeviceMesh`. It accepts the following values:

- `Shard(dimension)` shards a `DTensor` across a given dimension over the `DeviceMesh` it was constructed under. The example below shows how to shard weights over different dimensions for column-wise partitioning.

    ```python
    weight = ...
    weight = DTensor.from_local(weight, device_mesh["tp"], placements=[Shard(0)]) # Shard across the 1st (column-wise) dimension
    bias = ...
    bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Shard(-1)]) # Shard across the ONLY dimension
    ```

    This example shows how to shard weights over different dimensions for row-wise partitioning.

    ```python
    weight = ...
    weight = DTensor.from_local(weight, device_mesh["tp"], placements=[Shard(1)]) # Shard across the 2nd (row-wise) dimension
    bias = ...
    bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Replicate()]) # Replicate bias across all GPUs
    ```

- `Replicate()` replicates a `DTensor` across the `DeviceMesh`. It creates a full copy of the tensor on each device.

    ```py
    bias = ...
    bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Replicate()]) # Replicate bias across all GPUs
    ```

- `Partial()` indicates a tensor is pending a reduction operation (not typically relevant for Transformers usage).

## Resources

- The [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism) section on tensor parallelism provides more details.

- Check the [expert parallelism](./expert_parallelism) guide if you're using a mixture-of-experts (MoE) model. These models support tensor parallelism and expert parallelism.

- Read the [Tensor Parallelism (TP) in Transformers: 5 Minutes to Understand](https://huggingface.co/blog/qgallouedec/tp) blog post for a quick overview of tensor parallelism and learn how column and row parallel setups differ.