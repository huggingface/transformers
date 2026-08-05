<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
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
class ParallelInterface(GeneralInterface):
    _global_mapping = {
        "embedding_rowwise": EmbeddingParallel(embedding_dim_sharding=0),
        "embedding_colwise": EmbeddingParallel(embedding_dim_sharding=1),
        "colwise_gather_output": ColwiseParallel(gather_output=True),
        "colwise": ColwiseParallel(),
        "rowwise": RowwiseParallel(),
        "rowwise_split_input": RowwiseParallel(split_input=True),
        "packed_colwise": PackedColwiseParallel(),
        "packed_rowwise": PackedRowwiseParallel(),
        "sequence_parallel": SequenceParallel(),
        "grouped_gemm": GroupedGemmParallel(),
        "ep_router": RouterParallel(),
        "megamoe_router": RouterParallelMegaMoe(),
        "moe_tp_experts": MoeTensorParalellExperts(),
        "megamoe_experts": MoeTensorParalellMegaMoeExperts(),
        "moe_identity_expert": MoeIdentityExpertParallel(),
        "replicated_with_grad_allreduce": ReplicatedWithGradAllReduce(),
        "mla_kv_a_proj": MlaKvAProjParallel(),
        "all_reduce": AllReduceParallel(),
    }
```

The table below describes each strategy.

| Strategy | Description |
|---|---|
| `ColwiseParallel` | Shards weights on dim `-2` (output features). Expects replicated input and produces output sharded on the last dim. Set `gather_output=True` (the `colwise_gather_output` key) to all-gather the full output. |
| `RowwiseParallel` | Shards weights on dim `-1` (input features). Expects input sharded by a preceding column-wise layer and all-reduces the partial output. Set `split_input=True` (the `rowwise_split_input` key) when the input arrives replicated instead. |
| `PackedColwiseParallel` | A variant of `ColwiseParallel` for fused weights (for example, `up_proj` and `gate_proj` packed into `gate_up_proj`). |
| `PackedRowwiseParallel` | A variant of `RowwiseParallel` for fused weights. |
| `EmbeddingParallel` | Shards an embedding table and handles masked lookups. `embedding_dim_sharding=0` shards the vocabulary dim, `1` shards the hidden dim. |
| `SequenceParallel` | Shards inputs and outputs on the sequence dimension while replicating weights. |
| `ReplicatedWithGradAllReduce` | Replicates a parameter but all-reduces its gradient. Used for parameters like `q_norm` and `k_norm` that sit between column-wise and row-wise layers. |
| `AllReduceParallel` | All-reduces a module's forward output across the mesh. Use it as a sync point for a module whose compute ends in a partial sum. |
| `MlaKvAProjParallel` | Handles the split `kv_a_proj_with_mqa` output in DeepSeek-V2 style MLA attention. |
| `GroupedGemmParallel` | Applies expert parallelism to MoE experts by loading the correct experts on each device. |
| `RouterParallel` | Reshapes router scores so experts can run with expert parallelism. |
| `RouterParallelMegaMoe` | Router variant for DeepGEMM Mega MoE, which handles expert-parallel dispatch inside the kernel. |
| `MoeTensorParalellExperts` | Tensor parallel MoE experts, including the gradient syncs for hidden states, routing weights, and partial expert outputs. |
| `MoeTensorParalellMegaMoeExperts` | Inference-only experts layer for DeepGEMM Mega MoE. |
| `MoeIdentityExpertParallel` | Compensates for the parent all-reduce on zero/identity experts, which produce the same output on every rank. |

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

## Custom partitioning strategies

Inherit from [TensorParallelLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py) to create a custom partitioning strategy. Implement `shard_tensor`, `_prepare_input_fn` and `_prepare_output_fn`.

Register the strategy in the `ParallelInterface` mapping so the dispatching logic finds it when specified in `tp_plan`.

The example below shows how to implement `ColwiseParallel` with this workflow.

1. Inherit from `TensorParallelLayer`. Use the `__init__` method to store any options that change how the layer shards weights or handles its input and output. `ColwiseParallel` only needs `gather_output`, which controls whether the sharded output is all-gathered back into a full tensor.

    ```python
    class ColwiseParallel(TensorParallelLayer):
        def __init__(self, gather_output: bool = False, **kwargs):
            super().__init__(**kwargs) # Sets device_mesh, rank, and empty_param
            self.gather_output = gather_output
    ```

2. Implement the `shard_tensor`, `_prepare_input_fn`, and `_prepare_output_fn` methods.

    The `shard_tensor` method returns this rank's shard of a full parameter. Use the utility function `get_tensor_shard` to get the correct shard of the original parameter for a given rank, and `get_packed_weights` for packed weights. The mesh, rank, and target parameter are available as `self.device_mesh`, `self.rank`, and `self.empty_param`.

    ```python
    def shard_tensor(
        self,
        param, # Full tensor of the parameter
        tensor_idx=None, # Index of the tensor when a parameter is loaded from several tensors
        device=None, # The device to place the shard on
        dtype=None, # The dtype to cast the shard to
    ) -> torch.Tensor: # Return this rank's shard
        # Shard dim -2 for weights, dim -1 for 1D tensors such as a bias
        dim = -1 if param.dim() == 1 else -2
        parameter = get_tensor_shard(param, self.empty_param, self.device_mesh, self.rank, dim)
        return parameter.to(device=device, dtype=dtype)
    ```

    The `_prepare_input_fn` and `_prepare_output_fn` methods are used in the [pre-forward](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_pre_hook.html) and [forward](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) hooks. They apply the communication the strategy needs around the module's compute.

    ```python
    def _prepare_input_fn(self, mod, inputs, device_mesh):
        input_tensor = inputs[0] if inputs else inputs
        # Column-wise expects a replicated input, so only the backward pass needs a reduction
        return all_reduce_backward(input_tensor, device_mesh)

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        if self.gather_output:
            return all_gather(outputs, device_mesh)
        return outputs
    ```

    Optionally implement `get_expected_sharded_shape` and `update_module_attributes` so the loader knows the sharded shape and the module reports sharded attributes such as `out_features`.

3. Register the strategy on [`ParallelInterface`] to enable it for use with `tp_plan`. Pass an instance of the strategy, configured however you need it.

    ```python
    from transformers.integrations.tensor_parallel import ParallelInterface

    ParallelInterface.register("colwise_custom", ColwiseParallel())
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

- See the [Tensor parallelism](./tensor_parallelism) training guide to learn how to use it in a training setting.
