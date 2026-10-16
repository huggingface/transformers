<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Tensor parallelism for inference

[Tensor parallelism](./perf_train_gpu_many#tensor-parallelism) slices a model layer into pieces so multiple hardware accelerators work on it simultaneously. This lets you run models that exceed a single GPU's memory capacity and achieve higher throughput. You'll need fast intra-node communication because GPUs exchange partial results at each layer.

A model supports tensor parallelism if its config defines `base_model_tp_plan`. Check a loaded model with the `supports_tp_plan` property.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
print(model.supports_tp_plan)
```

This guide covers enabling tensor parallelism in Transformers and the available partitioning strategies.

## Partitioning a model

Configure the number of tensor parallel devices with `tp_size` in [`DistributedConfig`].

- Set `DistributedConfig(tp_size=N)` to use the model's predefined plan.
- Define a manual `tp_plan` and pass it to [`DistributedConfig`] along with `tp_size`.

You can also set `tp_plan="auto"` to request the predefined plan explicitly. When `tp_size` is omitted and a `tp_plan` is set, `tp_size` is derived from `WORLD_SIZE` divided by the other parallel sizes. Passing `tp_plan` directly to [`~PreTrainedModel.from_pretrained`] is deprecated and will be removed in v5.18.

<hfoptions id="tp_plan">
<hfoption id="auto plan">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DistributedConfig

# model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct" # better to visualize all the possible strategies
distributed_config = DistributedConfig(tp_size=4)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    dtype=torch.bfloat16,
    distributed_config=distributed_config,
)
print(model.tp_plan)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
prompt = "Can I help"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# distributed run
outputs = model(inputs)
```

Launch the inference script with [torchrun](https://pytorch.org/docs/stable/elastic/run.html). Use one process per GPU.

```bash
torchrun --nproc-per-node 4 demo.py
```

</hfoption>
<hfoption id="manual plan">

Define a tensor parallel plan for each layer in `tp_plan` and pass it through [`DistributedConfig`]. The example below uses column and row partitioning. See the [Partitioning strategies](#partitioning-strategies) section for other supported strategies.

Manual partitioning requires a deep understanding of model architecture and strategy interactions. Poor partitioning choices create slow models that fail or produce incorrect results. The [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism) explains partitioning strategies in detail.

Keys are module or parameter names, with `*` standing in for layer indices. An unrecognized strategy name raises a `ValueError` listing the supported names, and Transformers logs a warning for plan rules that matched nothing and for parameters that no rule covered.

```py
from transformers import AutoModelForCausalLM, DistributedConfig

tp_plan = {
    "model.layers.*.self_attn.q_proj": "colwise",
    "model.layers.*.self_attn.k_proj": "colwise",
    "model.layers.*.self_attn.v_proj": "colwise",
    "model.layers.*.self_attn.o_proj": "rowwise",
    ...
}

distributed_config = DistributedConfig(tp_size=4, tp_plan=tp_plan)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="auto",
    distributed_config=distributed_config,
)
print(model.tp_plan)
```

</hfoption>
</hfoptions>

## Partitioning strategies

The `ParallelInterface` class maps each strategy name you can use in a `tp_plan` to a configured strategy instance. You don't interact with it directly to shard a model, but it's the authoritative list of available names.

```py
class ParallelInterface(GeneralInterface):
    _global_mapping = {
        "embedding_rowwise": RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
        "colwise_gather_output": ColwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
        "colwise_rep": ColwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
        "colwise": ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(-1)),
        "rowwise": RowwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate()),
        "rowwise_split_input": RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
        "rowwise_rep": RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
        "packed_colwise": PackedColwiseParallel(),
        "packed_rowwise": PackedRowwiseParallel(),
        "sequence_parallel": SequenceParallel(use_local_output=True),
        "grouped_gemm": MoEParamShard(Shard(0), shards_expert_dim=True),
        "ep_router": EpRouterParallel(),
        "megamoe_router": RouterParallelMegaMoe(),
        "moe_tp_experts": MoeExpertsParallel(),
        "megamoe_experts": MoeTensorParalellMegaMoeExperts(),
        "moe_identity_expert": MoeIdentityParallel(),
        "replicated_with_grad_allreduce": ReplicatedWithGradAllReduce(),
        "mla_kv_a_proj": MlaKvAProjParallel(),
        "all_reduce": AllReduceParallel(),
    }
```

Every strategy is a subclass of `TensorParallelLayer` in [distributed/tensor_parallel.py](https://github.com/huggingface/transformers/blob/main/src/transformers/distributed/tensor_parallel.py). The two you'll reach for most, `ColwiseParallel` and `RowwiseParallel`, take `input_layouts` and `output_layouts` placements, which is how one class covers several plan names. `colwise` leaves its output sharded on the last dim for a following `rowwise` layer to consume, while `colwise_gather_output` all-gathers it back to a full tensor.

The table below describes each strategy.

| Strategy | Description |
|---|---|
| `ColwiseParallel` | Shards a weight on its output-feature dim (`Shard(0)` for a 2D `nn.Linear` weight, `Shard(1)` for `nn.Embedding`) and shards a 1D bias. Redistributes the input to `Replicate()`, then places the output at `output_layouts`. |
| `RowwiseParallel` | Shards a weight on its input-feature dim (`Shard(-1)`, or `Shard(0)` for `nn.Embedding`) and replicates the bias, which is added after the reduction. Each rank's forward produces a `Partial()` output holding its share of the sum, and `transform_output_post_forward` redistributes that to `output_layouts`. Reducing to `Replicate()` issues an all-reduce, and reducing to `Shard(1)` issues a reduce-scatter. |
| `PackedColwiseParallel` | A variant of `ColwiseParallel` for fused weights, for example `up_proj` and `gate_proj` packed into `gate_up_proj`. Use `split_factor` when a weight packs a number of layers other than two. |
| `PackedRowwiseParallel` | The row-wise counterpart, for weights packed along the final dim. Replicates 1D parameters. |
| `SequenceParallel` | Replicates the module's parameters and shards its input on `sequence_dim` (defaults to `1`). Used for norms that operate per-token, such as `LayerNorm` and `RMSNorm`. |
| `ReplicatedWithGradAllReduce` | Replicates a parameter but all-reduces its gradient. Needed for norms that sit between a column-wise and a row-wise layer and normalize along a sharded axis, where each rank only sees its own heads. |
| `AllReduceParallel` | All-reduces a module's `Partial()` forward output to `Replicate()`. Use it as a sync point for a module whose compute ends in a partial sum. |
| `MlaKvAProjParallel` | Splits the `kv_a_proj_with_mqa` output of DeepSeek-V2 style MLA attention and all-reduces the gradient of the RoPE half, which bypasses `kv_b_proj` and would otherwise keep a partial gradient. Requires `qk_rope_head_dim` in the model config. |
| `MoEParamShard` | Shards MoE expert weights on a given placement. Backs the `grouped_gemm` name, where `shards_expert_dim=True` also rewrites `module.num_experts` to the per-rank expert count. |
| `EpRouterParallel` | Masks router scores for non-local experts and remaps global expert IDs to local ones, so each rank runs only the experts it owns. Requires `num_experts` to be divisible by the mesh size. |
| `RouterParallelMegaMoe` | Router variant for DeepGEMM Mega MoE, which dispatches experts inside the kernel which wants the router output untouched. |
| `MoeExpertsParallel` | Tensor parallel MoE experts. All-reduces the expert output forward and adds the backward all-reduces for hidden states and routing weights. |
| `MoeTensorParalellMegaMoeExperts` | Inference-only experts layer for DeepGEMM Mega MoE. Skips the gradient syncs and passes the process group into the module so the kernel can set up its shared buffers on the first forward. |
| `MoeIdentityParallel` | Pre-divides the input of a zero or identity expert by the mesh size, cancelling the all-reduce that `moe_tp_experts` applies downstream. |

### Packed strategies

Weight packing combines multiple linear layers into a single, larger layer. The `PackedColwiseParallel` and `PackedRowwiseParallel` strategies shard packed weights correctly. Basic `ColwiseParallel` or `RowwiseParallel` strategies shard packed weights incorrectly.

The example below packs `up_proj` and `gate_proj` into a single `gate_up_proj` module and requires the `packed_rowwise` strategy to shard `gate_up_proj`.

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

A plain `Shard` splits that dimension into contiguous blocks, so a rank would receive the tail of `gate` and the head of `up` instead of a slice of each. The packed strategies use `_StridedShard` with `split_factor` (`2` by default) to interleave the split, giving every rank a matching slice of both halves so `chunk` still lines up after sharding.

## Custom partitioning strategies

Inherit from `TensorParallelLayer` in [distributed/tensor_parallel.py](https://github.com/huggingface/transformers/blob/main/src/transformers/distributed/tensor_parallel.py) to create a custom partitioning strategy. Override only the hooks your strategy needs, since every one has a no-op default.

| Hook | Purpose |
|---|---|
| `validate_param` | Reject a parameter this strategy can't shard, before any weights load. |
| `shard_param` | Replace one parameter with a `DTensor` placeholder so the loader knows which shard belongs to this rank. |
| `transform_inputs_pre_forward` | Redistribute the module's inputs to the layout its forward expects. |
| `context_around_forward` | Wrap the forward in a context manager, for example to expose local tensors to a kernel. |
| `transform_output_post_forward` | Redistribute or reduce the module's output. |
| `should_use_local_tensors` | Report that this module's forward needs plain tensors rather than `DTensor`s. |
| `install_forward` | Replace `module.forward` outright. Override this only when the hooks above aren't enough, as `ReplicatedWithGradAllReduce` does to register a backward hook. |

The example below walks through a trimmed version of `ColwiseParallel`.

1. Inherit from `TensorParallelLayer` and store the placements the strategy works with. The base class defines no `__init__`, so there's nothing to call `super()` on.

    ```python
    class ColwiseParallel(TensorParallelLayer):
        def __init__(self, *, input_layouts=None, output_layouts=None, use_local_output: bool = True):
            self.input_layouts = input_layouts or Replicate()
            self.output_layouts = output_layouts if output_layouts is not None else Shard(-1)
            self.use_local_output = use_local_output
    ```

2. Implement `shard_param` to wrap one parameter as a `DTensor` placeholder. It runs on meta tensors, so `distribute_tensor` only builds metadata and moves no data. Pass `src_data_rank=None` because there's no full tensor to scatter from yet.

    ```python
    def shard_param(self, module, param, mesh):
        meta = module._parameters.get(param)
        if meta is None:
            return
        # Output features live on dim 0 for a 2D weight and dim -1 for a 1D bias
        placement = Shard(1) if isinstance(module, torch.nn.Embedding) else Shard(meta.ndim - 2)
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [placement], src_data_rank=None),
            requires_grad=meta.requires_grad,
        )
    ```

3. Implement the input and output transforms. `install_forward` calls them around the module's original forward, so they only need to move tensors between layouts. Column-wise partitioning expects a replicated input and produces an output sharded on the last dim.

    ```python
    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        x = args[0]
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, mesh, [self.input_layouts], run_check=False)
        if x.placements != (Replicate(),):
            x = x.redistribute(placements=[Replicate()])
        return (x,) + args[1:], kwargs

    def transform_output_post_forward(self, module, output, mesh):
        if not isinstance(output, DTensor):
            output = DTensor.from_local(output, mesh, [Shard(-1)], run_check=False)
        if output.placements != (self.output_layouts,):
            output = output.redistribute(placements=[self.output_layouts])
        return output.to_local() if self.use_local_output else output
    ```

    The shipped `ColwiseParallel` adds fast paths on top of this that skip the `DTensor` round trip for plain `nn.Linear` inference and for quantized modules that need local tensors. Read the source before copying it if your strategy needs the same treatment.

4. Register the strategy so a `tp_plan` can name it. Registration takes a strategy instance, so the placements you pass to `__init__` are stored with the name. Note the `()` below.

    ```python
    import torch

    from transformers import AutoModelForCausalLM, DistributedConfig
    from transformers.distributed.tensor_parallel import ParallelInterface

    ParallelInterface.register("colwise_custom", ColwiseParallel())
    tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise_custom",
        ...
    }
    distributed_config = DistributedConfig(tp_size=4, tp_plan=tp_plan)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        dtype=torch.bfloat16,
        distributed_config=distributed_config,
    )
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
import torch
from torch.distributed.device_mesh import init_device_mesh

# Create a 1D mesh of 4 accelerators
device_type = torch.accelerator.current_accelerator().type
device_mesh = init_device_mesh(device_type, (4,), mesh_dim_names=["tp"])
```

Most `torch.distributed` parallelization strategies apply to the mesh itself or its sub-mesh. The mesh automatically handles communication patterns.

### DTensor

`DTensor` (Distributed Tensor) handles distributed logic on top of usual tensor operations. Model weights under tensor parallelism are stored as `DTensor`s, which is what lets a strategy describe communication as a change of layout instead of an explicit collective.

The `placements` attribute tells PyTorch how a tensor is laid out across the devices in a `DeviceMesh`. It accepts the following values:

- `Shard(dimension)` splits a `DTensor` across a given dimension over the `DeviceMesh` it was constructed under. Column-wise partitioning shards the output-feature dim of the weight and the only dim of the bias.

    ```python
    weight = DTensor.from_local(weight, device_mesh["tp"], placements=[Shard(0)]) # Shard the output features
    bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Shard(-1)]) # Shard the ONLY dimension
    ```

    Row-wise partitioning shards the input-feature dim instead and replicates the bias, because the bias is added once after the reduction rather than on every rank.

    ```python
    weight = DTensor.from_local(weight, device_mesh["tp"], placements=[Shard(-1)]) # Shard the input features
    bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Replicate()]) # Replicate bias across all GPUs
    ```

- `Replicate()` replicates a `DTensor` across the `DeviceMesh`, creating a full copy of the tensor on each device.

- `Partial()` marks a tensor as pending a reduction. A row-wise layer's forward output is `Partial()`, and redistributing it to `Replicate()` is what issues the all-reduce.

## Resources

- The [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism) section on tensor parallelism provides more details.

- Check the [expert parallelism](./expert_parallelism) guide if you're using a mixture-of-experts (MoE) model. These models support tensor parallelism and expert parallelism.

- Read the [Tensor Parallelism (TP) in Transformers: 5 Minutes to Understand](https://huggingface.co/blog/qgallouedec/tp) blog post for a quick overview of tensor parallelism and learn how column and row parallel setups differ.

- See the [Tensor parallelism](./tensor_parallelism) training guide to learn how to use it in a training setting.
