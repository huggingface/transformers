<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Tensor parallelism in transformers

[Tensor parallelism](./perf_train_gpu_many#tensor-parallelism) shards a model onto multiple GPUs and parallelizes computations such as matrix multiplication. It enables fitting larger model sizes into memory and is faster because each GPU can process a tensor slice.
This document assumes that you are already familiar with the basics of tensor parallelism. If you are not, please refer to the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism) section on tensor parallelism.

> [!TIP]
> Tensor parallelism is very communication intensive, therefore it is reccomended to use it on a single machine with multiple GPUs, utilizing fast intra-node communication. For multi-node training, methods as pipeline or data parallelism are more efficient (depending on your use case).

Tensor parallelism requires slight changes to the model parameters, therefore in transformers, we support some of the popular models out of the box.

> [!TIP]
> Expand the list below to see which models support tensor parallelism. Open a GitHub issue or pull request to add support for a model not currently below.

<details>
<summary>Supported models</summary>

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

## Using 🤗 transformers

Transformers provides a simple interface to use for tensor parallelism. We provide multiple classes implementing different partitioning
strategies and a simple entrypoint to parallelize `nn.Module` instance. You won't have to interact with this interface directly, everything is done in `PretrainedModel.from_pretrained` method for you. This section will first talk about the partitioning strategies
we support, then the user interface you will be interacting with, and finally it will teach you how to extend it with your own partitioning
strategies.

### Partitioning strategies

In transformers, partitioning strategies reside in a class `ParallelInterface` which works like a mapping from string to the strategy implementation.


```python
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

We support the following strategies:

- `ColwiseParallel` - A simple column-wise partitioning, being able to handle both weights and biases, does exactly what we've discussed before.
- `RowwiseParallel` - Again, row-wise partitioning as dicussed before, supports weights and biases, on top of that it also supports `nn.Embedding` modules.
- `SequenceParallel` - Sequence parallel implementation, for support of `LayerNorm` and `Dropout` layers. Also supports Python implementation of `RMSNorm` (see [this](https://github.com/facebookresearch/llama/blob/main/llama/model.py#L34))
- `PackedColwiseParallel` - A variant of column-wise partitioning, however it works on packed weights (i.e. `up_proj` and `gate_proj` being packed together). For more details, see [this comment](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py#L79-#L108)
- `PackedRowwiseParallel` - A variant of row-wise partitioning, works on packed weights, for more details check the comment linked above.
- `GatherParallel` - A very simple class, that only makes the outputs of the module to be gathered across devices.
- `IsolatedParallel` - This is a special case, where we want to *isolate* the module from the rest of the devices (world). This is used for Experts in MoE layers, basically creating Expert parallelism of sorts.
- `ReplicateParallel` - Many `torch.distributed` APIs break if model is partially sharded, so this class is used to replicate the module across all devices.

### Sharding a model

We provide two ways to shard a model, first one is to use `auto` tensor parallelism plan, which will automatically shard the model based on our predefined configuration. This requires the model to have predefined tensor parallel plan in transformers.

```python
from transformers import AutoModelForCausalLM

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct" # better for smaller number of GPUs
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct" # better to visualize all the possible strategies

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, tp_plan="auto")

print(model._tp_plan)
```

> [!TIP]
> For a list of models that support tensor parallelism, see the [Supported models](#supported-models) section above.

The second way is to manually specify your own partitioning plan.

```python
from transformers import AutoModelForCausalLM

tp_plan = {
    "model.layers.*.self_attn.q_proj": "colwise",
    "model.layers.*.self_attn.k_proj": "colwise",
    "model.layers.*.self_attn.v_proj": "colwise",
    "model.layers.*.self_attn.o_proj": "rowwise",
    ...
}

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, tp_plan=tp_plan)

print(model._tp_plan)
```

You might have noticed that there are some special cases in the `ParallelInterface` mapping, let's now talk about them. This will help you understand their purpose and help with extending to other strategies.

### PackedRowwiseParallel
This class is a special case of `RowwiseParallel`, it's used to shard packed weights. Weight packing is a common technique used in models. It's a technique where we pack multiple linear layers into a single, bigger one.

For example in `Llama4` model, we pack `up_proj` and `gate_proj` into a single `gate_up_proj` module.
```python
class Llama4TextExperts(nn.Module):
    ...
    self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
```

Then in forward, we can use batch matrix multiplication to compute the output of the `gate_up_proj` module.

```python
def forward(self, hidden_states):
    ...
    gate_up = torch.bmm(hidden_states, self.gate_up_proj) # Compute the output of the gate_up_proj module
    gate, up = gate_up.chunk(2, dim=-1) # Split the output into gate and up
```

In this case, we need to use the `PackedRowwiseParallel` strategy to shard the `gate_up_proj` module, as using a simple `RowwiseParallel` will shard the layers wrongly.

> [!TIP]
> If this is a bit difficult to wrap your head around, check out [this comment](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py#L79-#L108) for an amazing visual representation of why `Packed*` needs to be used.


### `local*` strategies

You could have noticed that there are `local*` strategies, which use the same layers as `*` strategy, but don't use `DTensor` at all.
This is because `DTensor` is not supported for some of the operations: such as `torch.chunk`. Therefore, sometimes we need to use the `local*` strategies, which use vanilla `torch.Tensor` and do some of the distributed logic manually.

<!---
Readd this when I get the exact error message
> [!TIP]
> If you are using a custom partitioning strategy, and it's not working with `... is not supported` error, try using the `local*` strategies to see if they work better.
-->

> [!WARNING]
> Manually specifying your own partitiong plan requires a good understanding of the model architecture and how the partitioning strategies interact together. If you are not sure about this, the resulting model can be very slow, even failing or incorrect. Again, refer to the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism) which can teach you everything required.

### Extending the interface with your own partitioning strategies

This is a very advanced topic, which requires a good understanding of distributed collectives and the model architecture.
Your custom partitioning strategy should inherit from `TensorParallelLayer` defined in [integrations/tensor_parallel.py](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py) and implement: `partition_tensor`, `_prepare_input_fn` and `_prepare_output_fn`. Then it should be registered in the `ParallelInterface` mapping, so our dispatching logic can find it when specified in the `tp_plan`.

Let's go through this workflow step by step, on an already existing example: `ColwiseParallel`.

1. Inherit from `TensorParallelLayer` and initialization

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

In the `__init__` method, we define these attributes, where `input_layouts` and `output_layouts` describing, how the input and output tensors should be placed on the devices. `desired_input_layouts` is used to specify, how the input *SHOULD* be placed on the devices.

2a. Implement `partition_tensor` method

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

This method is used to partition the tensor, and fill the `empty_param` with the partitioned tensor.
We provide some utility functions to help you with this, such as `get_tensor_shard` which will get you the correct shard of the original parameter for this rank or `get_packed_weights` to help with packed weights.

2b. Implement `_prepare_input_fn` and `_prepare_output_fn` methods

These methods are used as [`pre-forward`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_pre_hook.html) and [`forward`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) hooks respectively. Their purpose is to re-distribute the inputs and outputs to the desired layout, passed in the `__init__` method.

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

3. Register the strategy
Congratulations! You've implemented your own partitioning strategy. Now, to use it with your own `tp_plan`, you need to register it in the `ParallelInterface` mapping.

```python
from transformers.integrations.tensor_parallel import ParallelInterface

ParallelInterface.register_strategy("colwise_custom", ColwiseParallel)
```

And now you can use it in your `tp_plan` as such:

```python
tp_plan = {
    "model.layers.*.self_attn.q_proj": "colwise_custom",
    ...
}

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, tp_plan=tp_plan)
```


## Full example

Let's go through a full example of inference with tensor parallelism.
```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# enable tensor parallelism
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    tp_plan="auto",
)

# prepare input tokens
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
prompt = "Can I help"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# distributed run
outputs = model(inputs)
```

Launch the inference script above on [torchrun](https://pytorch.org/docs/stable/elastic/run.html) with 4 processes per GPU.

```bash
torchrun --nproc-per-node 4 demo.py
```

You can benefit from considerable speed ups for inference, especially for inputs with large batch size or long sequences.

For a single forward pass on [Llama](./model_doc/llama) with a sequence length of 512 and various batch sizes, you can expect the following speed ups.

<div style="text-align: center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Meta-Llama-3-8B-Instruct%2C%20seqlen%20%3D%20512%2C%20python%2C%20w_%20compile.png">
</div>

## Tensor parallelism in-depth
Our implementation of tensor parallelism is framework-agnostic in design, but the specific implementations we've developed rely on the torch.distributed package. We heavily utilize abstractions such as `DeviceMesh` or `DTensor` to provide a simple and extensible interface to the user.

### DeviceMesh
Imagine `DeviceMesh` as a multi-dimensional grid of devices that communicate together. Different parallelization strategies require different types of communication patterns, therefore we can create a `DeviceMesh` with multiple submeshes:
```python
from torch.distributed.device_mesh import init_device_mesh

# Create a 1D mesh of 4 GPUs
device_mesh = init_device_mesh("cuda", (4,), mesh_dim_names=["tp"])
```
Then, most of the `torch.distributed` defined parallelization strategies can be applied to a mesh itself, or its submesh, automatically handling the communication patterns.

### DTensor

Abbreviation for Distributed Tensor, `DTensor` is a tensor subclass that handles the distributed logic on-top of the usual tensor operations. Most of the model weights in case of tensor parallelism are stored as `DTensor`s (with some exceptions, more on that later).
The most important part of DTensor, that is crucial to understand, is the `placement` attribute. It's an attribute that tells PyTorch how is the tensor placed on the devices of the `DeviceMesh`.

It can have the following values:

- `Shard(dimension)` - Annotates that this `DTensor` is sharded across a given dimension, over the `DeviceMesh` it was constructed under. For example, if we would like to shard weights for column-wise partitioning, we would do:
```python
weight = ...
weight = DTensor.from_local(weight, device_mesh["tp"], placements=[Shard(0)]) # Shard across the 1st (column-wise) dimension
bias = ...
bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Shard(-1)]) # Shard across the ONLY dimension
```

To give another example, for row-wise partitioning, we would do:
```python
weight = ...
weight = DTensor.from_local(weight, device_mesh["tp"], placements=[Shard(1)]) # Shard across the 2nd (row-wise) dimension
bias = ...
bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Replicate()]) # Replicate bias across all GPUs
```

- `Replicate()` - Annotates that this `DTensor` is replicated across the `DeviceMesh`. Very straight-forward, only creates a full copy of the tensor on each device.
- `Partial()` - This placement is mostly of no interest to us, it's used to annotate that this tensor is pending a reduction operation.
