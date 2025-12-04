<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Dynamic weight loading

Transformers provides a set of composable operations in [`WeightConverter`] for defining how to map checkpoint tensors to model tensors. These operations allow you to reshape checkpoint tensors into any expected model format during loading, like fusing the query (Q), key (K), and value (V) layers for example.

This allows Transformers to adapt to new model or weight formats instead of manually adding code for them. It is especially useful for loading mixture-of-experts (MoE) models with different expert representations, handling quantized checkpoints with special tensor layouts, supporting tensor parallelism during loading, or a combination of all of these.

This guide demonstrates how to use the [`WeightConverter`] to convert tensors. Your [`WeightConverter`] should be added inside [_build_checkpoint_conversion_mapping()](https://github.com/huggingface/transformers/blob/4c9fde2a2a3aece0bcf1be93f696e88297da9397/src/transformers/conversion_mapping.py#L34) in the [conversion_mapping.py](https://github.com/huggingface/transformers/blob/main/src/transformers/conversion_mapping.py) file.

## Conversion operations

The [`WeightConverter`] class has several operations that are executed when [`~PreTrainedModel.from_pretrained`] is called for transforming checkpoint source tensors into model target tensors.

Operations are fully reversible. Saving reverses the conversions and returns the original checkpoint so you can easily work across different frameworks.

### Chunk

The [`Chunk`] operation is used to split a tensor. For example, if a model expects Q, K, and V as three separate tensors instead of a single tensor.

```py
WeightConverter(
    "self_attn.qkv_proj",
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    operations=[Chunk(dim=0)],
)
```

### Concatenate

The [`Concatenate`] operation allows you to fuse separate tensors into a single tensor. For example, if a model expects Q, K, and V as a single tensor instead of separate tensors.

```py
WeightConverter(
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "self_attn.qkv_proj",
    operations=[Concatenate(dim=0)],
)
```

### MergeModulelist

[`MergeModulelist`] merges a list of tensors into a single tensor. For example, you can compose [`MergeModulelist`] with [`Concatenate`] to stack the experts in a MoE and pack them into one tensor.

```py
WeightConverter(
    ["block_sparse_moe.experts.*.w1.weight", "block_sparse_moe.experts.*.w3.weight",],
    "mlp.experts.gate_up_proj",
    operations=[
        MergeModulelist(dim=0),
        Concatenate(dim=1),
    ],
)
```

### SplitModulelist

[`SplitModulelist`] splits a tensor back into a list of tensors. For example, you can split a stack of experts back into individual experts.

```py
WeightConverter(
    "mlp.experts.down_proj",
    "block_sparse_moe.experts.*.w2.weight",
    operations=[SplitModulelist(dim=0)],
)
```

### PermuteForRope

[`PermuteForRope`] converts weights from the interleaved format to use the sin/cos format. For example, you can compose [`Chunk`] with [`PermuteForRope`] to split a fused QKV tensor and apply the sin/cos RoPE permutation to Q and K.

```py
WeightConverter(
    ["model.layers.*.self_attn.qkv_proj.weight"],
    ["model.layers.*.self_attn.q_proj.weight", "model.layers.*.self_attn.k_proj.weight", "model.layers.*.self_attn.v_proj.weight",],
    operations=[
        Chunk(dim=0),
        PermuteForRope(),
    ],
)
```

## Fast model loading

Loading a model is faster and uses less memory because the loader knows which tensors are required for operations and schedules their materialization lazily.

The loader scans the checkpoint *once* to discover pattern matches and collect tensors. Tensors are collected as `Future` objects and not loaded into memory immediately. They're kept as lazy references until needed to defer memory allocation. Tensor loading is scheduled asynchronously without blocking the GIL.

Tensors are materialized once all `Future` objects are collected. Operations are batched together and applied to return the transformed tensors.