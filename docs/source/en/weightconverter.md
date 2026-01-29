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

Checkpoints are often serialized in a format that does not match what a model expects at runtime. Quantization and parallelism frequently require reshaping, splitting, or merging tensors into the expected model format instead of loading weights as-is.

Dynamic weight loading addresses this by applying scheduled, reversible operations to checkpoint tensors as they are loaded. Transformers makes this available through [`WeightConverter`], which maps one or more source keys to target keys by running a list of composable conversion operations. This approach adapts to new weight layouts, and supports loading quantized mixture-of-experts (MoEs) or enabling tensor parallelism and MoEs.

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

## Fast and efficient model loading

Loading a model is faster and uses less memory because the loader knows which tensors are required for operations and schedules their materialization lazily.

The loader scans the checkpoint *once* to discover pattern matches and collect tensors. It stores them as `Future` objects and submits them to a thread pool for asynchronous loading without blocking the GIL. A parameter starts loading as soon as a thread becomes available to it.

If your system runs other heavy processes, multiple threads may slow down loading instead of accelerating it. In this case, set the environment variable `HF_DEACTIVATE_ASYNC_LOAD=1` to load weights sequentially.

> [!NOTE]
> The default is 4 threads for asynchronous parameter loading. This provides the best trade-off across loading scenarios and hardware. The work is mostly I/O bound, but depending on accelerator hardware and the `dtype` required at loading, it can become CPU/GPU-bound if the `dtype` differs from the serialized one (this requires an additional copy operation).

When converting a weight, the converter waits for all required tensors to materialize if they haven't loaded yet. For example, the [`MergeModulelist`] operation requires all weights in `ModuleList` to be loaded before merging.

Concatenating tensors requires a temporary copy, so operations like [`MergeModulelist`] and [`Concatenate`] need 2x the memory of the underlying tensors during conversion. Once merged, only the resulting tensor stays in memory. The theoretical worst-case memory peak is the model size plus the tensors required for the largest [`MergeModulelist`] or [`Concatenate`] operation.

This worst case only occurs when all other parameters have loaded before the demanding conversion runs. Two scenarios trigger this.

1. All parameters loaded asynchronously before entering the demanding conversion (the thread pool was faster than the conversion queue).
2. The demanding conversion is the last one.

For example, a MoE model using [`MergeModulelist`] for experts on each layer, the theoretical worst-case memory peak is model size plus experts on one layer.

These worst-case scenarios are uncommon. The actual memory peak tends to stay close to the model size.

## Reusing the dynamic loading building blocks

Dynamic weight loading is not limited to full model checkpoints. The same building blocks let you load *any* set of
weights as long as you can describe how checkpoint keys map to parameters and ensure the target modules exist.

At a high level, the contract looks like this:

1. **Prepare the model namespace.** Make sure the modules/parameters you want to load are present and named the way your
   mapping will target them. For adapters, that means calling `inject_adapter_in_model(...)` so adapter modules exist
   before loading. For custom heads or extra modules, instantiate them on the model first.
2. **Describe how to map weights.** Build a conversion/renaming list (for example, in a helper like
   `_build_peft_weight_mapping(...)`) using [`WeightConverter`] or [`WeightRenaming`]. This is where you express how
   checkpoint keys should be converted, split, merged, or renamed to match your model namespace.
   You can do mostly 3 things:
    - add operations to the list of converters: these will be applied on all weights except for the ones collected in any of the `WeightConverter`. These in general should be `WeightRenaming` operations
    - add operations to the list of operations of each converter: this is what happens for `Quantization`, where we just add a quantization operation after the list of operations of any `WeightConverter`.
    - replace / map operations to your custom operations: this is what happens with `peft`. We replace the `Concatenate` operation of say `mixtral`, to be `PeftConcatenate`. This way, when the adapter checkpoint is read, the weights to be concatenated are collected, and are properly formated for `peft`
3. **Load + finalize + report.** Use the core loader to perform the conversion and populate tensors, then finalize and
   log results. Concretely, this flow is:
   - `LoadStateDictConfig(...)` + `_load_pretrained_model(...)` to load and convert.
   - `_finalize_load_state_dict(...)` to move any missing/mismatched tensors off `meta`, initialize them, and tie weights.
   - `log_state_dict_report(...)` to report missing/unexpected/mismatched keys (and conversion errors).

These APIs are expose to allow you to handle custom code, custom weight format, but also make sure you benefit from the highest and most efficient weight loading, sharding and good quality of life of `transformers` API!