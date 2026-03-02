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

Checkpoints are often serialized in a format that does not match what a model expects at runtime. Common scenarios include:

1. **Fused weights**: Checkpoints store separate `gate_proj` and `up_proj` weights, but the model uses a fused `gate_up_proj` for efficiency.
2. **MoE expert consolidation**: Individual expert weights (`experts.0.weight`, `experts.1.weight`, ...) need to be stacked into a single 3D tensor.
3. **Legacy naming**: Old checkpoints use different naming conventions (e.g., `LayerNorm.gamma` vs `LayerNorm.weight`).
4. **Quantization**: Weights may be stored in quantized formats that need deserialization.

Dynamic weight loading addresses this by applying scheduled, reversible operations to checkpoint tensors as they are loaded. Transformers makes this available through [`WeightConverter`], which maps one or more source keys to target keys by running a list of composable conversion operations. This approach adapts to new weight layouts, and supports loading quantized mixture-of-experts (MoEs) or enabling tensor parallelism and MoEs.

This guide demonstrates how to use the [`WeightConverter`] to convert tensors. Your [`WeightConverter`] should be added inside [_build_checkpoint_conversion_mapping()](https://github.com/huggingface/transformers/blob/4c9fde2a2a3aece0bcf1be93f696e88297da9397/src/transformers/conversion_mapping.py#L34) in the [conversion_mapping.py](https://github.com/huggingface/transformers/blob/main/src/transformers/conversion_mapping.py) file.

## Full loading pipeline

All models go through the dynamic weight loading system. Conversion mapping is an **optional step within that system** that only activates when the model has entries in `_MODEL_TO_CONVERSION_PATTERN`.

```
Checkpoint File → from_pretrained() → convert_and_load_state_dict_in_model()
                                              ↓
                         ┌───────────────────────────────────────────────────────────┐
                         │  For each weight in checkpoint:                           │
                         │  1. Match renamed/processed source key to model parameter │
                         │  2. Shard the weight and send to device (async)           │
                         │  3. Collect tensors with the same source_pattern together │
                         │     (e.g. MoE experts, gate_up_proj)                     │
                         │  4. Apply dequantization/deserialization (if pre-quant)   │
                         │  5. Apply conversion (if defined)                        │
                         │  6. Apply quantization (if enabled and step 4 not used)  │
                         │  7. Set parameter on model                               │
                         └───────────────────────────────────────────────────────────┘
```

| Step | When it activates |
|------|-------------------|
| Dynamic loading | Always, for all models |
| Conversion mapping | Only when `model_type` is in `_MODEL_TO_CONVERSION_PATTERN` |
| TP sharding | Only when `tp_plan="auto"` and model has `base_model_tp_plan` |
| Dequantization/deserialization | Only when loading a pre-quantized checkpoint |
| Quantization | Only when a quantization config is provided and weights are not pre-quantized |

### Dense models (e.g., Llama)

For most dense models, the checkpoint format matches the model format directly, so no conversion mapping is needed. Some models may still require renaming (e.g., legacy naming conventions). TP sharding still applies when enabled.

```
Checkpoint:                          Model:
model.layers.0.self_attn.q_proj.weight  →  model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight  →  model.layers.0.self_attn.k_proj.weight
model.layers.0.mlp.gate_proj.weight     →  model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight       →  model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight     →  model.layers.0.mlp.down_proj.weight
```x

Legacy checkpoints may use older naming conventions that are handled by built-in renamings applied to all models:

```
Checkpoint:                          Model:
LayerNorm.gamma              →       LayerNorm.weight
LayerNorm.beta               →       LayerNorm.bias
```

### MoE models (e.g., Mixtral)

For MoE models, the checkpoint format differs from the model format. Conversion mapping transforms separate expert weights into fused 3D tensors, and TP sharding applies after conversion.

```
Checkpoint:                              Model:
experts.0.w1.weight  ─┐
experts.1.w1.weight   │ MergeModulelist
...                   ├───────────────→  experts.gate_up_proj (8, hidden, 2*intermediate)
experts.0.w3.weight   │ + Concatenate
experts.1.w3.weight  ─┘
```

## Architecture

The system is built around several key components defined in `src/transformers/core_model_loading.py`:

**Phase 1 — Per-key processing** (iterates over checkpoint keys):

1. **Rename key** via `WeightRenaming` (e.g. `block_sparse_moe` -> `mlp`)
2. **Match pattern** via `WeightConverter` (e.g. `experts.*.w1.weight`)
3. **Shard (TP) and send to device** asynchronously via `ThreadPoolExecutor`
4. **Collect** tensors with the same `source_pattern` together (e.g. all MoE expert weights, gate + up projections)

**Phase 2 — Per-mapping processing** (iterates over collected mappings):

1. **Dequantize/deserialize** (pre-quantized checkpoints only)
2. **Apply `ConversionOps` chain**: `Chunk`, `Concatenate`, `MergeModulelist`, `Transpose`, etc.
3. **Quantize** on-the-fly (if not pre-quantized)
4. **Set parameter** on model

### WeightTransform

The base class that handles pattern matching and tensor collection:

- **Pattern compilation**: Converts glob-style patterns (`*.weight`) to regex.
- **Key renaming**: `rename_source_key()` transforms checkpoint keys to model keys.
- **Tensor collection**: `add_tensor()` gathers related tensors for batch processing.
- **Reversibility**: `reverse_transform()` creates the inverse operation for saving.

```python
@dataclass(slots=True)
class WeightTransform:
    source_patterns: str | list[str]      # Checkpoint key patterns
    target_patterns: str | list[str]      # Model key patterns
    compiled_sources: re.Pattern          # Compiled regex for matching
    distributed_operation: TensorParallelLayer | None
    quantization_operation: ConversionOps | None
    collected_tensors: dict[str, list[Future]]  # Gathered tensors
    layer_targets: dict[str, set[str]]          # Target key tracking
```

### WeightRenaming

[`WeightRenaming`] is a specialized [`WeightTransform`] for simple 1:1 key renaming without tensor operations:

```py
# Legacy checkpoint compatibility
WeightRenaming("LayerNorm.gamma", "LayerNorm.weight")

# Module path changes
WeightRenaming(".block_sparse_moe.", ".mlp.")

# Adding prefixes
WeightRenaming("(.+)", "timm_model.\\1")
```

### WeightConverter

[`WeightConverter`] extends [`WeightTransform`] with a list of [`ConversionOps`]:

```python
@dataclass(slots=True)
class WeightConverter(WeightTransform):
    operations: list[ConversionOps]  # Chain of operations
```

It supports many-to-one (e.g., concatenating `gate` + `up` → `gate_up`), one-to-many (e.g., splitting `qkv` → `q`, `k`, `v`), and chained operations applied sequentially.

## Conversion operations

The [`WeightConverter`] class has several operations that are executed when [`~PreTrainedModel.from_pretrained`] is called for transforming checkpoint source tensors into model target tensors.

Operations are fully reversible. Saving reverses the conversions and returns the original checkpoint so you can easily work across different frameworks.

| Operation | Reverse |
|-----------|---------|
| [`Chunk(dim)`] | [`Concatenate(dim)`] |
| [`Concatenate(dim)`] | [`Chunk(dim)`] |
| [`MergeModulelist(dim)`] | [`SplitModulelist(dim)`] |
| [`SplitModulelist(dim)`] | [`MergeModulelist(dim)`] |
| [`Transpose(d0, d1)`] | [`Transpose(d1, d0)`] |
| [`Force16BytesAlignment`] | [`Force16BytesAlignment`] (idempotent) |

### Chunk

The [`Chunk`] operation splits a tensor into equal parts along a dimension. For example, if a model expects Q, K, and V as three separate tensors instead of a single tensor.

```py
WeightConverter(
    "self_attn.qkv_proj",
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    operations=[Chunk(dim=0)],
)
```

### Concatenate

The [`Concatenate`] operation fuses separate tensors into a single tensor. For example, if a model expects Q, K, and V as a single tensor instead of separate tensors.

```py
WeightConverter(
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "self_attn.qkv_proj",
    operations=[Concatenate(dim=0)],
)
```

### MergeModulelist

[`MergeModulelist`] merges a list of 2D tensors into a single 3D tensor. For example, you can compose [`MergeModulelist`] with [`Concatenate`] to stack the experts in a MoE and pack them into one tensor.

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

[`SplitModulelist`] splits a 3D tensor back into a list of 2D tensors. For example, you can split a stack of experts back into individual experts.

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

### Transpose

[`Transpose`] swaps dimensions of a tensor. Useful for converting weight layouts between different conventions.

```py
WeightConverter(
    source_patterns="mlp.gate.weight",
    target_patterns="mlp.text_moe.gate.weight",
    operations=[Transpose(dim0=0, dim1=1)],
)
```

### Force16BytesAlignment

[`Force16BytesAlignment`] clones a tensor if it is not 16-byte aligned. This is required for `torch._grouped_mm` and TMA/SIMD operations. It is idempotent: applying it more than once has no additional effect.

## Operation chaining

Operations can be chained to perform complex transformations. The operations execute in order, with each operation's output becoming the next operation's input.

### Example: Mixtral MoE conversion

```python
WeightConverter(
    source_patterns=[
        ".experts.*.w1.weight",  # gate_proj per expert
        ".experts.*.w3.weight",  # up_proj per expert
    ],
    target_patterns=".experts.gate_up_proj",
    operations=[
        MergeModulelist(dim=0),  # Stack all experts: (n_experts, in, out)
        Concatenate(dim=1),      # Fuse gate+up: (n_experts, in, 2*out)
    ],
)
```

**Data flow:**
```
Input:
  ".experts.*.w1.weight": [tensor_0, tensor_1, ..., tensor_7]  # 8 experts
  ".experts.*.w3.weight": [tensor_0, tensor_1, ..., tensor_7]  # 8 experts

After MergeModulelist(dim=0):
  ".experts.*.w1.weight": (8, 4096, 14336)  # stacked gate
  ".experts.*.w3.weight": (8, 4096, 14336)  # stacked up

After Concatenate(dim=1):
  ".experts.gate_up_proj": (8, 4096, 28672)  # fused gate_up
```

### Pattern matching

The `*` in patterns acts as a wildcard:
- During loading, it matches any numeric index (`experts.0.`, `experts.1.`, etc.).
- Tensors with the same pattern (differing only in index) are grouped together.
- The order of collection is preserved for correct concatenation.

## Tensor parallelism integration

The dynamic loading system integrates with tensor parallelism (TP) through the `TensorParallelLayer` hierarchy defined in `src/transformers/integrations/tensor_parallel.py`.

When TP is enabled, tensors are sharded **during** materialization, not after. This means each rank only loads the portion of the tensor it needs.

```python
def spawn_tp_materialize(thread_pool, tensor, sharding_method, tensor_idx, device, dtype):
    def _job():
        return sharding_method.shard_tensor(tensor, tensor_idx=tensor_idx, device=device, dtype=dtype)
    return thread_pool.submit(_job)
```

### Available parallel styles

| Style | Weight Shard Dim | Description |
|-------|------------------|-------------|
| `colwise` | -2 | Column-wise: output features sharded |
| `rowwise` | -1 | Row-wise: input features sharded |
| `packed_colwise` | -2 | For fused weights (gate_up_proj) |
| `packed_rowwise` | -1 | For fused weights |
| `embedding_rowwise` | 0 | Vocabulary parallelism |
| `grouped_gemm` | 0 | Expert parallelism for MoE |
| `sequence_parallel` | None | No weight sharding |

### Packed weight handling

For fused weights like `gate_up_proj`, special care is needed to shard correctly:

```python
def get_packed_weights(param, empty_param, device_mesh, rank, dim):
    """
    Interleaves gate and up shards correctly.

    Packed tensor: [G0 G1 G2 G3 | U0 U1 U2 U3]

    With TP=2:
    - Rank 0 gets: [G0 G1 | U0 U1]
    - Rank 1 gets: [G2 G3 | U2 U3]
    """
```

The TP operation is stored in the [`WeightTransform`] and applied after conversion operations:

```python
if matched_tp_pattern := tp_plan_alt.search(renamed_key):
    tp_layer = ALL_PARALLEL_STYLES[model.tp_plan[matched_tp_pattern]]
    mapping.distributed_operation = tp_layer(
        device_mesh=device_mesh,
        rank=device_mesh.get_local_rank(),
        empty_param=empty_param.clone()
    )
```

## Quantization integration

Quantization hooks into the loading pipeline in two ways, depending on whether the checkpoint is already quantized:

- **Pre-quantized checkpoints**: The quantizer provides [`WeightConverter`] instances (via `get_weight_conversions()`) that deserialize quantized tensors. Checkpoint dtypes are preserved to avoid unwanted casts.
- **On-the-fly quantization**: The quantizer provides a quantization operation that is applied after conversion ops, quantizing weights as they are loaded.

## Fast and efficient model loading

Loading a model is faster and uses less memory because the loader knows which tensors are required for operations and schedules their materialization lazily.

The loader scans the checkpoint *once* to discover pattern matches and collect tensors. It stores them as `Future` objects and submits them to a thread pool for asynchronous loading without blocking the GIL. A parameter starts loading as soon as a thread becomes available to it.

If your system runs other heavy processes, multiple threads may slow down loading instead of accelerating it. In this case, set the environment variable `HF_DEACTIVATE_ASYNC_LOAD=1` to load weights sequentially.

> [!NOTE]
> The default is 4 threads for asynchronous parameter loading. This provides the best trade-off across loading scenarios and hardware. The work is mostly I/O bound, but depending on accelerator hardware and the `dtype` required at loading, it can become CPU/GPU-bound if the `dtype` differs from the serialized one (this requires an additional copy operation).

### Async vs sync loading

```python
def spawn_materialize(thread_pool, tensor, device, dtype) -> Future | Callable:
    def _job():
        return _materialize_copy(tensor, device, dtype)

    if thread_pool is not None:
        return thread_pool.submit(_job)  # Async: returns Future
    else:
        return _job  # Sync: returns Callable (deferred execution)
```

Sync loading is used when:
- `HF_DEACTIVATE_ASYNC_LOAD=1` environment variable is set.
- Disk offloading is enabled (memory constraints require sequential loading).

### Materialization flow

```
1. Checkpoint iteration (Phase 1):
   - For each key, submit materialization job to ThreadPoolExecutor
   - Job returns Future (async) or Callable (sync)
   - Collect into the matching WeightConverter/WeightRenaming

2. Per-mapping processing (Phase 2, one mapping at a time):
   - materialize_tensors() waits for this mapping's Futures only
   - Apply conversion operations chain (self.operations)
   - Apply quantization operation (if on-the-fly)
   - Set parameters on model
   - Delete realized tensors immediately

3. Cleanup:
   - Thread pool shutdown (with cancel_futures=True for interrupts)
```

### Memory efficiency

When converting a weight, the converter waits for all required tensors to materialize if they haven't loaded yet. For example, the [`MergeModulelist`] operation requires all weights in `ModuleList` to be loaded before merging.

Concatenating tensors requires a temporary copy, so operations like [`MergeModulelist`] and [`Concatenate`] need 2x the memory of the underlying tensors during conversion. Once merged, only the resulting tensor stays in memory. The theoretical worst-case memory peak is the model size plus the tensors required for the largest [`MergeModulelist`] or [`Concatenate`] operation.

This worst case only occurs when all other parameters have loaded before the demanding conversion runs. Two scenarios trigger this.

1. All parameters loaded asynchronously before entering the demanding conversion (the thread pool was faster than the conversion queue).
2. The demanding conversion is the last one.

For example, a MoE model using [`MergeModulelist`] for experts on each layer, the theoretical worst-case memory peak is model size plus experts on one layer.

These worst-case scenarios are uncommon. The actual memory peak tends to stay close to the model size.

## Reversibility

The system supports saving models with the inverse transformations, enabling round-trip save/load:

```python
def revert_weight_conversion(model, state_dict):
    """Applies reverse conversions for saving."""
    weight_conversions = getattr(model, "_weight_conversions", None)

    # Reverse all transforms
    reverse_weight_conversion = [
        conversion.reverse_transform() for conversion in weight_conversions
    ]

    # Apply in reverse
    for first_param_name, reversed_converter in conversion_mapping.items():
        realized_value = reversed_converter.convert(first_param_name, model=model)
```

Target patterns may contain regex elements that need processing for the reverse direction:

```python
def process_target_pattern(pattern: str) -> tuple[str, str | None]:
    """
    - Removes `^` and `$` anchors
    - Removes negative lookahead/lookbehind
    - Detects capturing groups, replaces with \1
    """
```

## Real examples

### Mixtral-style MoE

**Checkpoint format:**
```
model.layers.0.block_sparse_moe.experts.0.w1.weight  # gate per expert
model.layers.0.block_sparse_moe.experts.0.w2.weight  # down per expert
model.layers.0.block_sparse_moe.experts.0.w3.weight  # up per expert
...
model.layers.0.block_sparse_moe.experts.7.w1.weight
```

**Model format:**
```
model.layers.0.mlp.experts.gate_up_proj  # (8, 4096, 28672)
model.layers.0.mlp.experts.down_proj     # (8, 14336, 4096)
```

**Conversion mapping** (from `conversion_mapping.py`):
```python
"mixtral": [
    WeightRenaming(".block_sparse_moe.", ".mlp."),
    WeightConverter(
        source_patterns=[".experts.*.w1.weight", ".experts.*.w3.weight"],
        target_patterns=".experts.gate_up_proj",
        operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
    ),
    WeightConverter(
        source_patterns=[".experts.*.w2.weight"],
        target_patterns=".experts.down_proj",
        operations=[MergeModulelist(dim=0)],
    ),
],
```

### Custom operations (ERNIE 4.5 VL MoE)

When the built-in operations aren't sufficient, you can create a custom [`ConversionOps`] subclass. For example, ERNIE 4.5 VL MoE needs to split a shared expert list between text and vision modalities — something no single built-in op handles. The custom `ErnieFuseAndSplitTextVisionExperts` operation splits and re-stacks experts across two target keys:

```python
"ernie4_5_vl_moe": [
    WeightRenaming("vision_model", "vision_tower"),
    WeightConverter(
        source_patterns=["experts.*.down_proj.weight"],
        target_patterns=[
            "text_moe.experts.down_proj",
            "vision_moe.experts.down_proj",
        ],
        operations=[ErnieFuseAndSplitTextVisionExperts(stack_dim=0, concat_dim=1)],
    ),
],
```

Custom ops must implement `convert()` and the `reverse_op` property to support round-trip save/load.

### Model type aliases

Many models share conversion patterns:

```python
_MODEL_TO_CONVERSION_PATTERN = {
    "mixtral": "mixtral",
    "minimax": "mixtral",
    "qwen2_moe": "qwen2_moe",
    "deepseek_v2": "qwen2_moe",
    "deepseek_v3": "qwen2_moe",
    "qwen3_moe": "qwen2_moe",
    "olmoe": "qwen2_moe",
    ...
}
```

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
    - replace / map operations to your custom operations: this is what happens with `peft`. We replace the `Concatenate` operation of say `mixtral`, to be `PeftConcatenate`. This way, when the adapter checkpoint is read, the weights to be concatenated are collected, and are properly formatted for `peft`
3. **Load + finalize + report.** Use the core loader to perform the conversion and populate tensors, then finalize and
   log results. Concretely, this flow is:
   - `LoadStateDictConfig(...)` + `_load_pretrained_model(...)` to load and convert.
   - `_finalize_load_state_dict(...)` to move any missing/mismatched tensors off `meta`, initialize them, and tie weights.
   - `log_state_dict_report(...)` to report missing/unexpected/mismatched keys (and conversion errors).

These APIs are exposed to allow you to handle custom code, custom weight formats, but also make sure you benefit from the highest and most efficient weight loading, sharding and good quality of life of `transformers` API!

## Key files reference

| File | Purpose |
|------|---------|
| `src/transformers/core_model_loading.py` | Core loading logic, WeightConverter, ConversionOps |
| `src/transformers/conversion_mapping.py` | Built-in conversion patterns for all models |
| `src/transformers/integrations/tensor_parallel.py` | TP sharding classes and utilities |
| `src/transformers/quantizers/base.py` | Quantization hooks and base class |
