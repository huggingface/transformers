# FSDP Loader Design Summary

## Goal

The goal of this work is to make FSDP load through the standard Hugging Face `from_pretrained` pipeline in the same high-level way as tensor parallelism:

1. distribute the model before loading
2. let `core_model_loading.py` read checkpoint tensors
3. materialize only the local shard for the current rank
4. load that shard into the already-distributed parameter

This is intentionally not `dcp.load`.

## Current Architecture

The important split is:

- `distribute_model(...)` is responsible for creating the runtime distributed layout
- `core_model_loading.py` is responsible for loading checkpoint tensors into that layout

That means `core_model_loading.py` does **not** decide the FSDP layout from scratch. It reads the layout from the already-wrapped parameter and loads accordingly.

## Code Path: Distribute Before Load

In `src/transformers/modeling_utils.py`, FSDP is initialized and applied before `_load_pretrained_model(...)`:

```python
if fsdp_plan is not None:
    device_map, device_mesh, _ = initialize_fsdp(
        fsdp_plan=fsdp_plan,
        device_mesh=device_mesh,
        device_map=device_map,
    )
...
if _torch_distributed_available and device_mesh is not None and (tp_plan is not None or fsdp_plan is not None):
    model = distribute_model(model, tp_plan, distributed_config, device_mesh, tp_size, fsdp_plan=fsdp_plan)
...
loading_info, disk_offload_index = cls._load_pretrained_model(model, state_dict, checkpoint_files, load_config)
```

Relevant location:
- `src/transformers/modeling_utils.py:4010-4169`

In `src/transformers/integrations/tensor_parallel.py`, `distribute_model(...)` applies FSDP before loading:

```python
if fsdp_plan is not None:
    from .fsdp import apply_fsdp2

    model = apply_fsdp2(model, device_mesh, fsdp_plan)
```

Relevant location:
- `src/transformers/integrations/tensor_parallel.py:1525-1528`

So yes: the current code does what Arthur described, namely "distribute before load, then load according to that plan".

## What `core_model_loading.py` Does For FSDP

Once the model is already distributed, `core_model_loading.py`:

1. looks up the target runtime parameter
2. detects if it is DTensor-backed
3. derives the shard-on-read rule from that live parameter
4. reads only the local shard
5. rebuilds a DTensor from the local shard
6. copies or swaps it into the existing parameter object

The relevant lookup is:

```python
empty_param = meta_model_state_dict.get(renamed_key)
try:
    empty_param = model.get_parameter_or_buffer(renamed_key)
except (AttributeError, KeyError):
    pass
```

Relevant location:
- `src/transformers/core_model_loading.py:1394-1397`

The FSDP-aware shard selection is:

```python
if is_dtensor_like(empty_param):
    if getattr(mapping, "distributed_operation", None) is None:
        mapping.distributed_operation = FSDPShardOperation.from_param(empty_param)
    return ParallelMaterializationContext(
        mapping.distributed_operation,
        tensor_idx,
        get_device(device_map, renamed_key, valid_torch_device=True),
    )
```

Relevant location:
- `src/transformers/core_model_loading.py:1014-1021`

The actual shard-on-read spawn call is shared with TP:

```python
if parallel_context := get_parallel_materialization_context(...):
    future_or_tensor = spawn_parallel_materialize(
        thread_pool,
        tensor,
        parallel_context.distributed_operation,
        parallel_context.tensor_idx,
        parallel_context.device,
        _dtype,
    )
```

Relevant location:
- `src/transformers/core_model_loading.py:1451-1468`

And the loaded local shard is written back into the existing FSDP parameter like this:

```python
if is_dtensor_like(ref):
    local_param = param_value.detach() if isinstance(param_value, torch.nn.Parameter) else param_value
    fsdp_param = DTensor.from_local(
        local_param.contiguous(),
        ref.device_mesh,
        ref.placements,
        run_check=False,
        shape=ref.shape,
        stride=tuple(ref.stride()),
    )
    with torch.no_grad():
        if ref.is_meta:
            fsdp_param = torch.nn.Parameter(fsdp_param, requires_grad=ref.requires_grad)
            torch.utils.swap_tensors(ref, fsdp_param)
        else:
            ref.copy_(fsdp_param)
```

Relevant location:
- `src/transformers/core_model_loading.py:1127-1142`

## Why FSDP Needs The Live Parameter

Before this work, the loader could usually get away with:

```python
empty_param = model.state_dict()[key]
```

That was enough when loading was:

- plain loading: full tensor -> assign parameter
- TP loading: sharding decided by the external TP plan

For FSDP, the sharding layout is not coming from an external per-key plan inside `core_model_loading.py`. It comes from the already-wrapped runtime parameter:

- `device_mesh`
- `placements`
- meta/non-meta state
- existing parameter identity

So FSDP needs:

```python
ref = model.get_parameter_or_buffer(key)
```

because the live object tells us how to load the shard and where to install it.

### Simple Comparison

TP-style mental model:

```python
style = tp_plan[matched_pattern]
tp_op = ALL_PARALLEL_STYLES[style](...)
local_shard = tp_op.shard_tensor(checkpoint_tensor)
module.weight = torch.nn.Parameter(local_shard)
```

FSDP-style mental model:

```python
ref = model.get_parameter_or_buffer(key)
fsdp_op = FSDPShardOperation.from_param(ref)
local_shard = fsdp_op.shard_tensor(checkpoint_tensor)

fsdp_param = DTensor.from_local(
    local_shard,
    ref.device_mesh,
    ref.placements,
    shape=ref.shape,
    stride=tuple(ref.stride()),
    run_check=False,
)

ref.copy_(fsdp_param)  # or swap_tensors if ref.is_meta
```

The key distinction is:

- TP derives sharding from the plan
- FSDP derives sharding from the live parameter

## Why `deepcopy(...)` And `concretize_target_patterns(...)` Exist

This is a separate issue from FSDP sharding.

The point is that `WeightConverter` instances are stateful. They accumulate tensors in `collected_tensors`.

Example template:

```python
template = WeightConverter(
    ["experts.*.w1.weight", "experts.*.w3.weight"],
    "experts.gate_up_proj.weight",
    operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
)
```

If the same converter instance were reused across multiple layers, tensors from different layers would be mixed together.

Wrong:

```python
shared_converter.collected_tensors = {
    "experts.*.w1.weight": [layer0_expert0_w1, layer0_expert1_w1, layer1_expert0_w1, layer1_expert1_w1],
    "experts.*.w3.weight": [layer0_expert0_w3, layer0_expert1_w3, layer1_expert0_w3, layer1_expert1_w3],
}
```

Correct:

```python
converter_for_layer0.collected_tensors = {
    "experts.*.w1.weight": [layer0_expert0_w1, layer0_expert1_w1],
    "experts.*.w3.weight": [layer0_expert0_w3, layer0_expert1_w3],
}

converter_for_layer1.collected_tensors = {
    "experts.*.w1.weight": [layer1_expert0_w1, layer1_expert1_w1],
    "experts.*.w3.weight": [layer1_expert0_w3, layer1_expert1_w3],
}
```

That is why the loader does:

```python
new_converter = deepcopy(pattern_to_converter[source_pattern])
```

It creates one mutable converter instance per concrete destination key.

Then:

```python
new_converter = concretize_target_patterns(
    new_converter,
    original_key,
    source_pattern,
    prefix,
    meta_model_state_dict,
)
```

turns wildcard/generic target patterns into actual concrete destination keys for the current source key.

Example:

```python
source_key = "model.layers.7.self_attn.qkv_proj.weight"
target_pattern = "model.layers.*.self_attn.q_proj.weight"
```

becomes:

```python
"model.layers.7.self_attn.q_proj.weight"
```

So:

- `model.get_parameter_or_buffer(...)` is about live FSDP param layout
- `deepcopy(...)` is about isolating converter state per target
- `concretize_target_patterns(...)` is about resolving wildcard targets into concrete names

## Why This Was Not Needed Before

Two old assumptions used to hold well enough:

1. the loader only needed shape/dtype existence information, not runtime DTensor metadata
2. many conversions were simple enough that shared converter templates did not obviously collide

FSDP shard-on-read breaks assumption 1 because the loader now needs the live distributed parameter layout.

Mixtral-style grouped conversions exposed assumption 2 because converter state is now clearly multi-tensor and destination-specific.

## Why We Still Read DTensor Metadata

It is tempting to say "FSDP2 shards on dim 0, so just hardcode dim 0".

That is not robust enough.

For simple 1D FSDP2, parameters are usually `Shard(0)`. But that is not the only valid runtime layout:

- HSDP / 2D mesh can be `Replicate(), Shard(0)`
- custom `shard_placement_fn` can change the shard dimension
- future mixed parallel setups may use more complex placements

So the loader still needs to read:

- `ref.device_mesh`
- `ref.placements`

This makes the load path follow the runtime layout that PyTorch actually created, instead of assuming one.

## FSDP With CP Or EP

Do not assume "always plain `Shard(0)`" once other parallelisms are involved.

Practical rule:

- FSDP2 alone on a 1D mesh: usually `(Shard(0),)`
- HSDP / 2D mesh: can be `(Replicate(), Shard(0))`
- CP mostly affects sequence buffers/activations, but FSDP parameter layout still needs to be read from the param
- EP combinations should also not be hardcoded to a single assumed placement

So the safe abstraction remains:

```python
read the live DTensor placements and load accordingly
```

## Generic Refactor Done In `core_model_loading.py`

The load path is now more parallelism-agnostic than before.

Shared helper for shard-on-read:

```python
def spawn_parallel_materialize(...)
```

Relevant location:
- `src/transformers/core_model_loading.py:876`

Shared helper for resolving shard-on-read context:

```python
def get_parallel_materialization_context(
    mapping,
    renamed_key,
    source_pattern,
    empty_param,
    device_mesh,
    parallel_plan,
    parallel_pattern_matcher,
    parallel_pattern_by_group_name,
    device_map,
)
```

Relevant location:
- `src/transformers/core_model_loading.py:982`

This is intentionally named in a parallelism-agnostic way so future distributed loading modes are not forced into TP-specific naming.

## Short Takeaways

- Yes, FSDP is now distributed before load.
- Yes, `core_model_loading.py` now loads according to that FSDP layout.
- No, `core_model_loading.py` does not create the FSDP layout itself.
- The live parameter lookup is needed because FSDP shard metadata lives on the runtime DTensor-backed parameter.
- `deepcopy(...)` and `concretize_target_patterns(...)` are separate converter-state fixes, not the same issue as FSDP sharding.
- Reading DTensor metadata is still the right design even if many simple FSDP2 cases shard on dim 0.
