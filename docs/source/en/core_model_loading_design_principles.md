<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Core model loading: design principles

[`~PreTrainedModel.from_pretrained`] turns checkpoint tensors into model params through one generic pipeline
(`core_model_loading.py`). Quantization and tensor parallelism plug into it at fixed points; they never fork it. This
page walks one key of the hardest case through every step. See [Dynamic weight loading](./weightconverter) for the
`WeightRenaming` / `WeightConverter` API itself.

## Principles

1. **Renamings first, in order.** All `WeightRenaming`s (the model's, then the quantizers') run on every checkpoint key,
   one after another.
2. **Then at most one converter.** A `WeightConverter` matches a key on a glob pattern (`*` = the expert index) and
   gives its target param. Converters are general: having more of them than a checkpoint needs is free.
3. **One instance per target param.** The first key of a param deep-copies its converter; every other key of that param
   is collected into the same instance.
4. **Ops are per param.** Once the instance exists, its full name is known: the quantizer of that layer adds its ops
   (deserialize, dequantize, quantize).
5. **Shard on read.** With tensor parallelism, a key is sliced to this rank's piece when it is read, before any op.
6. **Nothing after loading.** No quantizer post-processes the loaded model: everything happens on the tensors, while
   loading.

## Before any key is read

```text
from_pretrained
 ├─ config, hf_quantizer = get_hf_quantizer(config)        a plan: {pattern: config} → one quantizer per distinct config
 ├─ model = cls(config)                                    on meta
 ├─ hf_quantizer.preprocess_model(model)
 │    ├─ match_modules     every module → its quantizer, matched once against the plan's patterns
 │    │                    (a checkpoint's own layer names, e.g. ModelOpt's, are first renamed to the model's)
 │    ├─ replace_modules   nn.Linear → FP8Linear, experts → NVFP4Experts, … (quantized_module)
 │    └─ update_parallel_plan   the quantized params (scales, …) follow their weight's TP/EP style
 ├─ maybe_distribute_model(model)                          PP → TP → EP → FSDP: params become DTensors (still meta)
 └─ conversions = model's conversion mapping, updated by the quantizer:
      renamings  : model's + each quantizer's `renamed_names`, scoped to its layers
      converters : per quantizer, every model converter + one copy per serialized name, + its general converters
```

## One key, end to end

ModelOpt's `nvidia/Qwen3.6-35B-A3B-NVFP4`: experts saved as a per-expert list, merged and concatenated into the
model's fused `gate_up_proj`; NVFP4 experts and FP8 attention in one checkpoint.

```text
checkpoint key   model.language_model.layers.0.mlp.experts.3.gate_proj.weight_scale
  │
  │ 1. RENAMINGS (all, in order)
  │      model's      model.language_model.X → model.X
  │      quantizers'  FP8: weight_scale → weight_scale_inv … only on FP8's layers: not this one (NVFP4)
  ▼
                 model.layers.0.mlp.experts.3.gate_proj.weight_scale
  │
  │ 2. CONVERTERS of this layer       hf_quantizer.get_weight_converters(key) → NVFP4's list
  │      model converter    experts.*.gate_proj.weight + experts.*.up_proj.weight → experts.gate_up_proj
  │      its copy per serialized name (same ops, the name swapped):
  │                         experts.*.gate_proj.weight_scale + …up_proj.weight_scale → experts.gate_up_proj_scale
  │                         (dequantized layer: same target as the weight, gate_up_proj, to dequantize them together)
  ▼
  target param   model.layers.0.mlp.experts.gate_up_proj_scale
  │
  │ 3. INSTANCE      the first key of this param: deepcopy(converter); all its keys (every expert × gate, up) collect here
  │ 4. OPS           hf_quantizer.update_weight_transform(instance, "…gate_up_proj_scale")   ← full name → NVFP4
  │                    [MergeModulelist, Concatenate] + [NVFP4Deserialize]      deserialize: after the model's ops
  │                    dequantize=True: [NVFP4Dequantize] + [MergeModulelist, Concatenate], on the weight + its scales
  │                    full precision checkpoint: quantization_operation = quantize op
  │ 5. READ          each key: get_slice → (TP) this rank's piece → dtype → device       (threaded)
  │ 6. CONVERT       ops on the collected tensors:  stack experts → cat gate|up → swizzle the block scales
  ▼
  set_param      NVFP4Experts.gate_up_proj_scale (+ gate_up_proj_sf, from the deserialize)
```

What each component knows:

| Component | Knows | Doesn't know |
|---|---|---|
| `WeightRenaming` | a key's name | layers, quantization |
| `WeightConverter` | how per-expert / split keys map to one param (merge, concat, chunk, …) | quantization: a scale goes through the same ops as its weight |
| the quantizer | its serialized names, its ops, its layers | the model's conversions |
| core loading | the order: rename → converters of the layer → instance → ops → read → convert | which format, which quantizer |

## Quantization plans: two name maps

A checkpoint and a model name the same layers differently (`model.language_model.layers.0.mlp.experts.3.gate_proj` vs
`model.layers.0.mlp.experts`). The model's conversion mapping goes from one to the other; the plan goes through it
once:

```text
plan (checkpoint names, e.g. ModelOpt's quantized_layers)
   │ rename_patterns: each pattern renamed like a key of it (`{pattern}.weight` through the model's conversions)
   ▼
plan (model names) ── match_modules ──► module_quantizers {module name → quantizer}      ← the only matcher
   used by:  replace_modules (which class) · get_weight_converters (which converters) ·
             update_weight_transform (which ops) · renamings (which layers)
```

A pattern matches like the dtype plan's globs; exact names win over globs, `"*"` takes the rest. Patterns with equal
configs share one quantizer.

## Tensor parallelism

```text
before loading   apply_tensor_parallelism(model, tp_mesh, tp_plan)        params → DTensors with placements, on meta
                   q_proj.weight          Shard(0)            rows split across ranks
                   o_proj.weight          Shard(1)            columns split
                   experts.gate_up_proj   _StridedShard(d, 2) gate | up split each, so a rank keeps both halves
                   quantized params       the style of their weight (`update_parallel_plan`); per-tensor scales Replicate
while loading    per key: DtensorShardOperation(empty param).shard_tensor(slice)
                   reads only this rank's rows/columns from disk (safetensors get_slice), before any op
                   per-expert keys (tensor_idx): experts this rank doesn't own are skipped
                 the ops then run on local shards: Merge / Concat / deserialize / dequantize see this rank's piece
                 set_param wraps the local result back into the DTensor
saving           the conversions run in reverse (converters, then renamings); dequantize reverses to identity
```

Because sharding happens on read, every op must work on a shard: merges and concatenations act along dims the placement
keeps whole, and a quantizer's op (e.g. NVFP4's swizzle, per expert) only reshapes what it receives.
