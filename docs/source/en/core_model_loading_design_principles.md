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

# Core model loading: a walkthrough

This page follows one checkpoint through [`~PreTrainedModel.from_pretrained`], step by step: what happens, and why.
See [Dynamic weight loading](./weightconverter) for the `WeightRenaming` / `WeightConverter` API itself.

The checkpoint is ModelOpt's `nvidia/Qwen3.6-35B-A3B-NVFP4`, the hardest common case:

- the experts are saved one by one (`experts.3.gate_proj.weight`), the model has them fused (`experts.gate_up_proj`);
- the checkpoint says `model.language_model.layers…`, the model says `model.layers…`;
- the layers are quantized differently: NVFP4 experts, FP8 attention.

## 1. The config becomes a plan

```text
config.json  quantization_config = {
               "quant_algo": "MIXED_PRECISION",
               "quantized_layers": {
                 "model.language_model.layers.0.mlp.experts.3.gate_proj": {"quant_algo": "NVFP4"},
                 "model.language_model.layers.0.self_attn.q_proj":        {"quant_algo": "FP8"},
                 …
               }}
                 │ modelopt_plan (PLAN_CONVERTERS)
                 ▼
plan         {"model.language_model.layers.0.mlp.experts.3.gate_proj": NVFP4Config(), …: FP8Config(), …}
                 │ one quantizer per distinct config
                 ▼
             QuantizationPlanHfQuantizer   →  NVFP4HfQuantizer, FineGrainedFP8HfQuantizer
```

Why: a per-layer format is just a plan, `{pattern: config}`, like the dtype plan. Each format is its own quantizer, so
a new format is a new entry in `PLAN_CONVERTERS`, not new loading code. Equal configs share one quantizer, so 40 layers
of NVFP4 are one quantizer, not 40.

## 2. The plan is renamed to the model's names, once

```text
model.language_model.layers.0.mlp.experts.3.gate_proj   (+ ".weight", through the model's conversions)
   → model.layers.0.mlp.experts.gate_up_proj.weight
   → model.layers.0.mlp.experts                          ← the plan's pattern now
```

Why: the plan was written against checkpoint names, the model only knows its own. Sending `{pattern}.weight` through
the same conversions the keys will go through renames the pattern exactly as its keys will be renamed: no guessing. The
64 per-expert patterns collapse onto the one fused module.

## 3. Every module gets its quantizer, once

```text
match_modules(model)       exact names first, then globs, then "*"
  model.layers.0.mlp.experts        → NVFP4
  model.layers.0.self_attn.q_proj   → FP8
  model.layers.0.mlp.gate           → None (unquantized)
```

Why: this is the only place a name is matched against the plan. Everything later (which module class, which
renamings, which converters, which ops) looks the answer up in `module_quantizers` instead of matching again.

## 4. The model is built on meta and its modules swapped

```text
model = cls(config)                        on meta, no memory
preprocess_model
  replace_modules     experts → NVFP4Experts (buffers gate_up_proj, gate_up_proj_scale, _scale_2, _input_scale, _sf)
                      q_proj  → FP8Linear    (weight, weight_scale_inv, activation_scale)
  update_parallel_plan  the new params follow their weight's TP/EP style
maybe_distribute_model    with TP: params become DTensors (still meta)
```

Why: the quantized params must exist before loading so each checkpoint tensor has a destination. Swapping is one pass
over the model for all quantizers.

## 5. The conversions are built: renamings, then converters

```text
renamings    the model's     model.language_model.X     → model.X
             FP8's, scoped   ^(…self_attn.q_proj|…)\.weight_scale$ → \1.weight_scale_inv
                             ^(…self_attn.q_proj|…)\.input_scale$  → \1.activation_scale
converters   the model's     experts.*.gate_proj.weight + experts.*.up_proj.weight      → experts.gate_up_proj
             NVFP4's copies  experts.*.gate_proj.weight_scale   + …up_proj.weight_scale   → experts.gate_up_proj_scale
                             experts.*.gate_proj.weight_scale_2 + …                       → experts.gate_up_proj_scale_2
                             experts.*.gate_proj.input_scale    + …                       → experts.gate_up_proj_input_scale
             general         weight + weight_scale_inv + activation_scale                 → weight   (FP8)
                             weight + weight_scale + weight_scale_2 + input_scale         → weight   (NVFP4)
```

Why:

- **Renamings are scoped to the quantizer's layers.** ModelOpt calls FP8's scale `weight_scale`, but NVFP4's block
  scale is also `weight_scale`. Renaming it everywhere would break the experts. Each quantizer only renames on its
  layers (from step 3).
- **A scale goes through the same ops as its weight.** The model's converter stacks 64 experts and concatenates gate
  and up; the scales need exactly that. So each quantizer copies the model's converters that load into its layers, one
  copy per serialized name, with the same ops. The model's conversion mapping never knows about quantization.
- **Converters only describe shapes.** No quantization op is in them yet: which op runs depends on the layer, known
  only once a key is matched (step 7). Having more converters than a checkpoint needs costs nothing.

## 6. One key: renamed, then matched

```text
model.language_model.layers.0.mlp.experts.3.gate_proj.weight_scale
  │ renamings (all, in order)    model's: drop "language_model"     FP8's: not an FP8 layer, untouched
  ▼
model.layers.0.mlp.experts.3.gate_proj.weight_scale
  │ converters (the first that matches)    NVFP4's copy: experts.*.gate_proj.weight_scale
  ▼
model.layers.0.mlp.experts.gate_up_proj_scale             ← the target param
```

The FP8 attention key next to it:

```text
model.language_model.layers.0.self_attn.q_proj.weight_scale
  │ renamings    model's: drop "language_model"     FP8's: q_proj is an FP8 layer → weight_scale_inv
  │ converters   general (FP8): weight_scale_inv → weight
  ▼
model.layers.0.self_attn.q_proj.weight_scale_inv
```

Why renamings first: a converter matches leaf patterns, so it needs the final names. Running every renaming before
any converter means a converter never sees a checkpoint's spelling.

## 7. One instance per param, and its ops

```text
first key of model.layers.0.mlp.experts.gate_up_proj_scale
  instance = deepcopy(converter)                       the next 127 keys (64 experts × gate, up) collect into it
  hf_quantizer.update_weight_transform(instance, "model.layers.0.mlp.experts.gate_up_proj_scale")
     get_quantizer(name) → NVFP4 (module_quantizers, step 3)
     ops = [MergeModulelist, Concatenate] + [NVFP4Deserialize]
```

| Load | Ops | Why |
|---|---|---|
| pre-quantized | model's ops, then `deserialize` | stack the experts first, then put the result in the kernel's layout (swizzled `*_sf`) |
| `dequantize=True` | `dequantize`, then model's ops | dequantizing needs each expert's weight with its own scales, before they are stacked; the instance collects the weight and all its scales together |
| full precision | model's ops, then `quantize` | quantize the fused param, as the module expects it |

Why per instance: here the full param name is known for the first time, so the layer's quantizer is a lookup, and the
op is added only where it applies. The converter in the list stays generic.

## 8. Read, shard, convert, set

```text
each key   get_slice → with TP: only this rank's rows/columns (DtensorShardOperation) → dtype → device   (threaded)
           with EP: experts this rank doesn't own are skipped (tensor_idx)
instance   [MergeModulelist, Concatenate, NVFP4Deserialize] on the collected tensors
set_param  gate_up_proj_scale, gate_up_proj_sf (from the deserialize)   with TP: wrapped back into the DTensor
```

Why shard on read: no rank ever holds a full tensor. Every op therefore works on a shard: merges and concatenations
act along dims the placement keeps whole, and a quantizer's op only reshapes what it receives.

Why nothing after loading: everything above happened on tensors while loading. No quantizer walks the model afterwards.

## 9. Saving

```text
the same conversions, reversed: converters (Concatenate → split, MergeModulelist → unstack), then renamings
dequantize reverses to identity: a dequantized model saves as an unquantized one
```

Why: the conversions used to load are the ones that save, so a model saves in its checkpoint's format.

## Who knows what

| Component | Knows | Doesn't know |
|---|---|---|
| `WeightRenaming` | a key's name | layers, quantization |
| `WeightConverter` | how per-expert / split keys map to one param | quantization |
| the quantizer | its serialized names, its ops, its layers | the model's conversions |
| the plan | which layer goes to which quantizer | how any format loads |
| core loading | the order: rename → convert → instance → ops → read → set | which format |
