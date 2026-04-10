<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Fusion mapping (experimental feature)

Fusion mapping provides an opt-in way to replace model submodules at load time while preserving the original checkpoint format.

It builds on:

- [Monkey patching](./monkey_patching) to swap module classes before model instantiation.
- [Dynamic weight loading](./weightconverter) to map weights between the original and fused runtime layouts.

> [!WARNING]
> Fusion mapping is an experimental loading feature. It changes the runtime module structure and may affect model behavior. Use it only when you explicitly want a fused runtime layout.

## Quick start

Fusion is enabled through [`~PreTrainedModel.from_pretrained`] with `fusion_config`:

```python
from transformers import AutoModelForImageTextToText


model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    fusion_config={"patch_embeddings": True},
)
```

By default, no fusion is applied.
If `fusion_config` is stored in the model config, `from_pretrained()` will reuse it automatically.

## How it works

Fusion registration happens before the model is instantiated:

1. [`~PreTrainedModel.from_pretrained`] uses the explicit `fusion_config` argument or falls back to `config.fusion_config`.
2. The fusion registry validates the requested fusion names.
3. Each enabled fusion meta-initializes the target model class, optionally filters candidate modules by name, and uses `is_fusable(...)` to discover compatible module classes.
4. Fused replacement classes are registered through [`~transformers.monkey_patching.register_patch_mapping`].
5. Matching [`~WeightTransform`] rules are generated from the config so checkpoint loading can map weights into the fused runtime layout.
6. By default, [`~PreTrainedModel.save_pretrained`] uses the reverse conversion path to restore the original checkpoint layout. Pass `save_original_format=False` to keep the converted runtime layout instead.

This lets a fusion use a different runtime module structure while still loading from the original checkpoint format, and by default saving back to it as well.

Note: With the current monkey-patching mechanism, fusion registration is class-level: one compatible module class maps to one fused replacement class.

## Current fusion families

Currently, `fusion_config` supports one fusion family:

- `patch_embeddings`
  Enable with:

  ```python
  fusion_config = {"patch_embeddings": True}
  ```

  Effect:
  Replaces compatible `nn.Conv3d` patch embedding projections with equivalent flattened `nn.Linear` projections at runtime.

## Extending fusion mapping

To add a new fusion family:

1. Add an `is_fusable` predicate.
   This decides whether a discovered module is compatible with the fusion.
2. Optionally add `target_modules_patterns`.
   This makes the discovery step more explicit by pre-filtering candidate module names before `is_fusable(...)`.
3. Add a `make_fused_class` factory.
   This returns the runtime replacement class for a compatible module class.
4. Add a `make_transforms` factory if the fused layout needs checkpoint conversion.
   This returns the [`~WeightTransform`] rules that map weights between the original and fused layouts for a given config.
5. Register the new `ModuleFusionSpec` in [`fusion_mapping.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/fusion_mapping.py).

Once registered, the new fusion becomes available through `fusion_config`.

## Internal API

[[autodoc]] fusion_mapping.ModuleFusionSpec

[[autodoc]] fusion_mapping.PatchEmbeddingsFusionSpec

[[autodoc]] fusion_mapping._register_module_fusion

[[autodoc]] fusion_mapping.register_fusion_patches
