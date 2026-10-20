<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Fine-grained FP8 (deprecated)

> [!WARNING]
> `FineGrainedFP8Config` is deprecated and will be removed in a future release. Existing code keeps working: a
> `FineGrainedFP8Config` passed to `from_pretrained` loads through its own frozen integration. A checkpoint declaring
> `quant_method="fp8"`, loaded without one, loads as [`Fp8Config`].

New code uses the config that describes the checkpoint's format:

| `FineGrainedFP8Config` | Use | Guide |
|---|---|---|
| `weight_block_size=[1, 32]` with `scale_fmt="ue8m0"` | [`Mxfp8Config`] | [MXFP8](./mxfp8) |
| any other `weight_block_size` | [`Fp8Config`] | [FP8](./fp8) |
| on a model whose config declares `expert_dtype="fp4"` (DeepSeek-V4) | [`Fp8Config`], which loads as a [`FineGrainedConfig`] of MXFP4 experts over block FP8 | [Fine-grained](./finegrained) |

The fields keep their names and meanings, so `FineGrainedFP8Config(weight_block_size=(128, 128),
activation_scheme="static")` becomes `Fp8Config(weight_block_size=(128, 128), activation_scheme="static")`.
