<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Generation

每个框架都在它们各自的 `GenerationMixin` 类中实现了文本生成的 `generate` 方法：

- PyTorch [`~generation.GenerationMixin.generate`] 在 [`~generation.GenerationMixin`] 中实现。
- TensorFlow [`~generation.TFGenerationMixin.generate`] 在 [`~generation.TFGenerationMixin`] 中实现。
- Flax/JAX [`~generation.FlaxGenerationMixin.generate`] 在 [`~generation.FlaxGenerationMixin`] 中实现。

无论您选择哪个框架，都可以使用 [`~generation.GenerationConfig`] 类实例对 generate 方法进行参数化。有关生成方法的控制参数的完整列表，请参阅此类。

要了解如何检查模型的生成配置、默认值是什么、如何临时更改参数以及如何创建和保存自定义生成配置，请参阅 [文本生成策略指南](../generation_strategies)。该指南还解释了如何使用相关功能，如token流。

## GenerationConfig

[[autodoc]] generation.GenerationConfig
	- from_pretrained
	- from_model_config
	- save_pretrained

## GenerationMixin

[[autodoc]] generation.GenerationMixin
	- generate
	- compute_transition_scores

## TFGenerationMixin

[[autodoc]] generation.TFGenerationMixin
	- generate
	- compute_transition_scores

## FlaxGenerationMixin

[[autodoc]] generation.FlaxGenerationMixin
	- generate
