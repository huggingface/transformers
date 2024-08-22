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

Each framework has a generate method for text generation implemented in their respective `GenerationMixin` class:

- PyTorch [`~generation.GenerationMixin.generate`] is implemented in [`~generation.GenerationMixin`].
- TensorFlow [`~generation.TFGenerationMixin.generate`] is implemented in [`~generation.TFGenerationMixin`].
- Flax/JAX [`~generation.FlaxGenerationMixin.generate`] is implemented in [`~generation.FlaxGenerationMixin`].

Regardless of your framework of choice, you can parameterize the generate method with a [`~generation.GenerationConfig`]
class instance. Please refer to this class for the complete list of generation parameters, which control the behavior
of the generation method.

To learn how to inspect a model's generation configuration, what are the defaults, how to change the parameters ad hoc,
and how to create and save a customized generation configuration, refer to the
[text generation strategies guide](../generation_strategies). The guide also explains how to use related features,
like token streaming.

## GenerationConfig

[[autodoc]] generation.GenerationConfig
	- from_pretrained
	- from_model_config
	- save_pretrained
	- update
	- validate
	- get_generation_mode

[[autodoc]] generation.WatermarkingConfig

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
