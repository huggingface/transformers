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

# 생성 [[generation]]

각 프레임워크에는 해당하는 `GenerationMixin` 클래스에서 구현된 텍스트 생성을 위한 generate 메소드가 있습니다:

- PyTorch에서는 [`~generation.GenerationMixin.generate`]가 [`~generation.GenerationMixin`]에 구현되어 있습니다.
- TensorFlow에서는 [`~generation.TFGenerationMixin.generate`]가 [`~generation.TFGenerationMixin`]에 구현되어 있습니다.
- Flax/JAX에서는 [`~generation.FlaxGenerationMixin.generate`]가 [`~generation.FlaxGenerationMixin`]에 구현되어 있습니다.

사용하는 프레임워크에 상관없이, generate 메소드는 [`~generation.GenerationConfig`] 클래스 인스턴스로 매개변수화 할 수 있습니다. generate 메소드의 동작을 제어하는 모든 생성 매개변수 목록을 확인하려면 이 클래스를 참조하세요. 

모델의 생성 설정을 어떻게 확인하고, 기본값이 무엇인지, 매개변수를 어떻게 임시로 변경하는지, 그리고 사용자 지정 생성 설정을 만들고 저장하는 방법을 배우려면 [텍스트 생성 전략 가이드](../generation_strategies)를 참조하세요. 이 가이드는 토큰 스트리밍과 같은 관련 기능을 사용하는 방법도 설명합니다.

## GenerationConfig [[transformers.GenerationConfig]]

[[autodoc]] generation.GenerationConfig
	- from_pretrained
	- from_model_config
	- save_pretrained
	- update
	- validate
	- get_generation_mode

[[autodoc]] generation.WatermarkingConfig

## GenerationMixin [[transformers.GenerationMixin]]

[[autodoc]] generation.GenerationMixin
	- generate
	- compute_transition_scores

## TFGenerationMixin [[transformers.TFGenerationMixin]]

[[autodoc]] generation.TFGenerationMixin
	- generate
	- compute_transition_scores

## FlaxGenerationMixin [[transformers.FlaxGenerationMixin]]

[[autodoc]] generation.FlaxGenerationMixin
	- generate
