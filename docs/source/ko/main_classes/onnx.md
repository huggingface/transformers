<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 🤗 Transformers 모델을 ONNX로 내보내기[[exporting--transformers-models-to-onnx]]

🤗 트랜스포머는 `transformers.onnx` 패키지를 제공하며, 이 패키지는 설정 객체를 활용하여 모델 체크포인트를 ONNX 그래프로 변환할 수 있게 합니다.

🤗 Transformers에 대한 자세한 내용은 [이 가이드](../serialization)를 참조하세요.

## ONNX 설정[[onnx-configurations]]

내보내려는(export) 모델 아키텍처의 유형에 따라 상속받아야 할 세 가지 추상 클래스를 제공합니다:

* 인코더 기반 모델은 [`~onnx.config.OnnxConfig`]을 상속받습니다.
* 디코더 기반 모델은 [`~onnx.config.OnnxConfigWithPast`]을 상속받습니다.
* 인코더-디코더 기반 모델은 [`~onnx.config.OnnxSeq2SeqConfigWithPast`]을 상속받습니다.

### OnnxConfig[[transformers.onnx.OnnxConfig]]

[[autodoc]] onnx.config.OnnxConfig

### OnnxConfigWithPast[[transformers.onnx.OnnxConfigWithPast]]

[[autodoc]] onnx.config.OnnxConfigWithPast

### OnnxSeq2SeqConfigWithPast[[OnnxSeq2SeqConfigWithPast]]

[[autodoc]] onnx.config.OnnxSeq2SeqConfigWithPast

## ONNX 특징[[onnx-features]]

각 ONNX 설정은 다양한 유형의 토폴로지나 작업에 대해 모델을 내보낼 수 있게(exporting) 해주는 _features_ 세트와 연관되어 있습니다.

### FeaturesManager[[transformers.onnx.FeaturesManager]]

[[autodoc]] onnx.features.FeaturesManager

