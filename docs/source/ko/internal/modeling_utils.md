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

# 사용자 정의 레이어 및 유틸리티 [[custom-layers-and-utilities]]

이 페이지는 라이브러리에서 사용되는 사용자 정의 레이어와 모델링을 위한 유틸리티 함수들을 나열합니다.

이 함수들 대부분은 라이브러리 내의 모델 코드를 연구할 때만 유용합니다.


## PyTorch 사용자 정의 모듈 [[transformers.Conv1D]]

[[autodoc]] pytorch_utils.Conv1D

[[autodoc]] modeling_utils.PoolerStartLogits
   - forward

[[autodoc]] modeling_utils.PoolerEndLogits
   - forward

[[autodoc]] modeling_utils.PoolerAnswerClass
   - forward

[[autodoc]] modeling_utils.SquadHeadOutput

[[autodoc]] modeling_utils.SQuADHead
   - forward

[[autodoc]] modeling_utils.SequenceSummary
   - forward

## PyTorch 헬퍼(helper) 함수 [[transformers.apply_chunking_to_forward]]

[[autodoc]] pytorch_utils.apply_chunking_to_forward

[[autodoc]] pytorch_utils.find_pruneable_heads_and_indices

[[autodoc]] pytorch_utils.prune_layer

[[autodoc]] pytorch_utils.prune_conv1d_layer

[[autodoc]] pytorch_utils.prune_linear_layer

## TensorFlow 사용자 정의 레이어 [[transformers.modeling_tf_utils.TFConv1D]]

[[autodoc]] modeling_tf_utils.TFConv1D

[[autodoc]] modeling_tf_utils.TFSequenceSummary

## TensorFlow 손실 함수 [[transformers.modeling_tf_utils.TFCausalLanguageModelingLoss]]

[[autodoc]] modeling_tf_utils.TFCausalLanguageModelingLoss

[[autodoc]] modeling_tf_utils.TFMaskedLanguageModelingLoss

[[autodoc]] modeling_tf_utils.TFMultipleChoiceLoss

[[autodoc]] modeling_tf_utils.TFQuestionAnsweringLoss

[[autodoc]] modeling_tf_utils.TFSequenceClassificationLoss

[[autodoc]] modeling_tf_utils.TFTokenClassificationLoss

## TensorFlow 도우미 함수 [[transformers.modeling_tf_utils.get_initializer]]

[[autodoc]] modeling_tf_utils.get_initializer

[[autodoc]] modeling_tf_utils.keras_serializable

[[autodoc]] modeling_tf_utils.shape_list
