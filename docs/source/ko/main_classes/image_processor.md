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

# 이미지 프로세서 [[image-processor]]

이미지 프로세서는 비전 모델의 입력 특성을 준비하고 출력값을 후처리하는 역할을 합니다. 여기에는 크기 조정, 정규화, PyTorch, TensorFlow, Flax, Numpy 텐서로의 변환과 같은 변환 작업이 포함됩니다. 또한, 로짓(logits)을 세그멘테이션 마스크로 변환하는 등 모델별 후처리가 포함될 수 있습니다.


## ImageProcessingMixin [[transformers.ImageProcessingMixin]]

[[autodoc]] image_processing_utils.ImageProcessingMixin
    - from_pretrained
    - save_pretrained

## BatchFeature [[transformers.BatchFeature]]

[[autodoc]] BatchFeature

## BaseImageProcessor [[transformers.BaseImageProcessor]]

[[autodoc]] image_processing_utils.BaseImageProcessor


## BaseImageProcessorFast [[transformers.BaseImageProcessorFast]]

[[autodoc]] image_processing_utils_fast.BaseImageProcessorFast
