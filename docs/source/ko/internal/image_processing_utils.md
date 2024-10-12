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

# 이미지 프로세서를 위한 유틸리티 [[utilities-for-image-processors]]

이 페이지는 이미지 프로세서에서 사용되는 유틸리티 함수들을 나열하며, 주로 이미지를 처리하기 위한 함수 기반의 변환 작업들을 다룹니다.

이 함수들 대부분은 라이브러리의 이미지 프로세서 코드를 연구할 때만 유용합니다.

## 이미지 변환 [[transformers.image_transforms.center_crop]]

[[autodoc]] image_transforms.center_crop

[[autodoc]] image_transforms.center_to_corners_format

[[autodoc]] image_transforms.corners_to_center_format

[[autodoc]] image_transforms.id_to_rgb

[[autodoc]] image_transforms.normalize

[[autodoc]] image_transforms.pad

[[autodoc]] image_transforms.rgb_to_id

[[autodoc]] image_transforms.rescale

[[autodoc]] image_transforms.resize

[[autodoc]] image_transforms.to_pil_image

## ImageProcessingMixin [[transformers.ImageProcessingMixin]]

[[autodoc]] image_processing_utils.ImageProcessingMixin
