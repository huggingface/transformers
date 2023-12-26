<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Feature Extractor

Feature Extractor负责为音频或视觉模型准备输入特征。这包括从序列中提取特征，例如，对音频文件进行预处理以生成Log-Mel频谱特征，以及从图像中提取特征，例如，裁剪图像文件，同时还包括填充、归一化和转换为NumPy、PyTorch和TensorFlow张量。


## FeatureExtractionMixin

[[autodoc]] feature_extraction_utils.FeatureExtractionMixin
    - from_pretrained
    - save_pretrained

## SequenceFeatureExtractor

[[autodoc]] SequenceFeatureExtractor
    - pad

## BatchFeature

[[autodoc]] BatchFeature

## ImageFeatureExtractionMixin

[[autodoc]] image_utils.ImageFeatureExtractionMixin
