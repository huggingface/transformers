<!--Copyright 2023 The HuggingFace Team. All rights reserved.

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


フィーチャーエクストラクタは、オーディオまたはビジョンモデルのための入力フィーチャーの準備を担当しています。これには、シーケンスからのフィーチャー抽出（例：オーディオファイルの前処理からLog-Melスペクトログラムフィーチャーへの変換）、画像からのフィーチャー抽出（例：画像ファイルのクロッピング）、またパディング、正規化、そしてNumpy、PyTorch、TensorFlowテンソルへの変換も含まれます。


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
