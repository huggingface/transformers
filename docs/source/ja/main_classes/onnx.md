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

# Exporting 🤗 Transformers models to ONNX

🤗 Transformers は `transformers.onnx` パッケージを提供します。
設定オブジェクトを利用することで、モデルのチェックポイントをONNXグラフに変換することができます。

詳細は[ガイド](../serialization) を参照してください。
を参照してください。

## ONNX Configurations

以下の3つの抽象クラスを提供しています。
エクスポートしたいモデルアーキテクチャのタイプに応じて、継承すべき3つの抽象クラスを提供します：

* エンコーダーベースのモデルは [`~onnx.config.OnnxConfig`] を継承します。
* デコーダーベースのモデルは [`~onnx.config.OnnxConfigWithPast`] を継承します。
* エンコーダー・デコーダーモデルは [`~onnx.config.OnnxSeq2SeqConfigWithPast`] を継承しています。


### OnnxConfig

[[autodoc]] onnx.config.OnnxConfig

### OnnxConfigWithPast

[[autodoc]] onnx.config.OnnxConfigWithPast

### OnnxSeq2SeqConfigWithPast

[[autodoc]] onnx.config.OnnxSeq2SeqConfigWithPast

## ONNX Features

各 ONNX 構成は、次のことを可能にする一連の _機能_ に関連付けられています。
さまざまなタイプのトポロジまたはタスクのモデルをエクスポートします。

### FeaturesManager

[[autodoc]] onnx.features.FeaturesManager

