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

＃ 構成

基本クラス [`PretrainedConfig`] は、設定をロード/保存するための一般的なメソッドを実装します。
ローカル ファイルまたはディレクトリから、またはライブラリ (ダウンロードされた) によって提供される事前トレーニング済みモデル構成から
HuggingFace の AWS S3 リポジトリから)。

各派生構成クラスはモデル固有の属性を実装します。すべての構成クラスに存在する共通の属性は次のとおりです。
`hidden_​​size`、`num_attention_heads`、および `num_hidden_​​layers`。テキスト モデルはさらに以下を実装します。
`vocab_size`。

## PretrainedConfig

[[autodoc]] PretrainedConfig
    - push_to_hub
    - all
