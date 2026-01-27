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

# 時系列ユーティリティ

このページには、時系列ベースのモデルに使用できるすべてのユーティリティ関数とクラスがリストされます。

これらのほとんどは、時系列モデルのコードを研究している場合、または分散出力クラスのコレクションに追加したい場合にのみ役立ちます。

## Distributional Output

[[autodoc]] time_series_utils.NormalOutput

[[autodoc]] time_series_utils.StudentTOutput

[[autodoc]] time_series_utils.NegativeBinomialOutput