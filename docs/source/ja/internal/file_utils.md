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

# 一般的なユーティリティ

このページには、ファイル `utils.py` にある Transformers の一般的なユーティリティ関数がすべてリストされています。

これらのほとんどは、ライブラリで一般的なコードを学習する場合にのみ役に立ちます。

## 列挙型と名前付きタプル

[[autodoc]] utils.ExplicitEnum

[[autodoc]] utils.PaddingStrategy

[[autodoc]] utils.TensorType

## 特別なデコレーター

[[autodoc]] utils.add_start_docstrings

[[autodoc]] utils.add_start_docstrings_to_model_forward

[[autodoc]] utils.add_end_docstrings

[[autodoc]] utils.add_code_sample_docstrings

[[autodoc]] utils.replace_return_docstrings

## 特殊なプロパティ

[[autodoc]] utils.cached_property

## その他のユーティリティ

[[autodoc]] utils._LazyModule
