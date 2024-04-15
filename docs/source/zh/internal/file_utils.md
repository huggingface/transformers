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

# 通用工具

此页面列出了在`utils.py`文件中找到的所有Transformers通用实用函数。

其中大多数仅在您研究库中的通用代码时才有用。


## Enums和namedtuples(命名元组)

[[autodoc]] utils.ExplicitEnum

[[autodoc]] utils.PaddingStrategy

[[autodoc]] utils.TensorType

## 特殊的装饰函数

[[autodoc]] utils.add_start_docstrings

[[autodoc]] utils.add_start_docstrings_to_model_forward

[[autodoc]] utils.add_end_docstrings

[[autodoc]] utils.add_code_sample_docstrings

[[autodoc]] utils.replace_return_docstrings

## 特殊的属性

[[autodoc]] utils.cached_property

## 其他实用程序

[[autodoc]] utils._LazyModule
