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

# Tokenizers的工具

并保留格式：此页面列出了tokenizers使用的所有实用函数，主要是类
[`~tokenization_utils_base.PreTrained TokenizerBase`] 实现了常用方法之间的
[`PreTrained Tokenizer`] 和 [`PreTrained TokenizerFast`] 以及混合类
[`~tokenization_utils_base.SpecialTokens Mixin`]。

其中大多数只有在您研究库中tokenizers的代码时才有用。


## PreTrainedTokenizerBase

[[autodoc]] tokenization_utils_base.PreTrainedTokenizerBase
    - __call__
    - all

## SpecialTokensMixin

[[autodoc]] tokenization_utils_base.SpecialTokensMixin

## Enums和namedtuples(命名元组)

[[autodoc]] tokenization_utils_base.TruncationStrategy

[[autodoc]] tokenization_utils_base.CharSpan

[[autodoc]] tokenization_utils_base.TokenSpan
