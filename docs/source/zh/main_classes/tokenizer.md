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

# Tokenizer

tokenizer负责准备输入以供模型使用。该库包含所有模型的tokenizer。大多数tokenizer都有两种版本：一个是完全的 Python 实现，另一个是基于 Rust 库 [🤗 Tokenizers](https://github.com/huggingface/tokenizers) 的“Fast”实现。"Fast" 实现允许：

1. 在批量分词时显著提速
2. 在原始字符串（字符和单词）和token空间之间进行映射的其他方法（例如，获取包含给定字符的token的索引或与给定token对应的字符范围）。

基类 [PreTrainedTokenizer] 和 [PreTrained TokenizerFast] 实现了在模型输入中编码字符串输入的常用方法（见下文），并从本地文件或目录或从库提供的预训练的 tokenizer（从 HuggingFace 的 AWS S3 存储库下载）实例化/保存 python 和“Fast” tokenizer。它们都依赖于包含常用方法的 [`~tokenization_utils_base.PreTrainedTokenizerBase`]和[`~tokenization_utils_base.SpecialTokensMixin`]。

因此，[`PreTrainedTokenizer`] 和 [`PreTrainedTokenizerFast`] 实现了使用所有tokenizers的主要方法：

- 分词（将字符串拆分为子词标记字符串），将tokens字符串转换为id并转换回来，以及编码/解码（即标记化并转换为整数）。
- 以独立于底层结构（BPE、SentencePiece……）的方式向词汇表中添加新tokens。
- 管理特殊tokens（如mask、句首等）：添加它们，将它们分配给tokenizer中的属性以便于访问，并确保它们在标记过程中不会被分割。

[`BatchEncoding`] 包含 [`~tokenization_utils_base.PreTrainedTokenizerBase`] 的编码方法（`__call__`、`encode_plus` 和 `batch_encode_plus`）的输出，并且是从 Python 字典派生的。当tokenizer是纯 Python tokenizer时，此类的行为就像标准的 Python 字典一样，并保存这些方法计算的各种模型输入（`input_ids`、`attention_mask` 等）。当分词器是“Fast”分词器时（即由 HuggingFace 的 [tokenizers 库](https://github.com/huggingface/tokenizers) 支持），此类还提供了几种高级对齐方法，可用于在原始字符串（字符和单词）与token空间之间进行映射（例如，获取包含给定字符的token的索引或与给定token对应的字符范围）。


## PreTrainedTokenizer

[[autodoc]] PreTrainedTokenizer
    - __call__
    - add_tokens
    - add_special_tokens
    - apply_chat_template
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## PreTrainedTokenizerFast

[`PreTrainedTokenizerFast`] 依赖于 [tokenizers](https://huggingface.co/docs/tokenizers) 库。可以非常简单地将从 🤗 tokenizers 库获取的tokenizers加载到 🤗 transformers 中。查看 [使用 🤗 tokenizers 的分词器](../fast_tokenizers) 页面以了解如何执行此操作。

[[autodoc]] PreTrainedTokenizerFast
    - __call__
    - add_tokens
    - add_special_tokens
    - apply_chat_template
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## BatchEncoding

[[autodoc]] BatchEncoding
