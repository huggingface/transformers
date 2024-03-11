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

# Data Collator

Data collators是一个对象，通过使用数据集元素列表作为输入来形成一个批次。这些元素与 `train_dataset` 或 `eval_dataset` 的元素类型相同。

为了能够构建批次，Data collators可能会应用一些预处理（比如填充）。其中一些（比如[`DataCollatorForLanguageModeling`]）还会在形成的批次上应用一些随机数据增强（比如随机掩码）。

在[示例脚本](../examples)或[示例notebooks](../notebooks)中可以找到使用的示例。


## Default data collator

[[autodoc]] data.data_collator.default_data_collator

## DefaultDataCollator

[[autodoc]] data.data_collator.DefaultDataCollator

## DataCollatorWithPadding

[[autodoc]] data.data_collator.DataCollatorWithPadding

## DataCollatorForTokenClassification

[[autodoc]] data.data_collator.DataCollatorForTokenClassification

## DataCollatorForSeq2Seq

[[autodoc]] data.data_collator.DataCollatorForSeq2Seq

## DataCollatorForLanguageModeling

[[autodoc]] data.data_collator.DataCollatorForLanguageModeling
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens

## DataCollatorForWholeWordMask

[[autodoc]] data.data_collator.DataCollatorForWholeWordMask
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens

## DataCollatorForPermutationLanguageModeling

[[autodoc]] data.data_collator.DataCollatorForPermutationLanguageModeling
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens
