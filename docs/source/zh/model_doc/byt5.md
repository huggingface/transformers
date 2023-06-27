<!--版权所有 2021 年 The HuggingFace 团队。保留所有权利。
根据 Apache License，Version 2.0（“许可证”）授权；除非符合许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”分发的基础，不附带任何形式的保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式，但包含特定于我们文档构建器（类似于 MDX）的语法，可能无法在您的 Markdown 查看器中正确呈现。
-->
# ByT5

## 概述

ByT5 模型在 [Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir 发布的[ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626) 中提出。Kale, Adam Roberts, Colin Raffel.

该论文的摘要如下：

*大多数广泛使用的预训练语言模型操作于对应于单词或子单词单位的标记序列。将文本编码为标记序列需要一个分词器 (Tokenizer)，通常将其作为与模型独立的工具创建。直接在原始文本（字节或字符）上操作的无标记模型具有许多优点：它们可以直接处理任何语言的文本，它们对噪声更具鲁棒性，并且通过消除复杂且容易出错的文本预处理流程来最小化技术债务。由于字节或字符序列比标记序列更长，过去关于无标记模型的工作通常引入了新的模型架构，以分摊直接在原始文本上操作的成本。在本文中，我们展示了标准 Transformer 架构可以通过最小修改用于处理字节序列。我们仔细研究了参数数量，训练 FLOPs 和推理速度方面的权衡，并表明字节级模型与其标记级对应模型相比具有竞争力。我们还证明了字节级模型对噪声更具鲁棒性，并在对拼写和发音敏感的任务上表现更好。作为我们的贡献的一部分，我们发布了一组基于 T5 架构的新的预训练字节级 Transformer 模型，以及我们在实验中使用的所有代码和数据。* 

此模型由 [patrickvonplaten](https://huggingface.co/patrickvonplaten) 贡献。原始代码可以在此处找到 [here](https://github.com/google-research/byt5)。

ByT5 的架构基于 T5v1.1 模型，因此可以参考 [T5v1.1 的文档页面](t5v1.1)。它们只在如何为模型准备输入方面有所不同，请参阅以下代码示例。

由于 ByT5 是无监督预训练的，在单任务微调过程中使用任务前缀没有真正的优势。如果进行多任务微调，应使用前缀。

### 示例

ByT5 可直接处理原始 UTF-8 字节，因此可以在不使用分词器 (Tokenizer)的情况下使用：

```python
>>> from transformers import T5ForConditionalGeneration
>>> import torch

>>> model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

>>> num_special_tokens = 3
>>> # Model has 3 special tokens which take up the input ids 0,1,2 of ByT5.
>>> # => Need to shift utf-8 character encodings by 3 before passing ids to model.

>>> input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens

>>> labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens

>>> loss = model(input_ids, labels=labels).loss
>>> loss.item()
2.66
```

但是，对于批量推理和训练，建议使用分词器 (Tokenizer)：
```python
>>> from transformers import T5ForConditionalGeneration, AutoTokenizer

>>> model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

>>> model_inputs = tokenizer(
...     ["Life is like a box of chocolates.", "Today is Monday."], padding="longest", return_tensors="pt"
... )
>>> labels_dict = tokenizer(
...     ["La vie est comme une boîte de chocolat.", "Aujourd'hui c'est lundi."], padding="longest", return_tensors="pt"
... )
>>> labels = labels_dict.input_ids

>>> loss = model(**model_inputs, labels=labels).loss
>>> loss.item()
17.9
```

与 [T5](t5) 类似，ByT5 也是在遮盖噪声的任务上进行训练的。然而，由于模型直接处理字符，因此预训练任务有所不同。让我们破坏一些字符 `"The dog chases a ball in the park."` 的输入句子，并要求 ByT5 预测它们给我们。
```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-base")

>>> input_ids_prompt = "The dog chases a ball in the park."
>>> input_ids = tokenizer(input_ids_prompt).input_ids

>>> # Note that we cannot add "{extra_id_...}" to the string directly
>>> # as the Byte tokenizer would incorrectly merge the tokens
>>> # For ByT5, we need to work directly on the character level
>>> # Contrary to T5, ByT5 does not use sentinel tokens for masking, but instead
>>> # uses final utf character ids.
>>> # UTF-8 is represented by 8 bits and ByT5 has 3 special tokens.
>>> # => There are 2**8+2 = 259 input ids and mask tokens count down from index 258.
>>> # => mask to "The dog [258]a ball [257]park."

>>> input_ids = torch.tensor([input_ids[:8] + [258] + input_ids[14:21] + [257] + input_ids[28:]])
>>> input_ids
tensor([[ 87, 107, 104,  35, 103, 114, 106,  35, 258,  35, 100,  35, 101, 100, 111, 111, 257,  35, 115, 100, 117, 110,  49,   1]])

>>> # ByT5 produces only one char at a time so we need to produce many more output characters here -> set `max_length=100`.
>>> output_ids = model.generate(input_ids, max_length=100)[0].tolist()
>>> output_ids
[0, 258, 108, 118,  35, 119, 107, 104,  35, 114, 113, 104,  35, 122, 107, 114,  35, 103, 114, 104, 118, 257,  35, 108, 113,  35, 119, 107, 104,  35, 103, 108, 118, 102, 114, 256, 108, 113,  35, 119, 107, 104, 35, 115, 100, 117, 110,  49,  35,  87, 107, 104,  35, 103, 114, 106, 35, 108, 118,  35, 119, 107, 104,  35, 114, 113, 104,  35, 122, 107, 114,  35, 103, 114, 104, 118,  35, 100,  35, 101, 100, 111, 111,  35, 108, 113, 255,  35, 108, 113,  35, 119, 107, 104,  35, 115, 100, 117, 110,  49]

>>> # ^- Note how 258 descends to 257, 256, 255

>>> # Now we need to split on the sentinel tokens, let's write a short loop for this
>>> output_ids_list = []
>>> start_token = 0
>>> sentinel_token = 258
>>> while sentinel_token in output_ids:
...     split_idx = output_ids.index(sentinel_token)
...     output_ids_list.append(output_ids[start_token:split_idx])
...     start_token = split_idx
...     sentinel_token -= 1

>>> output_ids_list.append(output_ids[start_token:])
>>> output_string = tokenizer.batch_decode(output_ids_list)
>>> output_string
['<pad>', 'is the one who does', ' in the disco', 'in the park. The dog is the one who does a ball in', ' in the park.']
```


## ByT5Tokenizer
[[autodoc]] ByT5Tokenizer

有关所有详细信息，请参阅 [`ByT5Tokenizer`]。