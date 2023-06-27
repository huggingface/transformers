<!--版权所有2021年The HuggingFace团队。保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；在符合许可证的情况下，您不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以获取特定语言下的权限和限制。⚠️请注意，此文件是 Markdown 格式，但包含特定于我们的文档生成器（类似于 MDX）的语法，可能无法在您的 Markdown 查看器中正确显示。
。-->

# BARTpho## 概述

BARTpho 模型是由 Nguyen Luong Tran、Duong Minh Le 和 Dat Quoc Nguyen 在《BARTpho：用于越南语的预训练序列到序列模型》中提出的。

论文摘要如下：
*我们提出了两个版本的 BARTpho——BARTpho_word 和 BARTpho_syllable——这是首个用于越南语的大规模单语序列到序列模型。
我们的 BARTpho 使用了 BART 的“large”架构和预训练方案，因此特别适用于生成式自然语言处理任务。越南语文本摘要的下游任务
的实验表明，在自动评估和人工评估方面，我们的 BARTpho 优于强基线 mBART，并改进了最先进技术。我们发布 BARTpho 以促进未来的研究和生成式越南语自然语言处理应用。


```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bartpho = AutoModel.from_pretrained("vinai/bartpho-syllable")

>>> tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

>>> line = "Chúng tôi là những nghiên cứu viên."

>>> input_ids = tokenizer(line, return_tensors="pt")

>>> with torch.no_grad():
...     features = bartpho(**input_ids)  # Models outputs are now tuples

>>> # With TensorFlow 2.0+:
>>> from transformers import TFAutoModel

>>> bartpho = TFAutoModel.from_pretrained("vinai/bartpho-syllable")
>>> input_ids = tokenizer(line, return_tensors="tf")
>>> features = bartpho(**input_ids)
```

提示：

- BARTpho 遵循 mBART 的做法，在 BART 文档中使用的示例，在适应 BARTpho 时，应通过将 BART 专用类替换为 mBART  专用的对应项进行调整。

例如：


```python
>>> from transformers import MBartForConditionalGeneration

>>> bartpho = MBartForConditionalGeneration.from_pretrained("vinai/bartpho-syllable")
>>> TXT = "Chúng tôi là <mask> nghiên cứu viên."
>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
>>> logits = bartpho(input_ids).logits
>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
>>> probs = logits[0, masked_index].softmax(dim=0)
>>> values, predictions = probs.topk(5)
>>> print(tokenizer.decode(predictions).split())
```

- 此实现仅用于标记化："monolingual_vocab_file" 由预训练的 SentencePiece 模型 "vocab_file" 提取的越南语专用类型组成，  该模型可从多语言 XLM-RoBERTa 获取。如果使用此预训练多语言 SentencePiece 模型 "vocab_file" 进行子词分割的其他语言  可以使用自己语言专用的 "monolingual_vocab_file" 重用 BartphoTokenizer。 

此模型由 [dqnguyen](https://huggingface.co/dqnguyen) 贡献。原始代码可在 [此处](https://github.com/VinAIResearch/BARTpho) 找到。

## BartphoTokenizer

[[autodoc]] BartphoTokenizer
