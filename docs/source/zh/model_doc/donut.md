<!--版权所有 2022 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得的许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的明示或暗示保证。有关许可证下的特定语言管理权限和限制的详细信息，请参阅许可证。
⚠️请注意，此文件是 Markdown 格式的，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确显示。渲染。
特定语言管理权限和限制的详细信息，请参阅许可证。-->
# Donut

## 概述
Donut 模型是由 Geewook Kim，Teakgyu Hong，Moonbin Yim，Jeongyeon Nam，Jinyoung Park，Jinyeong Yim，Wonseok Hwang，Sangdoo Yun，Dongyoon Han，Seunghyun Park 于 [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) 中提出的。

Donut 由图像 Transformer 编码器和自回归文本 Transformer 解码器组成，用于执行文档理解任务，如文档图像分类，表单理解和视觉问答。论文中的摘要如下：*理解文档图像（例如发票）是一项核心但具有挑战性的任务，因为它需要复杂的功能，例如读取文本和对文档的整体理解。当前的视觉文档理解（VDU）方法将读取文本的任务外包给现成的光学字符识别（OCR）引擎，并侧重于使用 OCR 输出的理解任务。尽管这些基于 OCR 的方法已经显示出有希望的性能，但它们存在以下问题：1）使用 OCR 的计算成本很高；2）OCR 模型在语言或文档类型上的灵活性有限；3）OCR 错误传播至后续过程。为了解决这些问题，在本文中，我们介绍了一种名为 Donut 的新颖的无 OCR VDU 模型，代表文档理解变压器。作为无 OCR VDU 研究的第一步，我们提出了一种简单的架构（即变压器），并具有预训练目标（即交叉熵损失）。Donut 在概念上简单而有效。通过大量的实验证明和分析，我们展示了一种简单的无 OCR VDU 模型 Donut，在各种 VDU 任务的速度和准确性方面实现了最先进的性能。此外，我们提供了一个合成数据生成器，帮助模型预训练在各种语言和领域中具有灵活性。*

论文中的摘要如下：

< img src =" https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/donut_architecture.jpg "
< img src =" https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/donut_architecture.jpg "
alt = "drawing" width = "600"/>

<small> Donut 总体概述。摘自 <a href="https://arxiv.org/abs/2111.15664"> 原始论文 </a>。</small>

该模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码可以在此处找到 [here](https://github.com/clovaai/donut)。

提示：

- 使用 Donut 最快速的方法是查看 [教程  notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut)，展示了如何在推断时使用模型  以及在自定义数据上进行微调。- Donut 始终在 [VisionEncoderDecoder](vision-encoder-decoder) 框架中使用。

## 推断

Donut 的 [`VisionEncoderDecoder`] 模型接受图像作为输入，并利用 [`~generation.GenerationMixin.generate`] 以自回归方式生成给定输入图像的文本。
[`DonutFeatureExtractor`] 类负责对输入图像进行预处理，并 [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`] 将生成的目标标记解码为目标字符串。[`DonutProcessor`] 将 [`DonutFeatureExtractor`] 和 [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`] 组合成一个实例，既提取输入特征，又解码预测的标记 ID。
- 逐步进行文档图像分类
```py
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # load document image
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[1]["image"]

>>> # prepare decoder inputs
>>> task_prompt = "<s_rvlcdip>"
>>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     early_stopping=True,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     num_beams=1,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
>>> print(processor.token2json(sequence))
{'class': 'advertisement'}
```

- 逐步进行文档解析
```py
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # load document image
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[2]["image"]

>>> # prepare decoder inputs
>>> task_prompt = "<s_cord-v2>"
>>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     early_stopping=True,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     num_beams=1,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
>>> print(processor.token2json(sequence))
{'menu': {'nm': 'CINNAMON SUGAR', 'unitprice': '17,000', 'cnt': '1 x', 'price': '17,000'}, 'sub_total': {'subtotal_price': '17,000'}, 'total': {'total_price': '17,000', 'cashprice': '20,000', 'changeprice': '3,000'}}
```

- 逐步进行文档视觉问答（DocVQA）
```py
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # load document image from the DocVQA dataset
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[0]["image"]

>>> # prepare decoder inputs
>>> task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
>>> question = "When is the coffee break?"
>>> prompt = task_prompt.replace("{user_input}", question)
>>> decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     early_stopping=True,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     num_beams=1,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
>>> print(processor.token2json(sequence))
{'question': 'When is the coffee break?', 'answer': '11-14 to 11:39 a.m.'}
```

请查看 [模型中心](https://huggingface.co/models?filter=donut) 以查找 Donut 检查点。

## 训练

请参阅 [教程笔记本](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut)。

## DonutSwinConfig

[[autodoc]] DonutSwinConfig

## DonutImageProcessor

[[autodoc]] DonutImageProcessor
    - preprocess

## DonutFeatureExtractor

[[autodoc]] DonutFeatureExtractor
    - __call__

## DonutProcessor

[[autodoc]] DonutProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## DonutSwinModel

[[autodoc]] DonutSwinModel
    - forward
