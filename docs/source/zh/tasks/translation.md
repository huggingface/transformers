<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 翻译

[[open-in-colab]]

<Youtube id="1JvfrvZgi6c"/>

翻译将一种语言的文本序列转换为另一种语言。它是可以表述为序列到序列问题的几个任务之一——这是一种从输入返回某些输出的强大框架，适用于翻译或摘要等任务。翻译系统通常用于不同语言文本之间的转换，但也可以用于语音，或者文本转语音、语音转文本等组合场景。

本指南将向您展示如何：

1. 在 [OPUS Books](https://huggingface.co/datasets/opus_books) 数据集的英法子集上微调 [T5](https://huggingface.co/google-t5/t5-small)，将英文文本翻译成法文。
2. 使用微调后的模型进行推断。

<Tip>

如果您想查看所有与本任务兼容的架构和检查点，最好查看[任务页](https://huggingface.co/tasks/translation)。

</Tip>

在开始之前，请确保您已安装所有必要的库：

```bash
pip install transformers datasets evaluate sacrebleu
```

建议您登录 Hugging Face 账户，以便将模型上传并分享给社区。在提示时，输入您的令牌进行登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 OPUS Books 数据集

首先从 🤗 Datasets 库中加载 [OPUS Books](https://huggingface.co/datasets/opus_books) 数据集的英法子集：

```py
>>> from datasets import load_dataset

>>> books = load_dataset("opus_books", "en-fr")
```

使用 [`~datasets.Dataset.train_test_split`] 方法将数据集划分为训练集和测试集：

```py
>>> books = books["train"].train_test_split(test_size=0.2)
```

然后查看一个示例：

```py
>>> books["train"][0]
{'id': '90560',
 'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
  'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}
```

`translation`：文本的英文和法文翻译。

## 预处理

<Youtube id="XAR8jnZZuUs"/>

下一步是加载 T5 分词器，处理英法语言对：

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

您要创建的预处理函数需要：

1. 在输入前添加提示词，让 T5 知道这是一个翻译任务。某些能够处理多种 NLP 任务的模型需要针对特定任务提示。
2. 在 `text_target` 参数中设置目标语言（法语），以确保分词器能正确处理目标文本。如果不设置 `text_target`，分词器会将目标文本作为英语处理。
3. 将序列截断至不超过 `max_length` 参数设置的最大长度。

```py
>>> source_lang = "en"
>>> target_lang = "fr"
>>> prefix = "translate English to French: "


>>> def preprocess_function(examples):
...     inputs = [prefix + example[source_lang] for example in examples["translation"]]
...     targets = [example[target_lang] for example in examples["translation"]]
...     model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
...     return model_inputs
```

使用 🤗 Datasets 的 [`~datasets.Dataset.map`] 方法将预处理函数应用于整个数据集。通过设置 `batched=True` 一次处理数据集的多个元素，可以加速 `map` 函数：

```py
>>> tokenized_books = books.map(preprocess_function, batched=True)
```

现在使用 [`DataCollatorForSeq2Seq`] 创建一批样本。在整理时将句子*动态填充*至批次中的最长长度，比将整个数据集填充至最大长度更高效。

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```

## 评估

在训练过程中加入评估指标有助于评估模型的性能。您可以使用 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载评估方法。对于此任务，加载 [SacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu) 指标（参阅 🤗 Evaluate [快速教程](https://huggingface.co/docs/evaluate/a_quick_tour)，了解更多关于加载和计算指标的信息）：

```py
>>> import evaluate

>>> metric = evaluate.load("sacrebleu")
```

然后创建一个函数，将您的预测结果和标签传递给 [`~evaluate.EvaluationModule.compute`] 来计算 SacreBLEU 分数：

```py
>>> import numpy as np


>>> def postprocess_text(preds, labels):
...     preds = [pred.strip() for pred in preds]
...     labels = [[label.strip()] for label in labels]

...     return preds, labels


>>> def compute_metrics(eval_preds):
...     preds, labels = eval_preds
...     if isinstance(preds, tuple):
...         preds = preds[0]
...     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

...     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
...     result = {"bleu": result["score"]}

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
...     result["gen_len"] = np.mean(prediction_lens)
...     result = {k: round(v, 4) for k, v in result.items()}
...     return result
```

您的 `compute_metrics` 函数已准备就绪，在设置训练时会用到它。

## 训练

<Tip>

如果您不熟悉使用 [`Trainer`] 微调模型，请查看[这里](../training#train-with-pytorch-trainer)的基础教程！

</Tip>

现在可以开始训练模型了！使用 [`AutoModelForSeq2SeqLM`] 加载 T5：

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

此时，只剩三个步骤：

1. 在 [`Seq2SeqTrainingArguments`] 中定义训练超参数。唯一必需的参数是 `output_dir`，它指定保存模型的位置。通过设置 `push_to_hub=True`，将模型推送到 Hub（您需要登录 Hugging Face 才能上传模型）。每个 epoch 结束时，[`Trainer`] 将评估 SacreBLEU 指标并保存训练检查点。
2. 将训练参数传递给 [`Seq2SeqTrainer`]，同时传入模型、数据集、分词器、数据整理器和 `compute_metrics` 函数。
3. 调用 [`~Trainer.train`] 微调您的模型。

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_opus_books_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=2,
...     predict_with_generate=True,
...     fp16=True, #change to bf16=True for XPU
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_books["train"],
...     eval_dataset=tokenized_books["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

训练完成后，使用 [`~transformers.Trainer.push_to_hub`] 方法将模型分享到 Hub，让所有人都能使用您的模型：

```py
>>> trainer.push_to_hub()
```

<Tip>

如需了解如何微调翻译模型的更深入示例，请参阅相应的
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb)。

</Tip>

## 推断

很好，现在您已经微调了模型，可以用它进行推断了！

准备一些您想要翻译成另一种语言的文本。对于 T5，您需要根据所处理的任务为输入添加前缀。对于从英语到法语的翻译，前缀如下所示：

```py
>>> text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
```

对文本进行分词并将 `input_ids` 作为 PyTorch 张量返回：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_opus_books_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

使用 [`~generation.GenerationMixin.generate`] 方法创建翻译结果。有关不同文本生成策略和控制生成参数的更多详情，请查阅[文本生成](../main_classes/text_generation) API。

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("username/my_awesome_opus_books_model")
>>> outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
```

将生成的词元 id 解码回文本：

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Les lignées partagent des ressources avec des bactéries enfixant l'azote.'
```
