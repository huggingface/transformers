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

# 文本分类

[[open-in-colab]]

<Youtube id="leNG9fN9FQU"/>

文本分类是一种常见的 NLP 任务，它为文本分配标签或类别。许多大型公司在生产环境中运行文本分类，用于各种实际应用。其中最流行的形式之一是情感分析，它为文本序列分配诸如 🙂 正面、🙁 负面或 😐 中性的标签。

本指南将向您展示如何：

1. 在 [IMDb](https://huggingface.co/datasets/imdb) 数据集上微调 [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased)，以判断电影评论是正面还是负面。
2. 使用微调后的模型进行推断。

<Tip>

如果您想查看所有与本任务兼容的架构和检查点，最好查看[任务页](https://huggingface.co/tasks/text-classification)。

</Tip>

在开始之前，请确保您已安装所有必要的库：

```bash
pip install transformers datasets evaluate accelerate
```

建议您登录 Hugging Face 账户，以便将模型上传并分享给社区。在提示时，输入您的令牌进行登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 IMDb 数据集

首先从 🤗 Datasets 库中加载 IMDb 数据集：

```py
>>> from datasets import load_dataset

>>> imdb = load_dataset("imdb")
```

然后查看一个示例：

```py
>>> imdb["test"][0]
{
    "label": 0,
    "text": "I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say \"Gene Roddenberry's Earth...\" otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again.",
}
```

该数据集有两个字段：

- `text`：电影评论文本。
- `label`：值为 `0` 表示负面评论，值为 `1` 表示正面评论。

## 预处理

下一步是加载 DistilBERT 分词器，对 `text` 字段进行预处理：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

创建一个预处理函数来对 `text` 进行分词，并将序列截断至不超过 DistilBERT 最大输入长度：

```py
>>> def preprocess_function(examples):
...     return tokenizer(examples["text"], truncation=True)
```

使用 🤗 Datasets 的 [`~datasets.Dataset.map`] 函数将预处理函数应用于整个数据集。通过设置 `batched=True` 一次处理数据集的多个元素，可以加速 `map`：

```py
tokenized_imdb = imdb.map(preprocess_function, batched=True)
```

现在使用 [`DataCollatorWithPadding`] 创建一批样本。在整理时将句子*动态填充*至批次中的最长长度，比将整个数据集填充至最大长度更高效。

```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

## 评估

在训练过程中加入评估指标有助于评估模型的性能。您可以使用 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载评估方法。对于此任务，加载[准确率](https://huggingface.co/spaces/evaluate-metric/accuracy)指标（参阅 🤗 Evaluate [快速教程](https://huggingface.co/docs/evaluate/a_quick_tour)，了解更多关于加载和计算指标的信息）：

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

然后创建一个函数，将您的预测结果和标签传递给 [`~evaluate.EvaluationModule.compute`] 来计算准确率：

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

您的 `compute_metrics` 函数已准备就绪，在设置训练时会用到它。

## 训练

在开始训练模型之前，使用 `id2label` 和 `label2id` 创建预期 id 到其标签的映射：

```py
>>> id2label = {0: "NEGATIVE", 1: "POSITIVE"}
>>> label2id = {"NEGATIVE": 0, "POSITIVE": 1}
```

<Tip>

如果您不熟悉使用 [`Trainer`] 微调模型，请查看[这里](../training#train-with-pytorch-trainer)的基础教程！

</Tip>

现在可以开始训练模型了！使用 [`AutoModelForSequenceClassification`] 加载 DistilBERT，并指定预期标签数量和标签映射：

```py
>>> from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

>>> model = AutoModelForSequenceClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
... )
```

此时，只剩三个步骤：

1. 在 [`TrainingArguments`] 中定义训练超参数。唯一必需的参数是 `output_dir`，它指定保存模型的位置。通过设置 `push_to_hub=True`，将模型推送到 Hub（您需要登录 Hugging Face 才能上传模型）。每个 epoch 结束时，[`Trainer`] 将评估准确率并保存训练检查点。
2. 将训练参数传递给 [`Trainer`]，同时传入模型、数据集、分词器、数据整理器和 `compute_metrics` 函数。
3. 调用 [`~Trainer.train`] 微调您的模型。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_model",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=2,
...     weight_decay=0.01,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_imdb["train"],
...     eval_dataset=tokenized_imdb["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

<Tip>

当您将 `tokenizer` 传递给 [`Trainer`] 时，它会默认应用动态填充。在这种情况下，您无需显式指定数据整理器。

</Tip>

训练完成后，使用 [`~transformers.Trainer.push_to_hub`] 方法将模型分享到 Hub，让所有人都能使用您的模型：

```py
>>> trainer.push_to_hub()
```

<Tip>

如需了解如何微调文本分类模型的更深入示例，请参阅相应的
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)。

</Tip>

## 推断

很好，现在您已经微调了模型，可以用它进行推断了！

准备一些您想要进行推断的文本：

```py
>>> text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
```

使用微调后的模型进行推断最简单的方式是在 [`pipeline`] 中使用它。用您的模型实例化一个情感分析 `pipeline`，并将文本传递给它：

```py
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
>>> classifier(text)
[{'label': 'POSITIVE', 'score': 0.9994940757751465}]
```

如果您愿意，也可以手动复现 `pipeline` 的结果：

对文本进行分词并返回 PyTorch 张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
>>> inputs = tokenizer(text, return_tensors="pt")
```

将输入传递给模型并返回 `logits`：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

获取概率最高的类别，并使用模型的 `id2label` 映射将其转换为文本标签：

```py
>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'POSITIVE'
```
