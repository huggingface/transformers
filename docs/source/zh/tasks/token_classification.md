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

# 词元分类

[[open-in-colab]]

<Youtube id="wVHdVlPScxA"/>

词元分类为句子中的每个词元分配标签。最常见的词元分类任务之一是命名实体识别（NER）。NER 尝试为句子中的每个实体找到对应标签，例如人名、地名或组织名。

本指南将向您展示如何：

1. 在 [WNUT 17](https://huggingface.co/datasets/wnut_17) 数据集上微调 [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased)，以检测新兴实体。
2. 使用微调后的模型进行推断。

<Tip>

如果您想查看所有与本任务兼容的架构和检查点，最好查看[任务页](https://huggingface.co/tasks/token-classification)。

</Tip>

在开始之前，请确保您已安装所有必要的库：

```bash
pip install transformers datasets evaluate seqeval
```

建议您登录 Hugging Face 账户，以便将模型上传并分享给社区。在提示时，输入您的令牌进行登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 WNUT 17 数据集

首先从 🤗 Datasets 库中加载 WNUT 17 数据集：

```py
>>> from datasets import load_dataset

>>> wnut = load_dataset("wnut_17")
```

然后查看一个示例：

```py
>>> wnut["train"][0]
{'id': '0',
 'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
 'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
}
```

`ner_tags` 中的每个数字代表一个实体。将数字转换为标签名称，以了解实体类型：

```py
>>> label_list = wnut["train"].features[f"ner_tags"].feature.names
>>> label_list
[
    "O",
    "B-corporation",
    "I-corporation",
    "B-creative-work",
    "I-creative-work",
    "B-group",
    "I-group",
    "B-location",
    "I-location",
    "B-person",
    "I-person",
    "B-product",
    "I-product",
]
```

每个 `ner_tag` 的前缀字母表示实体中词元的位置：

- `B-` 表示实体的开始。
- `I-` 表示词元包含在同一实体中（例如，`State` 词元是 `Empire State Building` 等实体的一部分）。
- `0` 表示该词元不对应任何实体。

## 预处理

<Youtube id="iY2AZYdZAr0"/>

下一步是加载 DistilBERT 分词器，对 `tokens` 字段进行预处理：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

如上面示例的 `tokens` 字段所示，看起来输入已经完成了分词。但实际上输入尚未分词，您需要设置 `is_split_into_words=True` 将词语分词为子词。例如：

```py
>>> example = wnut["train"][0]
>>> tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
>>> tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
>>> tokens
['[CLS]', '@', 'paul', '##walk', 'it', "'", 's', 'the', 'view', 'from', 'where', 'i', "'", 'm', 'living', 'for', 'two', 'weeks', '.', 'empire', 'state', 'building', '=', 'es', '##b', '.', 'pretty', 'bad', 'storm', 'here', 'last', 'evening', '.', '[SEP]']
```

然而，这会添加一些特殊词元 `[CLS]` 和 `[SEP]`，子词分词会造成输入与标签之间的不匹配——原本对应单个标签的单个词，现在可能被分割为两个子词。您需要通过以下方式重新对齐词元和标签：

1. 使用 [`word_ids`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.BatchEncoding.word_ids) 方法将所有词元映射到对应的词语。
2. 对特殊词元 `[CLS]` 和 `[SEP]` 分配标签 `-100`，使其被 PyTorch 的损失函数忽略（参见 [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)）。
3. 仅为给定词语的第一个词元打标签，对同一词语的其他子词元分配 `-100`。

下面是创建一个函数来重新对齐词元和标签、并将序列截断至不超过 DistilBERT 最大输入长度的方法：

```py
>>> def tokenize_and_align_labels(examples):
...     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

...     labels = []
...     for i, label in enumerate(examples[f"ner_tags"]):
...         word_ids = tokenized_inputs.word_ids(batch_index=i)  # 将词元映射到对应词语
...         previous_word_idx = None
...         label_ids = []
...         for word_idx in word_ids:  # 将特殊词元设置为 -100
...             if word_idx is None:
...                 label_ids.append(-100)
...             elif word_idx != previous_word_idx:  # 仅为给定词语的第一个词元打标签
...                 label_ids.append(label[word_idx])
...             else:
...                 label_ids.append(-100)
...             previous_word_idx = word_idx
...         labels.append(label_ids)

...     tokenized_inputs["labels"] = labels
...     return tokenized_inputs
```

使用 🤗 Datasets 的 [`~datasets.Dataset.map`] 函数将预处理函数应用于整个数据集。通过设置 `batched=True` 一次处理数据集的多个元素，可以加速 `map` 函数：

```py
>>> tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
```

现在使用 [`DataCollatorWithPadding`] 创建一批样本。在整理时将句子*动态填充*至批次中的最长长度，比将整个数据集填充至最大长度更高效。

```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

## 评估

在训练过程中加入评估指标有助于评估模型的性能。您可以使用 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载评估方法。对于此任务，加载 [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval) 框架（参阅 🤗 Evaluate [快速教程](https://huggingface.co/docs/evaluate/a_quick_tour)，了解更多关于加载和计算指标的信息）。seqeval 实际上会产生多个分数：精确率、召回率、F1 和准确率。

```py
>>> import evaluate

>>> seqeval = evaluate.load("seqeval")
```

首先获取 NER 标签，然后创建一个函数，将真实预测结果和真实标签传递给 [`~evaluate.EvaluationModule.compute`] 来计算分数：

```py
>>> import numpy as np

>>> labels = [label_list[i] for i in example[f"ner_tags"]]


>>> def compute_metrics(p):
...     predictions, labels = p
...     predictions = np.argmax(predictions, axis=2)

...     true_predictions = [
...         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]
...     true_labels = [
...         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]

...     results = seqeval.compute(predictions=true_predictions, references=true_labels)
...     return {
...         "precision": results["overall_precision"],
...         "recall": results["overall_recall"],
...         "f1": results["overall_f1"],
...         "accuracy": results["overall_accuracy"],
...     }
```

您的 `compute_metrics` 函数已准备就绪，在设置训练时会用到它。

## 训练

在开始训练模型之前，使用 `id2label` 和 `label2id` 创建预期 id 到其标签的映射：

```py
>>> id2label = {
...     0: "O",
...     1: "B-corporation",
...     2: "I-corporation",
...     3: "B-creative-work",
...     4: "I-creative-work",
...     5: "B-group",
...     6: "I-group",
...     7: "B-location",
...     8: "I-location",
...     9: "B-person",
...     10: "I-person",
...     11: "B-product",
...     12: "I-product",
... }
>>> label2id = {
...     "O": 0,
...     "B-corporation": 1,
...     "I-corporation": 2,
...     "B-creative-work": 3,
...     "I-creative-work": 4,
...     "B-group": 5,
...     "I-group": 6,
...     "B-location": 7,
...     "I-location": 8,
...     "B-person": 9,
...     "I-person": 10,
...     "B-product": 11,
...     "I-product": 12,
... }
```

<Tip>

如果您不熟悉使用 [`Trainer`] 微调模型，请查看[这里](../training#train-with-pytorch-trainer)的基础教程！

</Tip>

现在可以开始训练模型了！使用 [`AutoModelForTokenClassification`] 加载 DistilBERT，并指定预期标签数量和标签映射：

```py
>>> from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

>>> model = AutoModelForTokenClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
... )
```

此时，只剩三个步骤：

1. 在 [`TrainingArguments`] 中定义训练超参数。唯一必需的参数是 `output_dir`，它指定保存模型的位置。通过设置 `push_to_hub=True`，将模型推送到 Hub（您需要登录 Hugging Face 才能上传模型）。每个 epoch 结束时，[`Trainer`] 将评估 seqeval 分数并保存训练检查点。
2. 将训练参数传递给 [`Trainer`]，同时传入模型、数据集、分词器、数据整理器和 `compute_metrics` 函数。
3. 调用 [`~Trainer.train`] 微调您的模型。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_wnut_model",
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
...     train_dataset=tokenized_wnut["train"],
...     eval_dataset=tokenized_wnut["test"],
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

如需了解如何微调词元分类模型的更深入示例，请参阅相应的
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb)。

</Tip>

## 推断

很好，现在您已经微调了模型，可以用它进行推断了！

准备一些您想要进行推断的文本：

```py
>>> text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
```

使用微调后的模型进行推断最简单的方式是在 [`pipeline`] 中使用它。用您的模型实例化一个 NER `pipeline`，并将文本传递给它：

```py
>>> from transformers import pipeline

>>> classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
>>> classifier(text)
[{'entity': 'B-location',
  'score': 0.42658573,
  'index': 2,
  'word': 'golden',
  'start': 4,
  'end': 10},
 {'entity': 'I-location',
  'score': 0.35856336,
  'index': 3,
  'word': 'state',
  'start': 11,
  'end': 16},
 {'entity': 'B-group',
  'score': 0.3064001,
  'index': 4,
  'word': 'warriors',
  'start': 17,
  'end': 25},
 {'entity': 'B-location',
  'score': 0.65523505,
  'index': 13,
  'word': 'san',
  'start': 80,
  'end': 83},
 {'entity': 'B-location',
  'score': 0.4668663,
  'index': 14,
  'word': 'francisco',
  'start': 84,
  'end': 93}]
```

如果您愿意，也可以手动复现 `pipeline` 的结果：

对文本进行分词并返回 PyTorch 张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> inputs = tokenizer(text, return_tensors="pt")
```

将输入传递给模型并返回 `logits`：

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

获取概率最高的类别，并使用模型的 `id2label` 映射将其转换为文本标签：

```py
>>> predictions = torch.argmax(logits, dim=2)
>>> predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
>>> predicted_token_class
['O',
 'O',
 'B-location',
 'I-location',
 'B-group',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'B-location',
 'B-location',
 'O',
 'O']
```
