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

# 微调预训练模型

[[open-in-colab]]

使用预训练模型有许多显著的好处。它降低了计算成本，减少了碳排放，同时允许您使用最先进的模型，而无需从头开始训练一个。🤗 Transformers 提供了涉及各种任务的成千上万的预训练模型。当您使用预训练模型时，您需要在与任务相关的数据集上训练该模型。这种操作被称为微调，是一种非常强大的训练技术。在本教程中，您将使用您选择的深度学习框架来微调一个预训练模型：

* 使用 🤗 Transformers 的 [`Trainer`] 来微调预训练模型。
* 在 TensorFlow 中使用 Keras 来微调预训练模型。
* 在原生 PyTorch 中微调预训练模型。

<a id='data-processing'></a>

## 准备数据集

<Youtube id="_BZearw7f0w"/>

在您进行预训练模型微调之前，需要下载一个数据集并为训练做好准备。之前的教程向您展示了如何处理训练数据，现在您有机会将这些技能付诸实践！

首先，加载[Yelp评论](https://huggingface.co/datasets/Yelp/yelp_review_full)数据集：

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

正如您现在所知，您需要一个`tokenizer`来处理文本，包括填充和截断操作以处理可变的序列长度。如果要一次性处理您的数据集，可以使用 🤗 Datasets 的 [`map`](https://huggingface.co/docs/datasets/process#map) 方法，将预处理函数应用于整个数据集：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)


>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```
如果愿意的话，您可以从完整数据集提取一个较小子集来进行微调，以减少训练所需的时间：

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

<a id='trainer'></a>

## 训练

此时，您应该根据您训练所用的框架来选择对应的教程章节。您可以使用右侧的链接跳转到您想要的章节 - 如果您想隐藏某个框架对应的所有教程内容，只需使用右上角的按钮！


<Youtube id="nvBXf7s7vTI"/>

## 使用 PyTorch Trainer 进行训练

🤗 Transformers 提供了一个专为训练 🤗 Transformers 模型而优化的 [`Trainer`] 类，使您无需手动编写自己的训练循环步骤而更轻松地开始训练模型。[`Trainer`] API 支持各种训练选项和功能，如日志记录、梯度累积和混合精度。

首先加载您的模型并指定期望的标签数量。根据 Yelp Review [数据集卡片](https://huggingface.co/datasets/Yelp/yelp_review_full#data-fields)，您知道有五个标签：


```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

<Tip>

您将会看到一个警告，提到一些预训练权重未被使用，以及一些权重被随机初始化。不用担心，这是完全正常的！BERT 模型的预训练`head`被丢弃，并替换为一个随机初始化的分类`head`。您将在您的序列分类任务上微调这个新模型`head`，将预训练模型的知识转移给它。

</Tip>

### 训练超参数

接下来，创建一个 [`TrainingArguments`] 类，其中包含您可以调整的所有超参数以及用于激活不同训练选项的标志。对于本教程，您可以从默认的训练[超参数](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)开始，但随时可以尝试不同的设置以找到最佳设置。

指定保存训练检查点的位置：

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

### 评估

[`Trainer`] 在训练过程中不会自动评估模型性能。您需要向 [`Trainer`] 传递一个函数来计算和展示指标。[🤗 Evaluate](https://huggingface.co/docs/evaluate/index) 库提供了一个简单的 [`accuracy`](https://huggingface.co/spaces/evaluate-metric/accuracy) 函数，您可以使用 [`evaluate.load`] 函数加载它（有关更多信息，请参阅此[快速入门](https://huggingface.co/docs/evaluate/a_quick_tour)）：

```py
>>> import numpy as np
>>> import evaluate

>>> metric = evaluate.load("accuracy")
```
在 `metric` 上调用 [`~evaluate.compute`] 来计算您的预测的准确性。在将预测传递给 `compute` 之前，您需要将预测转换为`logits`（请记住，所有 🤗 Transformers 模型都返回对`logits`）：

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     predictions = np.argmax(logits, axis=-1)
...     return metric.compute(predictions=predictions, references=labels)
```

如果您希望在微调过程中监视评估指标，请在您的训练参数中指定 `eval_strategy` 参数，以在每个`epoch`结束时展示评估指标：

```py
>>> from transformers import TrainingArguments, Trainer

>>> training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
```

### 训练器

创建一个包含您的模型、训练参数、训练和测试数据集以及评估函数的 [`Trainer`] 对象：


```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```
然后调用[`~transformers.Trainer.train`]以微调模型：

```py
>>> trainer.train()
```

<a id='pytorch_native'></a>

## 在原生 PyTorch 中训练

<Youtube id="Dh9CL8fyG80"/>

[`Trainer`] 负责训练循环，允许您在一行代码中微调模型。对于喜欢编写自己训练循环的用户，您也可以在原生 PyTorch 中微调 🤗 Transformers 模型。

现在，您可能需要重新启动您的`notebook`，或执行以下代码以释放一些内存：

```py
del model
del trainer
torch.cuda.empty_cache()
```

接下来，手动处理 `tokenized_dataset` 以准备进行训练。

1. 移除 text 列，因为模型不接受原始文本作为输入：

    ```py
    >>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    ```

2. 将 label 列重命名为 labels，因为模型期望参数的名称为 labels：

    ```py
    >>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    ```

3. 设置数据集的格式以返回 PyTorch 张量而不是`lists`：

    ```py
    >>> tokenized_datasets.set_format("torch")
    ```

接着，创建一个先前展示的数据集的较小子集，以加速微调过程

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

### DataLoader

您的训练和测试数据集创建一个`DataLoader`类，以便可以迭代处理数据批次

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

加载您的模型，并指定期望的标签数量：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

### Optimizer and learning rate scheduler

创建一个`optimizer`和`learning rate scheduler`以进行模型微调。让我们使用 PyTorch 中的 [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) 优化器：

```py
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters(), lr=5e-5)
```

创建来自 [`Trainer`] 的默认`learning rate scheduler`：


```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
...     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
... )
```

最后，指定 `device` 以使用 GPU（如果有的话）。否则，使用 CPU 进行训练可能需要几个小时，而不是几分钟。


```py
>>> import torch

>>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> model.to(device)
```

<Tip>

如果没有 GPU，可以通过notebook平台如 [Colaboratory](https://colab.research.google.com/) 或 [SageMaker StudioLab](https://studiolab.sagemaker.aws/) 来免费获得云端GPU使用。

</Tip>

现在您已经准备好训练了！🥳

### 训练循环

为了跟踪训练进度，使用 [tqdm](https://tqdm.github.io/) 库来添加一个进度条，显示训练步数的进展：

```py
>>> from tqdm.auto import tqdm

>>> progress_bar = tqdm(range(num_training_steps))

>>> model.train()
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         batch = {k: v.to(device) for k, v in batch.items()}
...         outputs = model(**batch)
...         loss = outputs.loss
...         loss.backward()

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

### 评估

就像您在 [`Trainer`] 中添加了一个评估函数一样，当您编写自己的训练循环时，您需要做同样的事情。但与在每个`epoch`结束时计算和展示指标不同，这一次您将使用 [`~evaluate.add_batch`] 累积所有批次，并在最后计算指标。

```py
>>> import evaluate

>>> metric = evaluate.load("accuracy")
>>> model.eval()
>>> for batch in eval_dataloader:
...     batch = {k: v.to(device) for k, v in batch.items()}
...     with torch.no_grad():
...         outputs = model(**batch)

...     logits = outputs.logits
...     predictions = torch.argmax(logits, dim=-1)
...     metric.add_batch(predictions=predictions, references=batch["labels"])

>>> metric.compute()
```

<a id='additional-resources'></a>

## 附加资源

更多微调例子可参考如下链接：

- [🤗 Transformers 示例](https://github.com/huggingface/transformers/tree/main/examples) 包含用于在 PyTorch 和 TensorFlow 中训练常见自然语言处理任务的脚本。

- [🤗 Transformers 笔记](notebooks) 包含针对特定任务在 PyTorch 和 TensorFlow 中微调模型的各种`notebook`。
