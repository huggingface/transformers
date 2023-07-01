<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，按“原样”分发的软件根据许可证分发，并且没有任何形式的担保或条件。请参阅许可证以了解特定语言下的权限和限制。具体语言下的权限和限制。
⚠️请注意，此文件是使用 Markdown 编写的，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->

# Fine-tune a pretrained model

[[open-in-colab]]

使用预训练模型有很多好处。它降低了计算成本和碳排放量，并且可以让您使用最先进的模型，而无需从头开始训练一个模型。🤗 Transformers 提供了数千个针对各种任务的预训练模型。当您使用预训练模型时，您会在与您的任务相关的数据集上进行微调。这被称为微调，是一种非常强大的训练技术。在本教程中，您将使用您选择的深度学习框架对一个预训练模型进行微调：

* 使用🤗 Transformers 的 [`Trainer`] 对一个预训练模型进行微调。
* 在 TensorFlow 中使用 Keras 对一个预训练模型进行微调。
* 在原生 PyTorch 中对一个预训练模型进行微调。

<a id='data-processing'> </a>

## 准备数据集

<Youtube id="_BZearw7f0w"/>

在您对一个预训练模型进行微调之前，需要下载一个数据集并为其进行准备以用于训练。之前的教程向您展示了如何处理用于训练的数据，现在您有机会将这些技巧付诸实践！

首先加载 [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full) 数据集：

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

正如您现在所知道的，您需要一个分词器来处理文本，并包括填充和截断策略以处理任意长度的序列。为了一次处理整个数据集，使用🤗 Datasets 的 [`map`](https://huggingface.co/docs/datasets/process.html#map) 方法在整个数据集上应用一个预处理函数：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)


>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

如果愿意的话，您可以创建一个较小的子集来进行微调，以减少所需的时间：
```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

<a id='trainer'> </a>

## Train

此时，您应该按照您想要使用的框架对应的部分进行操作。您可以使用右侧边栏的链接跳转到您想要的部分 - 如果您希望隐藏特定框架的全部内容，只需使用该框架块顶部右侧的按钮！

<frameworkcontent>
<pt>
<Youtube id="nvBXf7s7vTI"/>

## 使用 PyTorch Trainer 进行训练

🤗 Transformers 提供了一个针对训练🤗 Transformers 模型进行优化的 [`Trainer`] 类，使得无需手动编写训练循环即可轻松开始训练。[` Trainer`] API 支持广泛的训练选项和功能，如日志记录、梯度累积和混合精度。

首先加载模型并指定期望的标签数量。根据 Yelp Review [数据集卡片](https://huggingface.co/datasets/yelp_review_full#data-fields)，您知道有五个标签：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

<Tip>

您会看到有关某些预训练权重未被使用和某些权重随机初始化的警告。不要担心，这是完全正常的！BERT 模型的预训练头部被丢弃，并用随机初始化的分类头部替换。您将在您的序列分类任务上对这个新的模型头部进行微调，将预训练模型的知识转移到它上面。

</Tip>

### 训练超参数

接下来，创建一个 [`TrainingArguments`] 类，其中包含了您可以调节的所有超参数，以及用于激活不同训练选项的标志。对于本教程，您可以使用默认的训练 [超参数](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)，但请随意尝试不同的设置，找到最佳的配置。

指定保存训练检查点的位置：

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

### Evaluate

[`Trainer`] 在训练过程中不会自动评估模型性能。您需要为 [`Trainer`] 传递一个函数来计算和报告指标。[🤗 Evaluate](https://huggingface.co/docs/evaluate/index) 库提供了一个简单的 [`accuracy`](https://huggingface.co/spaces/evaluate-metric/accuracy) 函数，您可以使用 [`evaluate.load`] 函数加载（有关更多信息，请参阅此 [快速入门](https://huggingface.co/docs/evaluate/a_quick_tour)）：

```py
>>> import numpy as np
>>> import evaluate

>>> metric = evaluate.load("accuracy")
```

在 `metric` 上调用 [`~evaluate.compute`] 计算您的预测的准确率。在将预测传递给 `compute` 之前，您需要将预测转换为 logits（记住，所有🤗 Transformers 模型返回的都是 logits）：

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     predictions = np.argmax(logits, axis=-1)
...     return metric.compute(predictions=predictions, references=labels)
```

如果您希望在微调过程中监控评估指标，请在训练参数中指定 `evaluation_strategy` 参数，以在每个 epoch 结束时报告评估指标：

```py
>>> from transformers import TrainingArguments, Trainer

>>> training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
```

### Trainer

使用您的模型、训练参数、训练和测试数据集以及评估函数创建一个 [`Trainer`] 对象：

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

然后通过调用 [`~transformers.Trainer.train`] 对模型进行微调：

```py
>>> trainer.train()
```
</pt>
<tf>
<a id='keras'> </a>

<Youtube id="rnTGBy2ax1c"/>

## 使用 Keras 训练 TensorFlow 模型

您也可以使用 Keras API 训练🤗 Transformers 模型！

### 为 Keras 加载数据

当您想要使用 Keras API 训练一个🤗 Transformers 模型时，您需要将数据集转换为 Keras 可以理解的格式。如果您的数据集很小，您可以将整个数据集转换为 NumPy 数组并将其传递给 Keras。在更复杂的操作之前，让我们首先尝试这样做。

首先加载一个数据集。我们将使用 [GLUE 基准](https://huggingface.co/datasets/glue) 中的 CoLA 数据集，因为这是一个简单的二分类文本分类任务，现在只需取训练集。

```py
from datasets import load_dataset

dataset = load_dataset("glue", "cola")
dataset = dataset["train"]  # 现在只取训练集
```

接下来，加载一个分词器并将数据进行分词，得到 NumPy 数组。请注意，标签已经是一个由 0 和 1 组成的列表，因此我们可以直接将其转换为 NumPy 数组而无需进行分词！

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_data = tokenizer(dataset["sentence"], return_tensors="np", padding=True)
# 分词器返回BatchEncoding，但我们将其转换为Keras的dict格式
tokenized_data = dict(tokenized_data)

labels = np.array(dataset["label"])  # 标签已经是一个由0和1组成的数组
```

最后，加载、[`compile`](https://keras.io/api/models/model_training_apis/#compile-method) 和 [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) 模型。请注意，🤗 Transformers 模型都有一个默认的与任务相关的损失函数，因此除非您想要指定一个损失函数，否则不需要指定：

```py
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")
# Lower learning rates are often better for fine-tuning transformers
model.compile(optimizer=Adam(3e-5))  # No loss argument!

model.fit(tokenized_data, labels)
```

<Tip>

当您调用 `compile()` 编译模型时，不需要为模型传递损失参数！如果将该参数留空，Hugging Face 模型会自动选择适合任务和模型架构的损失函数。如果您希望自行指定损失函数，您始终可以覆盖此行为！

</Tip>

这种方法在较小的数据集上效果很好，但对于较大的数据集，您可能会发现它开始成为一个问题。为什么呢？因为分词后的数组和标签需要完全加载到内存中，并且因为 NumPy 不能处理“jagged”数组，所以每个分词后的样本都必须填充到整个数据集中最长样本的长度。这会使您的数组变得更大，并且所有这些填充令牌也会减慢训练速度！

### 将数据加载为 tf.data.Dataset

如果您想避免训练速度变慢，您可以将数据加载为 `tf.data.Dataset`。虽然您可以自己编写 `tf.data` 流水线，但我们有两种方便的方法可以实现这一点：
- [`~TFPreTrainedModel.prepare_tf_dataset`]: 这是我们在大多数情况下推荐的方法，因为它是一个方法在您的模型上，它可以检查模型以自动确定哪些列可用作模型输入，并丢弃其他列以创建一个更简单、更高性能的数据集。
- [`~datasets.Dataset.to_tf_dataset`]: 这种方法更底层，适用于您想要精确控制数据集创建方式的情况，通过指定要包括的`columns`和`label_cols`。

在使用 [`~TFPreTrainedModel.prepare_tf_dataset`] 之前，您需要将分词器的输出添加到数据集中作为列，如下所示：以下代码示例：

```py
def tokenize_dataset(data):
    # Keys of the returned dictionary will be added to the dataset as columns
    return tokenizer(data["text"])


dataset = dataset.map(tokenize_dataset)
```

请记住，Hugging Face 数据集默认存储在磁盘上，因此这不会增加内存使用量！一旦列被添加，您可以从数据集中流式传输批次并为每个批次添加填充，这大大减少了与对整个数据集进行填充相比的填充标记数量。

```py
>>> tf_dataset = model.prepare_tf_dataset(dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer)
```

请注意，在上面的代码示例中，您需要将分词器传递给 `prepare_tf_dataset`，以便在加载批次时正确填充批次。如果数据集中的所有样本长度相同且不需要填充，则可以跳过此参数。如果您需要执行比填充样本更复杂的操作（例如对遮蔽语言进行破坏）建模），您可以使用 `collate_fn` 参数将调用一个函数来将样本列表转换为批次并应用任何预处理。请参阅我们的 [示例](https://github.com/huggingface/transformers/tree/main/examples) 或 [笔记本](https://huggingface.co/docs/transformers/notebooks) 以查看此方法的实际操作。

创建了一个 `tf.data.Dataset` 后，您可以像以前一样编译和训练模型：
```py
model.compile(optimizer=Adam(3e-5))  # No loss argument!

model.fit(tf_dataset)
```

</tf>
</frameworkcontent>
<a id='pytorch_native'> </a>

## 在原生 PyTorch 中训练

<frameworkcontent> <pt>
<Youtube id="Dh9CL8fyG80"/>

[`Trainer`] 负责训练循环，并允许您通过一行代码进行模型微调。对于喜欢编写自己训练循环的用户，您还可以在原生 PyTorch 中微调🤗 Transformers 模型。

此时，您可能需要重新启动笔记本或执行以下代码以释放一些内存：

```py
del model
del trainer
torch.cuda.empty_cache()
```

接下来，手动对 `tokenized_dataset` 进行后处理，以准备进行训练。
1. 删除 `text` 列，因为模型不接受原始文本作为输入：
    ```py
    >>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    ```
2. Rename the `label` column to `labels` because the model expects the argument to be named `labels`:


    ```py
    >>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    ```
3. 将数据集的格式设置为返回 PyTorch 张量而不是列表：

```py
>>> tokenized_datasets.set_format("torch")
```
然后按照之前展示的方式创建一个较小的数据集子集，以加快微调的速度：

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

### DataLoader

为您的训练和测试数据集创建一个 `DataLoader`，以便您可以迭代处理批量数据：

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```
使用预期标签数量加载您的模型：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

### Optimizer and learning rate scheduler

创建一个优化器和学习率调度器来微调模型。让我们使用 PyTorch 中的 [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) 优化器：
```py
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters(), lr=5e-5)
```

Create the default learning rate scheduler from [`Trainer`]:

```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
...     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
... )
```

最后，如果您可以访问 GPU，请指定 `device` 以使用 GPU。否则，在 CPU 上进行训练可能需要数小时而不是几分钟。

```py
>>> import torch

>>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> model.to(device)
```

<Tip>

如果您没有 GPU，可以通过使用像 [Colaboratory](https://colab.research.google.com/) 或 [SageMaker StudioLab](https://studiolab.sagemaker.aws/) 这样的在线笔记本来获得免费的云 GPU 访问。

</Tip>

太棒了，现在您已经准备好开始训练了！🥳

### 训练循环

为了跟踪训练进度，使用 [tqdm](https://tqdm.github.io/) 库在训练步骤数量上添加一个进度条：

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

### Evaluate

就像您在 [`Trainer`] 中添加了一个评估函数一样，当您编写自己的训练循环时，您需要做同样的事情。但是，与在每个 epoch 结束时计算和报告指标不同，这一次您将使用 [`~evaluate.add_batch`] 来累积所有批次，并在最后计算指标。

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

</pt>
</frameworkcontent>

<a id='additional-resources'> </a>

## 其他资源

要获取更多微调示例，请参考以下资源：

- [🤗 Transformers 示例代码库](https://github.com/huggingface/transformers/tree/main/examples)：包含了使用 PyTorch 和 TensorFlow 训练常见 NLP 任务的示例脚本。
- [🤗 Transformers 笔记本](notebooks)：包含了使用 PyTorch 和 TensorFlow 为特定任务微调模型的各种笔记本。
