<!--版权所有 2022 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）进行许可；除非符合许可证，否则您不得使用此文件。您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以获取特定语言下的权限和限制。⚠️请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确显示。-->

# 多项选择
# Multiple choice

[[在 Colab 中打开]]

多项选择任务类似于问答，只是除了上下文还提供了几个候选答案，模型经过训练后可以选择正确答案。

本指南将向您展示如何：

1. 在 `regular` 配置的 [SWAG](https://huggingface.co/datasets/swag) 数据集上微调 [BERT](https://huggingface.co/bert-base-uncased)，以选择最佳答案，给定多个选项和一些上下文。
2. 使用您微调的模型进行推理。

<Tip> 

此教程中演示的任务支持以下模型架构：
<!--此提示由`make fix-copies`自动生成，请勿手动填写！-->
[ALBERT](../model_doc/albert)，[BERT](../model_doc/bert)，[BigBird](../model_doc/big_bird)，[CamemBERT](../model_doc/camembert)，[CANINE](../model_doc/canine)，[ConvBERT](../model_doc/convbert)，[Data2VecText](../model_doc/data2vec-text)，[DeBERTa-v2](../model_doc/deberta-v2)，[DistilBERT](../model_doc/distilbert)，[ELECTRA](../model_doc/electra)，[ERNIE](../model_doc/ernie)，[ErnieM](../model_doc/ernie_m)，[FlauBERT](../model_doc/flaubert)，[FNet](../model_doc/fnet)，[Funnel Transformer](../model_doc/funnel)，[I-BERT](../model_doc/ibert)，[Longformer](../model_doc/longformer)，[LUKE](../model_doc/luke)，[MEGA](../model_doc/mega)，[Megatron-BERT](../model_doc/megatron-bert)，[MobileBERT](../model_doc/mobilebert)，[MPNet](../model_doc/mpnet)，[Nezha](../model_doc/nezha)，[Nystr ö mformer](../model_doc/nystromformer)，[QDQBert](../model_doc/qdqbert)，[RemBERT](../model_doc/rembert)，[RoBERTa](../model_doc/roberta)，[RoBERTa-PreLayerNorm](../model_doc/roberta-prelayernorm)，[RoCBert](../model_doc/roc_bert)，[RoFormer](../model_doc/roformer)，[SqueezeBERT](../model_doc/squeezebert)，[XLM](../model_doc/xlm)，[XLM-RoBERTa](../model_doc/xlm-roberta)，[XLM-RoBERTa-XL](../model_doc/xlm-roberta-xl)，[XLNet](../model_doc/xlnet)，[X-MOD](../model_doc/xmod)，[YOSO](../model_doc/yoso)
<!--生成提示结束-->
</Tip>

开始之前，请确保已安装所有必要的库：
```bash
pip install transformers datasets evaluate
```

我们鼓励您登录您的 Hugging Face 帐户，这样您可以与社区共享和上传您的模型。在提示时，请输入您的令牌进行登录：
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 SWAG 数据集
首先从🤗 Datasets 库加载 SWAG 数据集的 `regular` 配置：
```py
>>> from datasets import load_dataset

>>> swag = load_dataset("swag", "regular")
```

然后查看一个示例：
```py
>>> swag["train"][0]
{'ending0': 'passes by walking down the street playing their instruments.',
 'ending1': 'has heard approaching them.',
 'ending2': "arrives and they're outside dancing and asleep.",
 'ending3': 'turns the lead singer watches the performance.',
 'fold-ind': '3416',
 'gold-source': 'gold',
 'label': 0,
 'sent1': 'Members of the procession walk down the street holding small horn brass instruments.',
 'sent2': 'A drum line',
 'startphrase': 'Members of the procession walk down the street holding small horn brass instruments. A drum line',
 'video-id': 'anetv_jkn6uvmqwh4'}
```

虽然这里看起来有很多字段，但实际上很简单：

- `sent1` 和 `sent2`：这些字段显示一个句子的开头，如果将它们放在一起，就可以得到 `startphrase` 字段。- `ending`：建议一个句子的可能结尾，但只有一个是正确的。- `label`：标识正确的句子结尾。

## 预处理

下一步是加载 BERT 分词器 (Tokenizer)，以处理句子开头和四个可能的结尾：
```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

您要创建的预处理函数需要执行以下操作：
1. 复制 `sent1` 字段四次，并将每个副本与 `sent2` 组合以重新创建句子的开头。
2. 将 `sent2` 与四个可能的句子结尾之一组合。
3. 扁平化这两个列表，以便可以对它们进行分词，然后在分词后重新组合它们，以便每个示例都有相应的 `input_ids`、`attention_mask` 和 `labels` 字段。
```py
>>> ending_names = ["ending0", "ending1", "ending2", "ending3"]


>>> def preprocess_function(examples):
...     first_sentences = [[context] * 4 for context in examples["sent1"]]
...     question_headers = examples["sent2"]
...     second_sentences = [
...         [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
...     ]

...     first_sentences = sum(first_sentences, [])
...     second_sentences = sum(second_sentences, [])

...     tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
...     return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
```

要在整个数据集上应用预处理函数，使用🤗 Datasets 的 [`~datasets.Dataset.map`] 方法。

通过将 `batched=True` 设置为一次处理数据集的多个元素，可以加快 `map` 函数的速度：
```py
tokenized_swag = swag.map(preprocess_function, batched=True)
```

🤗 Transformers 没有适用于多项选择的数据整理器，因此您需要调整 [`DataCollatorWithPadding`] 以创建一批示例。在整理过程中，将句子动态填充到批次中的最大长度上，而不是将整个数据集填充到最大长度。

`DataCollatorForMultipleChoice` 对所有模型输入进行扁平化处理，应用填充，然后将结果重新展开：

<frameworkcontent> 
<pt> 

 ```py
>>> from dataclasses import dataclass
>>> from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
>>> from typing import Optional, Union
>>> import torch


>>> @dataclass
... class DataCollatorForMultipleChoice:
...     """
...     Data collator that will dynamically pad the inputs for multiple choice received.
...     """

...     tokenizer: PreTrainedTokenizerBase
...     padding: Union[bool, str, PaddingStrategy] = True
...     max_length: Optional[int] = None
...     pad_to_multiple_of: Optional[int] = None

...     def __call__(self, features):
...         label_name = "label" if "label" in features[0].keys() else "labels"
...         labels = [feature.pop(label_name) for feature in features]
...         batch_size = len(features)
...         num_choices = len(features[0]["input_ids"])
...         flattened_features = [
...             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
...         ]
...         flattened_features = sum(flattened_features, [])

...         batch = self.tokenizer.pad(
...             flattened_features,
...             padding=self.padding,
...             max_length=self.max_length,
...             pad_to_multiple_of=self.pad_to_multiple_of,
...             return_tensors="pt",
...         )

...         batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
...         batch["labels"] = torch.tensor(labels, dtype=torch.int64)
...         return batch
```
</pt> 
<tf> 

```py
>>> from dataclasses import dataclass
>>> from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
>>> from typing import Optional, Union
>>> import tensorflow as tf


>>> @dataclass
... class DataCollatorForMultipleChoice:
...     """
...     Data collator that will dynamically pad the inputs for multiple choice received.
...     """

...     tokenizer: PreTrainedTokenizerBase
...     padding: Union[bool, str, PaddingStrategy] = True
...     max_length: Optional[int] = None
...     pad_to_multiple_of: Optional[int] = None

...     def __call__(self, features):
...         label_name = "label" if "label" in features[0].keys() else "labels"
...         labels = [feature.pop(label_name) for feature in features]
...         batch_size = len(features)
...         num_choices = len(features[0]["input_ids"])
...         flattened_features = [
...             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
...         ]
...         flattened_features = sum(flattened_features, [])

...         batch = self.tokenizer.pad(
...             flattened_features,
...             padding=self.padding,
...             max_length=self.max_length,
...             pad_to_multiple_of=self.pad_to_multiple_of,
...             return_tensors="tf",
...         )

...         batch = {k: tf.reshape(v, (batch_size, num_choices, -1)) for k, v in batch.items()}
...         batch["labels"] = tf.convert_to_tensor(labels, dtype=tf.int64)
...         return batch
```
</tf>
</frameworkcontent>


## 评估

在训练过程中包含度量标准通常有助于评估模型的性能。您可以使用🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载评估方法。对于此任务，请加载 [准确度](https://huggingface.co/spaces/evaluate-metric/accuracy) 度量标准（详细了解如何加载和计算度量标准，请参阅🤗 Evaluate 的 [快速入门](https://huggingface.co/docs/evaluate/a_quick_tour)）：

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

然后创建一个函数，将您的预测和标签传递给 [`~evaluate.EvaluationModule.compute`] 以计算准确度：
```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

现在您的 `compute_metrics` 函数已经准备就绪，在设置训练时将返回它。

## 训练

<frameworkcontent> 
<pt> 
<Tip>

如果您对使用 [`Trainer`] 微调模型不熟悉，请查看基本教程 [此处](../training#train-with-pytorch-trainer)！
</Tip>

现在您已准备好开始训练模型了！使用 [`AutoModelForMultipleChoice`] 加载 BERT：
```py
>>> from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

>>> model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
```

此时，只剩下三个步骤：
1. 在 [`TrainingArguments`] 中定义训练超参数。唯一必需的参数是 `output_dir`，用于指定保存模型的位置。通过设置 `push_to_hub=True` 将此模型推送到 Hub（您需要登录 Hugging Face 才能上传模型）。在每个 epoch 结束时，[`Trainer`] 将评估准确度并保存训练检查点。
2. 将训练参数与模型、数据集、分词器 (Tokenizer)、数据整理器和 `compute_metrics` 函数一起传递给 [`Trainer`]。
3. 调用 [`~Trainer.train`] 以微调模型。
```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_swag_model",
...     evaluation_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_swag["train"],
...     eval_dataset=tokenized_swag["validation"],
...     tokenizer=tokenizer,
...     data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

完成训练后，使用 [`~transformers.Trainer.push_to_hub`] 方法将您的模型分享到 Hub，以便每个人都可以使用您的模型：
```py
>>> trainer.push_to_hub()
```
</pt> 
<tf> 

<Tip>
如果您对使用 Keras 微调模型不熟悉，请查看基本教程 [此处](../training#train-a-tensorflow-model-with-keras)！
</Tip> 

要在 TensorFlow 中微调模型，首先设置优化器函数、学习率计划和一些训练超参数：

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_train_epochs = 2
>>> total_train_steps = (len(tokenized_swag["train"]) // batch_size) * num_train_epochs
>>> optimizer, schedule = create_optimizer(init_lr=5e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
```

然后您可以使用 [`TFAutoModelForMultipleChoice`] 加载 BERT：
```py
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
```

使用 [`~transformers.TFPreTrainedModel.prepare_tf_dataset`] 将数据集转换为 `tf.data.Dataset` 格式：
```py
>>> data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_swag["train"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_swag["validation"],
...     shuffle=False,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )
```

使用 [`compile`](https://keras.io/api/models/model_training_apis/#compile-method) 配置训练模型。

请注意，Transformer 模型都有一个默认的与任务相关的损失函数，所以除非您想要指定一个，否则不需要指定。

```py
>>> model.compile(optimizer=optimizer)  # No loss argument!
```

在开始训练之前，需要完成最后两项设置，即从预测结果中计算准确率，并提供一种将模型推送到 Hub 的方式。这两项设置可以通过使用 [Keras 回调函数](../main_classes/keras_callbacks) 来实现。

将您的 `compute_metrics` 函数传递给 [`~transformers.KerasMetricCallback`]：

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

在 [`~transformers.PushToHubCallback`] 中指定将模型和分词器 (Tokenizer)推送到何处：
```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_model",
...     tokenizer=tokenizer,
... )
```

然后将回调函数捆绑在一起：
```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

最后，您可以通过调用 [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) 来开始训练模型，同时传递训练和验证数据集、训练轮数以及回调函数来微调模型：
```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2, callbacks=callbacks)
```

训练完成后，您的模型将自动上传到 Hub，以供所有人使用！
</tf>
</frameworkcontent>

<Tip>

如果您想要了解有关如何为多项选择问题微调模型的更详细示例，请参考相应的 [PyTorch 笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb) 或者 [TensorFlow 笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)。
</Tip>

## 推理

太棒了，现在您已经微调了一个模型，可以用它进行推理了！
准备一些文本和两个候选答案：

```py
>>> prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
>>> candidate1 = "The law does not apply to croissants and brioche."
>>> candidate2 = "The law applies to baguettes."
```


<frameworkcontent> 
<pt> 

 对每个提示和候选答案对进行分词，并返回 PyTorch 张量。还应该创建一些 `labels`：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
>>> labels = torch.tensor(0).unsqueeze(0)
```

将输入和标签传递给模型，并返回 `logits`：
```py
>>> from transformers import AutoModelForMultipleChoice

>>> model = AutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
>>> logits = outputs.logits
```

获取具有最高概率的类别：
```py
>>> predicted_class = logits.argmax().item()
>>> predicted_class
'0'
```
</pt> 
<tf> 

对每个提示和候选答案对进行分词，并返回 TensorFlow 张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="tf", padding=True)
```

将输入传递给模型，并返回 `logits`：
```py
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
>>> inputs = {k: tf.expand_dims(v, 0) for k, v in inputs.items()}
>>> outputs = model(inputs)
>>> logits = outputs.logits
```

获取具有最高概率的类别：
```py
>>> predicted_class = int(tf.math.argmax(logits, axis=-1)[0])
>>> predicted_class
'0'
```
</tf>
</frameworkcontent>

