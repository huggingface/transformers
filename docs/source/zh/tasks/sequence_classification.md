<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按原样”方式分发，不附带任何明示或暗示的保证或条件。请参阅许可证以了解具体的语言规定和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含了我们的文档生成器特定的语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确显示。-->

# 文本分类

[[在 Colab 中打开]]

<Youtube id="leNG9fN9FQU"/>


文本分类是一种常见的 NLP 任务，它将标签或类别分配给文本。一些最大的公司将文本分类用于各种实际应用的生产环境中。最受欢迎的文本分类形式之一是情感分析，它将标签（如🙂积极，🙁消极或😐中性）分配给一段文本。
本指南将向您展示如何：

1. 在 [IMDb](https://huggingface.co/datasets/imdb) 数据集上对 [DistilBERT](https://huggingface.co/distilbert-base-uncased) 进行微调，以确定电影评论是积极还是消极。
2. 使用您微调后的模型进行推理。

<Tip> 

本教程中所示的任务由以下模型架构支持：
<!--此提示是由`make fix-copies`自动生成的，请勿手动填写！-->
[ALBERT](../model_doc/albert)，[BART](../model_doc/bart)，[BERT](../model_doc/bert)，[BigBird](../model_doc/big_bird)，[BigBird-Pegasus](../model_doc/bigbird_pegasus)，[BioGpt](../model_doc/biogpt)，[BLOOM](../model_doc/bloom)，[CamemBERT](../model_doc/camembert)，[CANINE](../model_doc/canine)，[ConvBERT](../model_doc/convbert)，[CTRL](../model_doc/ctrl)，[Data2VecText](../model_doc/data2vec-text)，[DeBERTa](../model_doc/deberta)，[DeBERTa-v2](../model_doc/deberta-v2)，[DistilBERT](../model_doc/distilbert)，[ELECTRA](../model_doc/electra)，[ERNIE](../model_doc/ernie)，[ErnieM](../model_doc/ernie_m)，[ESM](../model_doc/esm)，[FlauBERT](../model_doc/flaubert)，[FNet](../model_doc/fnet)，[Funnel Transformer](../model_doc/funnel)，[GPT-Sw3](../model_doc/gpt-sw3)，[OpenAI GPT-2](../model_doc/gpt2)，[GPTBigCode](../model_doc/gpt_bigcode)，[GPT Neo](../model_doc/gpt_neo)，[GPT NeoX](../model_doc/gpt_neox)，[GPT-J](../model_doc/gptj)，[I-BERT](../model_doc/ibert)，[LayoutLM](../model_doc/layoutlm)，[LayoutLMv2](../model_doc/layoutlmv2)，[LayoutLMv3](../model_doc/layoutlmv3)，[LED](../model_doc/led)，[LiLT](../model_doc/lilt)，[LLaMA](../model_doc/llama)，[Longformer](../model_doc/longformer)，[LUKE](../model_doc/luke)，[MarkupLM](../model_doc/markuplm)，[mBART](../model_doc/mbart)，[MEGA](../model_doc/mega)，[Megatron-BERT](../model_doc/megatron-bert)，[MobileBERT](../model_doc/mobilebert)，[MPNet](../model_doc/mpnet)，[MVP](../model_doc/mvp)，[Nezha](../model_doc/nezha)，[Nystr ö mformer](../model_doc/nystromformer)，[OpenLlama](../model_doc/open-llama)，[OpenAI GPT](../model_doc/openai-gpt)，[OPT](../model_doc/opt)，[Perceiver](../model_doc/perceiver)，[PLBart](../model_doc/plbart)，[QDQBert](../model_doc/qdqbert)，[Reformer](../model_doc/reformer)，[RemBERT](../model_doc/rembert)，[RoBERTa](../model_doc/roberta)，[RoBERTa-PreLayerNorm](../model_doc/roberta-prelayernorm)，[RoCBert](../model_doc/roc_bert)，[RoFormer](../model_doc/roformer)，[SqueezeBERT](../model_doc/squeezebert)，[TAPAS](../model_doc/tapas)，[Transformer-XL](../model_doc/transfo-xl)，[XLM](../model_doc/xlm)，[XLM-RoBERTa](../model_doc/xlm-roberta)，[XLM-RoBERTa-XL](../model_doc/xlm-roberta-xl)，[XLNet](../model_doc/xlnet)，[X-MOD](../model_doc/xmod)，[YOSO](../model_doc/yoso)

<!--生成提示的结尾-->
</Tip>

开始之前，请确保您已安装所有必要的库：
```bash
pip install transformers datasets evaluate
```

我们鼓励您登录您的 Hugging Face 账户，这样您就可以将您的模型上传和共享给社区。
在提示时，输入您的令牌以登录：
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 IMDb 数据集

首先从🤗 Datasets 库加载 IMDb 数据集：
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

该数据集中有两个字段：

- `text`：电影评论文本。
- `label`：一个值，要么为 `0` 表示负面评论，要么为 `1` 表示正面评论。

## 预处理

下一步是加载 DistilBERT 分词器 (Tokenizer)以预处理 `text` 字段：
```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

创建一个预处理函数，对 `text` 进行分词和截断，以便长度不超过 DistilBERT 的最大输入长度：
```py
>>> def preprocess_function(examples):
...     return tokenizer(examples["text"], truncation=True)
```

要将预处理函数应用于整个数据集，请使用🤗 Datasets 的 [`~datasets.Dataset.map`] 函数。

通过将 `batched=True` 设置为 `map`，可以同时处理数据集的多个元素，从而加快处理速度：
```py
tokenized_imdb = imdb.map(preprocess_function, batched=True)
```

现在使用 [`DataCollatorWithPadding`] 创建一批示例。在整理过程中，将句子动态填充到一批中的最长长度，而不是将整个数据集填充到最大长度。

<frameworkcontent> 
<pt> 

 ```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
</pt> 
<tf> 

```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
```
</tf>
</frameworkcontent>


## 评估

在训练过程中，包含一个指标通常有助于评估模型的性能。您可以使用🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载一个评估方法。对于本任务，加载 [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) 指标（请参阅🤗 Evaluate [快速指南](https://huggingface.co/docs/evaluate/a_quick_tour) 以了解如何加载和计算指标）：

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

然后创建一个函数，将您的预测结果和标签传递给 [`~evaluate.EvaluationModule.compute`] 以计算准确度：
```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

您的 `compute_metrics` 函数现在已经准备就绪，在设置训练时将再次使用它。

## 训练

在开始训练模型之前，使用 `id2label` 和 `label2id` 创建一个预期 ID 到标签的映射表：
```py
>>> id2label = {0: "NEGATIVE", 1: "POSITIVE"}
>>> label2id = {"NEGATIVE": 0, "POSITIVE": 1}
```


<frameworkcontent> 
<pt> 
<Tip>

如果您对使用 [`Trainer`] 进行模型微调不熟悉，请查看这里的基本教程 [here](../training#train-with-pytorch-trainer)！
</Tip>

现在，您已准备好开始训练模型了！使用 [`AutoModelForSequenceClassification`] 加载 DistilBERT 以及预期标签的数量和标签映射：
```py
>>> from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

>>> model = AutoModelForSequenceClassification.from_pretrained(
...     "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
... )
```

此时，只剩下三个步骤：

1. 在 [`TrainingArguments`] 中定义您的训练超参数。唯一必需的参数是 `output_dir`，指定要保存模型的位置。您可以通过设置 `push_to_hub=True` 将此模型推送到 Hub（您需要登录 Hugging Face 才能上传模型）。在每个 epoch 结束时，[`Trainer`] 将评估准确性并保存训练检查点。
2. 将训练参数与模型、数据集、分词器 (Tokenizer)、数据整理器和 `compute_metrics` 函数一起传递给 [`Trainer`]。
3. 调用 [`~Trainer.train`] 来微调您的模型。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_model",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=2,
...     weight_decay=0.01,
...     evaluation_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_imdb["train"],
...     eval_dataset=tokenized_imdb["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

<Tip>

当您将 `tokenizer` 传递给 [`Trainer`] 时，默认情况下，[`Trainer`] 会应用动态填充。在这种情况下，您不需要显式指定数据整理器。
</Tip>

一旦训练完成，使用 [`~transformers.Trainer.push_to_hub`] 方法将您的模型共享到 Hub，这样每个人都可以使用您的模型：
```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

如果您不熟悉如何使用 Keras 对模型进行微调，请查看基本教程 [此处](../training#train-a-tensorflow-model-with-keras)！
</Tip> 要在 TensorFlow 中微调模型，请首先设置优化器函数、学习率调度和一些训练超参数：
```py
>>> from transformers import create_optimizer
>>> import tensorflow as tf

>>> batch_size = 16
>>> num_epochs = 5
>>> batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
>>> total_train_steps = int(batches_per_epoch * num_epochs)
>>> optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
```

然后，您可以使用 [`TFAutoModelForSequenceClassification`] 加载 DistilBERT，同时还需要指定预期标签的数量和标签映射：
```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(
...     "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
... )
```

使用 [`~transformers.TFPreTrainedModel.prepare_tf_dataset`] 将您的数据集转换为 `tf.data.Dataset` 格式：
```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_imdb["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_imdb["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

使用 [`compile`](https://keras.io/api/models/model_training_apis/#compile-method) 配置用于训练的模型。

请注意，Transformers 模型都有一个默认的与任务相关的损失函数，因此除非您想自定义，否则无需指定：

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # No loss argument!
```

在开始训练之前，还有最后两个设置任务，即从预测结果中计算准确率，并提供将模型推送到 Hub 的方法。两者都可以使用 [Keras 回调函数](../main_classes/keras_callbacks) 来完成。

将您的 `compute_metrics` 函数传递给 [`~transformers.KerasMetricCallback`]：

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

在 [`~transformers.PushToHubCallback`] 中指定要推送模型和分词器 (Tokenizer)的位置：
```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_model",
...     tokenizer=tokenizer,
... )
```

然后将您的回调函数打包在一起：
```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

最后，您已经准备好开始训练模型了！调用 [`fit`](https://keras.io/api/models/model_training_apis/#fit-method)，并传入训练集、验证集、训练轮数和回调函数来微调模型：
```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)
```

一旦训练完成，您的模型将自动上传到 Hub，这样每个人都可以使用它！

</tf>
</frameworkcontent>
<Tip>

有关如何为文本分类微调模型的更详细示例，请参阅相应的 [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb) 或者 [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)。
</Tip>

## 推断

太棒了，现在您已经微调了一个模型，可以用它进行推断了！

获取一些您希望进行推断的文本：

```py
>>> text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
```

尝试使用 [`pipeline`] 来进行推断是尝试您微调的模型最简单的方法。使用您的模型实例化一个情感分析的 `pipeline`，并将文本传递给它：
```py
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
>>> classifier(text)
[{'label': 'POSITIVE', 'score': 0.9994940757751465}]
```

如果您愿意，还可以手动复制 `pipeline` 的结果：

<frameworkcontent> 
<pt> 

 对文本进行分词，并返回 PyTorch 张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
>>> inputs = tokenizer(text, return_tensors="pt")
```

将输入传递给模型，并返回 `logits`：
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
</pt> 
<tf> 

对文本进行分词，并返回 TensorFlow 张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
>>> inputs = tokenizer(text, return_tensors="tf")
```

将输入传递给模型，并返回 `logits`：
```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
>>> logits = model(**inputs).logits
```

获取概率最高的类别，并使用模型的 `id2label` 映射将其转换为文本标签：
```py
>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
>>> model.config.id2label[predicted_class_id]
'POSITIVE'
```
</tf>
</frameworkcontent>

