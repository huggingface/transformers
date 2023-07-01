<!--版权所有2022年的HuggingFace团队。保留所有权利。
根据Apache许可证第2版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”分发，不附带任何形式的担保或条件，无论是明示还是暗示。请查看许可证了解具体的权限和限制。
⚠️ 请注意，该文件是Markdown格式，但包含我们的文档构建器（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确呈现。
-->

# 翻译

[[open-in-colab]]
<Youtube id="1JvfrvZgi6c"/>

翻译将一个文本序列从一种语言转换为另一种语言。它是多个任务之一，可以将其表述为序列到序列问题，这是一个从输入返回某些输出的强大框架，例如翻译或摘要。翻译系统通常用于不同语言文本之间的翻译，但也可以用于语音或文本到语音或语音到文本之间的某种组合。
本指南将向您展示如何：

1. 在[OPUS图书](https://huggingface.co/datasets/opus_books)数据集的英法子集上对[T5](https://huggingface.co/t5-small)进行微调，以将英文文本翻译成法语。2. 使用您微调的模型进行推理。

<Tip>
本教程中的任务受以下模型架构的支持：
<!--此提示由`make fix-copies`自动生成，请勿手动填写！-->
[BART](../model_doc/bart)，[BigBird-Pegasus](../model_doc/bigbird_pegasus)，[Blenderbot](../model_doc/blenderbot)，[BlenderbotSmall](../model_doc/blenderbot-small)，[Encoder decoder](../model_doc/encoder-decoder)，[FairSeq Machine-Translation](../model_doc/fsmt)，[GPTSAN-japanese](../model_doc/gptsan-japanese)，[LED](../model_doc/led)，[LongT5](../model_doc/longt5)，[M2M100](../model_doc/m2m_100)，[Marian](../model_doc/marian)，[mBART](../model_doc/mbart)，[MT5](../model_doc/mt5)，[MVP](../model_doc/mvp)，[NLLB](../model_doc/nllb)，[NLLB-MOE](../model_doc/nllb-moe)，[Pegasus](../model_doc/pegasus)，[PEGASUS-X](../model_doc/pegasus_x)，[PLBart](../model_doc/plbart)，[ProphetNet](../model_doc/prophetnet)，[SwitchTransformers](../model_doc/switch_transformers)，[T5](../model_doc/t5)，[XLM-ProphetNet](../model_doc/xlm-prophetnet)
<!--生成提示的末尾-->
</Tip>

开始之前，请确保已安装所有必要的库：
```bash
pip install transformers datasets evaluate sacrebleu
```

我们鼓励您登录您的Hugging Face账户，这样您就可以上传和共享您的模型。当提示时，请输入您的令牌以登录：
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载OPUS图书数据集

首先，使用🤗 Datasets库加载[OPUS图书](https://huggingface.co/datasets/opus_books)数据集的英法子集：
```py
>>> from datasets import load_dataset

>>> books = load_dataset("opus_books", "en-fr")
```

使用[`~datasets.Dataset.train_test_split`]方法将数据集拆分为训练集和测试集：
```py
>>> books = books["train"].train_test_split(test_size=0.2)
```

然后看一个例子：
```py
>>> books["train"][0]
{'id': '90560',
 'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
  'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}
```

`translation`：文本的英文和法文翻译。
## 预处理
<Youtube id="XAR8jnZZuUs"/>
下一步是加载T5分词器来处理英法语言对：
```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

您要创建的预处理函数需要执行以下操作：
1. 为输入添加提示，以便T5知道这是一个翻译任务。某些能够执行多个NLP任务的模型需要为特定任务提供提示。
2. 将输入（英文）和目标（法文）分别进行标记化，因为无法使用在英文词汇上预训练的标记器对法文文本进行标记化。
3. 将序列截断为由`max_length`参数设置的最大长度。
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

要在整个数据集上应用预处理函数，使用🤗 Datasets的[`~datasets.Dataset.map`]方法。通过将`batched=True`设置为一次处理数据集的多个元素，您可以加速`map`函数的处理速度：
```py
>>> tokenized_books = books.map(preprocess_function, batched=True)
```

使用[`DataCollatorForSeq2Seq`]创建一批示例。
在整理过程中，将句子动态填充到批中的最长长度，而不是将整个数据集填充到最大长度。
<frameworkcontent>
<pt>

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```
</pt>
<tf>
```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")
```
</tf>
</frameworkcontent>

## 评估

在训练过程中包含度量标准通常有助于评估模型的性能。您可以使用🤗 [Evaluate](https://huggingface.co/docs/evaluate/index)库快速加载评估方法。对于此任务，加载[SacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu)度量标准（请参阅🤗 Evaluate [快速导览](https://huggingface.co/docs/evaluate/a_quick_tour)以了解有关如何加载和计算度量标准的更多信息）：
```py
>>> import evaluate

>>> metric = evaluate.load("sacrebleu")
```

然后创建一个函数，将您的预测和标签传递给[`~evaluate.EvaluationModule.compute`]以计算SacreBLEU分数：
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

您的`compute_metrics`函数现在已经准备好了，在设置训练时会返回到它。

## 训练

<frameworkcontent><pt>
<Tip>
如果您对使用[`Trainer`]进行模型微调不熟悉，请查看基本教程[此处](../training#train-with-pytorch-trainer)！
</Tip>

现在，您已准备好开始训练您的模型了！使用[`AutoModelForSeq2SeqLM`]加载T5：
```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

此时，只剩下三个步骤：

1. 在[`Seq2SeqTrainingArguments`]中定义您的训练超参数。唯一需要的参数是`output_dir`，它指定保存模型的位置。通过设置`push_to_hub=True`将此模型推送到Hub（需要登录Hugging Face以上传您的模型）。在每个时期结束时，[`Trainer`]将评估SacreBLEU度量标准并保存训练检查点。
2. 将训练参数与模型、数据集、标记器、数据整理器和`compute_metrics`函数一起传递给[`Seq2SeqTrainer`]。
3. 调用[`~Trainer.train`]来微调您的模型。

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_opus_books_model",
...     evaluation_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=2,
...     predict_with_generate=True,
...     fp16=True,
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_books["train"],
...     eval_dataset=tokenized_books["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
````

完成训练后，使用[`~transformers.Trainer.push_to_hub`]方法将您的模型共享到Hub，以便每个人都可以使用您的模型：
```py
>>> trainer.push_to_hub()
```

</pt>
<tf>
<Tip>

如果您对使用Keras进行模型微调不熟悉，请查看基本教程[此处](../training#train-a-tensorflow-model-with-keras)！
</Tip>

要在TensorFlow中微调模型，请首先设置优化器函数、学习率计划和一些训练超参数：

```py
>>> from transformers import AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

然后，使用[`TFAutoModelForSeq2SeqLM`]加载T5：
```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

使用[`~transformers.TFPreTrainedModel.prepare_tf_dataset`]将数据集转换为`tf.data.Dataset`格式：
```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_books["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     tokenized_books["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

使用[`compile`](https://keras.io/api/models/model_training_apis/#compile-method)为训练配置模型。请注意，Transformers模型都具有默认的与任务相关的损失函数，因此您无需指定损失函数，除非您希望使用其他损失函数：
```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # No loss argument!
```

在开始训练之前，最后两件要设置的事情是从预测中计算SacreBLEU度量标准，并提供一种将模型推送到Hub的方式。这两个任务都可以通过使用[Keras回调](../main_classes/keras_callbacks)来完成。
将您的`compute_metrics`函数传递给[`~transformers.KerasMetricCallback`]：
```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

在[`~transformers.PushToHubCallback`]中指定要推送模型和分词器的位置：
```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_opus_books_model",
...     tokenizer=tokenizer,
... )
```

然后将回调函数捆绑在一起：
```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

最后，您可以开始训练模型了！使用[`fit`](https://keras.io/api/models/model_training_apis/#fit-method)方法，传入训练数据集、验证数据集、训练轮数和回调函数来微调模型：
```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)
```

训练完成后，您的模型将自动上传到Hub，这样每个人都可以使用它！</tf></frameworkcontent>
<Tip>
如果您想了解有关如何为翻译微调模型的更详细示例，请查看相应的[PyTorch笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb)或者[TensorFlow笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb)。

</Tip>

## 推断

太棒了，现在您已经微调了一个模型，可以用它进行推断了！
准备一些您想要翻译成其他语言的文本。对于T5模型，您需要根据正在处理的任务为输入添加前缀。例如，要将英文翻译为法文，您应该像下面这样为输入添加前缀：

```py
>>> text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
```

尝试使用您微调的模型进行推断的最简单方法是在[`pipeline`]中使用它。实例化一个用于翻译的`pipeline`，并将文本传递给它：
```py
>>> from transformers import pipeline

>>> translator = pipeline("translation", model="my_awesome_opus_books_model")
>>> translator(text)
[{'translation_text': 'Legumes partagent des ressources avec des bactéries azotantes.'}]
```

如果您想要手动复制`pipeline`的结果，也可以这样做：
<frameworkcontent><pt>将文本进行标记化，并将`input_ids`返回为PyTorch张量：
```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

使用[`~transformers.generation_utils.GenerationMixin.generate`]方法生成翻译结果。

有关不同文本生成策略和控制生成的参数的更多详细信息，请查看[Text Generation](../main_classes/text_generation) API。

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("my_awesome_opus_books_model")
>>> outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
```

将生成的标记ID解码为文本：
```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Les lignées partagent des ressources avec des bactéries enfixant l'azote.'
```
</pt><tf>将文本进行标记化，并将`input_ids`返回为TensorFlow张量：
```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")
>>> inputs = tokenizer(text, return_tensors="tf").input_ids
```

使用[`~transformers.generation_tf_utils.TFGenerationMixin.generate`]方法生成翻译结果。

有关不同文本生成策略和控制生成的参数的更多详细信息，请查看[Text Generation](../main_classes/text_generation) API。

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained("my_awesome_opus_books_model")
>>> outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
```

将生成的标记ID解码为文本：
```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Les lugumes partagent les ressources avec des bactéries fixatrices d'azote.'
```
</tf>
</frameworkcontent>