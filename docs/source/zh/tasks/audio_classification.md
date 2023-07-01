<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可，除非符合许可证的规定，否则您不能使用此文件。您可以在许可证处获得许可证副本。
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以 "按原样" 的方式分发，不附带任何明示或暗示的担保或条件。请参阅许可证以了解许可证下的特定语言的权限和限制。特定语言的权限和限制。特定语言的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档生成器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确渲染。
-->

# 音频分类

[[在 Colab 中打开]]
<Youtube id="KWwzcmG98Ds"/>

音频分类 - 与文本类似 - 从输入数据中输出一个类别标签。唯一的区别是，您拥有的是原始音频波形，而不是文本输入。音频分类的一些实际应用包括识别说话者意图、语言分类，甚至通过声音识别动物物种。

本指南将向您展示如何：

1. 在 [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 数据集上微调 [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) 以分类说话者意图。
2. 使用您微调的模型进行推理。

<Tip> 
本教程中所示的任务由以下模型架构支持：
<!--此提示是由`make fix-copies`自动生成的，请勿手动填写！-->
[音频频谱变换器](../model_doc/audio-spectrogram-transformer)，[Data2Vec 音频](../model_doc/data2vec-audio)，[Hubert](../model_doc/hubert)，[SEW](../model_doc/sew)，[SEW-D](../model_doc/sew-d)，[UniSpeech](../model_doc/unispeech)，[UniSpeechSat](../model_doc/unispeech-sat)，[Wav2Vec2](../model_doc/wav2vec2)，[Wav2Vec2-Conformer](../model_doc/wav2vec2-conformer)，[WavLM](../model_doc/wavlm)，[Whisper](../model_doc/whisper)
<!--生成提示的末尾-->
</Tip>

开始之前，请确保您已安装所有必要的库：
```bash
pip install transformers datasets evaluate
```

我们鼓励您登录 Hugging Face 帐户，这样您就可以与社区上传和分享您的模型。在提示时，输入您的令牌登录：
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 MInDS-14 数据集

首先从🤗 Datasets 库中加载 MInDS-14 数据集：
```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

使用 [`~datasets.Dataset.train_test_split`] 方法将数据集的 `train` 拆分为较小的训练集和测试集。这样您可以在处理完整数据集之前进行实验和确保一切正常。
```py
>>> minds = minds.train_test_split(test_size=0.2)
```

然后查看数据集：
```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 450
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 113
    })
})
```

尽管数据集包含许多有用的信息，比如 `lang_id` 和 `english_transcription`，但在本指南中，您将关注 `audio` 和 `intent_class`。

使用 [`~datasets.Dataset.remove_columns`] 方法删除其他列：
```py
>>> minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
```

现在来看一个示例：
```py
>>> minds["train"][0]
{'audio': {'array': array([ 0.        ,  0.        ,  0.        , ..., -0.00048828,
         -0.00024414, -0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 8000},
 'intent_class': 2}
```

有两个字段：
- `audio`：表示必须调用以加载和重采样音频文件的语音信号的一维 `array`。- `intent_class`：表示说话者意图的类别 ID。
为了使模型能够从标签 ID 获取标签名称，创建一个将标签名称映射到整数及其相反的字典：
```py
>>> labels = minds["train"].features["intent_class"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

现在您可以将标签 ID 转换为标签名称：
```py
>>> id2label[str(2)]
'app_error'
```

## 预处理
下一步是加载 Wav2Vec2 特征提取器来处理音频信号：
```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

MInDS-14 数据集的采样率为 8000kHz（您可以在其 [数据集卡片](https://huggingface.co/datasets/PolyAI/minds14) 中找到此信息），这意味着您需要将数据集重采样为 16000kHz 才能使用预训练的 Wav2Vec2 模型：
```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([ 2.2098757e-05,  4.6582241e-05, -2.2803260e-05, ...,
         -2.8419291e-04, -2.3305941e-04, -1.1425107e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 16000},
 'intent_class': 2}
```

现在创建一个预处理函数，它：
1. 调用 `audio` 列来加载音频文件，并在必要时对音频文件进行重采样。
2. 检查音频文件的采样率是否与模型预训练时的音频数据的采样率匹配。您可以在 Wav2Vec2 [模型卡片](https://huggingface.co/facebook/wav2vec2-base) 中找到此信息。
3. 设置最大输入长度，以批处理更长的输入而不截断它们。
```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
...     )
...     return inputs
```

要将预处理函数应用于整个数据集，请使用🤗 Datasets [`~datasets.Dataset.map`] 函数。通过将 `batched=True` 设置为一次处理数据集的多个元素，可以加快 `map` 的速度。删除不需要的列，并将 `intent_class` 重命名为 `label`，因为模型期望的是这个名称：
```py
>>> encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
>>> encoded_minds = encoded_minds.rename_column("intent_class", "label")
```

## 评估

在训练过程中包括指标通常有助于评估模型的性能。您可以使用🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载评估方法。对于此任务，请加载 [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) 指标（请参阅🤗 Evaluate [快速导览](https://huggingface.co/docs/evaluate/a_quick_tour) 以了解更多关于如何加载和计算指标的信息）：

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

然后创建一个函数，将您的预测和标签传递给 [`~evaluate.EvaluationModule.compute`] 以计算准确性：
```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions = np.argmax(eval_pred.predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
```

现在您的 `compute_metrics` 函数已准备就绪，在设置培训时将返回到它。

## 训练

<frameworkcontent> 
<pt> 

 <Tip>
如果您对使用 [`Trainer`] 对模型进行微调不熟悉，请查看基本教程 [此处](../training#train-with-pytorch-trainer)！
</Tip>
现在您已经准备好开始训练模型了！使用 [`AutoModelForAudioClassification`] 加载 Wav2Vec2，同时指定预期的标签数量和标签映射：
```py
>>> from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

>>> num_labels = len(id2label)
>>> model = AutoModelForAudioClassification.from_pretrained(
...     "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
... )
```

此时，只剩下三个步骤：

1. 在 [`TrainingArguments`] 中定义您的训练超参数。唯一必需的参数是 `output_dir`，指定要保存模型的位置。您可以通过设置 `push_to_hub=True` 将此模型上传到 Hub（需要登录 Hugging Face 才能上传模型）。在每个 epoch 结束时，[`Trainer`] 将评估准确性并保存训练检查点。
2. 将训练参数与模型、数据集、分词器 (Tokenizer)、数据整理器和 `compute_metrics` 函数一起传递给 [`Trainer`]。
3. 调用 [`~Trainer.train`] 来微调您的模型。


```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_mind_model",
...     evaluation_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=3e-5,
...     per_device_train_batch_size=32,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=32,
...     num_train_epochs=10,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     tokenizer=feature_extractor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

完成训练后，使用 [`~transformers.Trainer.push_to_hub`] 方法将您的模型共享到 Hub，以便每个人都可以使用您的模型：
```py
>>> trainer.push_to_hub()
```
</pt> 
</frameworkcontent>
<Tip>

有关如何为音频分类微调模型的更详细示例，请参阅相应的 [PyTorch 笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb)。
</Tip>

## 推理

很好，现在您已经微调了模型，可以将其用于推理！
加载要运行推理的音频文件。如果需要，请记得将音频文件的采样率进行重采样，以匹配模型的采样率！

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

尝试使用 [`pipeline`] 在推理中使用您微调的模型是最简单的方法。使用您的模型实例化一个音频分类的 `pipeline`，并将音频文件传递给它：
```py
>>> from transformers import pipeline

>>> classifier = pipeline("audio-classification", model="stevhliu/my_awesome_minds_model")
>>> classifier(audio_file)
[
    {'score': 0.09766869246959686, 'label': 'cash_deposit'},
    {'score': 0.07998877018690109, 'label': 'app_error'},
    {'score': 0.0781070664525032, 'label': 'joint_account'},
    {'score': 0.07667109370231628, 'label': 'pay_bill'},
    {'score': 0.0755252093076706, 'label': 'balance'}
]
```

如果愿意，您也可以手动复制 `pipeline` 的结果：

<frameworkcontent> 
<pt> 

 加载一个特征提取器来预处理音频文件，并将 `input` 返回为 PyTorch 张量：

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("stevhliu/my_awesome_minds_model")
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

将您的输入传递给模型，并返回逻辑值：
```py
>>> from transformers import AutoModelForAudioClassification

>>> model = AutoModelForAudioClassification.from_pretrained("stevhliu/my_awesome_minds_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

获取具有最高概率的类，并使用模型的 `id2label` 映射将其转换为标签：
```py
>>> import torch

>>> predicted_class_ids = torch.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'cash_deposit'
```
</pt> 
</frameworkcontent>