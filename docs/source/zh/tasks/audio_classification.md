<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 音频分类

[[open-in-colab]]

<Youtube id="KWwzcmG98Ds"/>

音频分类和文本分类一样，会根据输入数据输出一个类别标签。区别在于，输入不再是文本，而是原始音频波形。音频分类的实际应用包括识别说话者意图、语言分类，甚至通过声音识别动物物种。

本指南将向您展示如何：

1. 在 [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 数据集上微调 [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base)，以分类说话者意图。
2. 使用微调后的模型进行推理。

<Tip>

若要查看与此任务兼容的全部架构和检查点，建议查阅[任务页面](https://huggingface.co/tasks/audio-classification)。

</Tip>

开始前，请确认已安装所有必要的库：

```bash
pip install transformers datasets evaluate soundfile librosa torchcodec
```

建议登录 Hugging Face 账户，以便将模型上传并分享给社区。出现提示时，输入令牌以登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 MInDS-14 数据集

首先从 🤗 Datasets 库加载 MInDS-14 数据集：

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

使用 [`~datasets.Dataset.train_test_split`] 方法，将数据集的 `train` 划分为较小的训练集和测试集。这样可以先进行实验并确认流程正常，再投入更多时间处理完整数据集。

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

接着查看数据集：

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

数据集包含许多有用的信息，例如 `lang_id` 和 `english_transcription`；但本指南只使用 `audio` 和 `intent_class`。使用 [`~datasets.Dataset.remove_columns`] 方法移除其他列：

```py
>>> minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
```

示例如下：

```py
>>> minds["train"][0]
{'audio': {'array': array([ 0.        ,  0.        ,  0.        , ..., -0.00048828,
         -0.00024414, -0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 8000},
 'intent_class': 2}
```

其中有两个字段：

- `audio`：语音信号的一维 `array`；访问该字段会加载音频文件，并在需要时进行重采样。
- `intent_class`：表示说话者意图的类别 ID。

为方便模型由标签 ID 获取标签名称，创建一个在标签名称与整数之间双向映射的字典：

```py
>>> labels = minds["train"].features["intent_class"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

现在可以将标签 ID 转为标签名称：

```py
>>> id2label[str(2)]
'app_error'
```

## 预处理

下一步，加载 Wav2Vec2 特征提取器来处理音频信号：

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

MInDS-14 数据集的采样率为 8kHz（可在其[数据集卡片](https://huggingface.co/datasets/PolyAI/minds14)中查看）。预训练的 Wav2Vec2 模型需要 16kHz，因此需要对数据集进行重采样：

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([ 2.2098757e-05,  4.6582241e-05, -2.2803260e-05, ...,
         -2.8419291e-04, -2.3305941e-04, -1.1425107e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 16000},
 'intent_class': 2}
```

现在创建一个预处理函数，用于：

1. 访问 `audio` 列以加载音频文件，并在必要时进行重采样。
2. 检查音频文件的采样率是否与模型预训练所用音频数据的采样率一致。该信息可在 Wav2Vec2 的[模型卡片](https://huggingface.co/facebook/wav2vec2-base)中找到。
3. 设置最大输入长度，以便批处理较长输入并截断超长部分。

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
...     )
...     return inputs
```

使用 🤗 Datasets 的 [`~datasets.Dataset.map`] 函数，将预处理函数应用到整个数据集。设置 `batched=True` 可一次处理多个样本，从而加速 `map`。移除不必要的列，并按模型要求将 `intent_class` 重命名为 `label`：

```py
>>> encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
>>> encoded_minds = encoded_minds.rename_column("intent_class", "label")
```

## 评估

训练时加入评估指标通常有助于衡量模型表现。可以通过 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载评估方法。本任务使用 [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) 指标（有关加载和计算指标的方法，请参阅 🤗 Evaluate [快速浏览](https://huggingface.co/docs/evaluate/a_quick_tour)）：

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

然后创建一个函数，将预测结果和标签传给 [`~evaluate.EvaluationModule.compute`]，以计算准确率：

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions = np.argmax(eval_pred.predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
```

`compute_metrics` 函数已经准备就绪，配置训练时会再次用到它。

## 训练

<Tip>

如果您不熟悉如何使用 [`Trainer`] 微调模型，请先阅读[基础教程](../training)。

</Tip>

现在可以开始训练模型。加载 Wav2Vec2 的 [`AutoModelForAudioClassification`]，并传入预期标签数量及标签映射：

```py
>>> from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

>>> num_labels = len(id2label)
>>> model = AutoModelForAudioClassification.from_pretrained(
...     "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
... )
```

此时只剩三个步骤：

1. 在 [`TrainingArguments`] 中定义训练超参数。唯一必需的参数是 `output_dir`，用于指定模型的保存位置。设置 `push_to_hub=True` 后，模型会被推送到 Hub（上传模型前需要登录 Hugging Face）。每个 epoch 结束时，[`Trainer`] 会评估准确率并保存训练检查点。
2. 将训练参数、模型、数据集、分词器、数据整理器以及 `compute_metrics` 函数传给 [`Trainer`]。
3. 调用 [`~Trainer.train`] 微调模型。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_mind_model",
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=3e-5,
...     per_device_train_batch_size=32,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=32,
...     num_train_epochs=10,
...     warmup_steps=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
...     report_to="trackio",
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     processing_class=feature_extractor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

训练完成后，使用 [`~transformers.Trainer.push_to_hub`] 方法将模型分享到 Hub，供所有人使用：

```py
>>> trainer.push_to_hub()
```

<Tip>

如需更深入地了解如何为音频分类微调模型，请参阅对应的 [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb)。

</Tip>

## 推理

很好，现在已经完成模型微调，可以使用它进行推理了！

加载用于推理的音频文件。必要时，请记得将音频文件重采样到与模型相同的采样率。

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

试用微调模型进行推理的最简单方式是使用 [`pipeline`]。为音频分类实例化一个 `pipeline`，传入模型和音频文件：

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

如果需要，也可以手动复现 `pipeline` 的结果：

加载特征提取器以预处理音频文件，并将 `input` 作为 PyTorch 张量返回：

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("stevhliu/my_awesome_minds_model")
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

将输入传给模型并返回 logits：

```py
>>> from transformers import AutoModelForAudioClassification

>>> model = AutoModelForAudioClassification.from_pretrained("stevhliu/my_awesome_minds_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

取概率最高的类别，并通过模型的 `id2label` 映射将其转换为标签：

```py
>>> import torch

>>> predicted_class_ids = torch.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'cash_deposit'
```
