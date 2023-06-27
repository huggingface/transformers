<!--版权 2023 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可; 除非符合许可证的要求，否则您不得使用此文件许可证。您可以在下面获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律或书面约定，按“原样”基础分发的软件均按照许可证分发没有任何明示或暗示的保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️注意，该文件以 Markdown 格式编写，但包含我们的文档生成器的特定语法（类似于 MDX），可能不会在 Markdown 查看器中正确显示。
-->

# 自动语音识别

[[在 Colab 中打开]]
<Youtube id="TksaY_FDgnk"/>
自动语音识别（ASR）将语音信号转换为文本，将一系列音频输入映射到文本输出。像 Siri 和 Alexa 这样的虚拟助手使用 ASR 模型来帮助用户，还有许多其他有用的用户界面应用，比如会议期间的实时字幕和记录笔记。

本指南将向您展示如何：

1. 在 [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 数据集上微调 [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) 模型，将音频转录为文本。
2. 使用您微调的模型进行推理。

<Tip> 

本教程中演示的任务由以下模型架构支持：
<!--此提示由`make fix-copies`自动生成，不要手动填写！-->
[Data2VecAudio](../model_doc/data2vec-audio), [Hubert](../model_doc/hubert), [M-CTC-T](../model_doc/mctct), [SEW](../model_doc/sew), [SEW-D](../model_doc/sew-d), [UniSpeech](../model_doc/unispeech), [UniSpeechSat](../model_doc/unispeech-sat), [Wav2Vec2](../model_doc/wav2vec2), [Wav2Vec2-Conformer](../model_doc/wav2vec2-conformer), [WavLM](../model_doc/wavlm)
<!--生成提示的末尾-->
</Tip>

开始之前，请确保已安装所有必要的库：
```bash
pip install transformers datasets evaluate jiwer
```

我们鼓励您登录您的 Hugging Face 帐户，以便您可以将您的模型上传和分享给社区。当提示时，请输入您的令牌以登录：
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 MInDS-14 数据集

首先，使用🤗 Datasets 库加载 [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 数据集的较小子集。这样可以让您有机会在使用完整数据集进行更长时间的训练之前进行实验和确保一切正常。
```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
```

使用 [`~Dataset.train_test_split`] 方法将数据集的 `train` 部分拆分为训练集和测试集：
```py
>>> minds = minds.train_test_split(test_size=0.2)
```

然后查看数据集：
```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 16
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 4
    })
})
```

虽然数据集包含许多有用的信息，如 `lang_id` 和 `english_transcription`，但在本指南中，您将专注于 `audio` 和 `transcription`。
使用 [`~datasets.Dataset.remove_columns`] 方法删除其他列：
```py
>>> minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
```

再次查看示例：
```py
>>> minds["train"][0]
{'audio': {'array': array([-0.00024414,  0.        ,  0.        , ...,  0.00024414,
          0.00024414,  0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 8000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

有两个字段：
- `audio`：一个一维的 `array`，表示必须调用以加载和重新采样音频文件的语音信号。- `transcription`：目标文本。

## 预处理

下一步是加载一个 Wav2Vec2 处理器来处理音频信号：
```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
```

MInDS-14 数据集的采样率为 8000kHz（可以在其 [数据集卡片](https://huggingface.co/datasets/PolyAI/minds14) 中找到此信息），这意味着您需要将数据集重新采样为 16000kHz，以使用预训练的 Wav2Vec2 模型：
```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([-2.38064706e-04, -1.58618059e-04, -5.43987835e-06, ...,
          2.78103951e-04,  2.38446111e-04,  1.18740834e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 16000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

如上所示的 `transcription` 中，文本包含大小写字符的混合。Wav2Vec2 tokenizer 仅在大写字符上进行训练，因此您需要确保文本与 tokenizer 的词汇匹配：
```py
>>> def uppercase(example):
...     return {"transcription": example["transcription"].upper()}


>>> minds = minds.map(uppercase)
```

现在创建一个预处理函数，该函数：

1. 调用 `audio` 列以加载和重新采样音频文件。
2. 使用处理器从音频文件中提取 `input_values` 并对 `transcription` 列进行标记。
```py
>>> def prepare_dataset(batch):
...     audio = batch["audio"]
...     batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
...     batch["input_length"] = len(batch["input_values"][0])
...     return batch
```

要在整个数据集上应用预处理函数，使用🤗 Datasets [`~datasets.Dataset.map`] 函数。您可以通过增加 `num_proc` 参数来加快 `map` 的速度。

使用 [`~datasets.Dataset.remove_columns`] 方法删除不需要的列：
```py
>>> encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4)
```

🤗 Transformers 没有用于 ASR 的数据整理器，因此您需要适应 [`DataCollatorWithPadding`] 以创建一批示例。它还会动态填充文本和标签，使它们的长度与其批次中最长元素的长度相同（而不是整个数据集的长度），以便它们具有统一的长度。虽然可以通过在 `tokenizer` 函数中设置 `padding=True` 来填充文本，但动态填充更高效。

与其他数据整理器不同，该特定数据整理器需要为 `input_values` 和 `labels` 应用不同的填充方法：

```py
>>> import torch

>>> from dataclasses import dataclass, field
>>> from typing import Any, Dict, List, Optional, Union


>>> @dataclass
... class DataCollatorCTCWithPadding:
...     processor: AutoProcessor
...     padding: Union[bool, str] = "longest"

...     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
...         # split inputs and labels since they have to be of different lengths and need
...         # different padding methods
...         input_features = [{"input_values": feature["input_values"][0]} for feature in features]
...         label_features = [{"input_ids": feature["labels"]} for feature in features]

...         batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

...         labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

...         # replace padding with -100 to ignore loss correctly
...         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

...         batch["labels"] = labels

...         return batch
```

现在实例化您的 `DataCollatorForCTCWithPadding`：
```py
>>> data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
```

## 评估

在训练过程中包含度量标准通常有助于评估模型的性能。您可以使用🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载一个评估方法。对于此任务，加载 [word error rate](https://huggingface.co/spaces/evaluate-metric/wer)（WER）度量标准（有关如何加载和计算度量标准的更多信息，请参阅🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour)）：

```py
>>> import evaluate

>>> wer = evaluate.load("wer")
```

然后创建一个函数，将您的预测和标签传递给 [`~evaluate.EvaluationModule.compute`] 以计算 WER：
```py
>>> import numpy as np


>>> def compute_metrics(pred):
...     pred_logits = pred.predictions
...     pred_ids = np.argmax(pred_logits, axis=-1)

...     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

...     pred_str = processor.batch_decode(pred_ids)
...     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

...     wer = wer.compute(predictions=pred_str, references=label_str)

...     return {"wer": wer}
```

您的 `compute_metrics` 函数已准备就绪，当您设置训练时，将返回该函数。

## 训练

<frameworkcontent> 
<pt> 
 <Tip>
如果您对使用 [`Trainer`] 进行模型微调不熟悉，请查看基本教程 [此处](../training#train-with-pytorch-trainer)！
</Tip>

现在，您可以开始训练模型了！使用 [`AutoModelForCTC`] 加载 Wav2Vec2。

通过 `ctc_loss_reduction` 参数指定要应用的缩减方式。通常，使用平均而不是默认的求和更好：

```py
>>> from transformers import AutoModelForCTC, TrainingArguments, Trainer

>>> model = AutoModelForCTC.from_pretrained(
...     "facebook/wav2vec2-base",
...     ctc_loss_reduction="mean",
...     pad_token_id=processor.tokenizer.pad_token_id,
... )
```

此时，只剩下三个步骤：
1. 在 [`TrainingArguments`] 中定义您的训练超参数。唯一必需的参数是 `output_dir`，用于指定保存模型的位置。通过设置 `push_to_hub=True` 将此模型推送到 Hub（您需要登录 Hugging Face 以上传您的模型）。在每个 epoch 结束时，[`Trainer`] 将评估 WER 并保存训练检查点。
2. 将训练参数与模型、数据集、tokenizer、数据整理器和 `compute_metrics` 函数一起传递给 [`Trainer`]。
3. 调用 [`~Trainer.train`] 来微调您的模型。
```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_asr_mind_model",
...     per_device_train_batch_size=8,
...     gradient_accumulation_steps=2,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=2000,
...     gradient_checkpointing=True,
...     fp16=True,
...     group_by_length=True,
...     evaluation_strategy="steps",
...     per_device_eval_batch_size=8,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     load_best_model_at_end=True,
...     metric_for_best_model="wer",
...     greater_is_better=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     tokenizer=processor,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

完成训练后，使用 [`~transformers.Trainer.push_to_hub`] 方法将您的模型共享到 Hub，以便每个人都可以使用您的模型：
```py
>>> trainer.push_to_hub()
```
</pt> </frameworkcontent>
<Tip>
有关如何为自动语音识别微调模型的更详细示例，请参阅此博客 [文章](https://huggingface.co/blog/fine-tune-wav2vec2-english)（英语 ASR）和此 [文章](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)（多语言 ASR）。
</Tip>

## 推理

太棒了，现在您已经微调了一个模型，可以用它进行推理了！
加载要进行推理的音频文件。如果需要，请记得对音频文件的采样率进行重新采样以与模型的采样率匹配！

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

尝试使用 [`pipeline`] 来对您微调的模型进行推理是最简单的方法。使用您的模型实例化一个自动语音识别的 `pipeline`，并将音频文件传递给它：
```py
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="stevhliu/my_awesome_asr_minds_model")
>>> transcriber(audio_file)
{'text': 'I WOUD LIKE O SET UP JOINT ACOUNT WTH Y PARTNER'}
```

<Tip>

转录结果还不错，但可以更好！尝试使用更多的示例来微调您的模型以获得更好的结果！
</Tip>
如果需要，您也可以手动复制 `pipeline` 的结果：

<frameworkcontent> 
<pt> 

 加载处理器以对音频文件和转录进行预处理，并将 `input` 返回为 PyTorch 张量：

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

将您的输入传递给模型并返回对数概率：
```py
>>> from transformers import AutoModelForCTC

>>> model = AutoModelForCTC.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

获得具有最高概率的预测 `input_ids`，并使用处理器将预测的 `input_ids` 解码回文本：
```py
>>> import torch

>>> predicted_ids = torch.argmax(logits, dim=-1)
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription
['I WOUL LIKE O SET UP JOINT ACOUNT WTH Y PARTNER']
```
</pt> 
</frameworkcontent>