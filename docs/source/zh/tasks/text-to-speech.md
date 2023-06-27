<!--版权所有 2023 年的 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可; 除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面协议同意，否则根据许可证分发的软件将按照“按原样”基础分发，不附带任何明示或暗示的担保或条件。请参阅许可证的特定语言下的权限和限制。
⚠️请注意，该文件是 Markdown 格式，但包含特定的语法，供我们的文档构建器（类似于 MDX）使用，可能无法在您的 Markdown 查看器中正确显示。
-->

# 文字转语音

[[在 Colab 中打开]]

文字转语音（TTS）是将文本转换为自然语音的任务，其中语音可以以多种语言和多个讲话者生成。当前在🤗 Transformers 中只有一个文字转语音模型 [SpeechT5](model_doc/speecht5)，但未来将会添加更多。

SpeechT5 是基于语音到文本和文本到语音数据进行预训练的，因此它可以学习到文本和语音共享的隐藏表示的统一空间。这意味着同一个预训练模型可以用于不同的任务。此外，SpeechT5 通过 x-vector 讲话者嵌入支持多个讲话者。

本指南演示了如何：
1. 对在荷兰（`nl`）语言子集上使用英文语音进行原始训练的 [SpeechT5](model_doc/speecht5) 进行微调。
2. 使用已经微调的模型进行推断。

开始之前，请确保您已安装所有必要的库：
```bash
pip install datasets soundfile speechbrain accelerate
```

从源代码安装🤗 Transformers，因为尚未将所有 SpeechT5 功能合并到官方发布中：
```bash
pip install git+https://github.com/huggingface/transformers.git
```

<Tip>

要按照本指南操作，您需要一个 GPU。如果您在笔记本中工作，请运行以下代码行以检查是否有 GPU 可用：
```bash
!nvidia-smi
```

</Tip>

我们鼓励您登录您的 Hugging Face 帐户，与社区一起上传和共享您的模型。在提示时，请输入您的令牌进行登录：
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载数据集

[VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) 是一个大规模的多语言语音语料库，由 2009 年至 2020 年的欧洲议会活动记录中的数据组成。它包含了 15 种欧洲语言的标记音频转录数据。

在本指南中，我们使用荷兰语子集，您可以选择其他子集。
请注意，VoxPopuli 或任何其他自动语音识别（ASR）数据集可能不适合训练 TTS 模型。对于 ASR 有益的特性（如过多的背景噪声），在 TTS 中通常是不可取的。然而，找到高质量的多语言和多讲话者的 TTS 数据集可能相当具有挑战性。

让我们加载数据：


```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
>>> len(dataset)
20968
```

20968 个示例应该足够用于微调。SpeechT5 期望音频数据的采样率为 16 kHz，因此请确保数据集中的示例满足此要求：
```py
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

## 预处理数据

首先，让我们定义要使用的模型检查点并加载适当的处理器：
```py
>>> from transformers import SpeechT5Processor

>>> checkpoint = "microsoft/speecht5_tts"
>>> processor = SpeechT5Processor.from_pretrained(checkpoint)
```

### SpeechT5 标记化的文本清理

首先清理文本数据。您将需要处理器的分词器 (Tokenizer)部分来处理文本：
```py
>>> tokenizer = processor.tokenizer
```

数据集示例包含“raw_text”和“normalized_text”特征。在决定使用哪个特征作为文本输入时，请考虑 SpeechT5 分词器 (Tokenizer)没有任何数字标记。

在“normalized_text”中，数字被写为文本。因此，它更适合，我们建议使用“normalized_text”作为输入文本。
因为 SpeechT5 是在英语上进行训练的，它可能无法识别荷兰语数据集中的某些字符。如果保持不变，这些字符将被转换为 `<unk>` 标记。

然而，在荷兰语中，某些字符如 `à` 用于强调音节。为了保持文本的意义，我们可以用常规的 `a` 字符替换这个字符。

为了识别前一步骤中识别出的不受支持的标记，使用与字符作为标记进行工作的 `SpeechT5Tokenizer` 提取数据集中的所有唯一字符。

为此，编写 `extract_all_chars` 映射函数，将所有示例的转录连接成一个字符串，并将其转换为字符集合。

确保在 `dataset.map()` 中设置 `batched=True` 和 `batch_size=-1`，以便一次性获得所有转录。

```py
>>> def extract_all_chars(batch):
...     all_text = " ".join(batch["normalized_text"])
...     vocab = list(set(all_text))
...     return {"vocab": [vocab], "all_text": [all_text]}


>>> vocabs = dataset.map(
...     extract_all_chars,
...     batched=True,
...     batch_size=-1,
...     keep_in_memory=True,
...     remove_columns=dataset.column_names,
... )

>>> dataset_vocab = set(vocabs["vocab"][0])
>>> tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
```

现在您有两个字符集：一个是来自数据集的词汇表，另一个是来自分词器 (Tokenizer)的词汇表。

要识别数据集中的任何不受支持的字符，可以取这两个集合之间的差异。结果集将包含数据集中存在但不在分词器 (Tokenizer)中的字符。

```py
>>> dataset_vocab - tokenizer_vocab
{' ', 'à', 'ç', 'è', 'ë', 'í', 'ï', 'ö', 'ü'}
```

为了处理前一步骤中识别出的不受支持的字符，定义一个将这些字符映射为有效标记的函数。

请注意，分词器 (Tokenizer)中的空格已被替换为 `▁`，不需要单独处理。

```py
>>> replacements = [
...     ("à", "a"),
...     ("ç", "c"),
...     ("è", "e"),
...     ("ë", "e"),
...     ("í", "i"),
...     ("ï", "i"),
...     ("ö", "o"),
...     ("ü", "u"),
... ]


>>> def cleanup_text(inputs):
...     for src, dst in replacements:
...         inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
...     return inputs


>>> dataset = dataset.map(cleanup_text)
```

现在，您已经处理了文本中的特殊字符，是时候将注意力转向音频数据了。

### 讲话者

VoxPopuli 数据集包括来自多个讲话者的语音，但数据集中有多少讲话者呢？可以通过计算唯一讲话者的数量以及每个讲话者在数据集中贡献的示例数量来确定这一点。

在数据集中共有 20968 个示例，此信息将更好地了解数据中讲话者和示例的分布。通过绘制直方图，您可以了解每个讲话者的数据量。

```py
>>> from collections import defaultdict

>>> speaker_counts = defaultdict(int)

>>> for speaker_id in dataset["speaker_id"]:
...     speaker_counts[speaker_id] += 1
```

通过绘制直方图，您可以了解每个讲话者的数据量。
```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt.hist(speaker_counts.values(), bins=20)
>>> plt.ylabel("Speakers")
>>> plt.xlabel("Examples")
>>> plt.show()
```

<div class="flex justify-center">    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_speakers_histogram.png" alt="Speakers histogram"/> </div>

直方图显示，数据集中约有三分之一的讲话者拥有少于 100 个示例，而约有十个讲话者拥有超过 500 个示例。

为了提高训练效率并平衡数据集，我们可以将数据限制在具有 100 到 400 个示例之间的讲话者。

```py
>>> def select_speaker(speaker_id):
...     return 100 <= speaker_counts[speaker_id] <= 400


>>> dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
```

让我们检查剩下多少个讲话者：
```py
>>> len(set(dataset["speaker_id"]))
42
```

让我们看看还剩下多少个示例：
```py
>>> len(dataset)
9973
```

您还剩下将近 10,000 个示例，来自约 40 个唯一的讲话者，应该足够使用。

请注意，某些示例较少的讲话者如果示例较长，则实际上可能有更多的音频可用。但是，确定每个讲话者的总音频量需要扫描整个数据集，这是一个耗时的过程，需要加载和解码每个音频文件。因此，在此处我们选择跳过此步骤。

### 讲话者嵌入

为了使 TTS 模型能够区分多个讲话者，您需要为每个示例创建一个讲话者嵌入。说话者嵌入是模型的另一个输入，用于捕捉特定说话者的声音特征。要生成这些说话者嵌入，请使用预训练的 [spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) 模型。SpeechBrain 中的模型。

创建一个名为 `create_speaker_embedding()` 的函数，该函数接受输入音频波形并输出包含相应说话者嵌入的 512 元素向量。包含相应说话者嵌入的 512 元素向量。

```py
>>> import os
>>> import torch
>>> from speechbrain.pretrained import EncoderClassifier

>>> spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> speaker_model = EncoderClassifier.from_hparams(
...     source=spk_model_name,
...     run_opts={"device": device},
...     savedir=os.path.join("/tmp", spk_model_name),
... )


>>> def create_speaker_embedding(waveform):
...     with torch.no_grad():
...         speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
...         speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
...         speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
...     return speaker_embeddings
```

需要注意的是，`speechbrain/spkrec-xvect-voxceleb` 模型是在 VoxCeleb 数据集的英语语音上进行训练的，而本指南中的训练示例是荷兰语的。

尽管我们相信该模型仍会为我们的荷兰语数据集生成合理的说话者嵌入，但这种假设可能并非在所有情况下都成立。数据集，而这个指南中的训练示例是荷兰语的。尽管我们相信该模型仍会为我们的荷兰语数据集生成合理的说话者嵌入，但这种假设可能并非在所有情况下都成立。合理的说话者嵌入，但这种假设可能并非在所有情况下都成立。

为了获得最佳结果，建议首先在目标语音上训练一个 X-vector 模型。这样可以确保模型能够更好地捕捉荷兰语中独特的声音特征。捕捉荷兰语中独特的声音特征。

### 处理数据集

最后，让我们将数据处理成模型所期望的格式。创建一个 `prepare_dataset` 函数，该函数接受一个示例，并使用 `SpeechT5Processor` 对象对输入文本进行标记化，并将目标音频加载到对数梅尔频谱图中。对数梅尔频谱图中。

还应该将说话者嵌入作为额外的输入添加进去。

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example = processor(
...         text=example["normalized_text"],
...         audio_target=audio["array"],
...         sampling_rate=audio["sampling_rate"],
...         return_attention_mask=False,
...     )

...     # strip off the batch dimension
...     example["labels"] = example["labels"][0]

...     # use SpeechBrain to obtain x-vector
...     example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

...     return example
```

通过查看单个示例来验证处理是否正确：
```py
>>> processed_example = prepare_dataset(dataset[0])
>>> list(processed_example.keys())
['input_ids', 'labels', 'stop_labels', 'speaker_embeddings']
```

说话者嵌入应该是一个 512 元素的向量：
```py
>>> processed_example["speaker_embeddings"].shape
(512,)
```

标签应该是一个具有 80 个 mel 频带的对数梅尔频谱图。
```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt.imshow(processed_example["labels"].T)
>>> plt.show()
```

<div class="flex justify-center">    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_1.png" alt="具有80个mel频带的对数梅尔频谱图"/> </div>

顺便提一下：如果您觉得这个频谱图令人困惑，可能是因为您对将低频放在绘图底部、高频放在绘图顶部的约定很熟悉。然而，在使用 matplotlib 库将频谱图绘制为图像时，y 轴被翻转，频谱图会颠倒。

y 轴被翻转，频谱图会颠倒。频谱图会颠倒。

现在将处理函数应用于整个数据集。这将需要 5 到 10 分钟的时间。

```py
>>> dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
```

您将看到一个警告，该警告表示数据集中的一些示例长度超过了模型可以处理的最大输入长度（600 个标记）。

将数据集中的这些示例移除。在这里，我们进一步移除超过 200 个标记的内容，以便允许更大的批量大小。

```py
>>> def is_not_too_long(input_ids):
...     input_length = len(input_ids)
...     return input_length < 200


>>> dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
>>> len(dataset)
8259
```

接下来，创建一个基本的训练/测试拆分：
```py
>>> dataset = dataset.train_test_split(test_size=0.1)
```

### 数据整合器

为了将多个示例组合成一个批次，您需要定义一个自定义数据整合器。该整合器将使用填充标记填充较短的序列，确保所有示例的长度相同。对于频谱图标签，填充部分将被替换为特殊值 `-100`。

该特殊值指示模型在计算频谱图损失时忽略该部分。该特殊值指示模型在计算频谱图损失时忽略该部分。该特殊值指示模型在计算频谱图损失时忽略该部分。

```py
>>> from dataclasses import dataclass
>>> from typing import Any, Dict, List, Union


>>> @dataclass
... class TTSDataCollatorWithPadding:
...     processor: Any

...     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
...         input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
...         label_features = [{"input_values": feature["labels"]} for feature in features]
...         speaker_features = [feature["speaker_embeddings"] for feature in features]

...         # collate the inputs and targets into a batch
...         batch = processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

...         # replace padding with -100 to ignore loss correctly
...         batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)

...         # not used during fine-tuning
...         del batch["decoder_attention_mask"]

...         # round down target lengths to multiple of reduction factor
...         if model.config.reduction_factor > 1:
...             target_lengths = torch.tensor([len(feature["input_values"]) for feature in label_features])
...             target_lengths = target_lengths.new(
...                 [length - length % model.config.reduction_factor for length in target_lengths]
...             )
...             max_length = max(target_lengths)
...             batch["labels"] = batch["labels"][:, :max_length]

...         # also add in the speaker embeddings
...         batch["speaker_embeddings"] = torch.tensor(speaker_features)

...         return batch
```

在 SpeechT5 中，模型的解码器部分的输入减少了一半。换句话说，它会从目标序列中丢弃每隔一个时间步长。然后，解码器会预测一个长度为原始目标序列两倍的序列。由于原始目标序列长度可能是奇数，数据整合器确保将批次的最大长度向下取整为 2 的倍数。

从目标序列中丢弃每隔一个时间步长。预测一个长度为原始目标序列两倍的序列。将批次的最大长度向下取整为 2 的倍数。

```py 
>>> data_collator = TTSDataCollatorWithPadding(processor=processor)
```

## 训练模型

从与加载处理器相同的检查点加载预训练模型：

```py
>>> from transformers import SpeechT5ForTextToSpeech

>>> model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
```

`use_cache=True` 选项与渐变检查点不兼容。在训练过程中禁用它。
```py 
>>> model.config.use_cache = False
```

定义训练参数。在训练过程中，我们不计算任何评估指标。相反，我们只关注损失：只关注损失：
```python
>>> from transformers import Seq2SeqTrainingArguments

>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="speecht5_finetuned_voxpopuli_nl",  # change to a repo name of your choice
...     per_device_train_batch_size=4,
...     gradient_accumulation_steps=8,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=4000,
...     gradient_checkpointing=True,
...     fp16=True,
...     evaluation_strategy="steps",
...     per_device_eval_batch_size=2,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     report_to=["tensorboard"],
...     load_best_model_at_end=True,
...     greater_is_better=False,
...     label_names=["labels"],
...     push_to_hub=True,
... )
```

实例化 `Trainer` 对象，并将模型、数据集和数据整合器传递给它。
```py
>>> from transformers import Seq2SeqTrainer

>>> trainer = Seq2SeqTrainer(
...     args=training_args,
...     model=model,
...     train_dataset=dataset["train"],
...     eval_dataset=dataset["test"],
...     data_collator=data_collator,
...     tokenizer=processor,
... )
```

有了这些，您就可以开始训练了！训练将需要几个小时的时间。

根据您的 GPU，当您开始训练时，可能会遇到 CUDA 的 "内存不足" 错误。

在这种情况下，您可以逐步将 `per_device_train_batch_size` 减小为原来的 2 倍，并将 `gradient_accumulation_steps` 增加 2 倍以补偿。增加 2 倍以补偿。

```py
>>> trainer.train()
```

将最终的模型推送到🤗 Hub：
```py
>>> trainer.push_to_hub()
```

## 推断

太棒了，现在您已经对模型进行了微调，可以将其用于推断！从🤗 Hub 加载模型（确保在以下代码段中使用您的帐户名）：

```py
>>> model = SpeechT5ForTextToSpeech.from_pretrained("YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl")
```

选择一个示例，这里我们将从测试数据集中选择一个。获取一个说话者嵌入。
```py 
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

定义一些输入文本并对其进行标记化。
```py 
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
```

对输入文本进行预处理：
```py
>>> inputs = processor(text=text, return_tensors="pt")
```

使用您的模型创建一个频谱图：
```py
>>> spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
```

如果您愿意，可以可视化频谱图：
```py
>>> plt.figure()
>>> plt.imshow(spectrogram.T)
>>> plt.show()
```

<div class="flex justify-center">    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_2.png" alt="生成的对数梅尔频谱图"/> </div>

最后，使用声码器将频谱图转换为声音。

```py
>>> with torch.no_grad():
...     speech = vocoder(spectrogram)

>>> from IPython.display import Audio

>>> Audio(speech.numpy(), rate=16000)
```

根据我们的经验，从该模型获得令人满意的结果可能具有挑战性。说话者嵌入的质量似乎是一个重要因素。

由于 SpeechT5 是使用英语 x-vectors 进行预训练的，因此在使用英语说话者嵌入时，它的性能最佳。

如果合成的语音听起来很差，请尝试使用其他说话者嵌入。

增加训练时长也有可能提高结果的质量。

即便如此，该语音明显是荷兰语而不是英语，并且它确实捕捉到了说话者的声音特征（与示例中的原始音频进行比较）

尝试将 `config.reduction_factor = 1` 用于模型的配置，看看是否能改善结果。

最后，必须考虑道德问题。尽管 TTS 技术有许多有用的应用，但也可能被用于恶意目的，例如在未经他人知情或同意的情况下冒充某人的声音。请明智、负责地使用 TTS。