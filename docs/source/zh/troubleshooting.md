<!---版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非遵守许可证，否则您不得使用此文件。您可以在以下网址获取许可证副本
    http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则软件在许可证的基础上分发，该许可证是“按原样”分发的，无论是明示还是暗示，均不提供任何形式的担保或条件。有关特定语言的权限和限制，请参阅许可证。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们 doc-builder 的特定语法（类似于 MDX），在 Markdown 查看器中可能无法正确呈现。
-->

# 故障排除

有时会发生错误，但我们在这里帮助您！本指南涵盖了我们遇到的一些常见问题及其解决方法。但是，本指南并不意味着是每个🤗 Transformers 问题的全面集合。如果需要更多故障排除帮助，请尝试以下方法：
<Youtube id="S2EEG3JIt2A"/>

1. 在 [论坛](https://discuss.huggingface.co/) 上寻求帮助。您可以根据具体的类别发布问题，比如 [初学者](https://discuss.huggingface.co/c/beginners/5) 或 [🤗 Transformers](https://discuss.huggingface.co/c/transformers/9)。

请确保编写一个具有一些可重现代码的清晰描述的论坛帖子，以最大程度地提高问题解决的可能性！

<Youtube id="_PAli-V4wj0"/>

2. 在🤗 Transformers 存储库上创建 [问题](https://github.com/huggingface/transformers/issues/new/choose)，如果是与库相关的错误。请尽量提供尽可能多的描述错误的信息，以帮助我们更好地找出问题所在以及如何修复。

3. 如果您使用的是旧版本的🤗 Transformers，请查看 [迁移](migration) 指南，因为不同版本之间可能引入了一些重要的更改。

有关故障排除和获取帮助的更多详细信息，请参阅 Hugging Face 课程的 [第 8 章](https://huggingface.co/course/chapter8/1?fw=pt)。

## 防火墙环境

某些云端和内部网络设置的 GPU 实例被防火墙阻止对外部连接，导致连接错误。当您的脚本尝试下载模型权重或数据集时，下载将挂起，然后超时，并显示以下消息：
```
ValueError: Connection error, and we cannot find the requested files in the cached path.
Please try again or make sure your Internet connection is on.
```

在这种情况下，您应该尝试在 [离线模式](installation#offline-mode) 下运行🤗 Transformers，以避免连接错误。

## CUDA 内存不足

在训练参数有数百万个的大型模型时，如果没有适当的硬件，可能会面临内存不足的挑战。当 GPU 内存不足时，您可能会遇到以下常见错误：
```
CUDA out of memory. Tried to allocate 256.00 MiB (GPU 0; 11.17 GiB total capacity; 9.70 GiB already allocated; 179.81 MiB free; 9.85 GiB reserved in total by PyTorch)
```

以下是您可以尝试减少内存使用的一些解决方案：
- 在 [`TrainingArguments`] 中减少 [`per_device_train_batch_size`](main_classes/trainer#transformers.TrainingArguments.per_device_train_batch_size) 的值。- 尝试使用 [`TrainingArguments`] 中的 [`gradient_accumulation_steps`](main_classes/trainer#transformers.TrainingArguments.gradient_accumulation_steps)，以有效增加总批量大小。
<Tip>
有关节省内存的技巧的更多详细信息，请参阅性能 [指南](performance)。
</Tip>
## 无法加载保存的 TensorFlow 模型
TensorFlow 的 [model.save](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model) 方法会将整个模型（架构、权重、训练配置）保存在一个文件中。

但是，当您再次加载模型文件时，可能会遇到错误，因为🤗 Transformers 可能无法加载模型文件中的所有与 TensorFlow 相关的对象。为了避免保存和加载 TensorFlow 模型时出现问题，我们建议您：
- 使用 [`model.save_weights`](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model) 将模型权重保存为 `h5` 文件扩展名，然后使用 [`~TFPreTrainedModel.from_pretrained`] 重新加载模型：
```py
>>> from transformers import TFPreTrainedModel
>>> from tensorflow import keras

>>> model.save_weights("some_folder/tf_model.h5")
>>> model = TFPreTrainedModel.from_pretrained("some_folder")
```

- 使用 [`~TFPretrainedModel.save_pretrained`] 保存模型，然后使用 [`~TFPreTrainedModel.from_pretrained`] 再次加载它：
```py
>>> from transformers import TFPreTrainedModel

>>> model.save_pretrained("path_to/model")
>>> model = TFPreTrainedModel.from_pretrained("path_to/model")
```

## ImportError
另一个常见错误是 `ImportError`，特别是对于新发布的模型：
```
ImportError: cannot import name 'ImageGPTImageProcessor' from 'transformers' (unknown location)
```

对于这些错误类型，请确保您安装了最新版本的🤗 Transformers，以访问最新的模型：
```bash
pip install transformers --upgrade
```

## CUDA 错误：设备端触发了断言
有时，您可能会遇到有关设备代码错误的通用 CUDA 错误。
```
RuntimeError: CUDA error: device-side assert triggered
```

您应该首先在 CPU 上运行代码，以获取更详细的错误消息。在代码开头添加以下环境变量以切换到 CPU：
```py
>>> import os

>>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

另一种选择是从 GPU 获取更好的回溯信息。在代码开头添加以下环境变量，以使回溯指向错误的源代码：
```py
>>> import os

>>> os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

## 当填充标记未被掩码时输出不正确
在某些情况下，如果 `input_ids` 包含填充标记，则输出的 `hidden_state` 可能是不正确的。为了演示，加载一个模型和分词器 (Tokenizer)。您可以访问模型的 `pad_token_id` 以查看其值。

对于某些模型，`pad_token_id` 可能为 `None`，但您始终可以手动设置它。
```py
>>> from transformers import AutoModelForSequenceClassification
>>> import torch

>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
>>> model.config.pad_token_id
0
```

以下示例显示了未对填充标记进行掩码的输出：
```py
>>> input_ids = torch.tensor([[7592, 2057, 2097, 2393, 9611, 2115], [7592, 0, 0, 0, 0, 0]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [ 0.1317, -0.1683]], grad_fn=<AddmmBackward0>)
```

这是第二个序列的实际输出：
```py
>>> input_ids = torch.tensor([[7592]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

大多数情况下，您应该为模型提供一个 `attention_mask`，以忽略填充标记，以避免此静默错误。现在，第二个序列的输出与实际输出相匹配：
<Tip>
默认情况下，分词器 (Tokenizer)根据特定分词器 (Tokenizer)的默认值为您创建 `attention_mask`。
</Tip>
```py
>>> attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]])
>>> output = model(input_ids, attention_mask=attention_mask)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

🤗 Transformers 不会自动创建用于屏蔽填充标记的 `attention_mask`，因为：
- 某些模型没有填充标记。
- 对于某些用例，用户希望模型关注填充标记。

## ValueError：无法识别此类 AutoModel 的配置类 XYZ

一般而言，我们建议使用 [`AutoModel`] 类加载预训练模型的实例。该类可以根据给定的检查点的配置自动推断和加载正确的架构。如果在从检查点加载模型时看到此 `ValueError`，这意味着 Auto 类找不到从给定检查点的配置到您尝试加载的模型类型之间的映射。

最常见的情况是给定任务不支持给定的检查点。

例如，您将在以下示例中看到此错误，因为没有适用于问答的 GPT2 模型：

```py
>>> from transformers import AutoProcessor, AutoModelForQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("gpt2-medium")
>>> model = AutoModelForQuestionAnswering.from_pretrained("gpt2-medium")
ValueError: Unrecognized configuration class <class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'> for this kind of AutoModel: AutoModelForQuestionAnswering.
Model type should be one of AlbertConfig, BartConfig, BertConfig, BigBirdConfig, BigBirdPegasusConfig, BloomConfig, ...
```
