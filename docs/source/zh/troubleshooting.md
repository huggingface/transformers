<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 故障排除

错误是难免的，但我们随时为你提供帮助！本指南涵盖了一些我们最常见到的问题以及它们的解决方法。但本指南并不会涵盖所有的 🤗 Transformers 问题。如果你需要更多故障排除方面的帮助，请尝试：

<Youtube id="S2EEG3JIt2A"/>

1. 在[论坛](https://discuss.huggingface.co/)上寻求帮助。你可以将问题发布到特定类别下，例如 [初学者](https://discuss.huggingface.co/c/beginners/5) 或 [🤗 Transformers](https://discuss.huggingface.co/c/transformers/9)。请在论坛帖子内详细描述问题，并附上可重现的代码，以增加解决问题的可能性！

<Youtube id="_PAli-V4wj0"/>

2. 如果是与库相关的 bug，请在 🤗 Transformers 代码仓库中提交一个 [Issue](https://github.com/huggingface/transformers/issues/new/choose)。并且尽可能详细得描述与 bug 有关的信息，以帮助我们更好地定位以及修复问题。

3. 如果你使用的是较旧版本的 🤗 Transformers，请查阅[迁移](migration)指南，因为新旧版本之间引入了一些重要的更改。

有关故障排除和获取帮助的更多详情，请参阅 Hugging Face 课程的[第 8 章](https://huggingface.co/course/chapter8/1?fw=pt)。


## 防火墙环境

一些云和内联网上的 GPU 实例对外部连接设置了防火墙，导致连接错误。当你的脚本尝试下载模型权重或数据集时，下载将挂起，随后超时并显示以下报错信息：

```
ValueError: Connection error, and we cannot find the requested files in the cached path.
Please try again or make sure your Internet connection is on.
```

在这种情况下，你应该尝试以 [离线模式](installation#offline-mode) 运行 🤗 Transformers 以避免连接错误。

## CUDA 内存不足

在没有适当硬件的情况下，训练数百万参数的大型模型可能会很有挑战性。当 GPU 内存不足时，你可能会遇到以下常见错误：

```
CUDA out of memory. Tried to allocate 256.00 MiB (GPU 0; 11.17 GiB total capacity; 9.70 GiB already allocated; 179.81 MiB free; 9.85 GiB reserved in total by PyTorch)
```

以下的一些解决方案或许能够减少你的内存使用：

- 在 [`TrainingArguments`] 中减少 [`per_device_train_batch_size`](main_classes/trainer#transformers.TrainingArguments.per_device_train_batch_size) 值的大小。
- 尝试在 [`TrainingArguments`] 使用 [`gradient_accumulation_steps`](main_classes/trainer#transformers.TrainingArguments.gradient_accumulation_steps) 以有效地增加总的 batch size。

<Tip>

请参考性能[指南](performance)以获取更多节省 GPU 内存的技巧。

</Tip>

## 无法加载已保存的 TensorFlow 模型

TensorFlow 的 [model.save](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model) 方法会将整个模型 - 架构、权重、训练配置 - 保存在单个文件中。但是，你再次加载模型文件时，可能会遇到错误，因为 🤗 Transformers 可能不会加载模型文件中的所有与 TensorFlow 相关的对象。为了避免保存和加载 TensorFlow 模型时出现问题，我们建议你：

- 使用 [`model.save_weights`](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model) 将模型权重保存为 `h5` 扩展的文件格式，然后使用 [`~TFPreTrainedModel.from_pretrained`] 重新加载模型：

```py
>>> from transformers import TFPreTrainedModel
>>> from tensorflow import keras

>>> model.save_weights("some_folder/tf_model.h5")
>>> model = TFPreTrainedModel.from_pretrained("some_folder")
```

- 使用 [`~TFPretrainedModel.save_pretrained`] 保存模型，然后使用 [`~TFPreTrainedModel.from_pretrained`] 再次加载模型权重：

```py
>>> from transformers import TFPreTrainedModel

>>> model.save_pretrained("path_to/model")
>>> model = TFPreTrainedModel.from_pretrained("path_to/model")
```

## ImportError

你可能会遇到的另一个常见错误是 `ImportError`，尤其是使用新发布的模型时：

```
ImportError: cannot import name 'ImageGPTImageProcessor' from 'transformers' (unknown location)
```

对于这些错误类型，请确保您已安装了最新版本的 🤗 Transformers 以访问最新的模型：

```bash
pip install transformers --upgrade
```

## CUDA error：触发设备端断言

有时你可能会遇到一般 CUDA 错误，有关设备代码错误。

```
RuntimeError: CUDA error: device-side assert triggered
```

请先尝试在CPU上运行代码，以获取更详细的错误消息。将以下环境变量添加到代码开头，以将设备切换至 CPU：

```py
>>> import os

>>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

另一种选择是从 GPU 获得更好的回溯。将以下环境变量添加到代码的开头，以使回溯指向错误源：

```py
>>> import os

>>> os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

## padding token 未进行 mask 时的输出错误

在某些情况下，如果 `input_ids` 包括 padding token，输出的 `hidden_state` 可能是不正确的。为了进行演示，请加载一个模型和分词器。你可以访问模型的 `pad_token_id` 查看它的值。一些模型的 `pad_token_id` 也许为  `None`，但你始终能够手动设置它。

```py
>>> from transformers import AutoModelForSequenceClassification
>>> import torch

>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
>>> model.config.pad_token_id
0
```

以下示例展示了模型在没有屏蔽 padding token 情况下的输出：

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

大多数时候，你应该为模型提供一个 `attention_mask` 来忽略 padding token，以避免这种潜在错误。现在第二个序列的输出与其实际输出一致了：

<Tip>

默认情况下，分词器会根据该分词器的默认设置为你创建一个 `attention_mask`。

</Tip>

```py
>>> attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]])
>>> output = model(input_ids, attention_mask=attention_mask)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

当提供 padding token 时，🤗 Transformers 不会自动创建 `attention_mask` 对其进行掩盖（mask），原因如下：

- 一些模型没有 padding token。
- 某些用例中，用户希望模型处理 padding token。

## ValueError: 无法识别此类 AutoModel 的配置类 XYZ

通常，我们建议使用 [`AutoModel`] 类来加载模型的预训练实例。该类可以根据配置自动从给定的检查点（checkpoint）推断并加载正确的架构。从检查点加载模型时看到此 `ValueError`，意味着 Auto 类无法从给定检查点中，找到配置与你尝试加载的模型类型之间的映射。这个错误最常发生在加载的检查点不支持给定任务的时候。你将在以下示例中看到此错误，因为 GPT2 模型不支持问答（question answering）任务：

```py
>>> from transformers import AutoProcessor, AutoModelForQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("gpt2-medium")
>>> model = AutoModelForQuestionAnswering.from_pretrained("gpt2-medium")
ValueError: Unrecognized configuration class <class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'> for this kind of AutoModel: AutoModelForQuestionAnswering.
Model type should be one of AlbertConfig, BartConfig, BertConfig, BigBirdConfig, BigBirdPegasusConfig, BloomConfig, ...
```
