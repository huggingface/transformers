<!--版权所有2022年HuggingFace团队。保留所有权利。
根据Apache许可证第2.0版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，按“原样”分发的软件根据许可证分发，并且没有任何形式的担保或条件。请参阅许可证以了解特定语言下的权限和限制。具体语言下的权限和限制。
⚠️请注意，此文件是使用Markdown编写的，但包含我们的文档生成器（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确呈现。
-->

# 导出到TorchScript
<Tip>
这是我们与TorchScript的实验的最开始阶段，我们仍在探索其在可变输入大小模型上的能力。这是我们的兴趣重点，并将在即将发布的版本中进行深入分析，提供更多代码示例，更灵活的实现以及与编译后的TorchScript的Python代码进行比较的基准测试。
</Tip>
</Tip>

根据[TorchScript文档](https://pytorch.org/docs/stable/jit.html)：

>TorchScript是一种从PyTorch代码创建可序列化和可优化模型的方法。

有两个PyTorch模块，[JIT和TRACE](https://pytorch.org/docs/stable/jit.html)，使开发人员能够导出其模型以在其他程序中重复使用，比如面向效率的C++程序。

我们提供了一个接口，可以让您将🤗 Transformers模型导出到TorchScript以便它们可以在与基于PyTorch的Python程序不同的环境中重复使用。在这里，我们将解释如何使用TorchScript导出和使用我们的模型。

导出模型需要两个步骤：

-使用`torchscript`标志进行模型实例化
-使用虚拟输入进行前向传递

这些要求意味着开发人员需要注意以下几点，如下所述。

## TorchScript标志和绑定权重

`torchscript`标志是必需的，因为大多数🤗 Transformers语言模型其`Embedding`层和`Decoding` 层之间有绑定权重。

TorchScript不允许您导出具有绑定权重的模型，因此需要在导出之前解除绑定并克隆权重。necessary to untie and clone the weights beforehand.

使用`torchscript`标志实例化的模型将其“嵌入”层和“解码”层分开，这意味着它们不应该被训练。训练将导致两个层不同步，导致意外结果。

对于没有语言模型头的模型不是这种情况，因为这些模型没有绑定权重。这些模型可以安全地导出，而无需使用`torchscript`标志。

## 虚拟输入和标准长度 Dummy inputs and standard lengths

虚拟输入用于模型的前向传递。在输入的值通过层时，PyTorch会跟踪每个张量上执行的不同操作。然后使用这些记录的操作来创建模型的*trace*。

该trace是相对于输入的维度创建的。因此它受到虚拟输入的维度限制，不适用于任何其他序列长度或批次大小。尝试使用不同大小时，将引发以下错误：

```
`The expanded size of the tensor (3) must match the existing size (7) at non-singleton dimension 2`
```

我们建议您使用虚拟输入大小至少与将在推理期间馈送给模型的最大输入大小一样大进行跟踪。填充可以帮助填充缺失的值。

但是，由于使用较大的输入大小来跟踪模型，矩阵的维度也会变大，导致更多的计算。
在导出具有不同序列长度的模型时，请注意每个输入上执行的操作总数，并密切关注性能。

##  在Python中使用TorchScript

本节演示了如何保存和加载模型以及如何使用trace进行推理。for inference.

### 保存模型

要使用TorchScript导出`BertModel`，请从`BertConfig`类实例化`BertModel`，然后将其保存到名为`traced_bert.pt`的磁盘中：

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch

enc = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = "[MASK]"
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
config = BertConfig(
    vocab_size_or_config_json_file=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    torchscript=True,
)

# Instantiating the model
model = BertModel(config)

# The model needs to be in evaluation mode
model.eval()

# If you are instantiating the model with *from_pretrained* you can also easily set the TorchScript flag
model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

# Creating the trace
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
torch.jit.save(traced_model, "traced_bert.pt")
```

### 加载模型

现在可以从磁盘加载先前保存的`BertModel`，`traced_bert.pt`，并使用在先前初始化的`dummy_input`上进行使用：

```python
loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

all_encoder_layers, pooled_output = loaded_model(*dummy_input)
```

### 使用trace模型进行推理

通过使用其`__call__`方法使用trace模型进行推理：

```python
traced_model(tokens_tensor, segments_tensors)
```

## 使用Neuron SDK将Hugging Face TorchScript模型部署到AWS
AWS推出了[Amazon EC2 Inf1](https://aws.amazon.com/ec2/instance-types/inf1/)实例系列，用于云中低成本、高性能的机器学习推理。Inf1实例由AWS Inferentia芯片提供支持，这是一款定制的硬件加速器，专门用于深度学习推理工作负载。[AWSNeuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/#)是Inferentia的SDK，支持对transformers模型进行跟踪和优化部署到Inf1上。Neuron SDK提供以下功能：
1. 易于使用的API，只需更改一行代码即可对TorchScript进行跟踪和优化  用于云中推理的模型。
2. 针对[改进的  成本效益性的性能优化（https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/benchmark/）。
3. 支持使用  
    [PyTorch](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.html)  或  [TensorFlow](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/tensorflow/huggingface_bert/huggingface_bert.html)构建的Hugging Face transformers模型。

### 影响

基于[BERT（双向编码器表示来自Transformers）的transformer模型](https://huggingface.co/docs/transformers/main/model_doc/bert)架构，或其变体，如[distilBERT](https://huggingface.co/docs/transformers/main/model_doc/distilbert)和[roBERTa](https://huggingface.co/docs/transformers/main/model_doc/roberta)在Inf1上运行最佳，用于非生成任务，如抽取性问题回答、序列分类和标记分类。然而，文本生成任务仍然可以适应Inf1上运行，根据这个[AWS Neuron MarianMT
教程](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/transformers-marianmt.html)。
有关可以在Inferentia上直接转换的模型的更多信息，请参见[模型架构适配](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/models/models-inferentia.html#models-inferentia)部分的Neuron文档。

### 依赖项
使用AWS Neuron转换模型需要[Neuron SDK环境](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/index.html#installation-guide)已经预配置在[AW+AWS深度学习AMI](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia-launching.html)上。

### 将模型转换为AWS Neuron

使用与[在Python中使用TorchScript](torchscript#using-torchscript-in-python)相同的代码将模型转换为AWS NEURON。

导入`torch.neuron`框架扩展，以通过Python API访问Neuron SDK的组件：

只需修改以下行：这样可以使Neuron SDK跟踪模型并对其进行优化，以用于Inf1实例。要了解有关AWS Neuron SDK功能、工具、示例教程和最新更新的更多信息，请参阅[AWS Neuron SDK文档](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html)。
Python API:

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.neuron
```

You only need to modify the following line:

```diff
- torch.jit.trace(model, [tokens_tensor, segments_tensors])
+ torch.neuron.trace(model, [token_tensor, segments_tensors])
```

This enables the Neuron SDK to trace the model and optimize it for Inf1 instances.

To learn more about AWS Neuron SDK features, tools, example tutorials and latest
updates, please see the [AWS NeuronSDK
documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html).
