# RegNet

## 概述

RegNet模型是由Ilija Radosavovic，Raj Prateek Kosaraju，Ross Girshick，Kaim He，Piotr Dollár在[《设计网络设计空间](https://arxiv.org/abs/2003.13678)》中提出的。

作者设计了搜索空间来执行神经架构搜索 （NAS）。他们首先从高维搜索空间开始，通过根据当前搜索空间采样的最佳性能模型经验应用约束，迭代地减少搜索空间。

论文摘要如下：

_在这项工作中，我们提出了一种新的网络设计范式。我们的目标是帮助加深对网络设计的理解，并发现跨设置通用的设计原则。我们不是专注于设计单个网络实例，而是设计参数化网络群体的网络设计空间。整个过程类似于经典的网络手动设计，但提升到设计空间级别。使用我们的方法，我们探索了网络设计的结构方面，并得出了一个由简单、规则的网络组成的低维设计空间，我们称之为RegNet。RegNet参数化的核心见解非常简单：良好网络的宽度和深度可以用量化的线性函数来解释。我们分析了RegNet设计空间，并得出了与当前网络设计实践不符的有趣发现。RegNet 设计空间提供了简单快速的网络，可在各种翻牌制度中很好地工作。在可比的训练设置和失败率下，RegNet 模型的性能优于流行的 EfficientNet 模型，同时在 GPU 上的速度提高了 5 倍。_

该模型由 [Francesco](https://huggingface.co/Francesco) 提供。模型的 TensorFlow 版本 由 [sayakpaul](https://huggingface.co/sayakpaul) 和 [ariG23498](https://huggingface.co/ariG23498) 提供。 原始代码可以[在这里](https://github.com/facebookresearch/pycls)找到。

来自[野外视觉特征的自监督预训练](https://arxiv.org/abs/2103.01988)的巨大 10B 模型， 在 10 亿张 Instagram 图像上进行训练，可在[中心](https://huggingface.co/facebook/regnet-y-10b-seer)获得



## 资源

官方 Hugging Face 和社区（由 🌎 ）资源列表，可帮助您开始使用 RegNet。

图像分类

-   此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持 [RegNetForImageClassification](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/regnet#transformers.RegNetForImageClassification)。
-   另请参阅：[图像分类任务指南](https://huggingface.co/docs/transformers/tasks/image_classification)

如果您有兴趣提交要包含在此处的资源，请随时打开拉取请求，我们将对其进行审核！理想情况下，资源应该展示新内容，而不是复制现有资源。



# Pytorch实现

## RegNetConfig

### class transformers.RegNetConfig

```
(
        num_channels=3,
        embedding_size=32,
        hidden_sizes=[128, 192, 512, 1088],
        depths=[2, 6, 12, 2],
        groups_width=64,
        layer_type="y",
        hidden_act="relu",
        **kwargs
)
```

**参数**

-   **num_channels**（_可选_，默认为 3） — 输入通道数。`int`
-   **embedding\_size**（_可选_，默认为 64） — 嵌入层的维度（隐藏大小）。`int`
-   **hidden\_sizes** （， _可选_， 默认为 ） — 每个阶段的维度（隐藏大小）。`List[int]`，`[256, 512, 1024, 2048]`
-   **depths** （， _可选_， 默认为 ） — 每个阶段的深度（层数）。`List[int]`，`[3, 4, 6, 3]`
-   **layer\_type** （， _可选_， 默认为 ） — 要使用的层，它可以是“y”xreduction1yx'，但具有挤压和激励。论文详细解释这些层是如何构建的。`str`，`"y"`，`"x" or` `. An` `layer is a ResNet's BottleNeck layer with``fixed to``. While a` `layer is a`
-   **hidden\_act** （，_可选_，默认为 ） — 每个模块中的非线性激活函数。如果支持 string 和`str`，`"relu"`，`"gelu"`，`"relu"`，`"selu"`，`"gelu_new"`
-   **downsample\_in\_first\_stage** （， _可选_， 默认为 ） — 如果 ，第一阶段将使用 a 2 对输入进行下采样。`bool``False``True``stride`

这是用于存储 [RegNetModel](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/regnet#transformers.RegNetModel) 配置的配置类。它用于实例化 RegNet 根据指定的参数进行建模，定义模型架构。使用 默认值将产生与 RegNet [facebook/regnet-y-040](https://huggingface.co/facebook/regnet-y-040) 架构类似的配置。

配置对象继承自 [PretrainedConfig](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/configuration#transformers.PretrainedConfig)，可用于控制模型输出。阅读来自 [PretrainedConfig](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/configuration#transformers.PretrainedConfig) 的文档，了解更多信息。

例：

```python
from transformers import RegNetConfig, RegNetModel

# Initializing a RegNet regnet-y-40 style configuration
configuration = RegNetConfig()
# Initializing a model from the regnet-y-40 style configuration
model = RegNetModel(configuration)
# Accessing the model configuration
configuration = model.config
```



## RegNetModel

### class transformers.RegNetModel

**参数**

- **config** （[RegNetConfig](https://huggingface.co/docs/transformers/main/en/model_doc/regnet#transformers.RegNetConfig)） — 包含模型所有参数的模型配置类。 使用配置文件初始化不会加载与模型关联的权重，只会加载 配置。查看 [from_pretrained（）](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) 方法以加载模型权重。

RegNet 模型输出原始特征，顶部没有任何特定的头部。 此模型是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 子类。使用它作为常规的 PyTorch 模块，并参考 PyTorch 文档，了解与一般用法相关的所有事项，以及行为。

**forward**

( pixel_values: Tensoroutput_hidden_states: Optional = Nonereturn_dict: Optional = None ) **→** `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

**参数**

-   **pixel\_values** （ 形状 ） — 像素值。可以使用 [AutoImageProcessor](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/auto#transformers.AutoImageProcessor) 获取像素值。请参阅 [ConvNextImageProcessor.**call**（）](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/glpn#transformers.GLPNFeatureExtractor.__call__) 了解详细信息。`torch.FloatTensor``(batch_size, num_channels, height, width)`
-   **output\_hidden\_states** （_可选_） — 是否返回所有图层的隐藏状态。请参阅返回的张量下方 更多细节。`bool``hidden_states`
-   **return\_dict** （_可选_） — 是否返回 [ModelOutput](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/output#transformers.utils.ModelOutput) 而不是普通元组。`bool`

**返回**

`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention`或`tuple(torch.FloatTensor)`

A 或 （if is passed 或 when ） 的元组，包括各种 元素，具体取决于配置 （[RegNetConfig](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/regnet#transformers.RegNetConfig)） 和输入。`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention``torch.FloatTensor``return_dict=False``config.return_dict=False`

- **last\_hidden\_state** （ of shape ） — 模型最后一层输出处的隐藏状态序列。`torch.FloatTensor``(batch_size, num_channels, height, width)`

- **pooler\_output** （ of shape ） — 对空间维度进行池化操作后的最后一层隐藏状态。`torch.FloatTensor``(batch_size, hidden_size)`

- **hidden\_states** （， _可选_， 在传递时返回 或 当 ） — 元组 （一个用于嵌入的输出，如果模型有嵌入层，则为 + 每层一个输出）的形状。`tuple(torch.FloatTensor)``output_hidden_states=True``config.output_hidden_states=True``torch.FloatTensor``(batch_size, num_channels, height, width)`

  模型在每层输出端的隐藏状态以及可选的初始嵌入输出。

[RegNetModel](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/regnet#transformers.RegNetModel) forward 方法重写特殊方法.`__call__`

尽管需要在此函数中定义前向传递的配方，但应该在之后调用实例而不是此实例，因为前者负责运行预处理和后处理步骤，而 后者默默地忽略了他们.`模块`

**例：**

```python
from transformers import AutoImageProcessor, RegNetModel
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
model = RegNetModel.from_pretrained("facebook/regnet-y-040")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)
```



## RegNetForImageClassification

### class transformers.RegNetForImageClassification

**参数**

- **config** （[RegNetConfig](https://huggingface.co/docs/transformers/main/en/model_doc/regnet#transformers.RegNetConfig)） — 包含模型所有参数的模型配置类。 使用配置文件初始化不会加载与模型关联的权重，只会加载 配置。查看 [from_pretrained（）](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) 方法以加载模型权重。

RegNet模型，顶部有一个图像分类头（池化特征顶部的线性层），例如 ImageNet。

此模型是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 子类。使用它 作为常规的 PyTorch 模块，并参考 PyTorch 文档，了解与一般用法相关的所有事项，以及 行为。

**forward**

( pixel_values: Optional = Nonelabels: Optional = Noneoutput_hidden_states: Optional = Nonereturn_dict: Optional = None ) **→** [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](https://huggingface.co/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

**参数**

-   **pixel\_values** （ 形状 ） — 像素值。可以使用 [AutoImageProcessor](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/auto#transformers.AutoImageProcessor) 获取像素值。请参阅 [ConvNextImageProcessor。**call**（）](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/glpn#transformers.GLPNFeatureExtractor.__call__) 了解详细信息。`torch.FloatTensor``(batch_size, num_channels, height, width)`
-   **output\_hidden\_states** （_可选_） — 是否返回所有图层的隐藏状态。请参阅返回的张量下方 更多细节。`bool``hidden_states`
-   **return\_dict** （_可选_） — 是否返回 [ModelOutput](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/output#transformers.utils.ModelOutput) 而不是普通元组。`bool`
-   **labels** （ 形状 ， _可选_） — 用于计算图像分类/回归损失的标签。索引应在 中。如果计算分类损失（交叉熵）。`torch.LongTensor``(batch_size,)``[0, ..., config.num_labels - 1]``config.num_labels > 1`

**Returns**

[transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](https://huggingface.co/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

一个[transformers.modeling\_outputs。ImageClassifierOutputWithNoAttention](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) 或包含各种 元素，具体取决于配置 （[RegNetConfig](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/regnet#transformers.RegNetConfig)） 和输入。`torch.FloatTensor``return_dict=False``config.return_dict=False`

-   **loss** （ of shape ， _optional_， return when is provided） — 分类（如果 config.num\_labels==1） 则为回归损失。`torch.FloatTensor``(1,)``labels`
-   **logits** （ of shape ） — 分类（如果 config.num\_labels==1，则为回归）分数（在 SoftMax 之前）。`torch.FloatTensor``(batch_size, config.num_labels)`
-   **hidden\_states** （， _可选_， 在传递时返回 或 当 ） — 元组 （一个用于嵌入的输出，如果模型有嵌入层，则为 + 每个阶段的输出一个）的形状。隐藏状态（也 称为特征图）的模型在每个阶段的输出中。`tuple(torch.FloatTensor)``output_hidden_states=True``config.output_hidden_states=True``torch.FloatTensor``(batch_size, num_channels, height, width)`

[RegNetForImageClassification](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/regnet#transformers.RegNetForImageClassification) 正向方法重写特殊方法。`__call__`

尽管需要在此函数中定义前向传递的配方，但应该在之后调用实例而不是此实例，因为前者负责运行预处理和后处理步骤，而 后者默默地忽略了他们。`Module`

例：

```python
from transformers import AutoImageProcessor, RegNetForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
model = RegNetForImageClassification.from_pretrained("facebook/regnet-y-040")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
tabby, tabby cat
```







# TensorFlow实现

## TFRegNetModel

### class transformers.RegNetModel

**参数**

- **config** （[RegNetConfig](https://huggingface.co/docs/transformers/main/en/model_doc/regnet#transformers.RegNetConfig)） — 包含模型所有参数的模型配置类。 使用配置文件初始化不会加载与模型关联的权重，只会加载 配置。查看 [from_pretrained（）](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) 方法以加载模型权重。

RegNet 模型输出原始特征，顶部没有任何特定的头部。 此模型是 Tensorflow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) 子类。将其用作 常规 Tensorflow 模块，并参考 Tensorflow 文档，了解与一般用法相关的所有事项，以及 行为。

**forward**

( pixel_values: Tensoroutput_hidden_states: Optional = Nonereturn_dict: Optional = None ) **→** `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

**参数**

-   **pixel\_values** （ 形状 ） — 像素值。可以使用 [AutoImageProcessor](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/auto#transformers.AutoImageProcessor) 获取像素值。有关详细信息，请参阅。`tf.Tensor``(batch_size, num_channels, height, width)``ConveNextImageProcessor.__call__`
-   **output\_hidden\_states** （_可选_） — 是否返回所有图层的隐藏状态。请参阅返回的张量下方 更多细节。`bool``hidden_states`
-   **return\_dict** （_可选_） — 是否返回 [ModelOutput](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/output#transformers.utils.ModelOutput) 而不是普通元组。`bool`

**返回**

`transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndNoAttention`或`tuple(tf.Tensor)`

A 或 （if is passed 或 when ） 的元组，包含各种元素，具体取决于 配置 （[RegNetConfig](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/regnet#transformers.RegNetConfig)） 和输入。`transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndNoAttention``tf.Tensor``return_dict=False``config.return_dict=False`

- **last\_hidden\_state** （ of shape ） — 模型最后一层输出处的隐藏状态序列。`tf.Tensor``(batch_size, num_channels, height, width)`

- **pooler\_output** （ of shape ） — 对空间维度进行池化操作后的最后一层隐藏状态。`tf.Tensor``(batch_size, hidden_size)`

- **hidden\_states** （， _可选_， 在传递时返回 或 当 ） — 元组 （一个用于嵌入的输出，如果模型有一个嵌入层，+ 一个用于 每层的输出）的形状。`tuple(tf.Tensor)``output_hidden_states=True``config.output_hidden_states=True``tf.Tensor``(batch_size, num_channels, height, width)`

  模型在每层输出端的隐藏状态以及可选的初始嵌入输出。

[TFRegNetModel](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/regnet#transformers.TFRegNetModel) 正向方法重写特殊方法。`__call__`

尽管需要在此函数中定义前向传递的配方，但应该在之后调用实例而不是此实例，因为前者负责运行预处理和后处理步骤，而 后者默默地忽略了他们。`Module`

例：

```python
from transformers import AutoImageProcessor, TFRegNetModel
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
model = TFRegNetModel.from_pretrained("facebook/regnet-y-040")

inputs = image_processor(image, return_tensors="tf")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)
```



## TFRegNetForImageClassification

### class transformers.RegNetForImageClassification

**参数**

-   **config** （[RegNetConfig](https://huggingface.co/docs/transformers/main/en/model_doc/regnet#transformers.RegNetConfig)） — 包含模型所有参数的模型配置类。 使用配置文件初始化不会加载与模型关联的权重，只会加载 配置。查看 [from_pretrained（）](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) 方法以加载模型权重。

RegNet模型，顶部有一个图像分类头（池化特征顶部的线性层），例如 图像网。

此模型是 Tensorflow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) 子类。将其用作 常规 Tensorflow 模块，并参考 Tensorflow 文档，了解与一般用法相关的所有事项，以及 行为。

**forward**

( pixel_values: Optional = Nonelabels: Optional = Noneoutput_hidden_states: Optional = Nonereturn_dict: Optional = None ) **→** [transformers.modeling_outputs.ImageClassifierOutputWithNoAttention](https://huggingface.co/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

**参数**

-   **pixel\_values** （ 形状 ） — 像素值。可以使用 [AutoImageProcessor](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/auto#transformers.AutoImageProcessor) 获取像素值。有关详细信息，请参阅。`tf.Tensor``(batch_size, num_channels, height, width)``ConveNextImageProcessor.__call__`
-   **output\_hidden\_states** （_可选_） — 是否返回所有图层的隐藏状态。请参阅返回的张量下方 更多细节。`bool``hidden_states`
-   **return\_dict** （_可选_） — 是否返回 [ModelOutput](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/output#transformers.utils.ModelOutput) 而不是普通元组。`bool`
-   **标签** （ 形状 ， _可选_） — 用于计算图像分类/回归损失的标签。索引应在 中。如果计算分类损失（交叉熵）。`tf.Tensor``(batch_size,)``[0, ..., config.num_labels - 1]``config.num_labels > 1`

一个[transformers.modeling\_tf\_outputs。TFSequenceClassifierOutput](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput) 或包含各种元素的元组 （if is passed or when ） ，具体取决于 配置 （[RegNetConfig](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/regnet#transformers.RegNetConfig)） 和输入。`tf.Tensor``return_dict=False``config.return_dict=False`

- **loss** （ of shape ， _optional_， return when is provided） — 分类（如果 config.num\_labels==1） 则为回归损失。`tf.Tensor``(batch_size, )``labels`

- **logits** （ of shape ） — 分类（如果 config.num\_labels==1，则为回归）分数（在 SoftMax 之前）。`tf.Tensor``(batch_size, config.num_labels)`

- **hidden\_states** （， _可选_， 返回 when is passed or when ） — 形状的元组 （一个用于嵌入的输出 + 一个用于每层的输出） .`tuple(tf.Tensor)``output_hidden_states=True``config.output_hidden_states=True``tf.Tensor``(batch_size, sequence_length, hidden_size)`

  模型在每层输出处的隐藏状态加上初始嵌入输出。

- **attentions** （， _可选_， return when is passed or when ） — 形状的元组（每层一个）。`tuple(tf.Tensor)``output_attentions=True``config.output_attentions=True``tf.Tensor``(batch_size, num_heads, sequence_length, sequence_length)`

  注意力 softmax 之后的注意力权重，用于计算自注意力的加权平均值 头。

[TFRegNetForImageClassification](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/regnet#transformers.TFRegNetForImageClassification) 正向方法重写特殊方法。`__call__`

尽管需要在此函数中定义前向传递的配方，但应该在之后调用实例而不是此实例，因为前者负责运行预处理和后处理步骤，而 后者默默地忽略了他们。`Module`

例：

```python
from transformers import AutoImageProcessor, TFRegNetForImageClassification
import tensorflow as tf
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
model = TFRegNetForImageClassification.from_pretrained("facebook/regnet-y-040")

inputs = image_processor(image, return_tensors="tf")
logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = int(tf.math.argmax(logits, axis=-1))
print(model.config.id2label[predicted_label])
```





# JAX实现

## FlaxRegNetModel

### class transformers.FlaxRegNetModel

```
(
        self,
        config: RegNetConfig,
        input_shape=(1, 224, 224, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    )
```

- **config** （[RegNetConfig](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/regnet#transformers.RegNetConfig)） — 包含模型所有参数的模型配置类。 使用配置文件初始化不会加载与模型关联的权重，只会加载 配置。查看 [from\_pretrained（）](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) 方法以加载模型权重。

- **dtype** （， _可选_， 默认为 ） — 计算的数据类型。可以是 、（在 GPU 上）和（在 TPU 上）之一。`jax.numpy.dtype``jax.numpy.float32``jax.numpy.float32``jax.numpy.float16``jax.numpy.bfloat16`

  这可用于在 GPU 或 TPU 上启用混合精度训练或半精度推理。如果 指定所有计算都将使用给定的 .`dtype`

  **请注意，这仅指定计算的 dtype，不会影响模型的 dtype 参数。**

  如果要更改模型参数的 dtype，请参见 [to\_fp16（）](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) 和 [to\_bf16（）。](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16)

RegNet 模型输出原始特征，顶部没有任何特定的头部。

此模型继承自 [FlaxPreTrainedModel](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/model#transformers.FlaxPreTrainedModel)。查看超类文档，了解泛型方法 库实现其所有模型（例如从 PyTorch 模型下载、保存和转换权重）

此模型也是 [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) 子类。将其用作一个普通的亚麻布麻模块，并参考亚麻文档，了解与一般用途相关的所有事项，以及 行为。

最后，此模型支持固有的 JAX 功能，例如：

-   [实时 （JIT） 编译](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [自动微分](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [矢 量化](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [并行](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

**\_\_call\_\_**

( pixel_valuesparams: dict = Nonetrain: bool = Falseoutput_hidden_states: Optional = Nonereturn_dict: Optional = None ) **→** [transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling](https://huggingface.co/docs/transformers/main/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

一个[transformers.modeling\_flax\_outputs。FlaxBaseModelOutputWithPooling](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling) 或包含各种 元素，具体取决于配置 （） 和输入。`torch.FloatTensor``return_dict=False``config.return_dict=False``<class 'transformers.models.regnet.configuration_regnet.RegNetConfig'>`

- **last\_hidden\_state** （ of shape ） — 模型最后一层输出处的隐藏状态序列。`jnp.ndarray``(batch_size, sequence_length, hidden_size)`

- **pooler\_output** （ of shape ） — 序列的第一个标记（分类标记）的最后一层隐藏状态，由 线性层和 Tanh 激活功能。线性层权重是从下一句开始训练的 预训练期间的预测（分类）目标。`jnp.ndarray``(batch_size, hidden_size)`

- **hidden\_states** （， _可选_， 返回 when is passed or when ） — 形状的元组 （一个用于嵌入的输出 + 一个用于每层的输出） .`tuple(jnp.ndarray)``output_hidden_states=True``config.output_hidden_states=True``jnp.ndarray``(batch_size, sequence_length, hidden_size)`

  模型在每层输出处的隐藏状态加上初始嵌入输出。

- **attentions** （， _可选_， return when is passed or when ） — 形状的元组（每层一个）。`tuple(jnp.ndarray)``output_attentions=True``config.output_attentions=True``jnp.ndarray``(batch_size, num_heads, sequence_length, sequence_length)`

  注意力 softmax 之后的注意力权重，用于计算自注意力的加权平均值 头。

forward 方法将覆盖特殊方法。`FlaxRegNetPreTrainedModel``__call__`

尽管需要在此函数中定义前向传递的配方，但应该在之后调用实例而不是此实例，因为前者负责运行预处理和后处理步骤，而 后者默默地忽略了他们。`Module`

例子：

```python
from transformers import AutoImageProcessor, FlaxRegNetModel
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
model = FlaxRegNetModel.from_pretrained("facebook/regnet-y-040")

inputs = image_processor(images=image, return_tensors="np")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
```

## FlaxRegNetForImageClassification

### class transformers.FlaxRegNetForImageClassification

```
(
        self,
        config: RegNetConfig,
        input_shape=(1, 224, 224, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    )
```

**参数**

- **config** （[RegNetConfig](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/regnet#transformers.RegNetConfig)） — 包含模型所有参数的模型配置类。 使用配置文件初始化不会加载与模型关联的权重，只会加载 配置。查看 [from\_pretrained（）](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) 方法以加载模型权重。

- **dtype** （， _可选_， 默认为 ） — 计算的数据类型。可以是 、（在 GPU 上）和（在 TPU 上）之一。`jax.numpy.dtype``jax.numpy.float32``jax.numpy.float32``jax.numpy.float16``jax.numpy.bfloat16`

  这可用于在 GPU 或 TPU 上启用混合精度训练或半精度推理。如果 指定所有计算都将使用给定的 .`dtype`

  **请注意，这仅指定计算的 dtype，不会影响模型的 dtype 参数。**

  如果要更改模型参数的 dtype，请参见 [to\_fp16（）](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) 和 [to\_bf16（）。](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16)

RegNet模型，顶部有一个图像分类头（池化特征顶部的线性层），例如 图像网。

此模型继承自 [FlaxPreTrainedModel](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/model#transformers.FlaxPreTrainedModel)。查看超类文档，了解泛型方法 库实现其所有模型（例如从 PyTorch 模型下载、保存和转换权重）

此模型也是 [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) 子类。将其用作 一个普通的亚麻布麻模块，并参考亚麻文档，了解与一般用途相关的所有事项，以及 行为。

最后，此模型支持固有的 JAX 功能，例如：

-   [实时 （JIT） 编译](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [自动微分](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [矢 量化](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [并行](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

**\_\_call\_\_**

( pixel_valuesparams: dict = Nonetrain: bool = Falseoutput_hidden_states: Optional = Nonereturn_dict: Optional = None ) **→** `transformers.modeling_flax_outputs.FlaxImageClassifierOutputWithNoAttention` or `tuple(torch.FloatTensor)`

**返回**

`transformers.modeling_flax_outputs.FlaxImageClassifierOutputWithNoAttention`或`tuple(torch.FloatTensor)`

A 或 （if is passed 或 when ） 的元组，包括各种 元素，具体取决于配置 （） 和输入。`transformers.modeling_flax_outputs.FlaxImageClassifierOutputWithNoAttention``torch.FloatTensor``return_dict=False``config.return_dict=False``<class 'transformers.models.regnet.configuration_regnet.RegNetConfig'>`

-   **logits** （ of shape ） — 分类（如果 config.num\_labels==1，则为回归）分数（在 SoftMax 之前）。`jnp.ndarray``(batch_size, config.num_labels)`
-   **hidden\_states** （，_可选_，在传递时返回或在`tuple(jnp.ndarray)``output_hidden_states=True`
-   **`config.output_hidden_states=True`）：**元组（1 表示嵌入的输出，如果模型具有嵌入层，则 + 1 对于每个阶段的输出）的形状。隐藏状态（也 称为特征图）的模型在每个阶段的输出中。`jnp.ndarray``(batch_size, num_channels, height, width)`

forward 方法将覆盖特殊方法。`FlaxRegNetPreTrainedModel``__call__`

尽管需要在此函数中定义前向传递的配方，但应该在之后调用实例而不是此实例，因为前者负责运行预处理和后处理步骤，而 后者默默地忽略了他们。`Module`

例：

```python
from transformers import AutoImageProcessor, FlaxRegNetForImageClassification
from PIL import Image
import jax
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
model = FlaxRegNetForImageClassification.from_pretrained("facebook/regnet-y-040")

inputs = image_processor(images=image, return_tensors="np")
outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
```

