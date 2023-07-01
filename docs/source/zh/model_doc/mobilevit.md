<!--版权所有2022年HuggingFace团队。保留所有权利。
根据Apache许可证第2.0版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“原样”分发，不附带任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。

⚠️请注意，此文件采用Markdown格式，但包含特定于我们doc-builder（类似于MDX）的语法，您的Markdown查看器可能无法正确显示。
-->

# MobileViT

## 概览

MobileViT模型是由Sachin Mehta和Mohammad Rastegari在[MobileViT：轻量级、通用和适用于移动设备的视觉Transformer](https://arxiv.org/abs/2110.02178)中提出的。MobileViT通过使用Transformer将局部处理替换为全局处理，引入了一个新的层。

以下是论文的摘要:

*轻量级卷积神经网络（CNN）是移动视觉任务的事实标准。它们的空间归纳偏差使它们能够在不同的视觉任务中使用更少的参数来学习表示。但是，这些网络是局部的。为了学习全局表示，已经采用了基于自注意力的视觉Transformer（ViTs）。与CNN不同，ViTs是重量级的。在本文中，我们提出了以下问题：是否可能将CNN和ViTs的优点结合起来构建一个轻量级且低延迟的移动视觉任务网络？为此，我们引入了MobileViT，这是一个轻量级且通用的适用于移动设备的视觉Transformer。MobileViT提出了一种不同的视角来进行具有Transformer的信息的全局处理，即将Transformer作为卷积操作。我们的结果表明，MobileViT在不同的任务和数据集上明显优于基于CNN和ViT的网络。在ImageNet-1k数据集上，MobileViT在约600万个参数的情况下实现了78.4%的top-1准确率，这比MobileNetv3（基于CNN的）和DeIT（基于ViT的）的准确率分别高出3.2%和6.2%。在MS-COCO目标检测任务中，MobileViT的准确率比MobileNetv3高出5.7%，参数数量相近。*

提示:

- MobileViT更像是CNN而不是Transformer模型。它不适用于序列数据，而是适用于图像批次。与ViT不同，没有嵌入。骨干模型输出一个特征图。您可以参考[此教程](https://keras.io/examples/vision/mobilevit)进行简单介绍。
- 您可以使用[`MobileViTImageProcessor`]来为模型准备图像。请注意，如果您自行进行预处理，预训练检查点要求图像按BGR像素顺序排列（而不是RGB）。
- 可用的图像分类检查点是在[ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)上进行预训练的（也称为ILSVRC 2012，包含130万张图像和1,000个类别）。
- 分割模型使用[DeepLabV3](https://arxiv.org/abs/1706.05587)头部。可用的语义分割检查点是在[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)上进行预训练的。
- 正如其名称所示，MobileViT旨在在移动手机上具有高性能和高效率。MobileViT模型的TensorFlow版本与[TensorFlow Lite](https://www.tensorflow.org/lite)完全兼容。

  您可以使用以下代码将MobileViT检查点（无论是图像分类还是语义分割）转换为  TensorFlow Lite模型:
```py
from transformers import TFMobileViTForImageClassification
import tensorflow as tf


model_ckpt = "apple/mobilevit-xx-small"
model = TFMobileViTForImageClassification.from_pretrained(model_ckpt)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()
tflite_filename = model_ckpt.split("/")[-1] + ".tflite"
with open(tflite_filename, "wb") as f:
    f.write(tflite_model)
```

  生成的模型的大小约为**1MB**，非常适合资源和网络  带宽有限的移动应用程序。

此模型由[matthijs](https://huggingface.co/Matthijs)贡献。模型的TensorFlow版本由[sayakpaul](https://huggingface.co/sayakpaul)贡献。原始代码和权重可以在[此处](https://github.com/apple/ml-cvnets)找到。

## 资源

以下是官方Hugging Face资源和社区（由🌎表示）资源的列表，可帮助您开始使用MobileViT。
<PipelineTag pipeline="image-classification"/>

- [`MobileViTForImageClassification`]在此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)中得到支持。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)
**语义分割**- [语义分割任务指南](../tasks/semantic_segmentation)

如果您有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将对其进行审核！该资源应该展示出与现有资源不同的新功能，而不是重复现有资源。

## MobileViTConfig

[[autodoc]] MobileViTConfig

## MobileViTFeatureExtractor

[[autodoc]] MobileViTFeatureExtractor
    - __call__
    - post_process_semantic_segmentation

## MobileViTImageProcessor

[[autodoc]] MobileViTImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## MobileViTModel

[[autodoc]] MobileViTModel
    - forward

## MobileViTForImageClassification

[[autodoc]] MobileViTForImageClassification
    - forward

## MobileViTForSemanticSegmentation

[[autodoc]] MobileViTForSemanticSegmentation
    - forward

## TFMobileViTModel

[[autodoc]] TFMobileViTModel
    - call

## TFMobileViTForImageClassification

[[autodoc]] TFMobileViTForImageClassification
    - call

## TFMobileViTForSemanticSegmentation

[[autodoc]] TFMobileViTForSemanticSegmentation
    - call
