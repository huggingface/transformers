<!--版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则不得使用本文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的担保或条件。请参阅许可证以获取特定语言下的权限和限制。特别注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确
显示。-->


# 图像分类

[[在 Colab 中打开]]
<Youtube id="tjAIM7BOYhw"/>

图像分类是将标签或类别分配给图像的过程。与文本或音频分类不同，输入是构成图像的像素值。图像分类有许多应用，例如在自然灾害后检测损坏情况、监测作物健康状况或帮助筛查医学图像中的疾病迹象。本指南介绍如何执行以下操作：


1. 在 [Food-101](https://huggingface.co/datasets/food101) 数据集上微调 [ViT](model_doc/vit) 模型，以对图像中的食物进行分类。2. 使用微调后的模型进行推理。

<Tip> 

本教程展示的任务由以下模型架构支持：
<!--此提示由`make fix-copies`自动生成，请勿手动填写！-->
[BEiT](../model_doc/beit), [BiT](../model_doc/bit), [ConvNeXT](../model_doc/convnext), [ConvNeXTV2](../model_doc/convnextv2), [CvT](../model_doc/cvt), [Data2VecVision](../model_doc/data2vec-vision), [DeiT](../model_doc/deit), [DiNAT](../model_doc/dinat), [EfficientFormer](../model_doc/efficientformer), [EfficientNet](../model_doc/efficientnet), [FocalNet](../model_doc/focalnet), [ImageGPT](../model_doc/imagegpt), [LeViT](../model_doc/levit), [MobileNetV1](../model_doc/mobilenet_v1), [MobileNetV2](../model_doc/mobilenet_v2), [MobileViT](../model_doc/mobilevit), [MobileViTV2](../model_doc/mobilevitv2), [NAT](../model_doc/nat), [Perceiver](../model_doc/perceiver), [PoolFormer](../model_doc/poolformer), [RegNet](../model_doc/regnet), [ResNet](../model_doc/resnet), [SegFormer](../model_doc/segformer), [SwiftFormer](../model_doc/swiftformer), [Swin Transformer](../model_doc/swin), [Swin Transformer V2](../model_doc/swinv2), [VAN](../model_doc/van), [ViT](../model_doc/vit), [ViT Hybrid](../model_doc/vit_hybrid), [ViTMSN](../model_doc/vit_msn) 
<!--生成提示的末尾-->
</Tip>

开始之前，请确保已安装所有必要的库：
```bash
pip install transformers datasets evaluate
```

我们鼓励您登录 Hugging Face 帐户，与社区共享和上传您的模型。当提示时，请输入您的令牌以登录：
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 Food-101 数据集

首先，从🤗 Datasets 库中加载 Food-101 数据集的较小子集。这样可以让您有机会在训练完整数据集之前进行实验和确保一切正常。使用 [`~datasets.Dataset.train_test_split`] 方法将数据集的“train”部分拆分为训练集和测试集：
```py
>>> from datasets import load_dataset

>>> food = load_dataset("food101", split="train[:5000]")
```

然后，查看一个示例：
```py
>>> food = food.train_test_split(test_size=0.2)
```

Then take a look at an example:

```py
>>> food["train"][0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>,
 'label': 79}
```

数据集中的每个示例都包含两个字段：

- `image`：食物项目的 PIL 图像
- `label`：食物项目的标签类别

为了使模型能够从标签 ID 获取标签名称，创建一个将标签名称映射到整数及其反向映射的字典：
```py
>>> labels = food["train"].features["label"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

现在，您可以将标签 ID 转换为标签名称：
```py
>>> id2label[str(79)]
'prime_rib'
```

## 预处理

接下来，加载一个 ViT 图像处理器 (Image Processor)，将图像处理为张量：
```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "google/vit-base-patch16-224-in21k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```


<frameworkcontent> 
<pt> 

 对图像应用一些变换，使模型对过度拟合更具鲁棒性。这里使用 torchvision 的 [`transforms`](https://pytorch.org/vision/stable/transforms.html) 模块，但您也可以使用其他喜欢的图像库。

随机裁剪图像的一部分，调整大小，并使用图像的平均值和标准差进行归一化：
```py
>>> from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

>>> normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )
>>> _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
```

然后创建一个预处理函数，应用变换并返回图像的 `pixel_values`（模型的输入）：
```py
>>> def transforms(examples):
...     examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     del examples["image"]
...     return examples
```

要在整个数据集上应用预处理函数，使用🤗 Datasets 的 [`~datasets.Dataset.with_transform`] 方法。

当您加载数据集的元素时，变换将在

```py
>>> food = food.with_transform(transforms)
```

加载时动态应用。
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```
</pt> </frameworkcontent>

<frameworkcontent> 
<tf>

为了避免过拟合并使模型更具鲁棒性，在训练数据集中添加一些数据增强。这里我们使用 Keras 预处理层来定义训练数据的变换（包括数据增强），以及验证数据的变换（仅包括中心裁剪、调整大小和归一化）。您可以使用 `tf.image` 或其他您喜欢的库。
```py
>>> from tensorflow import keras
>>> from tensorflow.keras import layers

>>> size = (image_processor.size["height"], image_processor.size["width"])

>>> train_data_augmentation = keras.Sequential(
...     [
...         layers.RandomCrop(size[0], size[1]),
...         layers.Rescaling(scale=1.0 / 127.5, offset=-1),
...         layers.RandomFlip("horizontal"),
...         layers.RandomRotation(factor=0.02),
...         layers.RandomZoom(height_factor=0.2, width_factor=0.2),
...     ],
...     name="train_data_augmentation",
... )

>>> val_data_augmentation = keras.Sequential(
...     [
...         layers.CenterCrop(size[0], size[1]),
...         layers.Rescaling(scale=1.0 / 127.5, offset=-1),
...     ],
...     name="val_data_augmentation",
... )
```

接下来，创建批量图像的适当变换函数，而不是逐个图像进行变换。
```py
>>> import numpy as np
>>> import tensorflow as tf
>>> from PIL import Image


>>> def convert_to_tf_tensor(image: Image):
...     np_image = np.array(image)
...     tf_image = tf.convert_to_tensor(np_image)
...     # `expand_dims()` is used to add a batch dimension since
...     # the TF augmentation layers operates on batched inputs.
...     return tf.expand_dims(tf_image, 0)


>>> def preprocess_train(example_batch):
...     """Apply train_transforms across a batch."""
...     images = [
...         train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
...     ]
...     example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
...     return example_batch


... def preprocess_val(example_batch):
...     """Apply val_transforms across a batch."""
...     images = [
...         val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
...     ]
...     example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
...     return example_batch
```

使用🤗 Datasets 的 [`~datasets.Dataset.set_transform`] 方法，动态应用变换：
```py
food["train"].set_transform(preprocess_train)
food["test"].set_transform(preprocess_val)
```

作为最后预处理步骤，使用 `DefaultDataCollator` 创建一个示例批次。与🤗 Transformers 中的其他数据整理器不同，`DefaultDataCollator` 不会应用额外的预处理，如填充。</tf>
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```
</tf>
</frameworkcontent>

## 评估

在训练过程中包含度量指标通常有助于评估模型的性能。您可以使用🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载一个评估方法。

对于此任务，请加载 [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) 度量指标（请参阅🤗 Evaluate [快速导览](https://huggingface.co/docs/evaluate/a_quick_tour) 以了解有关如何加载和计算度量指标的更多信息）：

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

然后创建一个函数，将您的预测和标签传递给 [`~evaluate.EvaluationModule.compute`] 以计算准确度：
```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

您的 `compute_metrics` 函数已准备就绪，当设置训练时，您将返回它。

## 训练

<frameworkcontent>
<pt>
<Tip>

如果您对使用 [`Trainer`] 微调模型不熟悉，请参阅此处的基本教程 [here](../training#train-with-pytorch-trainer)！
</Tip>

现在，您可以开始训练模型了！使用 [`AutoModelForImageClassification`] 加载 ViT。指定标签数和预期标签数以及标签映射：
```py
>>> from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

>>> model = AutoModelForImageClassification.from_pretrained(
...     checkpoint,
...     num_labels=len(labels),
...     id2label=id2label,
...     label2id=label2id,
... )
```

此时，只剩下三个步骤：

1. 在 [`TrainingArguments`] 中定义您的训练超参数。重要的是不要移除未使用的列，因为那样会删除 `image` 列。没有 `image` 列，您就无法创建 `pixel_values`。设置 `remove_unused_columns=False` 以阻止此行为！唯一其他必需的参数是 `output_dir`，它指定了保存模型的位置。您可以通过设置 `push_to_hub=True` 将此模型推送到 Hub（您需要登录 Hugging Face 才能上传模型）。在每个 epoch 结束时，[`Trainer`] 将评估准确性并保存训练检查点。
2. 将训练参数与模型、数据集、分词器 (Tokenizer)、数据整理器和 `compute_metrics` 函数一起传递给 [`Trainer`]。
3. 调用 [`~Trainer.train`] 来微调您的模型。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_food_model",
...     remove_unused_columns=False,
...     evaluation_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=food["train"],
...     eval_dataset=food["test"],
...     tokenizer=image_processor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

训练完成后，使用 [`~transformers.Trainer.push_to_hub`] 方法将您的模型共享到 Hub，让每个人都可以使用您的模型：
```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>

<Tip>

如果您对使用 Keras 进行模型微调不熟悉，请先查看 [基本教程](./training#train-a-tensorflow-model-with-keras)！
</Tip>

在 TensorFlow 中进行模型微调，请按照以下步骤进行：

1. 定义训练超参数，并设置优化器和学习率调度。
2. 实例化一个预训练模型。
3. 将🤗数据集转换为 `tf.data.Dataset`。
4. 编译您的模型。
5. 添加回调函数，并使用 `fit()` 方法运行训练。
6. 将模型上传到🤗 Hub 与社区共享。

首先，定义超参数、优化器和学习率调度：

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_epochs = 5
>>> num_train_steps = len(food["train"]) * num_epochs
>>> learning_rate = 3e-5
>>> weight_decay_rate = 0.01

>>> optimizer, lr_schedule = create_optimizer(
...     init_lr=learning_rate,
...     num_train_steps=num_train_steps,
...     weight_decay_rate=weight_decay_rate,
...     num_warmup_steps=0,
... )
```

然后，使用 [`TFAutoModelForImageClassification`] 加载 ViT 以及标签映射：
```py
>>> from transformers import TFAutoModelForImageClassification

>>> model = TFAutoModelForImageClassification.from_pretrained(
...     checkpoint,
...     id2label=id2label,
...     label2id=label2id,
... )
```

使用 [`~datasets.Dataset.to_tf_dataset`] 和您的 `data_collator` 将数据集转换为 `tf.data.Dataset` 格式：
```py
>>> # converting our train dataset to tf.data.Dataset
>>> tf_train_dataset = food["train"].to_tf_dataset(
...     columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
... )

>>> # converting our test dataset to tf.data.Dataset
>>> tf_eval_dataset = food["test"].to_tf_dataset(
...     columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
... )
```

使用 `compile()` 配置模型进行训练：
```py
>>> from tensorflow.keras.losses import SparseCategoricalCrossentropy

>>> loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
>>> model.compile(optimizer=optimizer, loss=loss)
```

要从预测中计算准确性并将模型推送到🤗 Hub，使用 [Keras 回调](../main_classes/keras_callbacks)。将您的 `compute_metrics` 函数传递给 [KerasMetricCallback](../main_classes/keras_callbacks#transformers.KerasMetricCallback)，并使用 [PushToHubCallback](../main_classes/keras_callbacks#transformers.PushToHubCallback) 上传模型：
```py
>>> from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_dataset)
>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="food_classifier",
...     tokenizer=image_processor,
...     save_strategy="no",
... )
>>> callbacks = [metric_callback, push_to_hub_callback]
```

最后，您已经准备好训练模型了！使用您的训练和验证数据集、epoch 数量和回调函数来微调模型：和你的回调函数微调模型：
```py
>>> model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=num_epochs, callbacks=callbacks)
Epoch 1/5
250/250 [==============================] - 313s 1s/step - loss: 2.5623 - val_loss: 1.4161 - accuracy: 0.9290
Epoch 2/5
250/250 [==============================] - 265s 1s/step - loss: 0.9181 - val_loss: 0.6808 - accuracy: 0.9690
Epoch 3/5
250/250 [==============================] - 252s 1s/step - loss: 0.3910 - val_loss: 0.4303 - accuracy: 0.9820
Epoch 4/5
250/250 [==============================] - 251s 1s/step - loss: 0.2028 - val_loss: 0.3191 - accuracy: 0.9900
Epoch 5/5
250/250 [==============================] - 238s 949ms/step - loss: 0.1232 - val_loss: 0.3259 - accuracy: 0.9890
```

恭喜！您已经对模型进行了微调，并在🤗 Hub 上共享了它。现在您可以用它进行推理了！
</tf>
 </frameworkcontent>

<Tip>

有关如何为图像分类微调模型的更详细示例，请参阅相应的 [PyTorch 笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。
</Tip>

## 推理

太棒了，现在您已经对模型进行了微调，可以用它进行推理了！

加载要进行推理的图像：

```py
>>> ds = load_dataset("food101", split="validation[:10]")
>>> image = ds["image"][0]
```

<div class="flex justify-center">    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" alt="image of beignets"/> </div>

尝试使用 [`pipeline`] 对微调的模型进行推理是最简单的方法。使用您的模型实例化一个图像分类的 `pipeline`，并将图像传递给它：

```py
>>> from transformers import pipeline

>>> classifier = pipeline("image-classification", model="my_awesome_food_model")
>>> classifier(image)
[{'score': 0.31856709718704224, 'label': 'beignets'},
 {'score': 0.015232225880026817, 'label': 'bruschetta'},
 {'score': 0.01519392803311348, 'label': 'chicken_wings'},
 {'score': 0.013022331520915031, 'label': 'pork_chop'},
 {'score': 0.012728818692266941, 'label': 'prime_rib'}]
```

如果您愿意，也可以手动复制 `pipeline` 的结果：

<frameworkcontent> 
<pt> 

 加载图像处理器 (Image Processor)以预处理图像并将 `input` 返回为 PyTorch 张量：

```py
>>> from transformers import AutoImageProcessor
>>> import torch

>>> image_processor = AutoImageProcessor.from_pretrained("my_awesome_food_model")
>>> inputs = image_processor(image, return_tensors="pt")
```

将输入传递给模型并返回 logits：
```py
>>> from transformers import AutoModelForImageClassification

>>> model = AutoModelForImageClassification.from_pretrained("my_awesome_food_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

获取概率最高的预测标签，并使用模型的 `id2label` 映射将其转换为标签：
```py
>>> predicted_label = logits.argmax(-1).item()
>>> model.config.id2label[predicted_label]
'beignets'
```
</pt> 
</frameworkcontent>
<frameworkcontent> 
<tf> 

加载图像处理器 (Image Processor)以预处理图像并将 `input` 返回为 TensorFlow 张量：

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/food_classifier")
>>> inputs = image_processor(image, return_tensors="tf")
```

将输入传递给模型并返回 logits：
```py
>>> from transformers import TFAutoModelForImageClassification

>>> model = TFAutoModelForImageClassification.from_pretrained("MariaK/food_classifier")
>>> logits = model(**inputs).logits
```

获取概率最高的预测标签，并使用模型的 `id2label` 映射将其转换为标签：
```py
>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
>>> model.config.id2label[predicted_class_id]
'beignets'
```

</tf>
</frameworkcontent>

