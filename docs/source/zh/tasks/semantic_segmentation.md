<!--版权所有2022年HuggingFace团队保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非遵守许可证，否则不得使用此文件。您可以在许可证下获取许可证的副本。
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样” BASIS，不提供任何形式的担保或条件，无论是明示还是暗示。有关许可证下的特定语言的权限和限制，请参阅许可证。注意：此文件为 Markdown 文件，但包含我们的 doc-builder（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确呈现。-->

# 语义分割

[[在 Colab 中打开]]

<Youtube id="dKE8SIt9C-w"/>

语义分割将标签或类别分配给图像的每个像素。语义分割有几种类型，在语义分割的情况下，不区分同一对象的唯一实例。

两个对象都被赋予相同的标签（例如，“汽车”而不是“汽车-1”和“汽车-2”）。语义分割的常见实际应用包括培训自动驾驶汽车识别行人和重要交通信息，识别医学图像中的细胞和异常，以及监测卫星图像中的环境变化。

本指南将向您展示如何：
1. 在 [SceneParse150](https://huggingface.co/datasets/scene_parse_150) 数据集上微调 [SegFormer](https://huggingface.co/docs/transformers/main/en/model_doc/segformer#segformer) 模型。
2. 使用您微调的模型进行推理。

<Tip> 

本教程中所示的任务支持以下模型架构：
<!--此提示由'make fix-copies'自动生成，请勿手动填写！-->
[BEiT](../model_doc/beit), [Data2VecVision](../model_doc/data2vec-vision), [DPT](../model_doc/dpt), [MobileNetV2](../model_doc/mobilenet_v2), [MobileViT](../model_doc/mobilevit), [MobileViTV2](../model_doc/mobilevitv2), [SegFormer](../model_doc/segformer), [UPerNet](../model_doc/upernet)
<!--生成提示的结尾-->
</Tip>

开始之前，请确保已安装所有必要的库：
```bash
pip install -q datasets transformers evaluate
```

我们鼓励您登录 Hugging Face 帐户，以便与社区上传和共享您的模型。在提示时，输入您的令牌以登录：
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 SceneParse150 数据集

首先从🤗 Datasets 库中加载 SceneParse150 数据集的较小子集。这样您就有机会在在完整数据集上进行更多时间的训练之前进行实验和确保一切正常。
```py
>>> from datasets import load_dataset

>>> ds = load_dataset("scene_parse_150", split="train[:50]")
```

使用 [`~datasets.Dataset.train_test_split`] 方法将数据集的 `train` 拆分为训练集和测试集：
```py
>>> ds = ds.train_test_split(test_size=0.2)
>>> train_ds = ds["train"]
>>> test_ds = ds["test"]
```

然后查看一个示例：
```py
>>> train_ds[0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x683 at 0x7F9B0C201F90>,
 'annotation': <PIL.PngImagePlugin.PngImageFile image mode=L size=512x683 at 0x7F9B0C201DD0>,
 'scene_category': 368}
```

- `image`：场景的 PIL 图像。
- `annotation`：分割地图的 PIL 图像，也是模型的目标。
- `scene_category`：描述图像场景的类别 ID，如“厨房”或“办公室”。在本指南中，您只需要 `image` 和 `annotation`，这两者都是 PIL 图像。

您还需要创建一个将标签 ID 映射到标签类的字典，在稍后设置模型时将非常有用。

从 Hub 下载映射并创建 `id2label` 和 `label2id` 字典：

```py
>>> import json
>>> from huggingface_hub import cached_download, hf_hub_url

>>> repo_id = "huggingface/label-files"
>>> filename = "ade20k-id2label.json"
>>> id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
>>> id2label = {int(k): v for k, v in id2label.items()}
>>> label2id = {v: k for k, v in id2label.items()}
>>> num_labels = len(id2label)
```

## 预处理

下一步是加载 SegFormer 图像处理器 (Image Processor)，为模型准备图像和注释。某些数据集（如此数据集）使用零索引作为背景类。但是，实际上，背景类并不包含在 150 个类别中，因此您需要将 `reduce_labels=True` 设置为从所有标签中减去 1。零索引将被替换为 `255`，因此 SegFormer 的损失函数将忽略它：
```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "nvidia/mit-b0"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)
```


<frameworkcontent> 
<pt> 

通常会对图像数据集应用一些数据增强以使模型对过拟合更具鲁棒性。在本指南中，您将使用来自 [torchvision](https://pytorch.org/vision/stable/index.html) 的 [`ColorJitter`](https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html) 函数随机更改图像的颜色属性，但您也可以使用任何您喜欢的图像库。
```py
>>> from torchvision.transforms import ColorJitter

>>> jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
```

现在创建两个预处理函数，以准备图像和注释供模型使用。这些函数将图像转换为 `pixel_values`，将注释转换为 `labels`。对于训练集，会在将图像提供给图像处理器 (Image Processor)之前应用 `jitter`。

对于测试集，图像处理器 (Image Processor)会裁剪和规范化 `images`，仅裁剪 `labels`，因为在测试期间不应用任何数据增强。

```py
>>> def train_transforms(example_batch):
...     images = [jitter(x) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs


>>> def val_transforms(example_batch):
...     images = [x for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs
```

要在整个数据集上应用 `jitter`，请使用🤗 Datasets 的 [`~datasets.Dataset.set_transform`] 函数。

转换是在运行时应用的，速度更快，占用的磁盘空间更少：
```py
>>> train_ds.set_transform(train_transforms)
>>> test_ds.set_transform(val_transforms)
```

</pt>
</frameworkcontent>

<frameworkcontent>
<tf>

通常会对图像数据集应用一些数据增强以使模型对过拟合更具鲁棒性。在本指南中，您将使用 [`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image) 随机更改图像的颜色属性，但您也可以使用任何图像库。定义两个单独的转换函数：- 训练数据转换，包括图像增强- 验证数据转换，仅转置图像，因为🤗 Transformers 中的计算机视觉模型期望通道优先布局

```py
>>> import tensorflow as tf


>>> def aug_transforms(image):
...     image = tf.keras.utils.img_to_array(image)
...     image = tf.image.random_brightness(image, 0.25)
...     image = tf.image.random_contrast(image, 0.5, 2.0)
...     image = tf.image.random_saturation(image, 0.75, 1.25)
...     image = tf.image.random_hue(image, 0.1)
...     image = tf.transpose(image, (2, 0, 1))
...     return image


>>> def transforms(image):
...     image = tf.keras.utils.img_to_array(image)
...     image = tf.transpose(image, (2, 0, 1))
...     return image
```

接下来，创建两个预处理函数，以为模型准备图像和注释的批次。这些函数应用了图像转换，并使用先前加载的 `image_processor` 将图像转换为 `pixel_values` 和将注释转换为 `labels`。

`ImageProcessor` 还负责调整图像的尺寸和规范化。

```py
>>> def train_transforms(example_batch):
...     images = [aug_transforms(x.convert("RGB")) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs


>>> def val_transforms(example_batch):
...     images = [transforms(x.convert("RGB")) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs
```

要在整个数据集上应用预处理转换，请使用🤗 Datasets 的 [`~datasets.Dataset.set_transform`] 函数。

转换是在运行时应用的，速度更快，占用的磁盘空间更少：

```py
>>> train_ds.set_transform(train_transforms)
>>> test_ds.set_transform(val_transforms)
```
</tf>
</frameworkcontent>


## 评估

在训练过程中包含度量标准通常有助于评估模型的性能。您可以使用🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载评估方法。

对于此任务，加载 [mean Intersection over Union](https://huggingface.co/spaces/evaluate-metric/accuracy)（IoU）度量标准（请参阅🤗 Evaluate [快速导览](https://huggingface.co/docs/evaluate/a_quick_tour) 以了解有关如何加载和计算度量标准的更多信息）：

```py
>>> import evaluate

>>> metric = evaluate.load("mean_iou")
```

然后创建一个函数来 [`~evaluate.EvaluationModule.compute`] 计算度量标准。您的预测需要首先转换为 logits，然后调整其形状以与标签的大小相匹配，然后才能调用 [`~evaluate.EvaluationModule.compute`]：

<frameworkcontent> 
<pt> 

```py
>>> def compute_metrics(eval_pred):
...     with torch.no_grad():
...         logits, labels = eval_pred
...         logits_tensor = torch.from_numpy(logits)
...         logits_tensor = nn.functional.interpolate(
...             logits_tensor,
...             size=labels.shape[-2:],
...             mode="bilinear",
...             align_corners=False,
...         ).argmax(dim=1)

...         pred_labels = logits_tensor.detach().cpu().numpy()
...         metrics = metric.compute(
...             predictions=pred_labels,
...             references=labels,
...             num_labels=num_labels,
...             ignore_index=255,
...             reduce_labels=False,
...         )
...         for key, value in metrics.items():
...             if type(value) is np.ndarray:
...                 metrics[key] = value.tolist()
...         return metrics
```

</pt>
</frameworkcontent>


<frameworkcontent>
<tf>

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     logits = tf.transpose(logits, perm=[0, 2, 3, 1])
...     logits_resized = tf.image.resize(
...         logits,
...         size=tf.shape(labels)[1:],
...         method="bilinear",
...     )

...     pred_labels = tf.argmax(logits_resized, axis=-1)
...     metrics = metric.compute(
...         predictions=pred_labels,
...         references=labels,
...         num_labels=num_labels,
...         ignore_index=-1,
...         reduce_labels=image_processor.do_reduce_labels,
...     )

...     per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
...     per_category_iou = metrics.pop("per_category_iou").tolist()

...     metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
...     metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
...     return {"val_" + k: v for k, v in metrics.items()}
```

</tf>
</frameworkcontent>

您的 `compute_metrics` 函数已准备就绪，当您设置训练时将返回该函数。

## Train
<frameworkcontent>
<pt>
<Tip>

如果您对使用 [`Trainer`] 微调模型不熟悉，请查看基本教程 [此处](../training#finetune-with-trainer)！
</Tip>

您现在可以开始训练模型了！使用 [`AutoModelForSemanticSegmentation`] 加载 SegFormer，并将标签 ID 与标签类之间的映射传递给模型：
```py
>>> from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

>>> model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
```

此时，只剩下三个步骤：

1. 在 [`TrainingArguments`] 中定义您的训练超参数。请务必不要删除未使用的列，因为这会删除 `image` 列。没有 `image` 列，您无法创建 `pixel_values`。将 `remove_unused_columns=False` 以防止此行为！另一个必需的参数是 `output_dir`，它指定了保存模型的位置。通过设置 `push_to_hub=True` 将此模型推送到 Hub 上（您需要登录到 Hugging Face 才能上传模型）。在每个 epoch 结束时，[`Trainer`] 将评估 IoU 指标并保存训练检查点。

2. 将训练参数与模型、数据集、tokenizer、数据整理器和 `compute_metrics` 函数一起传递给 [`Trainer`]。

3. 调用 [`~Trainer.train`] 来微调您的模型。

```py
>>> training_args = TrainingArguments(
...     output_dir="segformer-b0-scene-parse-150",
...     learning_rate=6e-5,
...     num_train_epochs=50,
...     per_device_train_batch_size=2,
...     per_device_eval_batch_size=2,
...     save_total_limit=3,
...     evaluation_strategy="steps",
...     save_strategy="steps",
...     save_steps=20,
...     eval_steps=20,
...     logging_steps=1,
...     eval_accumulation_steps=5,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=train_ds,
...     eval_dataset=test_ds,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

训练完成后，使用 [`~transformers.Trainer.push_to_hub`] 方法将您的模型共享到 Hub 上，以便每个人都可以使用您的模型：
```py
>>> trainer.push_to_hub()
```

</pt> </frameworkcontent>
<frameworkcontent> <tf> <Tip>

如果您对使用 Keras 进行模型微调不熟悉，请先查看 [基本教程](./training#train-a-tensorflow-model-with-keras)！

</Tip>

要在 TensorFlow 中进行模型微调，请按照以下步骤进行：
1. 定义训练超参数，并设置优化器和学习率调度器。
2. 实例化预训练模型。
3. 将🤗数据集转换为 `tf.data.Dataset`。
4. 编译您的模型。
5. 添加回调函数以计算指标并上传模型到🤗 Hub。
6. 使用 `fit()` 方法运行训练。

首先，定义超参数、优化器和学习率调度器：
```py
>>> from transformers import create_optimizer

>>> batch_size = 2
>>> num_epochs = 50
>>> num_train_steps = len(train_ds) * num_epochs
>>> learning_rate = 6e-5
>>> weight_decay_rate = 0.01

>>> optimizer, lr_schedule = create_optimizer(
...     init_lr=learning_rate,
...     num_train_steps=num_train_steps,
...     weight_decay_rate=weight_decay_rate,
...     num_warmup_steps=0,
... )
```

然后，使用 [`TFAutoModelForSemanticSegmentation`] 加载 SegFormer，并与标签映射一起编译它的优化器。

请注意，Transformers 模型都有一个默认的任务相关损失函数，因此您无需指定，除非您想要自定义：使用 [`~datasets.Dataset.to_tf_dataset`] 和 [`DefaultDataCollator`] 将数据集转换为 `tf.data.Dataset` 格式：

```py
>>> from transformers import TFAutoModelForSemanticSegmentation

>>> model = TFAutoModelForSemanticSegmentation.from_pretrained(
...     checkpoint,
...     id2label=id2label,
...     label2id=label2id,
... )
>>> model.compile(optimizer=optimizer)  # No loss argument!
```

使用 [`~datasets.Dataset.to_tf_dataset`] 和 [`DefaultDataCollator`] 将数据集转换为 `tf.data.Dataset` 格式：
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")

>>> tf_train_dataset = train_ds.to_tf_dataset(
...     columns=["pixel_values", "label"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )

>>> tf_eval_dataset = test_ds.to_tf_dataset(
...     columns=["pixel_values", "label"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )
```

使用 [Keras 回调](../main_classes/keras_callbacks) 计算准确性并将模型推送到🤗 Hub，以计算预测结果的准确性：将您的 `compute_metrics` 函数传递给 [`KerasMetricCallback`]，并使用 [`PushToHubCallback`] 上传模型：
```py
>>> from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

>>> metric_callback = KerasMetricCallback(
...     metric_fn=compute_metrics, eval_dataset=tf_eval_dataset, batch_size=batch_size, label_cols=["labels"]
... )

>>> push_to_hub_callback = PushToHubCallback(output_dir="scene_segmentation", tokenizer=image_processor)

>>> callbacks = [metric_callback, push_to_hub_callback]
```

最后，您已准备好训练模型了！使用您的训练和验证数据集、epoch 数量和回调函数来微调模型：</tf>
```py
>>> model.fit(
...     tf_train_dataset,
...     validation_data=tf_eval_dataset,
...     callbacks=callbacks,
...     epochs=num_epochs,
... )
```

恭喜！您已经微调了模型并在🤗 Hub 上共享了它。现在您可以用它进行推理！</tf> </frameworkcontent>

## 推理

太棒了，现在您已经微调了模型，可以用它进行推理了！
加载一张图片进行推理：

```py
>>> image = ds[0]["image"]
>>> image
```

<div class="flex justify-center">    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-image.png" alt="卧室图片"/> </div>

<frameworkcontent> 
<pt> 

 尝试使用 [`pipeline`] 对您的微调模型进行推理是最简单的方法。使用您的模型实例化一个用于图像分割的 `pipeline`，并将图片传递给它：

```py
>>> from transformers import pipeline

>>> segmenter = pipeline("image-segmentation", model="my_awesome_seg_model")
>>> segmenter(image)
[{'score': None,
  'label': 'wall',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062690>},
 {'score': None,
  'label': 'sky',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062A50>},
 {'score': None,
  'label': 'floor',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062B50>},
 {'score': None,
  'label': 'ceiling',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062A10>},
 {'score': None,
  'label': 'bed ',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062E90>},
 {'score': None,
  'label': 'windowpane',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062390>},
 {'score': None,
  'label': 'cabinet',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062550>},
 {'score': None,
  'label': 'chair',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062D90>},
 {'score': None,
  'label': 'armchair',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062E10>}]
```

如果您愿意，也可以手动复制 `pipeline` 的结果。使用图像处理器 (Image Processor)处理图像，并将 `pixel_values` 放在 GPU 上：
```py
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available, otherwise use a CPU
>>> encoding = image_processor(image, return_tensors="pt")
>>> pixel_values = encoding.pixel_values.to(device)
```

将您的输入传递给模型并返回 `logits`：
```py
>>> outputs = model(pixel_values=pixel_values)
>>> logits = outputs.logits.cpu()
```

接下来，将 `logits` 重新缩放为原始图像大小：
```py
>>> upsampled_logits = nn.functional.interpolate(
...     logits,
...     size=image.size[::-1],
...     mode="bilinear",
...     align_corners=False,
... )

>>> pred_seg = upsampled_logits.argmax(dim=1)[0]
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>

加载图像处理器 (Image Processor)以预处理图像并将输入返回为 TensorFlow 张量：
```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/scene_segmentation")
>>> inputs = image_processor(image, return_tensors="tf")
```

将您的输入传递给模型并返回 `logits`：
```py
>>> from transformers import TFAutoModelForSemanticSegmentation

>>> model = TFAutoModelForSemanticSegmentation.from_pretrained("MariaK/scene_segmentation")
>>> logits = model(**inputs).logits
```

接下来，将 `logits` 重新缩放为原始图像大小，并在类维度上应用 argmax：
```py
>>> logits = tf.transpose(logits, [0, 2, 3, 1])

>>> upsampled_logits = tf.image.resize(
...     logits,
...     # We reverse the shape of `image` because `image.size` returns width and height.
...     image.size[::-1],
... )

>>> pred_seg = tf.math.argmax(upsampled_logits, axis=-1)[0]
```

</tf>
</frameworkcontent>


要可视化结果，请加载 [数据集颜色调色板](https://github.com/tensorflow/models/blob/3f1ca33afe3c1631b733ea7e40c294273b9e406d/research/deeplab/utils/get_dataset_colormap.py#L51) 作为 `ade_palette()`，将每个类别映射到其 RGB 值。
然后，您可以组合并绘制图像和预测的分割图：

```py
>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
>>> palette = np.array(ade_palette())
>>> for label, color in enumerate(palette):
...     color_seg[pred_seg == label, :] = color
>>> color_seg = color_seg[..., ::-1]  # convert to BGR

>>> img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
>>> img = img.astype(np.uint8)

>>> plt.figure(figsize=(15, 10))
>>> plt.imshow(img)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-preds.png" alt="Image of bedroom overlaid with segmentation map"/>
</div>