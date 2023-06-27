<!--版权所有2022年The HuggingFace团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；您除非符合许可证的规定，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以获取特定语言下的权限和限制。请注意，此文件是 Markdown 格式的，但包含我们文档生成器（类似于 MDX）的特定语法，您的 Markdown 查看器可能无法正确
呈现。-->


# 目标检测

[[在 Colab 中打开]]

目标检测是计算机视觉任务，用于检测图像中的实例（如人、建筑物或汽车）。

目标检测模型接收图像作为输入，并输出检测到的目标的边界框的坐标和相关标签。一幅图像可以包含多个对象，每个对象都有自己的边界框和标签（例如，一幅图像中可以有一辆汽车和一座建筑物），而且每个对象可以出现在图像的不同部分（例如，图像中可以有多辆汽车）。

这个任务通常用于自动驾驶，用于检测行人、道路标志和交通信号灯等。其他应用包括对图像中的对象进行计数、图像搜索等。在本指南中，您将学习以下内容：


1. 在 [DETR](https://huggingface.co/docs/transformers/model_doc/detr) 上微调模型，该模型将卷积 骨干与编码器-解码器 Transformer 结合在一起，使用 [CPPE-5](https://huggingface.co/datasets/cppe-5) 数据集 进行微调。 
2. 使用您微调的模型进行推理。

<Tip> 

本教程中演示的任务由以下模型架构支持：
<!--此提示由`make fix-copies`自动生成，请勿手动填写！-->
[Conditional DETR](../model_doc/conditional_detr), [Deformable DETR](../model_doc/deformable_detr), [DETA](../model_doc/deta), [DETR](../model_doc/detr), [Table Transformer](../model_doc/table-transformer), [YOLOS](../model_doc/yolos)
<!--生成提示的结尾-->
</Tip>

开始之前，请确保已安装所有必要的库：
```bash
pip install -q datasets transformers evaluate timm albumentations
```

您将使用🤗 Datasets 从 Hugging Face Hub 加载数据集，使用🤗 Transformers 训练模型，以及使用 `albumentations` 对数据进行增强。目前需要 `timm` 来加载 DETR 模型的卷积骨干。

我们鼓励您与社区分享您的模型。登录您的 Hugging Face 账户将其上传到 Hub。在提示时，输入您的令牌进行登录：
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 CPPE-5 数据集

[CPPE-5 数据集](https://huggingface.co/datasets/cppe-5) 包含了在 COVID-19 大流行背景下，标识医疗个人防护装备（PPE）的图像和注释。

首先加载数据集：
```py
>>> from datasets import load_dataset

>>> cppe5 = load_dataset("cppe-5")
>>> cppe5
DatasetDict({
    train: Dataset({
        features: ['image_id', 'image', 'width', 'height', 'objects'],
        num_rows: 1000
    })
    test: Dataset({
        features: ['image_id', 'image', 'width', 'height', 'objects'],
        num_rows: 29
    })
})
```

您会发现该数据集已经带有包含 1000 张图像的训练集和包含 29 张图像的测试集。

为了熟悉数据，可以查看示例的外观。
```py
>>> cppe5["train"][0]
{'image_id': 15,
 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=943x663 at 0x7F9EC9E77C10>,
 'width': 943,
 'height': 663,
 'objects': {'id': [114, 115, 116, 117],
  'area': [3796, 1596, 152768, 81002],
  'bbox': [[302.0, 109.0, 73.0, 52.0],
   [810.0, 100.0, 57.0, 28.0],
   [160.0, 31.0, 248.0, 616.0],
   [741.0, 68.0, 202.0, 401.0]],
  'category': [4, 4, 0, 0]}}
```

数据集中的示例具有以下字段：
- `image_id`：示例图像的 ID
- `image`：包含图像的 `PIL.Image.Image` 对象
- `width`：图像的宽度
- `height`：图像的高度
- `objects`：包含图像中对象的边界框元数据的字典：  
- `id`：注释的 ID  
- `area`：边界框的面积  
- `bbox`：对象的边界框（按 [COCO 格式](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco)）  
- `category`：对象的类别，可能的值包括 `Coverall (0)`、`Face_Shield (1)`、`Gloves (2)`、`Goggles (3)` 和 `Mask (4)`

您可能会注意到 `bbox` 字段遵循 COCO 格式，这是 DETR 模型期望的格式。

但是，`objects` 中字段的分组与 DETR 所需的注释格式不同。您需要在使用此数据进行训练之前应用一些预处理转换。为了更好地了解数据，可视化数据集中的示例。


```py
>>> import numpy as np
>>> import os
>>> from PIL import Image, ImageDraw

>>> image = cppe5["train"][0]["image"]
>>> annotations = cppe5["train"][0]["objects"]
>>> draw = ImageDraw.Draw(image)

>>> categories = cppe5["train"].features["objects"].feature["category"].names

>>> id2label = {index: x for index, x in enumerate(categories, start=0)}
>>> label2id = {v: k for k, v in id2label.items()}

>>> for i in range(len(annotations["id"])):
...     box = annotations["bbox"][i - 1]
...     class_idx = annotations["category"][i - 1]
...     x, y, w, h = tuple(box)
...     draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
...     draw.text((x, y), id2label[class_idx], fill="white")

>>> image
```

<div class="flex justify-center">
    <img src="https://i.imgur.com/TdaqPJO.png" alt="CPPE-5 Image Example"/>
</div>


要可视化带有关联标签的边界框，可以从数据集的元数据（特别是 `category` 字段）获取标签。您还需要创建将标签 ID 映射到标签类别（`id2label`）以及反过来映射的字典（`label2id`）。

如果将模型分享到 Hugging Face Hub 上，包括这些映射将使其他人可以重复使用您的模型。作为熟悉数据的最后一步，检查潜在问题。

目标检测数据集常见的一个问题是边界框“延伸”到图像的边缘之外。这种“逃逸”边界框可能导致训练时出现错误，因此应在此阶段解决。该数据集中有几个示例出现了该问题。

为了简化本指南，请将这些图像从数据中删除。

```py
>>> remove_idx = [590, 821, 822, 875, 876, 878, 879]
>>> keep = [i for i in range(len(cppe5["train"])) if i not in remove_idx]
>>> cppe5["train"] = cppe5["train"].select(keep)
```

## 数据预处理

要微调模型，必须对要使用的数据进行预处理，以完全匹配预训练模型使用的方法。[`AutoImageProcessor`] 负责处理图像数据，创建 `pixel_values`、`pixel_mask` 和 `labels`，DETR 模型可以使用这些数据进行训练。图像处理器 (Image Processor)具有一些属性，您无需担心：

- `image_mean = [0.485, 0.456, 0.406 ]`- `image_std = [0.229, 0.224, 0.225]`
这些是在模型预训练期间用于归一化图像的均值和标准差。在进行推理或微调预训练的图像模型时，这些值至关重要。


从与要微调的模型相同的检查点实例化图像处理器 (Image Processor)。
```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "facebook/detr-resnet-50"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

在将图像传递给 `image_processor` 之前，对数据集应用两个预处理转换：- 对图像进行增强- 重新格式化注释以满足 DETR 的期望

首先，为了确保模型不会对训练数据过拟合，可以使用任何数据增强库对图像进行增强。

这里我们使用 [Albumentations](https://albumentations.ai/docs/)...该库确保转换影响图像，并相应地更新边界框。🤗 Datasets 库文档中有一份详细的 [如何为目标检测增强图像的指南](https://huggingface.co/docs/datasets/object_detection)，并且它使用相同的数据集作为示例。在此处应用相同的方法，将每个图像调整为（480, 480），水平翻转图像，并增加亮度：

```py
>>> import albumentations
>>> import numpy as np
>>> import torch

>>> transform = albumentations.Compose(
...     [
...         albumentations.Resize(480, 480),
...         albumentations.HorizontalFlip(p=1.0),
...         albumentations.RandomBrightnessContrast(p=1.0),
...     ],
...     bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
... )
```

`image_processor` 希望注释采用以下格式：`{'image_id': int, 'annotations': List[Dict]}`，其中每个字典是一个 COCO 对象注释。让我们添加一个函数来重新格式化单个示例的注释：
```py
>>> def formatted_anns(image_id, category, area, bbox):
...     annotations = []
...     for i in range(0, len(category)):
...         new_ann = {
...             "image_id": image_id,
...             "category_id": category[i],
...             "isCrowd": 0,
...             "area": area[i],
...             "bbox": list(bbox[i]),
...         }
...         annotations.append(new_ann)

...     return annotations
```

现在，您可以将图像和注释转换组合在一起，用于批量示例：
```py
>>> # transforming a batch
>>> def transform_aug_ann(examples):
...     image_ids = examples["image_id"]
...     images, bboxes, area, categories = [], [], [], []
...     for image, objects in zip(examples["image"], examples["objects"]):
...         image = np.array(image.convert("RGB"))[:, :, ::-1]
...         out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

...         area.append(objects["area"])
...         images.append(out["image"])
...         bboxes.append(out["bboxes"])
...         categories.append(out["category"])

...     targets = [
...         {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
...         for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
...     ]

...     return image_processor(images=images, annotations=targets, return_tensors="pt")
```

使用🤗 Datasets [`~datasets.Dataset.with_transform`] 方法将此预处理函数应用于整个数据集。在加载数据集的元素时，此方法会动态地应用变换。
此时，您可以检查经过转换后数据集的示例外观。

您应该可以看到一个包含 `pixel_values` 张量、一个包含 `pixel_mask` 张量和 `labels` 的张量。

您已经成功增强了单个图像并准备好它们的注释。然而，预处理还没有完成。

在最后一步中，创建一个自定义的 `collate_fn` 来将图像批量化。

```py
>>> cppe5["train"] = cppe5["train"].with_transform(transform_aug_ann)
>>> cppe5["train"][15]
{'pixel_values': tensor([[[ 0.9132,  0.9132,  0.9132,  ..., -1.9809, -1.9809, -1.9809],
          [ 0.9132,  0.9132,  0.9132,  ..., -1.9809, -1.9809, -1.9809],
          [ 0.9132,  0.9132,  0.9132,  ..., -1.9638, -1.9638, -1.9638],
          ...,
          [-1.5699, -1.5699, -1.5699,  ..., -1.9980, -1.9980, -1.9980],
          [-1.5528, -1.5528, -1.5528,  ..., -1.9980, -1.9809, -1.9809],
          [-1.5528, -1.5528, -1.5528,  ..., -1.9980, -1.9809, -1.9809]],

         [[ 1.3081,  1.3081,  1.3081,  ..., -1.8431, -1.8431, -1.8431],
          [ 1.3081,  1.3081,  1.3081,  ..., -1.8431, -1.8431, -1.8431],
          [ 1.3081,  1.3081,  1.3081,  ..., -1.8256, -1.8256, -1.8256],
          ...,
          [-1.3179, -1.3179, -1.3179,  ..., -1.8606, -1.8606, -1.8606],
          [-1.3004, -1.3004, -1.3004,  ..., -1.8606, -1.8431, -1.8431],
          [-1.3004, -1.3004, -1.3004,  ..., -1.8606, -1.8431, -1.8431]],

         [[ 1.4200,  1.4200,  1.4200,  ..., -1.6476, -1.6476, -1.6476],
          [ 1.4200,  1.4200,  1.4200,  ..., -1.6476, -1.6476, -1.6476],
          [ 1.4200,  1.4200,  1.4200,  ..., -1.6302, -1.6302, -1.6302],
          ...,
          [-1.0201, -1.0201, -1.0201,  ..., -1.5604, -1.5604, -1.5604],
          [-1.0027, -1.0027, -1.0027,  ..., -1.5604, -1.5430, -1.5430],
          [-1.0027, -1.0027, -1.0027,  ..., -1.5604, -1.5430, -1.5430]]]),
 'pixel_mask': tensor([[1, 1, 1,  ..., 1, 1, 1],
         [1, 1, 1,  ..., 1, 1, 1],
         [1, 1, 1,  ..., 1, 1, 1],
         ...,
         [1, 1, 1,  ..., 1, 1, 1],
         [1, 1, 1,  ..., 1, 1, 1],
         [1, 1, 1,  ..., 1, 1, 1]]),
 'labels': {'size': tensor([800, 800]), 'image_id': tensor([756]), 'class_labels': tensor([4]), 'boxes': tensor([[0.7340, 0.6986, 0.3414, 0.5944]]), 'area': tensor([519544.4375]), 'iscrowd': tensor([0]), 'orig_size': tensor([480, 480])}}
```


你已成功增强了单个图像并准备了它们的注释。然而，预处理还没有完成。在最后一步中，创建一个自定义的 `collate_fn` 函数来将图像批量处理在一起。将图像（现在是 `pixel_values`）填充到批处理中最大的图像大小，并创建相应的 `pixel_mask`，以指示哪些像素是真实的（1），哪些是填充的（0）。

```py
>>> def collate_fn(batch):
...     pixel_values = [item["pixel_values"] for item in batch]
...     encoding = image_processor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
...     labels = [item["labels"] for item in batch]
...     batch = {}
...     batch["pixel_values"] = encoding["pixel_values"]
...     batch["pixel_mask"] = encoding["pixel_mask"]
...     batch["labels"] = labels
...     return batch
```

## 训练 DETR 模型

在前面的部分中，您已经完成了大部分的重要工作，现在可以开始训练您的模型了！即使经过调整大小，该数据集中的图像仍然相当大，这意味着微调该模型至少需要一个 GPU。

训练包括以下步骤：

1. 使用与预处理相同的检查点加载模型，使用 [`AutoModelForObjectDetection`]。
2. 在 [`TrainingArguments`] 中定义训练超参数。
3. 将训练参数与模型、数据集、图像处理器和数据整理器一起传递给 [`Trainer`]。
4. 调用 [`~Trainer.train`] 来微调您的模型。

在从用于预处理的相同检查点加载模型时，请记得传递您之前从数据集的元数据创建的 `label2id` 和 `id2label` 映射。此外，我们指定 `ignore_mismatched_sizes=True`，以替换现有的分类头部为新的头部。

```py
>>> from transformers import AutoModelForObjectDetection

>>> model = AutoModelForObjectDetection.from_pretrained(
...     checkpoint,
...     id2label=id2label,
...     label2id=label2id,
...     ignore_mismatched_sizes=True,
... )
```

在 [`TrainingArguments`] 中，使用 `output_dir` 来指定保存模型的位置，然后根据需要配置超参数。
重要的是不要删除未使用的列，因为这将删除图像列。没有图像列，您无法创建 `pixel_values`。因此，请将 `remove_unused_columns` 设置为 `False`。
如果您希望通过将模型推送到 Hub 来共享模型，请将 `push_to_hub` 设置为 `True`（您必须登录 Hugging Face 才能上传模型）。
```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(
...     output_dir="detr-resnet-50_finetuned_cppe5",
...     per_device_train_batch_size=8,
...     num_train_epochs=10,
...     fp16=True,
...     save_steps=200,
...     logging_steps=50,
...     learning_rate=1e-5,
...     weight_decay=1e-4,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

最后，将所有内容整合在一起，并调用 [`~transformers.Trainer.train`]：

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=collate_fn,
...     train_dataset=cppe5["train"],
...     tokenizer=image_processor,
... )

>>> trainer.train()
```

如果在 `training_args` 中将 `push_to_hub` 设置为 `True`，则训练检查点将被推送到 Hugging Face Hub。在训练完成后，通过调用 [`~transformers.Trainer.push_to_hub`] 方法将最终模型推送到 Hub。

```py
>>> trainer.push_to_hub()
```

## 评估

目标检测模型通常使用一组 <a href="https://cocodataset.org/#detection-eval"> COCO 风格的度量指标 </a> 进行评估。

您可以使用现有的度量指标实现之一，但在这里，您将使用从 `torchvision` 导入的指标来评估您推送到 Hub 的最终模型。model that you pushed to the Hub.

要使用 `torchvision` 评估器，您需要准备一个 ground truth COCO 数据集。构建 COCO 数据集的 API 要求数据以特定的格式存储，因此您需要首先将图像和注释保存到磁盘上。就像在准备训练数据时一样，需要格式化 `cppe5["test"]` 中的注释。然而，图像应保持原样。
评估步骤需要一些工作，但可以分为三个主要步骤。

首先，准备 `cppe5["test"]` 数据集：格式化注释并将数据保存到磁盘上。

```py
>>> import json


>>> # format annotations the same as for training, no need for data augmentation
>>> def val_formatted_anns(image_id, objects):
...     annotations = []
...     for i in range(0, len(objects["id"])):
...         new_ann = {
...             "id": objects["id"][i],
...             "category_id": objects["category"][i],
...             "iscrowd": 0,
...             "image_id": image_id,
...             "area": objects["area"][i],
...             "bbox": objects["bbox"][i],
...         }
...         annotations.append(new_ann)

...     return annotations


>>> # Save images and annotations into the files torchvision.datasets.CocoDetection expects
>>> def save_cppe5_annotation_file_images(cppe5):
...     output_json = {}
...     path_output_cppe5 = f"{os.getcwd()}/cppe5/"

...     if not os.path.exists(path_output_cppe5):
...         os.makedirs(path_output_cppe5)

...     path_anno = os.path.join(path_output_cppe5, "cppe5_ann.json")
...     categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
...     output_json["images"] = []
...     output_json["annotations"] = []
...     for example in cppe5:
...         ann = val_formatted_anns(example["image_id"], example["objects"])
...         output_json["images"].append(
...             {
...                 "id": example["image_id"],
...                 "width": example["image"].width,
...                 "height": example["image"].height,
...                 "file_name": f"{example['image_id']}.png",
...             }
...         )
...         output_json["annotations"].extend(ann)
...     output_json["categories"] = categories_json

...     with open(path_anno, "w") as file:
...         json.dump(output_json, file, ensure_ascii=False, indent=4)

...     for im, img_id in zip(cppe5["image"], cppe5["image_id"]):
...         path_img = os.path.join(path_output_cppe5, f"{img_id}.png")
...         im.save(path_img)

...     return path_output_cppe5, path_anno
```

接下来，准备 `CocoDetection` 类的实例，以便与 `cocoevaluator` 一起使用。
```py
>>> import torchvision


>>> class CocoDetection(torchvision.datasets.CocoDetection):
...     def __init__(self, img_folder, feature_extractor, ann_file):
...         super().__init__(img_folder, ann_file)
...         self.feature_extractor = feature_extractor

...     def __getitem__(self, idx):
...         # read in PIL image and target in COCO format
...         img, target = super(CocoDetection, self).__getitem__(idx)

...         # preprocess image and target: converting target to DETR format,
...         # resizing + normalization of both image and target)
...         image_id = self.ids[idx]
...         target = {"image_id": image_id, "annotations": target}
...         encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
...         pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
...         target = encoding["labels"][0]  # remove batch dimension

...         return {"pixel_values": pixel_values, "labels": target}


>>> im_processor = AutoImageProcessor.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")

>>> path_output_cppe5, path_anno = save_cppe5_annotation_file_images(cppe5["test"])
>>> test_ds_coco_format = CocoDetection(path_output_cppe5, im_processor, path_anno)
```

最后，加载度量指标并运行评估。
```py
>>> import evaluate
>>> from tqdm import tqdm

>>> model = AutoModelForObjectDetection.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")
>>> module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
>>> val_dataloader = torch.utils.data.DataLoader(
...     test_ds_coco_format, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn
... )

>>> with torch.no_grad():
...     for idx, batch in enumerate(tqdm(val_dataloader)):
...         pixel_values = batch["pixel_values"]
...         pixel_mask = batch["pixel_mask"]

...         labels = [
...             {k: v for k, v in t.items()} for t in batch["labels"]
...         ]  # these are in DETR format, resized + normalized

...         # forward pass
...         outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

...         orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
...         results = im_processor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api

...         module.add(prediction=results, reference=labels)
...         del batch

>>> results = module.compute()
>>> print(results)
Accumulating evaluation results...
DONE (t=0.08s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.150
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.280
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.130
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.038
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.036
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.182
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.166
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.317
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.104
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.146
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.382
```

调整 [`~transformers.TrainingArguments`] 中的超参数可以进一步改善这些结果。试试看吧！

## 推理

现在，您已经微调了一个 DETR 模型，对其进行了评估，并将其上传到 Hugging Face Hub，可以用它进行推理。在推理中尝试您微调的模型最简单的方法是在 [`Pipeline`] 中使用它。实例化一个带有您的模型的对象检测管道，并将图像传递给它：

```py
>>> from transformers import pipeline
>>> import requests

>>> url = "https://i.imgur.com/2lnWoly.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> obj_detector = pipeline("object-detection", model="MariaK/detr-resnet-50_finetuned_cppe5")
>>> obj_detector(image)
```

如果您愿意，也可以手动复制管道的结果：
```py
>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")
>>> model = AutoModelForObjectDetection.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")

>>> with torch.no_grad():
...     inputs = image_processor(images=image, return_tensors="pt")
...     outputs = model(**inputs)
...     target_sizes = torch.tensor([image.size[::-1]])
...     results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

>>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
...     box = [round(i, 2) for i in box.tolist()]
...     print(
...         f"Detected {model.config.id2label[label.item()]} with confidence "
...         f"{round(score.item(), 3)} at location {box}"
...     )
Detected Coverall with confidence 0.566 at location [1215.32, 147.38, 4401.81, 3227.08]
Detected Mask with confidence 0.584 at location [2449.06, 823.19, 3256.43, 1413.9]
```

让我们绘制结果：

```py
>>> draw = ImageDraw.Draw(image)

>>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
...     box = [round(i, 2) for i in box.tolist()]
...     x, y, x2, y2 = tuple(box)
...     draw.rectangle((x, y, x2, y2), outline="red", width=1)
...     draw.text((x, y), model.config.id2label[label.item()], fill="white")

>>> image
```


<div class="flex justify-center">
    <img src="https://i.imgur.com/4QZnf9A.png" alt="Object detection result on a new image"/>
</div>
