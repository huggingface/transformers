<!--版权所有 2023 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）的规定，您除非遵守许可证，否则不得使用本文件。您可以在以下网址获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“原样”分发的，不附带任何形式的保证或条件。请参阅许可证以获得特定语言下的权限和限制。特别注意，此文件以 Markdown 格式编写，但包含我们文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确显示。
-->


# 图像字幕

[[在 Colab 中打开]]

图像字幕是预测给定图像的字幕的任务。其常见的实际应用包括帮助视障人士在不同情况下进行导航。因此，图像字幕通过为人们描述图像来提高内容的可访问性。

本指南将向您展示如何：

* 对图像字幕模型进行微调。
* 使用微调后的模型进行推理。

开始之前，请确保您已安装所有必要的库：
```bash
pip install transformers datasets evaluate -q
pip install jiwer -q
```

我们鼓励您登录您的 Hugging Face 账户，这样您就可以与社区共享您的模型。在提示时，输入您的令牌以登录：

```python
from huggingface_hub import notebook_login

notebook_login()
```

## 加载 Pok é mon BLIP 字幕数据集

使用🤗数据集库加载由{image-caption}对组成的数据集。要在 PyTorch 中创建自己的图像字幕数据集，可以参考 [此笔记本](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/GIT/Fine_tune_GIT_on_an_image_captioning_dataset.ipynb)。

```python
from datasets import load_dataset

ds = load_dataset("lambdalabs/pokemon-blip-captions")
ds
```
```bash
DatasetDict({
    train: Dataset({
        features: ['image', 'text'],
        num_rows: 833
    })
})
```

数据集有两个特征，`image` 和 `text`。
<Tip>
许多图像字幕数据集每个图像都有多个字幕。在这种情况下，一种常见的策略是在训练过程中在可用字幕中随机选择一个字幕。
</Tip>
使用 [~datasets.Dataset.train_test_split] 方法将数据集的训练集拆分为训练集和测试集：

```python
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]
```

让我们从训练集中可视化几个样本。

```python
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np


def plot_images(images, captions):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        caption = captions[i]
        caption = "\n".join(wrap(caption, 12))
        plt.title(caption)
        plt.imshow(images[i])
        plt.axis("off")


sample_images_to_visualize = [np.array(train_ds[i]["image"]) for i in range(5)]
sample_captions = [train_ds[i]["text"] for i in range(5)]
plot_images(sample_images_to_visualize, sample_captions)
```
    
<div class="flex justify-center">    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_training_images_image_cap.png" alt="Sample training images"/> </div>

## 预处理数据集

由于数据集有两种模态（图像和文本），预处理流程将对图像和字幕进行预处理。

为此，加载与即将进行微调的模型相关联的处理器类。
```python
from transformers import AutoProcessor

checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)
```

处理器将在内部预处理图像（包括调整大小和像素缩放）并对字幕进行标记化。
```python
def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs


train_ds.set_transform(transforms)
test_ds.set_transform(transforms)
```

数据集准备就绪后，您现在可以为微调设置模型。

## 加载基础模型

将 ["microsoft/git-base"](https://huggingface.co/microsoft/git-base) 加载到 [`AutoModelForCausalLM`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) 对象中。


```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(checkpoint)
```

## 评估

图像字幕模型通常使用 [Rouge 分数](https://huggingface.co/spaces/evaluate-metric/rouge) 或 [词错误率](https://huggingface.co/spaces/evaluate-metric/wer) 进行评估。在本指南中，您将使用词错误率（WER）。

我们使用🤗评估库来进行评估。有关 WER 的潜在限制和其他注意事项，请参阅 [此指南](https://huggingface.co/spaces/evaluate-metric/wer)。

```python
from evaluate import load
import torch

wer = load("wer")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}
```

## 训练！
现在，您已准备好开始微调模型了。您将使用🤗 [`Trainer`] 进行此操作。

首先，使用 [`TrainingArguments`] 定义训练参数。

```python
from transformers import TrainingArguments, Trainer

model_name = checkpoint.split("/")[1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-pokemon",
    learning_rate=5e-5,
    num_train_epochs=50,
    fp16=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=50,
    remove_unused_columns=False,
    push_to_hub=True,
    label_names=["labels"],
    load_best_model_at_end=True,
)
```

然后将参数与数据集和模型一起传递给🤗 Trainer。
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
```

要开始训练，只需在 [`Trainer`] 对象上调用 [`~Trainer.train`]。
```python 
trainer.train()
```

您应该看到训练损失随着训练的进行而平滑下降。
训练完成后，使用 [`~Trainer.push_to_hub`] 方法将您的模型共享到 Hub，以便每个人都可以使用您的模型：

```python
trainer.push_to_hub()
```

## 推理
从 `test_ds` 中获取一个样本图像以测试模型。

```python
from PIL import Image
import requests

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
image = Image.open(requests.get(url, stream=True).raw)
image
```

<div class="flex justify-center">    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/test_image_image_cap.png" alt="Test image"/> </div>    

为模型准备图像。
```python
device = "cuda" if torch.cuda.is_available() else "cpu"

inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values
```

调用 [`generate`] 并解码预测结果。
```python
generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)
```
```bash
a drawing of a pink and blue pokemon
```

看起来微调后的模型生成了一个非常好的字幕！