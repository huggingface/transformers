<!--版权所有2021年的HuggingFace团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证的要求，否则您不得使用此文件。您可以在许可证中获取许可证副本。
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按原样”分发，在没有任何形式的保证或条件的情况下进行分发。请参阅许可证以了解特定语言下的权限和限制。# 特别提示：请注意，该文件是 Markdown 格式的，但包含我们文档生成器的特定语法（类似于 MDX），在 Markdown 查看器中可能无法正确渲染。-->


# MatCha

## 概述

MatCha 是由 Fangyu Liu、Francesco Piccinno、Syrine Krichene、Chenxi Pang、Kenton Lee、Mandar Joshi、Yasemin Altun、Nigel Collier 和 Julian Martin Eisenschlos 共同提出的。

该论文的摘要如下所述：

*在人类世界中，诸如情节、图表和信息图等的视觉语言数据无处不在。然而，最先进的视觉语言模型在处理这些数据时表现不佳。我们提出了 MatCha（数学推理和图表去渲染预训练），以增强视觉语言模型在联合建模图表/情节和语言数据方面的能力。具体而言，我们提出了几个预训练任务，涵盖了图表解构和数值推理，这是视觉语言建模中的关键能力。我们从最近提出的图像到文本视觉语言模型 Pix2Struct 开始进行 MatCha 预训练。在 PlotQA 和 ChartQA 等标准基准测试中，MatCha 模型的性能超过了最先进的方法近 20%。我们还研究了 MatCha 预训练在屏幕截图、教科书图示和文档图表等领域的转移情况，并观察到了整体改进情况，验证了 MatCha 预训练在更广泛的视觉语言任务中的实用性.*




## 模型描述

MatCha 是使用 `Pix2Struct` 架构训练的模型。您可以在 [Pix2Struct 文档](https://huggingface.co/docs/transformers/main/en/model_doc/pix2struct) 中找到更多信息。MatCha 是 `Pix2Struct` 架构的视觉问答子集。它将输入问题渲染到图像上并预测答案。

## 用法

目前有 6 个可用于 MatCha 的检查点：
- `google/matcha`：基础 MatCha 模型，用于在下游任务上微调 MatCha- `google/matcha-chartqa`：在 ChartQA 数据集上微调的 MatCha 模型。可用于回答有关图表的问题。
- `google/matcha-plotqa-v1`：在 PlotQA 数据集上微调的 MatCha 模型。可用于回答有关情节的问题。- `google/matcha-plotqa-v2`：在 PlotQA 数据集上微调的 MatCha 模型。可用于回答有关情节的问题。
- `google/matcha-chart2text-statista`：在 Statista 数据集上微调的 MatCha 模型。- `google/matcha-chart2text-pew`：在 Pew 数据集上微调的 MatCha 模型。

在 `chart2text-pew` 和 `chart2text-statista` 上微调的模型更适用于摘要，而在 `plotqa` 和 `chartqa` 上微调的模型更适用于问答。

您可以按以下方式使用这些模型（以 ChatQA 数据集为例）：

```python
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-chartqa").to(0)
processor = AutoProcessor.from_pretrained("google/matcha-chartqa")
url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/20294671002019.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="Is the sum of all 4 places greater than Laos?", return_tensors="pt").to(0)
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
```

## 微调 Fine-tuning

要进行 MatCha 的微调，请参考 pix2struct 的 [微调笔记本](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb)。对于 `Pix2Struct` 模型，我们发现使用 Adafactor 和余弦学习率调度器对模型进行微调可以加快收敛速度：
```python
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

optimizer = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, lr=0.01, weight_decay=1e-05)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=40000)
```