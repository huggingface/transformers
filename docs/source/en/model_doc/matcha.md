<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-12-19 and added to Hugging Face Transformers on 2023-06-20.*

# MatCha

[MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering](https://huggingface.co/papers/2212.09662) is a visual-language model designed to improve understanding of charts, plots, and infographics, areas where existing models struggle. It builds on Pix2Struct and introduces specialized pretraining tasks focused on plot deconstruction and numerical reasoning. On benchmarks like PlotQA and ChartQA, MatCha achieves up to a 20% performance gain over prior state-of-the-art models. The pretraining also generalizes well to other visual-language domains, including screenshots, textbook diagrams, and document figures.

<hfoptions id="usage">
<hfoption id="">

```py
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration

model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-chartqa", dtype="auto")
processor = AutoProcessor.from_pretrained("google/matcha-chartqa")

url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/20294671002019.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="Is the sum of all 4 places greater than Laos?", return_tensors="pt").to(0)
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Usage tips

- Fine-tune MatCha using the [pix2struct fine-tuning notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb).
- Fine-tune Pix2Struct models with Adafactor and cosine learning rate scheduler for faster convergence.