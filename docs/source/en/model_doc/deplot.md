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
*This model was released on 2022-12-20 and added to Hugging Face Transformers on 2023-06-20.*

# DePlot

[DePlot](https://huggingface.co/papers/2212.10505) presents a one-shot solution for visual language reasoning by decomposing the task into plot-to-text translation and reasoning over the translated text. The model, DePlot, translates images of plots or charts into linearized tables using a modality conversion module. This output is then used to prompt a pretrained large language model (LLM), leveraging the LLM's few-shot reasoning capabilities. DePlot is trained end-to-end on a standardized plot-to-table task and can be used with LLMs in a plug-and-play fashion. Compared to a state-of-the-art model fine-tuned on over 28,000 data points, DePlot combined with LLM achieves a 24.0% improvement on human-written queries in chart QA tasks.

<hfoptions id="usage">
<hfoption id="Pix2StructForConditionalGeneration">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration

model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot", dtype="auto")
processor = AutoProcessor.from_pretrained("google/deplot")

url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Usage tips

- Fine-tune DePlot using the pix2struct fine-tuning [notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb).
- Fine-tune Pix2Struct models with Adafactor and cosine learning rate scheduler for faster convergence.