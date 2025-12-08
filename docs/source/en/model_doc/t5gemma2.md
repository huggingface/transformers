
<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-12-01.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# T5Gemma 2

T5Gemma 2 is a family of pretrained encoder-decoder large language models with strong multilingual, multimodal and long-context capability, available in 270M-270M, 1B-1B and 4B-4B parameters. Following T5Gemma, it is built via model adaptation (based on Gemma 3) using UL2. The architecture is similar to T5Gemma and Gemma 3, enhanced with tied word embeddings and merged self- and cross-attention to save model parameters.

> [!TIP]
> Click on the T5Gemma 2 models in the right sidebar for more examples of how to apply T5Gemma 2 to different language tasks.

The example below demonstrates how to chat with the model with [`Pipeline`] or the [`AutoModel`] class, and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

generator = pipeline(
    "image-text-to-text",
    model="google/t5gemma-2-270m-270m",
    dtype=torch.bfloat16,
    device_map="auto",
)

generator(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    text="<start_of_image> in this image, there is",
    generate_kwargs={"do_sample": False, "max_new_tokens": 50},
)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

processor = AutoProcessor.from_pretrained("google/t5gemma-2-270m-270m")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/t5gemma-2-270m-270m",
    device_map="auto",
    dtype=torch.bfloat16,
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "<start_of_image> in this image, there is"

model_inputs = processor(text=prompt, images=image, return_tensors="pt")
generation = model.generate(**model_inputs, max_new_tokens=20, do_sample=False)
print(processor.decode(generation[0]))
```

</hfoption>
</hfoptions>

## T5Gemma2Config

[[autodoc]] T5Gemma2Config

## T5Gemma2TextConfig

[[autodoc]] T5Gemma2TextConfig

## T5Gemma2EncoderConfig

[[autodoc]] T5Gemma2EncoderConfig

## T5Gemma2DecoderConfig
[[autodoc]] T5Gemma2DecoderConfig

## T5Gemma2Model

[[autodoc]] T5Gemma2Model
    - forward

## T5Gemma2ForConditionalGeneration

[[autodoc]] T5Gemma2ForConditionalGeneration
    - forward

## T5Gemma2ForSequenceClassification

[[autodoc]] T5Gemma2ForSequenceClassification
    - forward

## T5Gemma2ForTokenClassification

[[autodoc]] T5Gemma2ForTokenClassification
    - forward
