<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-05-20 and added to Hugging Face Transformers on 2026-03-16.*

# PP-Chart2Table

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**PP-Chart2Table** is a SOTA multimodal model developed by the PaddlePaddle team, specializing in chart parsing for both Chinese and English. Its high performance is driven by a novel "Shuffled Chart Data Retrieval" training task, which, combined with a refined token masking strategy, significantly improves its efficiency in converting charts to data tables. The model is further strengthened by an advanced data synthesis pipeline that uses high-quality seed data, RAG, and LLMs persona design to create a richer, more diverse training set. To address the challenge of large-scale unlabeled, out-of-distribution (OOD) data, the team implemented a two-stage distillation process, ensuring robust adaptability and generalization on real-world data.

## Model Architecture 
PP-Chart2Table adopts a multimodal fusion architecture that combines a vision tower for chart feature extraction and a language model for table structure generation, enabling end-to-end chart-to-table conversion.


## Usage

### Single input inference

The example below demonstrates how to classify image with PP-Chart2Table using [`Pipeline`] or the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import requests
from PIL import Image
from transformers import pipeline
model_path = "PaddlePaddle/PP-Chart2Table_safetensors"
pipe = pipeline(
    task="image-text-to-text", 
    model=model_path,
    device_map="auto",
)
image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png", stream=True).raw)
result = pipe(
    images=image, 
    text="", 
    do_sample=False, 
    max_new_tokens=256
)
print(result)

```

</hfoption>

<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

model_path = "PaddlePaddle/PP-Chart2Table_safetensors"
model = AutoModelForImageTextToText.from_pretrained(
    model_path, 
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png", stream=True).raw)
inputs = processor(images=image).to(model.device)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
result = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(result)

```

</hfoption>
</hfoptions>

### Batched inference

Here is how you can do it with PP-Chart2Table using [`Pipeline`] or the [`AutoModel`]:

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import requests
from transformers import pipeline
from PIL import Image
model_path = "PaddlePaddle/PP-Chart2Table_safetensors"
pipe = pipeline(
    task="image-text-to-text", 
    model=model_path,
    device_map="auto",
)
image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png", stream=True).raw)
result = pipe(
    images=[image, image],
    text="",
    do_sample=False,
    max_new_tokens=256
)
print(result)
```

</hfoption>

<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

model_path = "PaddlePaddle/PP-Chart2Table_safetensors"
model = AutoModelForImageTextToText.from_pretrained(
    model_path, 
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/chart_parsing_02.png", stream=True).raw)
inputs = processor(images=[image, image]).to(model.device)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
result = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(result)
```

</hfoption>
</hfoptions>

## PPChart2TableForConditionalGeneration

[[autodoc]] PPChart2TableForConditionalGeneration

## PPChart2TableModel

[[autodoc]] PPChart2TableModel

## PPChart2TableConfig

[[autodoc]] PPChart2TableConfig

## PPChart2TableImageProcessorFast

[[autodoc]] PPChart2TableImageProcessorFast

## PPChart2TableProcessor

[[autodoc]] PPChart2TableProcessor
