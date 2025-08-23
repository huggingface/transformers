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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Florence-2

[Florence-2](https://huggingface.co/papers/2311.06242) is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks. Florence-2 can interpret simple text prompts to perform tasks like captioning, object detection, and segmentation. It leverages the FLD-5B dataset, containing 5.4 billion annotations across 126 million images, to master multi-task learning. The model's sequence-to-sequence architecture enables it to excel in both zero-shot and fine-tuned settings, proving to be a competitive vision foundation model.

You can find all the original Florence-2 checkpoints under the [Florence-2](https://huggingface.co/models?other=florence-2) collection.

> [!TIP]
> This model was contributed by [ducviet00](https://huggingface.co/ducviet00).
> Click on the Florence-2 models in the right sidebar for more examples of how to apply Florence-2 to different vision and language tasks.

The example below demonstrates how to perform object detection with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
import requests
from PIL import Image
from transformers import pipeline

pipeline = pipeline(
    "image-text-to-text",
    model="ducviet00/Florence-2-base-hf",
    device=0,
    dtype=torch.bfloat16
)

pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
    text="<OD>"
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-base", dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base")

task_prompt = "<OD>"
inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    num_beams=3,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

image_size = image.size
parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=image_size)
print(parsed_answer)
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to quantize the model to 4-bit.

```py
# pip install bitsandbytes
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = Florence2ForConditionalGeneration.from_pretrained(
    "microsoft/Florence-2-large",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

task_prompt = "<OD>"
inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(model.device, torch.bfloat16)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    num_beams=3,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

image_size = image.size
parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=image_size)

print(parsed_answer)
```

<div class="flex justify-center">
    <img src=""/>
</div>

## Notes

- Florence-2 is a prompt-based model. You need to provide a task prompt to tell the model what to do. Supported tasks are:
    - `<OCR>`
    - `<OCR_WITH_REGION>`
    - `<CAPTION>`
    - `<DETAILED_CAPTION>`
    - `<MORE_DETAILED_CAPTION>`
    - `<OD>`
    - `<DENSE_REGION_CAPTION>`
    - `<CAPTION_TO_PHRASE_GROUNDING>`
    - `<REFERRING_EXPRESSION_SEGMENTATION>`
    - `<REGION_TO_SEGMENTATION>`
    - `<OPEN_VOCABULARY_DETECTION>`
    - `<REGION_TO_CATEGORY>`
    - `<REGION_TO_DESCRIPTION>`
    - `<REGION_TO_OCR>`
    - `<REGION_PROPOSAL>`
- The raw output of the model is a string that needs to be parsed. The [`Florence2Processor`] has a [`~Florence2Processor.post_process_generation`] method that can parse the string into a more usable format, like bounding boxes and labels for object detection.

## Resources

- [Florence-2 technical report](https://huggingface.co/papers/2311.06242)
- [Jupyter Notebook for inference and visualization of Florence-2-large model](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb)

## Florence2VisionConfig

[[autodoc]] Florence2VisionConfig

## Florence2Config

[[autodoc]] Florence2Config

## Florence2Processor

[[autodoc]] Florence2Processor

## Florence2Model

[[autodoc]] Florence2Model
    - forward

## Florence2ForConditionalGeneration

[[autodoc]] Florence2ForConditionalGeneration
    - forward

## Florence2VisionBackbone

[[autodoc]] Florence2VisionBackbone
    - forward
