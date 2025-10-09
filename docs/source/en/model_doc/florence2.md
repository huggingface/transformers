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
*This model was released on 2024-06-16 and added to Hugging Face Transformers on 2025-08-20 and contributed by [ducviet00](https://huggingface.co/ducviet00).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Florence-2

[Florence-2](https://huggingface.co/papers/2311.06242) is a vision foundation model that uses a unified, prompt-based framework to perform diverse computer vision and vision-language tasks such as captioning, object detection, grounding, and segmentation. It follows a sequence-to-sequence architecture, allowing the model to take text prompts as task instructions and output text-based results. The model is trained on FLD-5B, a massive dataset containing 5.4 billion visual annotations across 126 million images, created through iterative automated labeling and model refinement. Evaluations show that Florence-2 achieves strong zero-shot and fine-tuned performance across a wide range of visual tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
import requests
from PIL import Image
from transformers import pipeline

pipeline = pipeline("image-text-to-text", model="florence-community/Florence-2-base", dtype="auto")

pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", text="<OD>")
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

model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-base", dtype="auto")
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base")

task_prompt = "<OD>"
inputs = processor(text=task_prompt, images=image, return_tensors="pt")

generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    num_beams=3,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

image_size = image.size
parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=image_size)
print(parsed_answer)"
```

</hfoption>
</hfoptions>

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
