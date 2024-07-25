<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# KOSMOS-2.5

## Overview

Kosmos-2.5 is a multimodal literate model for machine reading of text-intensive images. Pre-trained on large-scale text-intensive images, Kosmos-2.5 excels in two distinct yet cooperative transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned its spatial coordinates within the image, and (2) producing structured text output that captures styles and structures into the markdown format. This unified multimodal literate capability is achieved through a shared decoder-only auto-regressive Transformer architecture, task-specific prompts, and flexible text representations. We evaluate Kosmos-2.5 on end-to-end document-level text recognition and image-to-markdown text generation. Furthermore, the model can be readily adapted for any text-intensive image understanding task with different prompts through supervised fine-tuning, making it a general-purpose tool for real-world applications involving text-rich images. This work also paves the way for the future scaling of multimodal large language models.

The abstract from the paper is the following:

*We present Kosmos-2.5, a multimodal literate model for machine reading of text-intensive images. Pre-trained on large-scale text-intensive images, Kosmos-2.5 excels in two distinct yet cooperative transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned its spatial coordinates within the image, and (2) producing structured text output that captures styles and structures into the markdown format. This unified multimodal literate capability is achieved through a shared Transformer architecture, task-specific prompts, and flexible text representations. We evaluate Kosmos-2.5 on end-to-end document-level text recognition and image-to-markdown text generation. Furthermore, the model can be readily adapted for any text-intensive image understanding task with different prompts through supervised fine-tuning, making it a general-purpose tool for real-world applications involving text-rich images. This work also paves the way for the future scaling of multimodal large language models.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/kosmos2_5_ocr.png"
alt="drawing" width="600"/>

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/kosmos2_5_md.png"
alt="drawing" width="600"/>

<small> Overview of tasks that KOSMOS-2.5 can handle. Taken from the <a href="https://arxiv.org/abs/2309.11419">original paper</a>. </small>

## Example

```python
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration
import re
repo = "microsoft/kosmos-2.5"
device = "cuda:0"
dtype = torch.bfloat16
model = Kosmos2_5ForConditionalGeneration.from_pretrained(repo, device_map=device, torch_dtype=dtype)
processor = AutoProcessor.from_pretrained(repo)
url = "https://huggingface.co/kirp/kosmos2_5/resolve/main/receipt_00008.png"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "<ocr>" # <md>
inputs = processor(text=prompt, images=image, return_tensors="pt")
height, width = inputs.pop("height"), inputs.pop("width")
raw_width, raw_height = image.size
scale_height = raw_height / height
scale_width = raw_width / width
inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)
generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
def postprocess(y, scale_height, scale_width):
    y = y.replace(prompt, "")
    if "<md>" in prompt:
        return y
    pattern = r"<bbox><x_\d+><y_\d+><x_\d+><y_\d+></bbox>"
    bboxs_raw = re.findall(pattern, y)
    lines = re.split(pattern, y)[1:]
    bboxs = [re.findall(r"\d+", i) for i in bboxs_raw]
    bboxs = [[int(j) for j in i] for i in bboxs]
    info = ""
    for i in range(len(lines)):
        box = bboxs[i]
        x0, y0, x1, y1 = box
        if not (x0 >= x1 or y0 >= y1):
            x0 = int(x0 * scale_width)
            y0 = int(y0 * scale_height)
            x1 = int(x1 * scale_width)
            y1 = int(y1 * scale_height)
            info += f"{x0},{y0},{x1},{y0},{x1},{y1},{x0},{y1},{lines[i]}"
    return info
output_text = postprocess(generated_text[0], scale_height, scale_width)
print(output_text)
```
```text
55,595,71,595,71,629,55,629,1
82,595,481,595,481,635,82,635,[REG] BLACK SAKURA
716,590,841,590,841,629,716,629,45,455
55,637,71,637,71,672,55,672,1
82,637,486,637,486,675,82,675,COOKIE DOH SAUCES
818,632,843,632,843,668,818,668,0
51,683,71,683,71,719,51,719,1
82,683,371,683,371,719,82,719,NATA DE COCO
820,677,845,677,845,713,820,713,0
32,770,851,770,851,811,32,811,Sub Total 45,455
28,811,853,811,853,858,28,858,PB1 (10%) 4,545
28,857,855,857,855,905,28,905,Rounding 0
24,905,858,905,858,956,24,956,Total 50,000
17,1096,868,1096,868,1150,17,1150,Card Payment 50,000
```



## Kosmos2_5Config

[[autodoc]] Kosmos2_5Config

## Kosmos2_5ImageProcessor

[[autodoc]] Kosmos2_5ImageProcessor

## Kosmos2_5Processor

[[autodoc]] Kosmos2_5Processor
    - __call__

## Kosmos2_5Model

[[autodoc]] Kosmos2_5Model
    - forward

## Kosmos2_5ForConditionalGeneration

[[autodoc]] Kosmos2_5ForConditionalGeneration
    - forward
