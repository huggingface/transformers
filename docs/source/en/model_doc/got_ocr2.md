<!--Copyright 2024 StepFun and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-09-03 and added to Hugging Face Transformers on 2025-01-31 and contributed by [yonigozlan](https://huggingface.co/yonigozlan).*

# GOT-OCR2

[GOT-OCR2](https://huggingface.co/papers/2409.01704) is a unified, end-to-end model with 580M parameters designed to handle a wide range of OCR tasks, including plain text, scene text, formatted documents, tables, charts, mathematical formulas, geometric shapes, molecular formulas, and sheet music. It features a high-compression encoder and a long-contexts decoder, supports interactive OCR through region-level recognition guided by coordinates or colors, and incorporates dynamic resolution and multipage OCR technologies. The model can generate plain or formatted results using easy prompts and is adaptable for various practical applications.

<hfoptions id="usage">
<hfoption id="AutoModelForImageTextToText">

```py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText,

model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", dtype="auto")
processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
inputs = processor(image, return_tensors="pt", device=device).to(device)

generate_ids = model.generate(
    **inputs,
    do_sample=False,
    tokenizer=processor.tokenizer,
    stop_strings="<|im_end|>",
    max_new_tokens=4096,
)

processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

</hfoption>
</hfoptions>

## GotOcr2Config

[[autodoc]] GotOcr2Config

## GotOcr2VisionConfig

[[autodoc]] GotOcr2VisionConfig

## GotOcr2ImageProcessor

[[autodoc]] GotOcr2ImageProcessor

## GotOcr2ImageProcessorFast

[[autodoc]] GotOcr2ImageProcessorFast

## GotOcr2Processor

[[autodoc]] GotOcr2Processor

## GotOcr2ForConditionalGeneration

[[autodoc]] GotOcr2ForConditionalGeneration
    - forward

## GotOcr2Model

[[autodoc]] GotOcr2Model
    - forward

