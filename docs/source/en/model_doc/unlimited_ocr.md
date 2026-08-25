<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# UnlimitedOCR


## Overview

UnlimitedOCR is an OCR-specialized vision-language model with a dual vision tower (SAM ViT-B + CLIP-L) and a DeepSeek-V2 Mixture-of-Experts language backbone. Global and optional local tiled views are packed into the language model with learnable newline and view-separator tokens.

Hub checkpoints may still advertise `model_type: "unlimited-ocr"`; the in-tree identifier is `unlimited_ocr`.

## Usage example

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "baidu/Unlimited-OCR", device_map="auto", dtype="auto"
)
processor = AutoProcessor.from_pretrained("baidu/Unlimited-OCR")

inputs = processor(
    images="https://huggingface.co/baidu/Unlimited-OCR/resolve/main/assets/baidu.png",
    text="<image>document parsing.",
    return_tensors="pt",
).to(model.device)

generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
```

## UnlimitedOCRConfig

[[autodoc]] UnlimitedOCRConfig

## UnlimitedOCRVisionConfig

[[autodoc]] UnlimitedOCRVisionConfig

## UnlimitedOCRSamVisionConfig

[[autodoc]] UnlimitedOCRSamVisionConfig

## UnlimitedOCRVisionEncoderConfig

[[autodoc]] UnlimitedOCRVisionEncoderConfig

## UnlimitedOCRTextConfig

[[autodoc]] UnlimitedOCRTextConfig

## UnlimitedOCRImageProcessor

[[autodoc]] UnlimitedOCRImageProcessor

## UnlimitedOCRImageProcessorPil

[[autodoc]] UnlimitedOCRImageProcessorPil

## UnlimitedOCRProcessor

[[autodoc]] UnlimitedOCRProcessor

## UnlimitedOCRTextModel

[[autodoc]] UnlimitedOCRTextModel

## UnlimitedOCRVisionModel

[[autodoc]] UnlimitedOCRVisionModel

## UnlimitedOCRModel

[[autodoc]] UnlimitedOCRModel

## UnlimitedOCRForConditionalGeneration

[[autodoc]] UnlimitedOCRForConditionalGeneration
