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
*This model was released on 2025-03-24 and added to Hugging Face Transformers on 2026-04-30.*

# PP-FormulaNet


## Overview

**PP-FormulaNet-L** and **PP-FormulaNet_plus-L** are part of a series of dedicated lightweight models for table structure recognition, focusing on accurately recognizing table structures in documents and natural scenes. For more details about the SLANet series model, please refer to the [official documentation](https://www.paddleocr.ai/latest/en/version3.x/module_usage/table_structure_recognition.html).

## Usage

### Single input inference

The example below demonstrates how to detect text with PP-PP-FormulaNet_plus-L using the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
from io import BytesIO

import httpx
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "PaddlePaddle/PP-FormulaNet_plus-L_safetensors" # or "PaddlePaddle/PP-FormulaNet-L_safetensors"
model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

image_url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_formula_rec_001.png"
image = Image.open(BytesIO(httpx.get(image_url).content)).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(model.device)
outputs = model(**inputs)
result = processor.post_process(outputs)
print(result)
```

</hfoption>
</hfoptions>

## PPFormulaNetConfig

[[autodoc]] PPFormulaNetConfig

## PPFormulaNetForConditionalGeneration

[[autodoc]] PPFormulaNetForConditionalGeneration

## PPFormulaNetTextModel

[[autodoc]] PPFormulaNetTextModel

## PPFormulaNetVisionModel

[[autodoc]] PPFormulaNetVisionModel

## PPFormulaNetModel

[[autodoc]] PPFormulaNetModel

## PPFormulaNetTextConfig

[[autodoc]] PPFormulaNetTextConfig

## PPFormulaNetVisionConfig

[[autodoc]] PPFormulaNetVisionConfig

## PPFormulaNetImageProcessor

[[autodoc]] PPFormulaNetImageProcessor

## PPFormulaNetProcessor

[[autodoc]] PPFormulaNetProcessor
