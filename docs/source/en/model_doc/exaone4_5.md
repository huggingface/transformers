<!--Copyright 2026 The LG AI Research and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-04-09 and added to Hugging Face Transformers on 2026-05-04.*

# EXAONE 4.5

## Overview

[EXAONE 4.5](https://github.com/LG-AI-EXAONE/EXAONE-4.5) model is the first open-weight vision language model developed by LG AI Research.
Integrating a dedicated visual encoder into the existing EXAONE 4.0 framework, we expand the model's capability toward multimodality.
EXAONE 4.5 features 33 billion parameters in total, including 1.2 billion parameters from the vision encoder. 
EXAONE 4.5 achieves competitive performance in general benchmark while outperforming SOTA models of similar size in document understanding and Korean contextual reasoning, inheriting powerful language capabilities from our previous language models.

EXAONE 4.5 builds on the foundation of EXAONE 4.0 with several key enhancements. The vocabulary size has been expanded to 153,600, and the context window now supports up to 256K tokens. In addition, a Multi-Token Prediction (MTP) mechanism has been introduced, further improving the model's performance.

For more details, please refer to the [technical report](https://huggingface.co/papers/2604.08644), [blog](https://www.lgresearch.ai/blog/view?seq=641) and [GitHub](https://github.com/LG-AI-EXAONE/EXAONE-4.5).

All model weights including quantized version are available at [Huggingface Collections](https://huggingface.co/collections/LGAI-EXAONE/exaone-45).

## Usage tips

> To achieve the expected performance, we recommend using the following configurations:
> - We recommend to use `temperature=1.0`, `top_p=0.95`, `presence_penalty=1.5` for general purpose.
> - We recommend to use `temperature=0.6`, `top_p=0.95`, `presence_penalty=1.5`, `top_k=20` for OCR/document-related tasks, and Korean inputs.
> - We recommend to use `temperature=1.0`, `top_p=0.95` for text-only inputs.
> - Different from EXAONE-4.0, EXAONE 4.5 uses `enable_thinking=True` as default. Thus, you need to set `enable_thinking=False` when you want to use non-reasoning mode.
> - EXAONE 4.5 prefers using `\boxed{}` format to answer the question. We recommend using this format with the corresponding format instruction for better parsing accuracy. 

For tasks that require accurate results, you can run the EXAONE 4.5 model in reasoning mode, whereas for tasks where latency matters more than accuracy, you can run the EXAONE 4.5 model in non-reasoning mode.

Here is the example code for using EXAONE 4.5 model in reasoning mode:

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image

model_id = "LGAI-EXAONE/EXAONE-4.5-33B"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
)

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(image_url)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_url},
            {"type": "text", "text": "Describe the image."},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,   # default: True
)
inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_text = processor.batch_decode(
    generated_ids[:, inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)[0]
print(generated_text)
```


## Exaone4_5_Config

[[autodoc]] Exaone4_5_Config

## Exaone4_5_VisionConfig

[[autodoc]] Exaone4_5_VisionConfig

## Exaone4_5_Processor

[[autodoc]] Exaone4_5_Processor

## Exaone4_5_VisionModel

[[autodoc]] Exaone4_5_VisionModel
    - forward

## Exaone4_5_Model

[[autodoc]] Exaone4_5_Model
    - forward

## Exaone4_5_ForConditionalGeneration

[[autodoc]] Exaone4_5_ForConditionalGeneration
    - forward