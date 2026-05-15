<!--Copyright 2026 the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-05-08 and added to Hugging Face Transformers on 2026-05-15.*

# ZAYA1-VL

## Overview

ZAYA1-VL is a vision-language model from Zyphra built on top of the ZAYA1 text decoder and the Qwen2.5-VL vision
encoder. It adds vision-token-specific LoRA parameters in the text decoder and uses bidirectional attention between
image placeholder tokens.

For more details, see the [ZAYA1-VL model card](https://huggingface.co/Zyphra/ZAYA1-VL-8B).

This model was contributed by [JJJYmmm](https://github.com/JJJYmmm).

## Usage examples

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "Zyphra/ZAYA1-VL-8B"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "What do you see in the image?"},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
print(processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
```

## Zaya1VLConfig

[[autodoc]] Zaya1VLConfig

## Zaya1VLTextConfig

[[autodoc]] Zaya1VLTextConfig

## Zaya1VLVisionConfig

[[autodoc]] Zaya1VLVisionConfig

## Zaya1VLProcessor

[[autodoc]] Zaya1VLProcessor

## Zaya1VLModel

[[autodoc]] Zaya1VLModel
    - forward

## Zaya1VLVisionModel

[[autodoc]] Zaya1VLVisionModel
    - forward

## Zaya1VLTextModel

[[autodoc]] Zaya1VLTextModel
    - forward

## Zaya1VLForConditionalGeneration

[[autodoc]] Zaya1VLForConditionalGeneration
    - forward
