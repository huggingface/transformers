<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

specific language governing permissions and limitations under the License. -->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-01-14.*

# LightOnOcr


**LightOnOcr** is a compact, end-to-end vision–language model for Optical Character Recognition (OCR) and document understanding. It achieves state-of-the-art accuracy in its weight class while being several times faster and cheaper than larger general-purpose VLMs.

📝 **[Read the full blog post](https://huggingface.co/blog/lightonai/lightonocr/)** | 📓 **[Finetuning notebook](https://colab.research.google.com/drive/1WjbsFJZ4vOAAlKtcCauFLn_evo5UBRNa?usp=sharing)**

**Model Overview**

LightOnOcr combines a Vision Transformer encoder (Pixtral-based) with a lightweight text decoder (Qwen3-based) distilled from high-quality open VLMs. It is optimized for document parsing tasks, producing accurate, layout-aware text extraction from high-resolution pages.

## Usage

```python
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor


model = LightOnOcrForConditionalGeneration.from_pretrained("lightonai/LightOnOCR-1B-1025", device_map="auto")
processor = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-1B-1025")

url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg"

conversation = [{"role": "user", "content": [{"type": "image", "url": url}]}]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
output_text = processor.decode(generated_ids, skip_special_tokens=True)
print(output_text)
```

## LightOnOcrConfig

[[autodoc]] LightOnOcrConfig

## LightOnOcrProcessor

[[autodoc]] LightOnOcrProcessor
    - __call__

## LightOnOcrModel

[[autodoc]] LightOnOcrModel
    - forward
    - get_image_features

## LightOnOcrForConditionalGeneration

[[autodoc]] LightOnOcrForConditionalGeneration
    - forward
    - get_image_features
