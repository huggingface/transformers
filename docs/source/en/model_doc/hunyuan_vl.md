<!--Copyright 2026 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was contributed to Hugging Face Transformers on 2026-06-09.*

# HunYuanVL

HunYuanVL is a vision-language model for image-text understanding and generation. The open-source `hunyuan_vl`
integration in Transformers is a dense-only image-text variant tailored for OCR and document understanding style
workloads such as `tencent/HunyuanOCR`.

## Scope of the open-source implementation

This Transformers integration intentionally exposes the image-text path that is exercised by public OCR-style
checkpoints.

- Supported: dense-only text backbone, image-text prompting, OCR/document-understanding style generation.
- Not supported as part of this open-source variant: video inputs and runtime MoE execution paths.
- Compatibility note: some legacy Tencent-export configuration fields are still accepted so existing checkpoints can be
  loaded, but those fields do not imply that the open-source implementation enables extra runtime capabilities.

## Recommended checkpoints

- `tencent/HunyuanOCR` for OCR and document extraction workloads.

## Usage

```python
from PIL import Image
import torch

from transformers import AutoProcessor, HunYuanVLForConditionalGeneration


model_name_or_path = "tencent/HunyuanOCR"
processor = AutoProcessor.from_pretrained(model_name_or_path, backend="pil")
model = HunYuanVLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

image = Image.open("path/to/image.jpg").convert("RGB")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": "Extract the text from the image."},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
if getattr(model, "hf_device_map", None) is None:
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=1024)

generated_ids_trimmed = generated_ids[0][len(inputs["input_ids"][0]) :]
output = processor.decode(generated_ids_trimmed, skip_special_tokens=True)
print(output)
```

## Notes

- For the currently validated OCR path, `attn_implementation="eager"` is the recommended starting point.
- `backend="pil"` is recommended when loading the processor for the current public OCR checkpoints.
- When batching variable-length prompts, pass `padding=True` if you need tensor outputs from the processor.

## Limitations

- The current open-source variant is not a drop-in replacement for internal full-capability HunYuanVL stacks.
- Public checkpoints may still carry legacy configuration keys for compatibility.
- If you are extending the model family upstream, make changes in `modular_hunyuan_vl.py` and regenerate the derived
  files instead of editing generated modeling/configuration files directly.

## HunYuanVLConfig

[[autodoc]] HunYuanVLConfig

## HunYuanVLVisionConfig

[[autodoc]] HunYuanVLVisionConfig

## HunYuanVLTextConfig

[[autodoc]] HunYuanVLTextConfig

## HunYuanVLProcessor

[[autodoc]] HunYuanVLProcessor
    - __call__

## HunYuanVLImageProcessor

[[autodoc]] HunYuanVLImageProcessor

## HunYuanVLImageProcessorPil

[[autodoc]] HunYuanVLImageProcessorPil

`HunYuanVLForConditionalGeneration` is the main public entrypoint for image-text generation. `HunYuanVLForCausalLM`
and `HunYuanVLTextModel` expose the text backbone and are mainly useful for lower-level text-only workflows.

## HunYuanVLTextModel

[[autodoc]] HunYuanVLTextModel
    - forward

## HunYuanVLForCausalLM

[[autodoc]] HunYuanVLForCausalLM
    - forward

## HunYuanVLForConditionalGeneration

[[autodoc]] HunYuanVLForConditionalGeneration
    - forward
    - get_image_features
