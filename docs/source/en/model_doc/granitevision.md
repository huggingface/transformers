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

# Granite Vision

## Overview

The Granite Vision model is a variant of [LLaVA-NeXT](llava_next), leveraging a [Granite](granite) language model alongside a [SigLIP](SigLIP) visual encoder. It utilizes multiple concatenated vision hidden states as its image features, similar to [VipLlava](vipllava). It also uses a larger set of image grid pinpoints than the original LlaVa-NeXT models to support additional aspect ratios.

Tips:
- This model is loaded into Transformers as an instance of LlaVA-Next. The usage and tips from [LLaVA-NeXT](llava_next) apply to this model as well.

- You can apply the chat template on the tokenizer / processor in the same way as well. Example chat format:
```bash
"<|user|>\nWhat’s shown in this image?\n<|assistant|>\nThis image shows a red stop sign.<|end_of_text|><|user|>\nDescribe the image in more details.\n<|assistant|>\n"
```

Sample inference:
```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

model_path = "ibm-granite/granite-vision-3.1-2b-preview"
processor = LlavaNextProcessor.from_pretrained(model_path)

model = LlavaNextForConditionalGeneration.from_pretrained(model_path).to("cuda")

# prepare image and text prompt, using the appropriate prompt template
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to("cuda")


# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))
```

This model was contributed by [Alexander Brooks](https://huggingface.co/abrooks9944).

## LlavaNextConfig

[[autodoc]] LlavaNextConfig

## LlavaNextImageProcessor

[[autodoc]] LlavaNextImageProcessor
    - preprocess

## LlavaNextProcessor

[[autodoc]] LlavaNextProcessor

## LlavaNextForConditionalGeneration

[[autodoc]] LlavaNextForConditionalGeneration
    - forward
