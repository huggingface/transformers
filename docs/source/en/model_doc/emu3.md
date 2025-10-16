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
*This model was released on 2024-09-27 and added to Hugging Face Transformers on 2025-01-10 and contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Emu3

[Emu3: Next-Token Prediction is All You Need](https://huggingface.co/papers/2409.18869) is a multimodal LLM that employs vector quantization to tokenize images and videos into discrete tokens, which are then fused with text tokens for generation tasks. Emu3 excels in both image and text generation, outperforming models like SDXL and LLaVA-1.6, without relying on diffusion or compositional architectures. The model can also generate high-fidelity videos by predicting the next token in a video sequence, simplifying multimodal model designs by focusing on token prediction.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-text-to-text", model="BAAI/Emu3-Chat-hf", dtype="auto")
messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        {"type": "text", "text": "What is shown in this image?<image>"},
    ]},
]
pipeline(text=messages, max_new_tokens=300, return_full_text=False)
```

</hfoption>
<hfoption id="Emu3ForConditionalGeneration">

```py
import torch
import reqursts
from PIL import Image
from transformers import Emu3Processor, Emu3ForConditionalGeneration

processor = Emu3Processor.from_pretrained("BAAI/Emu3-Chat-hf")
model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Chat-hf", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "What do you see in this image?<image>"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(dtype=torch.bfloat16)

output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Usage tips

- Set `processor.tokenizer.padding_side = "left"` before batched generation for more accurate results.
- The model trains with a specific prompt format for chatting. Use `processor.apply_chat_template(my_conversation_dict)` to format prompts correctly.
- Emu3 has separate checkpoints for image generation and text generation. Use the correct checkpoint when loading the model. For image generation, use `prefix_constraints` to sample only from possible image tokens.
- Emu3 uses a special image token to indicate where to merge image embeddings. The implementation uses the reserved token `<|extra_0|>` instead of creating a new one. Add `<image>` to your prompt where the image should be embedded for correct generation.

## Emu3Config

[[autodoc]] Emu3Config

## Emu3VQVAEConfig

[[autodoc]] Emu3VQVAEConfig

## Emu3TextConfig

[[autodoc]] Emu3TextConfig

## Emu3Processor

[[autodoc]] Emu3Processor

## Emu3ImageProcessor

[[autodoc]] Emu3ImageProcessor
    - preprocess

## Emu3VQVAE

[[autodoc]] Emu3VQVAE
    - forward

## Emu3TextModel

[[autodoc]] Emu3TextModel
    - forward

## Emu3ForCausalLM

[[autodoc]] Emu3ForCausalLM
    - forward

## Emu3ForConditionalGeneration

[[autodoc]] Emu3ForConditionalGeneration
    - forward

## Emu3Model

[[autodoc]] Emu3Model
    - forward

