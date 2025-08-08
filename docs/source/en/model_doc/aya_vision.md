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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Aya Vision

[Aya Vision](https://huggingface.co/papers/2505.08751) is a family of open-weight multimodal vision-language models from Cohere Labs. It is trained with a synthetic annotation framework that generates high-quality multilingual image captions, improving Aya Vision's generated responses. In addition, a cross-modal model merging technique is used to prevent the model from losing its text capabilities after adding vision capabilities. The model combines a CommandR-7B language model with a SigLIP vision encoder.

You can find all the original Aya Vision checkpoints under the [Aya Vision](https://huggingface.co/collections/CohereLabs/cohere-labs-aya-vision-67c4ccd395ca064308ee1484) collection.

> [!TIP]
> This model was contributed by [saurabhdash](https://huggingface.co/saurabhdash) and [yonigozlan](https://huggingface.co/yonigozlan).
> 
> Click on the Aya Vision models in the right sidebar for more examples of how to apply Aya Vision to different image-to-text tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(model="CohereLabs/aya-vision-8b", task="image-text-to-text", device_map="auto")

# Format message with the aya-vision chat template
messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://media.istockphoto.com/id/458012057/photo/istanbul-turkey.jpg?s=612x612&w=0&k=20&c=qogAOVvkpfUyqLUMr_XJQyq-HkACXyYUSZbKhBlPrxo="},
        {"type": "text", "text": "Bu resimde hangi anıt gösterilmektedir?"},
    ]},
    ]
outputs = pipe(text=messages, max_new_tokens=300, return_full_text=False)

print(outputs)
```

</hfoption>
<hfoption id="AutoModel">

```python
# pip install 'git+https://github.com/huggingface/transformers.git@v4.49.0-Aya Vision'
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "CohereLabs/aya-vision-8b"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map="auto", dtype=torch.float16
)

# Format message with the aya-vision chat template
messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://pbs.twimg.com/media/Fx7YvfQWYAIp6rZ?format=jpg&name=medium"},
        {"type": "text", "text": "चित्र में लिखा पाठ क्या कहता है?"},
    ]},
    ]

inputs = processor.apply_chat_template(
    messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

gen_tokens = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.3,
)

print(processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

Quantization reduces the memory footprint of large models by representing weights at lower precision. Refer to the [Quantization](../quantization/overview) overview for supported backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to 4-bits.

```python
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

processor = AutoProcessor.from_pretrained("CohereLabs/aya-vision-32b", use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    "CohereLabs/aya-vision-32b",
    quantization_config=bnb_config,
    device_map="auto"
)

inputs = processor.apply_chat_template(
    [
    {"role": "user", "content": [
        {"type": "image", "url": "https://huggingface.co/roschmid/dog-races/resolve/main/images/Border_Collie.jpg"},
        {"type": "text",  "text":"Describe what you see."}
    ]}
    ],
    padding=True,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
).to("cuda")

generated = model.generate(**inputs, max_new_tokens=50)
print(processor.tokenizer.decode(generated[0], skip_special_tokens=True))
```

## Notes

- Images are represented with the `<image>` tag in the chat template.

- Use the [`~ProcessorMixin.apply_chat_template`] method to correctly format inputs.

- The example below demonstrates inference with multiple images.
  
    ```py
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch
        
    processor = AutoProcessor.from_pretrained("CohereForAI/aya-vision-8b")
    model = AutoModelForImageTextToText.from_pretrained(
        "CohereForAI/aya-vision-8b", device_map="cuda", dtype=torch.float16
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                },
                {
                    "type": "image",
                    "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
                },
                {
                    "type": "text",
                    "text": "These images depict two different landmarks. Can you identify them?",
                },
            ],
        },
    ]
    
    inputs = processor.apply_chat_template(
        messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    
    gen_tokens = model.generate(
        **inputs, 
        max_new_tokens=300, 
        do_sample=True, 
        temperature=0.3,
    )
    
    gen_text = processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(gen_text)
    ```

- The example below demonstrates inference with batched inputs.
  
    ```py
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch
        
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        "CohereForAI/aya-vision-8b", device_map="cuda", dtype=torch.float16
    )
    
    batch_messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
                    {"type": "text", "text": "Write a haiku for this image"},
                ],
            },
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                    },
                    {
                        "type": "image",
                        "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
                    },
                    {
                        "type": "text",
                        "text": "These images depict two different landmarks. Can you identify them?",
                    },
                ],
            },
        ],
    ]
    
    batch_inputs = processor.apply_chat_template(
        batch_messages, 
        padding=True, 
        add_generation_prompt=True, 
        tokenize=True, 
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device)
    
    batch_outputs = model.generate(
        **batch_inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.3,
    )
    
    for i, output in enumerate(batch_outputs):
        response = processor.tokenizer.decode(
            output[batch_inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        print(f"Response {i+1}:\n{response}\n")
    ```

## AyaVisionProcessor

[[autodoc]] AyaVisionProcessor

## AyaVisionConfig

[[autodoc]] AyaVisionConfig

## AyaVisionModel

[[autodoc]] AyaVisionModel

## AyaVisionForConditionalGeneration

[[autodoc]] AyaVisionForConditionalGeneration
    - forward
