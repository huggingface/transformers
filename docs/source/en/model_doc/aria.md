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
*This model was released on 2024-10-08 and added to Hugging Face Transformers on 2024-12-06 and contributed by [m-ric](https://huggingface.co/m-ric).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Aria

[Aria](https://huggingface.co/papers/2410.05993) is an open multimodal-native model designed to integrate diverse information sources and deliver comprehensive understanding. It employs a Mixture-of-Experts architecture with 3.9B and 3.5B activated parameters per visual and text token, respectively. Aria outperforms models like Pixtral-12B and Llama3.2-11B across various multimodal, language, and coding tasks. The model is pre-trained through a 4-stage pipeline that enhances language understanding, multimodal capabilities, long context handling, and instruction following. Aria's weights and codebase are open-sourced to facilitate adoption and adaptation in real-world applications.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-to-text", model="rhymes-ai/Aria", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", text="What is shown in this image?")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("rhymes-ai/Aria", dtype="auto")
processor = AutoProcessor.from_pretrained("rhymes-ai/Aria")

messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
ipnuts = inputs.to(model.device, torch.bfloat16)

output = model.generate(**inputs,
    max_new_tokens=15,
    stop_strings=["<|im_end|>"],
    tokenizer=processor.tokenizer,
    do_sample=True,
    temperature=0.9,
)
output_ids = output[0][inputs["input_ids"].shape[1]:]
response = processor.decode(output_ids, skip_special_tokens=True)
print(response)
```

</hfoption>
</hfoptions>

## AriaImageProcessor

[[autodoc]] AriaImageProcessor

## AriaProcessor

[[autodoc]] AriaProcessor

## AriaTextConfig

[[autodoc]] AriaTextConfig

## AriaConfig

[[autodoc]] AriaConfig

## AriaTextModel

[[autodoc]] AriaTextModel

## AriaTextForCausalLM

[[autodoc]] AriaTextForCausalLM

## AriaModel

[[autodoc]] AriaModel
    - forward

## AriaForConditionalGeneration

[[autodoc]] AriaForConditionalGeneration
    - forward

