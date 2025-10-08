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
*This model was released on 2025-05-13 and added to Hugging Face Transformers on 2025-03-04 and contributed by [saurabhdash](https://huggingface.co/saurabhdash) and [yonigozlan](https://huggingface.co/yonigozlan).*

# AyaVision

[Aya Vision](https://huggingface.co/papers/2505.08751) ntroduce two key innovations for multilingual multimodal learning: a synthetic annotation framework that generates high-quality, diverse instruction data across languages, and a cross-modal model merging technique that prevents catastrophic forgetting while preserving strong text-only performance. These methods enable effective alignment between vision and language without degrading existing capabilities. Aya-Vision-8B surpasses comparable models like Qwen-2.5-VL-7B, Pixtral-12B, and even larger models such as Llama-3.2-90B-Vision, while the larger Aya-Vision-32B outperforms models more than twice its size, including Molmo-72B. Overall, the approach demonstrates efficient scaling and state-of-the-art multilingual multimodal performance with reduced computational demands.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-text-to-text", model="CohereLabs/aya-vision-8b", dtype="auto")
messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        {"type": "text", "text": "Que montre cette image?"},
    ]},
]
pipeline(text=messages, max_new_tokens=300, return_full_text=False)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("CohereLabs/aya-vision-8b)
model = AutoModelForImageTextToText.from_pretrained("CohereLabs/aya-vision-8b", dtype="auto")

messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        {"type": "text", "text": "Que montre cette image?"},
    ]},
]

inputs = processor.apply_chat_template(
    messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.3,
)
print(processor.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## AyaVisionProcessor

[[autodoc]] AyaVisionProcessor

## AyaVisionConfig

[[autodoc]] AyaVisionConfig

## AyaVisionModel

[[autodoc]] AyaVisionModel
    - forward

## AyaVisionForConditionalGeneration

[[autodoc]] AyaVisionForConditionalGeneration
    - forward
