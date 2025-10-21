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
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-03-18 and contributed by [cyrilvallez](https://huggingface.co/cyrilvallez) and [yonigozlan](https://huggingface.co/yonigozlan).*

# Mistral 3

[Mistral 3](https://mistral.ai/news/mistral-small-3) is a highly efficient open-source language model designed for general generative AI tasks, offering performance comparable to much larger models like Llama 3.3 70B and Qwen 32B while running over three times faster on the same hardware. It achieves this efficiency by using significantly fewer layers, enabling low-latency inference at around 150 tokens per second and over 81% accuracy on MMLU benchmarks. The model is available as both a pretrained and instruction-tuned checkpoint under Apache 2.0, making it a versatile base for further development. Unlike some models, it is not trained with reinforcement learning or synthetic data, positioning it as an early-stage model suitable for building advanced reasoning capabilities.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

messages = [
    {"role": "user",
        "content":[
            {"type": "image",
            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",},
            {"type": "text", "text": "Describe this image."}
        ,]
    ,}
,]

pipeline = pipeline(task="image-text-to-text", model="mistralai/Mistral-Small-3.1-24B-Instruct-2503", dtype="auto")
pipeline(text=messages, max_new_tokens=50, return_full_text=False)
```

</hfoption>
<hfoption id="AutoModelForImageTextToText">

```py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")
model = AutoModelForImageTextToText.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503", dtype="auto")

messages = [
    {"role": "user",
        "content":[
            {"type": "image",
            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",},
            {"type": "text", "text": "Describe this image."}
        ,]
    ,}
,]

inputs = processor.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    tokenize=True, return_dict=True, 
    return_tensors="pt").to(model.device, dtype=torch.bfloat16)

generate_ids = model.generate(**inputs, max_new_tokens=20)
print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

</hfoption>
</hfoptions>


## Mistral3Config

[[autodoc]] Mistral3Config

## Mistral3ForConditionalGeneration

[[autodoc]] Mistral3ForConditionalGeneration
    - forward

## Mistral3Model

[[autodoc]] Mistral3Model
    - forward
