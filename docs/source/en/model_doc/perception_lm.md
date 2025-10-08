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

*This model was released on 2025-04-17 and added to Hugging Face Transformers on 2025-07-11 and contributed by [shumingh](https://huggingface.co/shumingh).*

# PerceptionLM

[PerceptionLM](https://huggingface.co/papers/2504.13180) is a fully open and reproducible vision-language model designed for transparent research in image and video understanding. It features a vision encoder paired with a small-scale LLM decoder. The model addresses the issue of closed-source vision-language models by avoiding distillation from proprietary models. To enhance detailed video understanding, 2.8M human-labeled instances of fine-grained video question-answer pairs and spatio-temporally grounded video captions are released. Additionally, PLM–VideoBench, a suite for evaluating challenging video understanding tasks, is introduced, focusing on reasoning about "what," "where," "when," and "how" in videos. The work is fully reproducible, with data, training recipes, code, and models provided.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import hf_hub_download

processor = AutoProcessor.from_pretrained("facebook/Perception-LM-1B")
model = AutoModelForImageTextToText.from_pretrained("facebook/Perception-LM-1B", dtype="auto")
test_image_file = hf_hub_download(
            repo_id="shumingh/perception_lm_test_images",
            filename="14496_0.PNG",
            repo_type="dataset",
)
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": test_image_file,
            },
            {"type": "text", "text": "Describe the bar plot in the image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    [conversation],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=256)
input_length = inputs["input_ids"].shape[1]
generate_ids_without_inputs = generate_ids[:, input_length:]

for output in processor.batch_decode(generate_ids_without_inputs, skip_special_tokens=True):
    print(output)
```

</hfoption>
</hfoptions>

## PerceptionLMConfig

[[autodoc]] PerceptionLMConfig

## PerceptionLMProcessor

[[autodoc]] PerceptionLMProcessor

## PerceptionLMImageProcessorFast

[[autodoc]] PerceptionLMImageProcessorFast

## PerceptionLMVideoProcessor

[[autodoc]] PerceptionLMVideoProcessor

## PerceptionLMModel

[[autodoc]] PerceptionLMModel

## PerceptionLMForConditionalGeneration

[[autodoc]] PerceptionLMForConditionalGeneration
    - forward

