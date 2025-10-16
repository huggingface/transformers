<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-12-01 and added to Hugging Face Transformers on 2023-12-13 and contributed by [ybelkada](https://huggingface.co/ybelkada).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# VipLlava

[VipLlava](https://huggingface.co/papers/2312.00784) introduces a novel multimodal model that enhances region-specific comprehension in vision-language tasks. It achieves this by allowing users to mark images with natural cues such as "red bounding boxes" or "pointed arrows" during training. This approach eliminates the need for complex region encodings and achieves state-of-the-art performance on benchmarks like Visual7W, PointQA, and Visual Commonsense Reasoning. Additionally, ViP-Bench is presented as a comprehensive benchmark to evaluate models' capabilities in understanding visual prompts across various dimensions.

<hfoptions id="usage">
<hfoption id="VipLlavaForConditionalGeneration">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, VipLlavaForConditionalGeneration

model = VipLlavaForConditionalGeneration.from_pretrained("llava-hf/vip-llava-7b-hf", dtype="auto")
processor = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf")

prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{}###Assistant:"
question = "Can you please describe this image?"
prompt = prompt.format(question)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=text, images=image, return_tensors="pt").to(0, torch.float16)
generate_ids = model.generate(**inputs, max_new_tokens=20)
print(processor.decode(generate_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Usage tips

- Use `padding_side="left"` for batched generation to get more accurate results. Set `processor.tokenizer.padding_side = "left"` before generating.
- The model doesn't explicitly train to process multiple images in the same prompt. While technically possible, you may get inaccurate results.
- LLaVA models after release v4.46 raise warnings about adding `processor.patch_size = {{patch_size}}`, `processor.num_additional_image_tokens = {{num_additional_image_tokens}}`, and `processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. Add these attributes to the processor if you own the model checkpoint, or open a PR if you don't.
- Adding these attributes means LLaVA infers the number of image tokens required per image and expands text with `<image>` placeholders. Usually around 500 tokens per image, so ensure text isn't truncated to avoid embedding merge failures.
- Get attributes from `model.config.vision_config.patch_size` or `model.config.vision_feature_select_strategy`. Set `num_additional_image_tokens` to 1 if the vision backbone adds a CLS token or 0 if nothing extra adds to the vision patches.
- Use the processor's [`apply_chat_template`] method to format prompts correctly. Construct a conversation history instead of passing a plain string. Each message in the conversation history is a dictionary with keys "role" and "content". The "content" should be a list of dictionaries for "text" and "image" modalities.

## VipLlavaConfig

[[autodoc]] VipLlavaConfig

## VipLlavaForConditionalGeneration

[[autodoc]] VipLlavaForConditionalGeneration
    - forward

## VipLlavaModel

[[autodoc]] VipLlavaModel
    - forward

