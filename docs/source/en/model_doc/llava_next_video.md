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
*This model was released on 2024-05-31 and added to Hugging Face Transformers on 2024-06-26 and contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).*

# LLaVa-NeXT-Video

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

[LLaVA-NeXT: A Strong Zero-shot Video Understanding Model](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/) introduces key technical advances that enable strong zero-shot video understanding despite being primarily trained on images. Its AnyRes technique converts high-resolution images—or video frames—into grids of sub-images that a pretrained Vision Transformer can process as a unified token sequence, allowing image-trained models to generalize naturally to videos. To handle longer sequences, LLaVA-NeXT applies length generalization via a linear scaling modification to rotary position embeddings, effectively doubling the model’s token capacity from 4096 to 8192, supporting up to 56 frames. Finally, Direct Preference Optimization (DPO) with AI-generated feedback fine-tunes the model (LLaVA-NeXT-Video-DPO) for improved response alignment and accuracy, while SGLang offers a 5× inference speedup for scalable video tasks.

<hfoptions id="usage">
<hfoption id="LlavaNextVideoForConditionalGeneration">

```py
import torch
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", dtype="auto")
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Why is this video funny?"},
            {"type": "video", "path": video_path},
            ],
    },
]

inputs = processor.apply_chat_template(conversation, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=60)
print(processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True))
```

</hfoption>
</hfoptions>

## Usage tips

- Use `padding_side="left"` for batched generation to get more accurate results. Set `processor.tokenizer.padding_side = "left"` before generating.
- LLaVA-NeXT uses different numbers of patches for images and pads inputs inside the modeling code, aside from padding done during processing. The default setting is left-padding if the model is in `eval()` mode, otherwise right-padding.
- LLaVA models after release v4.46 raise warnings about adding `processor.patch_size = {{patch_size}}`, `processor.num_additional_image_tokens = {{num_additional_image_tokens}}`, and `processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. Add these attributes to the processor if you own the model checkpoint, or open a PR if you don't.
- Adding these attributes means LLaVA infers the number of image tokens required per image and expands text with `<image>` placeholders. Usually around 500 tokens per image, so ensure text isn't truncated to avoid embedding merge failures.
- Get attributes from `model.config.vision_config.patch_size` or `model.config.vision_feature_select_strategy`. Set `num_additional_image_tokens` to 1 if the vision backbone adds a CLS token or 0 if nothing extra is added to vision patches.
- Each checkpoint trains with a specific prompt format depending on the underlying large language model backbone. Use the processor's [`apply_chat_template`] method to ensure correct formatting.

## LlavaNextVideoConfig

[[autodoc]] LlavaNextVideoConfig

## LlavaNextVideoProcessor

[[autodoc]] LlavaNextVideoProcessor

## LlavaNextVideoForConditionalGeneration

[[autodoc]] LlavaNextVideoForConditionalGeneration
    - forward

## LlavaNextVideoModel

[[autodoc]] LlavaNextVideoModel
    - forward

## LlavaNextVideoVideoProcessor

[[autodoc]] LlavaNextVideoVideoProcessor
