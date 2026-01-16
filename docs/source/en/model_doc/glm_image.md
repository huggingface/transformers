<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on 2026-01-10 and added to Hugging Face Transformers on 2026-01-13.*

# GlmImage

## Overview

GLM-Image is an image generation model adopts a hybrid autoregressive + diffusion decoder architecture, effectively pushing the upper bound of visual fidelity and fine-grained details. In general image generation quality, it aligns with industry-standard LDM-based approaches, while demonstrating significant advantages in knowledge-intensive image generation scenarios.

Model architecture: a hybrid autoregressive + diffusion decoder design、

+ Autoregressive generator: a 9B-parameter model initialized from [GLM-4-9B-0414](https://huggingface.co/zai-org/GLM-4-9B-0414), with an expanded vocabulary to incorporate visual tokens. The model first generates a compact encoding of approximately 256 tokens, then expands to 1K–4K tokens, corresponding to 1K–2K high-resolution image outputs.
+ Diffusion Decoder: a 7B-parameter decoder based on a single-stream DiT architecture for latent-space image decoding. It is equipped with a Glyph Encoder text module, significantly improving accurate text rendering within images.

Post-training with decoupled reinforcement learning: the model introduces a fine-grained, modular feedback strategy using the GRPO algorithm, substantially enhancing both semantic understanding and visual detail quality.

+ Autoregressive module: provides low-frequency feedback signals focused on aesthetics and semantic alignment, improving instruction following and artistic expressiveness.
+ Decoder module: delivers high-frequency feedback targeting detail fidelity and text accuracy, resulting in highly realistic textures, lighting, and color reproduction, as well as more precise text rendering.

GLM-Image supports both text-to-image and image-to-image generation within a single model

+ Text-to-image: generates high-detail images from textual descriptions, with particularly strong performance in information-dense scenarios.
+ Image-to-image: supports a wide range of tasks, including image editing, style transfer, multi-subject consistency, and identity-preserving generation for people and objects.

+ `GlmImageForConditionalGeneration` is the AR part of GLM-Image model, and for full image generation pipeline, please refer to [here](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/glm_image).

This model was contributed by [Raushan Turganbay](https://huggingface.co/RaushanTurganbay) and [Yuxuan Zhang](https://huggingface.co/ZHANGYUXUAN-zR).

## Usage examples

Using GLM-Image with image input to generate vision token for DIT using.

```python
from transformers import GlmImageForConditionalGeneration, AutoProcessor
import torch

model = GlmImageForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path="zai-org/GLM-Image/vision_language_encoder",
    dtype=torch.bfloat16,
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path="zai-org/GLM-Image/processor",
    use_fast=True
)

# Case1 T2I
prompt = "现代美食杂志风格的甜点制作教程图，主题为覆盆子慕斯蛋糕。整体布局干净明亮，分为四个主要区域：顶部左侧是黑色粗体标题“覆盆子慕斯蛋糕制作指南”，右侧搭配光线柔和的成品蛋糕特写照片，蛋糕呈淡粉色，表面点缀新鲜覆盆子与薄荷叶；左下方为配料清单区域，标题“配料”使用简洁字体，下方列有“面粉 150g”“鸡蛋 3个”“细砂糖 120g”“覆盆子果泥 200g”“明胶片 10g”“淡奶油 300ml”“新鲜覆盆子”等配料，每种配料旁配有简约线图标（如面粉袋、鸡蛋、糖罐等）；右下方是四个等大的步骤方框，每个方框内含高清微距实拍图及对应操作说明，从上到下依次为：步骤1展示打蛋器打发白色泡沫（对应说明“打发蛋白至干性发泡”），步骤2展示红白相间的混合物被刮刀翻拌（对应说明“轻柔翻拌果泥与面糊”），步骤3展示粉色液体被倒入圆形模具（对应说明“倒入模具并冷藏4小时”），步骤4展示成品蛋糕表面装饰覆盆子与薄荷叶（对应说明“用覆盆子和薄荷装饰”）；底部边缘设浅棕色信息条，左侧图标分别代表“准备时间：30分钟”“烹饪时间：20分钟”“份量：8人份”。整体色调以奶油白、淡粉色为主，背景带轻微纸质纹理，图文排版紧凑有序，信息层级分明。"
target_h, target_w = 1152, 768
use_reference_images = False
reference_image_paths = None

# ## Case2
# prompt = "Replace the background of the snow forest with an underground station featuring an automatic escalator."
# cond_0 = "cond.jpg"
# target_h, target_w = 1152, 768
# use_reference_images = True
# reference_image_paths = [cond_0]

## Case3
# prompt = "Make the man in the first figure and the child from the second image bow at the same time in a respectful KTV."
# cond_0 = "cond_0.jpg"
# cond_1 = "cond_1.jpg"
# target_h, target_w = 1152, 768
# use_reference_images = True
# reference_image_paths = [cond_0, cond_1]


def build_messages(prompt, use_reference_images, reference_image_paths):
    content = []
    if use_reference_images:
        for img_path in reference_image_paths:
            content.append({"type": "image", "url": img_path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def compute_generation_params(image_grid_thw, use_reference_images):
    grid_sizes = []
    for i in range(image_grid_thw.shape[0]):
        t, h, w = image_grid_thw[i].tolist()
        grid_sizes.append(int(h * w))

    target_output_length = grid_sizes[0]

    if use_reference_images:
        max_new_tokens = grid_sizes[-1] + 1
        output_start_offset = 0
        output_length = grid_sizes[-1]
    else:
        total_tokens = sum(grid_sizes)
        max_new_tokens = total_tokens + 1
        output_start_offset = sum(grid_sizes[1:])
        output_length = target_output_length

    return max_new_tokens, output_start_offset, output_length


messages = build_messages(prompt, use_reference_images, reference_image_paths if use_reference_images else None)

inputs = processor.apply_chat_template(
    messages,
    target_h=target_h,
    target_w=target_w,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

image_grid_thw = inputs.get('image_grid_thw')
print(f"image_grid_thw: {image_grid_thw}")

max_new_tokens, output_start_offset, output_length = compute_generation_params(
    image_grid_thw, use_reference_images
)

print(f"use_reference_images: {use_reference_images}")
print(f"max_new_tokens: {max_new_tokens}")
print(f"output_start_offset: {output_start_offset}")
print(f"output_length: {output_length}")

outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=True
)

input_length = inputs["input_ids"].shape[-1]
output_tokens = outputs[0][input_length:][output_start_offset:output_start_offset + output_length]
print(f"Input length: {input_length}")
print(f"Total generated tokens: {outputs[0].shape[-1] - input_length}")
print(f"Extracted output tokens shape: {output_tokens.shape}")
print(f"Output tokens: {output_tokens}")
```

## GlmImageConfig

[[autodoc]] GlmImageConfig

## GlmImageVisionConfig

[[autodoc]] GlmImageVisionConfig

## GlmImageTextConfig

[[autodoc]] GlmImageTextConfig

## GlmImageVQVAEConfig

[[autodoc]] GlmImageVQVAEConfig

## GlmImageImageProcessor

[[autodoc]] GlmImageImageProcessor
    - preprocess

## GlmImageImageProcessorFast

[[autodoc]] GlmImageImageProcessorFast
    - preprocess

## GlmImageProcessor

[[autodoc]] GlmImageProcessor

## GlmImageVisionModel

[[autodoc]] GlmImageVisionModel
    - forward

## GlmImageTextModel

[[autodoc]] GlmImageTextModel
    - forward

## GlmImageVQVAE

[[autodoc]] GlmImageVQVAE
    - forward

## GlmImageModel

[[autodoc]] GlmImageModel
    - forward

## GlmImageForConditionalGeneration

[[autodoc]] GlmImageForConditionalGeneration
    - forward
