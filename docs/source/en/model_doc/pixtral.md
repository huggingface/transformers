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

# Pixtral

## Overview

The Pixtral model was released by the Mistral AI team on [vLLM](https://github.com/vllm-project/vllm/pull/8377), where a version of the code can be found!

Tips:

- Pixtral is a multimodal model, taking images and text as input, and producing text as output.
- This model follows the [Llava](llava) family, meaning image embeddings are placed instead of the `[IMG]` token placeholders. The model uses [`PixtralVisionModel`] for its vision encoder, and [`MistralForCausalLM`] for its language decoder.
- The main contribution is the 2d ROPE (rotary postiion embeddings) on the images, and support for arbitrary image sizes (the images are not padded together nor are they resized).
- The format for one or mulitple prompts is the following:
```
"<s>[INST][IMG]\nWhat are the things I should be cautious about when I visit this place?[/INST]"
```
Then, the processor will replace each `[IMG]` token with  a number of `[IMG]` token that depends on the height and the width of the image. Each *row* of the image is separated by a `[IMG_BREAK]` token, and each image is separated by a  `[IMG_END]` token.

This model was contributed by [amyeroberts](https://huggingface.co/amyeroberts) and [ArthurZ](https://huggingface.co/ArthurZ). The original code can be found [here](https://github.com/vllm-project/vllm/pull/8377).

## Usage

Here is an example of how to run it:

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

model_id = "mistral-community/pixtral-12b"
model = LlavaForConditionalGeneration.from_pretrained(model_id).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

IMG_URLS = [
    "https://picsum.photos/id/237/400/300",
    "https://picsum.photos/id/231/200/300",
    "https://picsum.photos/id/27/500/500",
    "https://picsum.photos/id/17/150/600",
]
PROMPT = "<s>[INST]Describe the images.\n[IMG][IMG][IMG][IMG][/INST]"

inputs = processor(images=IMG_URLS, text=PROMPT, return_tensors="pt").to("cuda")
generate_ids = model.generate(**inputs, max_new_tokens=500)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

EXPECTED_GENERATION = """
Describe the images.
Sure, let's break down each image description:

1. **Image 1:**
   - **Description:** A black dog with a glossy coat is sitting on a wooden floor. The dog has a focused expression and is looking directly at the camera.
   - **Details:** The wooden floor has a rustic appearance with visible wood grain patterns. The dog's eyes are a striking color, possibly brown or amber, which contrasts with its black fur.

2. **Image 2:**
   - **Description:** A scenic view of a mountainous landscape with a winding road cutting through it. The road is surrounded by lush green vegetation and leads to a distant valley.
   - **Details:** The mountains are rugged with steep slopes, and the sky is clear, indicating good weather. The winding road adds a sense of depth and perspective to the image.

3. **Image 3:**
   - **Description:** A beach scene with waves crashing against the shore. There are several people in the water and on the beach, enjoying the waves and the sunset.
   - **Details:** The waves are powerful, creating a dynamic and lively atmosphere. The sky is painted with hues of orange and pink from the setting sun, adding a warm glow to the scene.

4. **Image 4:**
   - **Description:** A garden path leading to a large tree with a bench underneath it. The path is bordered by well-maintained grass and flowers.
   - **Details:** The path is made of small stones or gravel, and the tree provides a shaded area with the bench invitingly placed beneath it. The surrounding area is lush and green, suggesting a well-kept garden.

Each image captures a different scene, from a close-up of a dog to expansive natural landscapes, showcasing various elements of nature and human interaction with it.
"""

```
## PixtralVisionConfig

[[autodoc]] PixtralVisionConfig

## PixtralVisionModel

[[autodoc]] PixtralVisionModel
    - forward

## PixtralImageProcessor

[[autodoc]] PixtralImageProcessor
    - preprocess

## PixtralProcessor

[[autodoc]] PixtralProcessor
