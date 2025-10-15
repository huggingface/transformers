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
*This model was released on 2023-04-06 and added to Hugging Face Transformers on 2024-02-26 and contributed by [EduardoPacheco](https://huggingface.co/EduardoPacheco).*

# SegGPT

[SegGPT](https://huggingface.co/papers/2304.03284) is a generalist model designed for segmenting various elements in context using a decoder-only Transformer. It generates segmentation masks from input images, prompt images, and their corresponding prompt masks. SegGPT achieves high accuracy with 56.1 mIoU on COCO-20 and 85.6 mIoU on FSS-1000. The model is trained as an in-context coloring problem with random color mapping, enabling it to handle diverse segmentation tasks such as object instance, stuff, part, contour, and text segmentation. It is evaluated on tasks including few-shot semantic segmentation, video object segmentation, semantic segmentation, and panoptic segmentation, demonstrating strong capabilities in both in-domain and out-of-domain scenarios.

<hfoptions id="usage">
<hfoption id="SegGptForImageSegmentation">

```py
import torch
import requests
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegGptForImageSegmentation, SegGptImageProcessor

image_input_url = "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_2.jpg"
image_prompt_url = "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1.jpg"
mask_prompt_url = "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1_target.png"

image_input = Image.open(requests.get(image_input_url, stream=True).raw)
image_prompt = Image.open(requests.get(image_prompt_url, stream=True).raw)
mask_prompt = Image.open(requests.get(mask_prompt_url, stream=True).raw).convert("L")

processor = SegGptImageProcessor.from_pretrained("BAAI/seggpt-vit-large")
model = SegGptForImageSegmentation.from_pretrained("BAAI/seggpt-vit-large", dtype="auto")

inputs = processor(images=image_input, prompt_images=image_prompt, prompt_masks=mask_prompt, return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs)

target_sizes = [(image_input.height, image_input.width)]
outputs = processor.post_process_semantic_segmentation(
    outputs,
    target_sizes=target_sizes,
)

plt.imshow(outputs[0])
plt.axis("off")
plt.show()
```

</hfoption>
</hfoptions>

## Usage tips

- Use [`SegGptImageProcessor`] to prepare image input, prompt, and mask for the model.
- Use segmentation maps or RGB images as prompt masks. If using RGB images, set `do_convert_rgb=False` in the preprocess method.
- Pass `num_labels` when using segmentation maps (excluding background) during preprocessing and postprocessing with [`SegGptImageProcessor`] for your use case.
- When doing inference with [`SegGptForImageSegmentation`], if your batch size is greater than 1, use feature ensemble across images by passing `feature_ensemble=True` in the forward method.

## SegGptConfig

[[autodoc]] SegGptConfig

## SegGptImageProcessor

[[autodoc]] SegGptImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## SegGptModel

[[autodoc]] SegGptModel
    - forward

## SegGptForImageSegmentation

[[autodoc]] SegGptForImageSegmentation
    - forward

