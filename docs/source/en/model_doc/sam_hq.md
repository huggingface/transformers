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

*This model was released on 2023-06-02 and added to Hugging Face Transformers on 2025-04-28 and contributed by [sushmanth](https://huggingface.co/sushmanth).*

# SAM-HQ

[SAM-HQ](https://huggingface.co/papers/2306.01567) enhances the original SAM model by producing higher quality segmentation masks, especially for intricate objects. It introduces a learnable High-Quality Output Token and global-local feature fusion to improve mask details. Trained on a dataset of 44K high-quality masks, SAM-HQ adds only 0.5% additional parameters while maintaining SAM's promptable design, efficiency, and zero-shot generalizability. The model predicts binary masks with accurate boundaries and handles thin structures better, and it supports multiple points for a single mask prediction.

Tips:

- SAM-HQ produces higher quality masks than the original SAM model, particularly for objects with intricate structures and fine details
- The model predicts binary masks with more accurate boundaries and better handling of thin structures
- Like SAM, the model performs better with input 2D points and/or input bounding boxes
- You can prompt multiple points for the same image and predict a single high-quality mask
- The model maintains SAM's zero-shot generalization capabilities
- SAM-HQ only adds ~0.5% additional parameters compared to SAM
- Fine-tuning the model is not supported yet

This model was contributed by [sushmanth](https://huggingface.co/sushmanth).
The original code can be found [here](https://github.com/SysCV/SAM-HQ).

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-segmentation", model="syscv-community/sam-hq-vit-base", dtype="auto")
pipeline("path/to/image.png")
```

Below is an example on how to run mask generation given an image and a 2D point:

```python
import torch
from PIL import Image
import requests
from transformers import SamHQModel, SamHQProcessor
from accelerate import Accelerator

device = Accelerator().device
model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base").to(device)
processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

You can also process your own masks alongside the input images in the processor to be passed to the model:

```python
import torch
from PIL import Image
import requests
from transformers import SamHQModel, SamHQProcessor
from accelerate import Accelerator

device = Accelerator().device
model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base").to(device)
processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
mask_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
segmentation_map = Image.open(requests.get(mask_url, stream=True).raw).convert("1")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, segmentation_maps=segmentation_map, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with SAM-HQ:

- Demo notebook for using the model (coming soon)
- Paper implementation and code: [SAM-HQ GitHub Repository](https://github.com/SysCV/SAM-HQ)

## SamHQConfig

[[autodoc]] SamHQConfig

## SamHQVisionConfig

[[autodoc]] SamHQVisionConfig

## SamHQMaskDecoderConfig

[[autodoc]] SamHQMaskDecoderConfig

## SamHQPromptEncoderConfig

[[autodoc]] SamHQPromptEncoderConfig

## SamHQProcessor

[[autodoc]] SamHQProcessor

## SamHQVisionModel

[[autodoc]] SamHQVisionModel

## SamHQModel

[[autodoc]] SamHQModel
    - forward

