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

*This model was released on 2023-04-05 and added to Hugging Face Transformers on 2023-04-19.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# SAM
[SAM (Segment Anything Model)](https://huggingface.co/papers/2304.02643) is a promptable segmentation model designed for interactive and flexible object segmentation. The model accepts various input prompts - points, bounding boxes, or masks - and generates high-quality segmentation masks in response. Its architecture combines a heavyweight image encoder with a lightweight mask decoder, enabling real-time interactive performance. When prompts are ambiguous, SAM can return multiple valid masks (for example, when clicking on a person, it might return masks for the whole person, their clothing, or specific body parts), allowing users to select the most appropriate one.

You can find all the original SAM checkpoints under the [Facebook](https://huggingface.co/facebook/models?search=sam-vit) organization.

> [!TIP]
> This model was contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ).
>
> Click on the SAM models in the right sidebar for more examples of how to apply SAM to different segmentation tasks.

The example below demonstrates how to segment objects with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline
import matplotlib.pyplot as plt
from PIL import Image
import requests
import numpy as np

generator =  pipeline(
    task="mask-generation", 
    model="facebook/sam-vit-base", 
    device = 0, 
    points_per_batch = 64)

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"

# Segment with points
outputs = generator(img_url, points_per_batch = 64)

# Show results
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

def show_mask(mask, ax):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

plt.imshow(np.array(raw_image))
ax = plt.gca()
for mask in outputs["masks"]:
    show_mask(mask, ax=ax)
plt.axis("off")
plt.show()
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoModel, AutoProcessor
import matplotlib.pyplot as plt
from PIL import Image
import requests
import numpy as np

model = AutoModel.from_pretrained("facebook/sam-vit-base")
processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

input_points = [[[400, 650]]]  # 2D location of rear window on the vehicle

inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")
outputs = model(**inputs)

masks = processor.post_process_masks(
    outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
)

# Choose the most probable mask
iou_scores = outputs.iou_scores
best_idx = iou_scores[0].argmax().item()
best_mask = masks[0][0][best_idx].cpu().numpy()

# Show results
plt.imshow(np.array(raw_image))
color = np.array([0, 1.0, 0, 0.6])
overlay = best_mask[..., None] * color
plt.imshow(overlay)
plt.axis("off")
plt.show()
```

</hfoption>
</hfoptions>


## Notes

- SAM predicts binary masks indicating the presence or absence of an object given an image.
- The model produces better results when provided with 2D points and/or bounding boxes as prompts.
- You can prompt multiple points for the same image and predict a single mask.
- Fine-tuning the model is not supported yet.
- According to the paper, the model supports textual input, but this capability was not released to the public, as it can be seen in [the official repository](https://github.com/facebookresearch/segment-anything/issues/4#issuecomment-1497626844).
- You can also process your own masks alongside the input images in the processor to be passed to the model.
    ```py
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    import requests
    from transformers import SamModel, SamProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    mask_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    segmentation_map = Image.open(requests.get(mask_url, stream=True).raw).convert("1")
    input_points = [[[450, 600]]]  # 2D location of a window in the image

    inputs = processor(raw_image, input_points=input_points, segmentation_maps=segmentation_map, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores

    # Choose the most probable mask
    iou_scores = outputs.iou_scores
    best_idx = iou_scores[0].argmax().item()
    best_mask = masks[0][0][best_idx].cpu().numpy()

    # Show results
    plt.imshow(np.array(raw_image))
    color = np.array([0, 1.0, 0, 0.6])
    overlay = best_mask[..., None] * color
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()
    ```

## Resources

- [Demo notebook](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb) for using the model
- [Demo notebook](https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb) for using the automatic mask generation pipeline
- [Demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Run_inference_with_MedSAM_using_HuggingFace_Transformers.ipynb) for inference with MedSAM, a fine-tuned version of SAM on the medical domain
- [Demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb) for fine-tuning the model on custom data
- [0.1% Data Makes Segment Anything Slim](https://huggingface.co/papers/2312.05284) - SlimSAM, a pruned version that reduces model size while maintaining performance
- [Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks](https://huggingface.co/papers/2401.14159) - Combining Grounding DINO with SAM for text-based mask generation
- [Demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb) for Grounded SAM

## SamConfig

[[autodoc]] SamConfig

## SamVisionConfig

[[autodoc]] SamVisionConfig

## SamMaskDecoderConfig

[[autodoc]] SamMaskDecoderConfig

## SamPromptEncoderConfig

[[autodoc]] SamPromptEncoderConfig

## SamProcessor

[[autodoc]] SamProcessor

## SamImageProcessor

[[autodoc]] SamImageProcessor

## SamImageProcessorFast

[[autodoc]] SamImageProcessorFast

## SamVisionModel

[[autodoc]] SamVisionModel
    - forward

## SamModel

[[autodoc]] SamModel
    - forward
