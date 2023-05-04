from typing import List

from transformers import AutoProcessor, CLIPSegForImageSegmentation, is_vision_available, ViTImageProcessor

from .base import PipelineTool, send_to_device
import torch
import numpy as np

if is_vision_available():
    from PIL import Image


IMAGE_SEGMENTATION_DESCRIPTION = (
    "This is a tool that creates a segmentation mask using an image and a prompt. It takes an original image, as well as a prompt which "
    "is a textual description of what should be identified in the mask. The tool returns the mask as an image."
)


class ImageSegmentationTool(PipelineTool):
    description = IMAGE_SEGMENTATION_DESCRIPTION
    default_checkpoint = "CIDAS/clipseg-rd64-refined"
    pre_processor_class = AutoProcessor
    model_class = CLIPSegForImageSegmentation

    def __init__(self, *args, **kwargs):
        if not is_vision_available():
            raise ImportError("Pillow should be installed in order to use the ImageSegmentationTool.")

        super().__init__(*args, **kwargs)

    def encode(self, text: str, image: "Image"):
        self.pre_processor.image_processor.size = {'width': 512, 'height': 512}
        return self.pre_processor(text=[text], images=[image], padding=True, return_tensors="pt")

    def forward(self, inputs):
        logits = self.model(**inputs).logits
        return logits

    def decode(self, outputs):
        array = outputs.cpu().detach().numpy()
        array[array <= 0] = 0
        array[array > 0] = 1
        return Image.fromarray((np.dstack([array, array, array]) * 255).astype(np.uint8))