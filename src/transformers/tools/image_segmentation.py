from typing import List

from transformers import AutoProcessor, CLIPSegForImageSegmentation, is_vision_available

from .base import PipelineTool


if is_vision_available():
    from PIL import Image


class ImageSegmentationTool(PipelineTool):
    pre_processor_class = AutoProcessor
    model_class = CLIPSegForImageSegmentation

    description = (
        "This is a tool that generates a description of an image. It takes an input named `image` which should be the "
        "image to caption, and returns a text that contains the description in English."
    )
    default_checkpoint = "CIDAS/clipseg-rd64-refined"

    def __init__(self, *args, **kwargs):
        if not is_vision_available():
            raise ImportError("Pillow should be installed in order to use the StableDiffusionTool.")

        super().__init__(*args, **kwargs)

    def encode(self, texts: List[str], image: "Image"):
        return self.pre_processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

    def decode(self, outputs):
        logits_array = outputs.logits
        segmentation_maps = []

        for logits in logits_array:
            array = logits.cpu().detach().numpy()
            array[array < 0] = 0
            array[array >= 0] = 1
            segmentation_maps.append(array)

        return segmentation_maps
