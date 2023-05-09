import numpy as np

from transformers import AutoProcessor, CLIPSegForImageSegmentation, is_vision_available

from .base import PipelineTool


if is_vision_available():
    from PIL import Image


class ImageSegmentationTool(PipelineTool):
    description = (
        "This is a tool that creates a segmentation mask identifiying elements inside an image according to a prompt. "
        "It takes two arguments named `image` which should be the original image, and `prompt` which should be a text "
        "describing the elements what should be identified in the segmentation mask. The tool returns the mask as a "
        "black-and-white image."
    )
    default_checkpoint = "CIDAS/clipseg-rd64-refined"
    name = "image_segmenter"
    pre_processor_class = AutoProcessor
    model_class = CLIPSegForImageSegmentation

    inputs = ["image", "text"]
    outputs = ["image"]

    def __init__(self, *args, **kwargs):
        if not is_vision_available():
            raise ImportError("Pillow should be installed in order to use the ImageSegmentationTool.")

        super().__init__(*args, **kwargs)

    def encode(self, image: "Image", prompt: str):
        self.pre_processor.image_processor.size = {"width": image.size[0], "height": image.size[1]}
        return self.pre_processor(text=[prompt], images=[image], padding=True, return_tensors="pt")

    def forward(self, inputs):
        logits = self.model(**inputs).logits
        return logits

    def decode(self, outputs):
        array = outputs.cpu().detach().numpy()
        array[array <= 0] = 0
        array[array > 0] = 1
        return Image.fromarray((np.dstack([array, array, array]) * 255).astype(np.uint8))
