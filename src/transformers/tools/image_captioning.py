from ..models.auto import AutoModelForVision2Seq, AutoProcessor
from ..utils import is_vision_available
from .base import PipelineTool


try:
    from PIL.Image import Image
except ImportError:
    pass


class ImageCaptioningTool(PipelineTool):
    pre_processor_class = AutoProcessor
    model_class = AutoModelForVision2Seq

    description = (
        "This is a tool that generates a description of an image. It takes an input named `image` which should be the "
        "image to caption, and returns a text that contains the description in English."
    )
    default_checkpoint = "Salesforce/blip-image-captioning-base"

    def __init__(self, *args, **kwargs):
        if not is_vision_available():
            raise ValueError("Pillow must be installed to use the ImageCaptioningTool.")

        super().__init__(*args, **kwargs)

    def encode(self, image: "Image"):
        return self.pre_processor(images=image, return_tensors="pt")

    def forward(self, inputs):
        return self.model.generate(**inputs)

    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
