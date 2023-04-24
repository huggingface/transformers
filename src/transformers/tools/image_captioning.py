from transformers import AutoProcessor, BlipForConditionalGeneration, is_vision_available

from .base import PipelineTool


try:
    from PIL.Image import Image
except ImportError:
    pass


class ImageCaptioningTool(PipelineTool):
    pre_processor_class = AutoProcessor
    model_class = BlipForConditionalGeneration

    description = """
    image captioning tool, which can analyze images and caption them in English according to their content. It takes an
    image as input, and returns a English text caption as an output.
    """
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
