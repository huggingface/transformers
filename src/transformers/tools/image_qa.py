from ..models.auto import AutoModelForVisualQuestionAnswering, AutoProcessor
from ..utils import is_vision_available
from .base import PipelineTool


if is_vision_available():
    from PIL import Image


IMAGE_QUESTION_ANSWERING_DESCRIPTION = (
    "This is a tool that answers a question about an image. It takes an input named `image` which should be the "
    "image containing the information, as well as a `query` that is the question about the image. It returns a text "
    "that contains the answer to the question."
)


class ImageQuestionAnsweringTool(PipelineTool):
    default_checkpoint = "dandelin/vilt-b32-finetuned-vqa"
    description = IMAGE_QUESTION_ANSWERING_DESCRIPTION
    pre_processor_class = AutoProcessor
    model_class = AutoModelForVisualQuestionAnswering

    inputs = ["image", "text"]
    outputs = ["text"]

    def __init__(self, *args, **kwargs):
        if not is_vision_available():
            raise ValueError("Pillow must be installed to use the ImageQuestionAnsweringTool.")

        super().__init__(*args, **kwargs)

    def encode(self, image: "Image", query: str):
        return self.pre_processor(image, query, return_tensors="pt")

    def forward(self, inputs):
        return self.model(**inputs)

    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

