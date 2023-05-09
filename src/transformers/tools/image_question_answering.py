from ..models.auto import AutoModelForVisualQuestionAnswering, AutoProcessor
from ..utils import is_vision_available
from .base import PipelineTool


if is_vision_available():
    from PIL import Image


class ImageQuestionAnsweringTool(PipelineTool):
    default_checkpoint = "dandelin/vilt-b32-finetuned-vqa"
    description = (
        "This is a tool that answers a question about an image. It takes an input named `image` which should be the "
        "image containing the information, as well as a `question` which should be the question in English. It "
        "returns a text that is the answer to the question."
    )
    name = "image_qa"
    pre_processor_class = AutoProcessor
    model_class = AutoModelForVisualQuestionAnswering

    inputs = ["image", "text"]
    outputs = ["text"]

    def __init__(self, *args, **kwargs):
        if not is_vision_available():
            raise ValueError("Pillow must be installed to use the ImageQuestionAnsweringTool.")

        super().__init__(*args, **kwargs)

    def encode(self, image: "Image", question: str):
        return self.pre_processor(image, question, return_tensors="pt")

    def forward(self, inputs):
        with torch.no_grad():
            return self.model(**inputs).logits

    def decode(self, outputs):
        idx = outputs.argmax(-1).item()
        return self.model.config.id2label[idx]
