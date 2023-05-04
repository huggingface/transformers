
from transformers.tools.base import Tool
from transformers.utils import is_accelerate_available, is_diffusers_available


if is_accelerate_available():
    from accelerate.state import PartialState

if is_diffusers_available():
    from diffusers import StableDiffusionInpaintPipeline


IMAGE_INPAINTING_DESCRIPTION = (
    "This is a tool that replaces something in an image. It takes three inputs: the original image, a mask, and the prompt detailing what will end up in the image. "
    "The mask is a black-and-white image of the section to replace in the image. It can be retrieved using an image segmentation tool. "
    "The input types are therefore: an image, an image, and a text. It returns an image."
)


class ImageInpaintingTool(Tool):
    default_checkpoint = "runwayml/stable-diffusion-inpainting"
    description = IMAGE_INPAINTING_DESCRIPTION

    def __init__(self, device=None, **hub_kwargs) -> None:
        if not is_accelerate_available():
            raise ImportError("Accelerate should be installed in order to use tools.")
        if not is_diffusers_available():
            raise ImportError("Diffusers should be installed in order to use the StableDiffusionTool.")

        super().__init__()

        self.device = device
        self.pipeline = None
        self.hub_kwargs = hub_kwargs

    def setup(self):
        if self.device is None:
            self.device = PartialState().default_device

        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(self.default_checkpoint)
        self.pipeline.to(self.device)

        self.is_initialized = True

    def __call__(self, image, mask, prompt):
        if not self.is_initialized:
            self.setup()

        return self.pipeline(prompt, image=image, mask_image=mask).images[0]
