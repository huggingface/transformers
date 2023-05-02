import torch

from transformers.tools.base import Tool
from transformers.utils import is_accelerate_available, is_diffusers_available


if is_accelerate_available():
    from accelerate.state import PartialState

if is_diffusers_available():
    from diffusers import DiffusionPipeline


TEXT_TO_VIDEO_DESCRIPTION = (
    "This is a tool that creates a video according to a text description. It takes an input named `prompt` which "
    "contains the image description, as well as an optional input `seconds` which will be the duration of the video. "
    "The default is of two seconds. The tool outputs a video object."
)


class TextToVideoTool(Tool):
    default_checkpoint = "damo-vilab/text-to-video-ms-1.7b"
    description = TEXT_TO_VIDEO_DESCRIPTION

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

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.default_checkpoint, torch_dtype=torch.float16, variant="fp16"
        )
        self.pipeline.to(self.device)

        self.is_initialized = True

    def __call__(self, prompt, seconds=2):
        if not self.is_initialized:
            self.setup()

        return self.pipeline(prompt, num_frames=8 * seconds).frames
