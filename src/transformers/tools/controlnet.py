import cv2
import numpy as np
import torch
from accelerate.state import PartialState
from PIL import Image

from transformers.tools.base import Tool


try:
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    diffusers_available = True
except ImportError:
    diffusers_available = False


class ControlNetTool(Tool):
    default_stable_diffusion_checkpoint = "runwayml/stable-diffusion-v1-5"
    default_controlnet_checkpoint = "lllyasviel/sd-controlnet-canny"

    description = """This is a tool that creates an image according to a text description.
    It takes text as input, and it outputs an image.
    """

    def __init__(self, device=None, controlnet=None, stable_diffusion=None) -> None:
        if not diffusers_available:
            raise ImportError('Diffusers should be installed in order to use the ControlNetTool.')

        super().__init__()

        if controlnet is None:
            controlnet = self.default_controlnet_checkpoint
        self.controlnet_checkpoint = controlnet

        if stable_diffusion is None:
            stable_diffusion = self.default_stable_diffusion_checkpoint
        self.stable_diffusion_checkpoint = stable_diffusion

        self.device = device


    def setup(self):
        if self.device is None:
            self.device = PartialState().default_device

        self.controlnet = ControlNetModel.from_pretrained(self.controlnet_checkpoint, torch_dtype=torch.float16)
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.stable_diffusion_checkpoint, controlnet=self.controlnet, torch_dtype=torch.float16
        )
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.enable_model_cpu_offload()

        self.is_initialized = True

    def __call__(self, image, prompt):
        if not self.is_initialized:
            self.setup()

        initial_prompt = 'super-hero character, best quality, extremely detailed'
        prompt = initial_prompt + prompt

        low_threshold = 100
        high_threshold = 200

        image = np.array(image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        generator = torch.Generator(device="cpu").manual_seed(2)

        return self.pipeline(
            prompt,
            canny_image,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            num_inference_steps=20,
            generator=generator,
        ).images[0]
