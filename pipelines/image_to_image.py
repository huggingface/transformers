# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Union

import numpy as np

from ..utils import (
    add_end_docstrings,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
from .base import Pipeline, build_pipeline_init_args


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES

logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class ImageToImagePipeline(Pipeline):
    """
    Image to Image pipeline using any `AutoModelForImageToImage`. This pipeline generates an image based on a previous
    image input.

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests

    >>> from transformers import pipeline

    >>> upscaler = pipeline("image-to-image", model="caidas/swin2SR-classical-sr-x2-64")
    >>> img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    >>> img = img.resize((64, 64))
    >>> upscaled_img = upscaler(img)
    >>> img.size
    (64, 64)

    >>> upscaled_img.size
    (144, 144)
    ```

    This image to image pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-to-image"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=image-to-image).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        postprocess_params = {}
        forward_params = {}

        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]
        if "head_mask" in kwargs:
            forward_params["head_mask"] = kwargs["head_mask"]

        return preprocess_params, forward_params, postprocess_params

    def __call__(
        self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs
    ) -> Union["Image.Image", List["Image.Image"]]:
        """
        Transform the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is used and
                the call may block forever.

        Return:
            An image (Image.Image) or a list of images (List["Image.Image"]) containing result(s). If the input is a
            single image, the return will be also a single image, if the input is a list of several images, it will
            return a list of transformed images.
        """
        return super().__call__(images, **kwargs)

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout=timeout)
        inputs = self.image_processor(images=[image], return_tensors="pt")
        return inputs

    def postprocess(self, model_outputs):
        images = []
        if "reconstruction" in model_outputs.keys():
            outputs = model_outputs.reconstruction
        for output in outputs:
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.moveaxis(output, source=0, destination=-1)
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
            images.append(Image.fromarray(output))

        return images if len(images) > 1 else images[0]
