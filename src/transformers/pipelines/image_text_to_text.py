# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_processor=True))
class ImageTextToTextPipeline(Pipeline):
    """
    Image-text-to-text pipeline using an `AutoModelForImageTextToText`. This pipeline generates text given an image and text.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> pipe = pipeline(task="image-text-to-text", model="Salesforce/blip-image-captioning-base")
    >>> pipe("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", text="A photo of")
    [{'generated_text': 'a photo of two birds'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image-text to text pipeline can currently be loaded from pipeline() using the following task identifier:
    "image-text-to-text".

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-text-to-text).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)

    def _sanitize_parameters(self, max_new_tokens=None, generate_kwargs=None, text=None, timeout=None):
        forward_kwargs = {}
        preprocess_params = {}

        if text is not None:
            preprocess_params["text"] = text
        if timeout is not None:
            preprocess_params["timeout"] = timeout

        if generate_kwargs is not None:
            forward_kwargs["generate_kwargs"] = generate_kwargs
        if max_new_tokens is not None:
            if "generate_kwargs" not in forward_kwargs:
                forward_kwargs["generate_kwargs"] = {}
            if "max_new_tokens" in forward_kwargs["generate_kwargs"]:
                raise ValueError(
                    "'max_new_tokens' is defined twice, once in 'generate_kwargs' and once as a direct parameter,"
                    " please use only one"
                )
            forward_kwargs["generate_kwargs"]["max_new_tokens"] = max_new_tokens

        return preprocess_params, forward_kwargs, {}

    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]] = None, **kwargs):
        """
        Generate a text given text and the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a HTTP(s) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images.

            text (`str`):
                The text to be used as a prompt for the generation.

            max_new_tokens (`int`, *optional*):
                The amount of maximum tokens to generate. By default it will use `generate` default.

            generate_kwargs (`Dict`, *optional*):
                Pass it to send all of these arguments directly to `generate` allowing full control of this function.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following key:

            - **generated_text** (`str`) -- The generated text.
        """
        return super().__call__(images, **kwargs)

    def preprocess(self, image=None, text=None, timeout=None):
        if image is not None:
            image = load_image(image, timeout=timeout)

        model_type = self.model.config.model_type

        kwargs = {}

        if model_type == "pix2struct":
            kwargs = {"add_special_tokens": False}

        if model_type == "idefics":
            model_inputs = self.processor(text, return_tensors=self.framework, **kwargs)
        else:
            model_inputs = self.processor(images=image, text=text, return_tensors=self.framework, **kwargs)

        if model_type == "git":
            # remove EOS token from input_ids and attention_mask
            model_inputs["input_ids"] = model_inputs["input_ids"][:, :-1]
            model_inputs["attention_mask"] = model_inputs["attention_mask"][:, :-1]

        if model_type == "vision-encoder-decoder" and self.processor.__class__.__name__ == "DonutProcessor":
            model_inputs["decoder_input_ids"] = self.processor.tokenizer(
                text,
                add_special_tokens=False,
                return_tensors=self.framework,
            ).input_ids

        return model_inputs

    def _forward(self, model_inputs, generate_kwargs=None):
        if generate_kwargs is None:
            generate_kwargs = {}

        model_outputs = self.model.generate(**model_inputs, **generate_kwargs)
        return model_outputs

    def postprocess(self, model_outputs):
        records = []
        generated_texts = self.processor.batch_decode(
            model_outputs,
            skip_special_tokens=True,
        )

        records = [{"generated_text": text} for text in generated_texts]

        return records
