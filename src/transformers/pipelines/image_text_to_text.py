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

from typing import Dict, List, Union

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
    from .pt_utils import KeyDataset

logger = logging.get_logger(__name__)

IMAGE_TOKEN = "<image>"


class Chat:
    """This class is intended to just be used internally in this pipeline and not exposed to users. We convert chats
    to this format because the rest of the pipeline code tends to assume that lists of messages are
    actually a batch of samples rather than messages in the same conversation."""

    def __init__(self, messages: Dict, images: Union[str, List[str], "Image.Image", List["Image.Image"]]):
        for message in messages:
            if not ("role" in message and "content" in message):
                raise ValueError("When passing chat dicts as input, each dict must have a 'role' and 'content' key.")
        if count_images_in_chat(messages) != len(images):
            raise ValueError("The number of images should be the same as the number of images in the chat.")

        self.messages = messages
        self.images = images


class ImageText:
    """This class is intended to just be used internally in this pipeline and not exposed to users. We used this class
    as the base pipeline does not support multiple inputs, so we need to convert multiple inputs to a single input."""

    def __init__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], text: Union[str, List[str]]):
        self.images = images
        self.text = text


def count_images_in_chat(chat):
    num_images = 0
    for message in chat:
        num_images += sum(1 for content in message["content"] if content.get("type") == "image")
    return num_images


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

    def _sanitize_parameters(
        self,
        max_new_tokens=None,
        generate_kwargs=None,
        text=None,
        truncation=None,
        padding=None,
        max_length=None,
        timeout=None,
    ):
        forward_kwargs = {}
        preprocess_params = {}
        post_process_params = {}

        if timeout is not None:
            preprocess_params["timeout"] = timeout

        if truncation is not None:
            preprocess_params["truncation"] = truncation

        if padding is not None:
            preprocess_params["padding"] = padding

        if max_length is not None:
            preprocess_params["max_length"] = max_length

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

        return preprocess_params, forward_kwargs, post_process_params

    def __call__(
        self,
        images: Union[str, List[str], List[List[str]], "Image.Image", List["Image.Image"], List[List["Image.Image"]]],
        text: Union[str, List[str], List[dict]],
        **kwargs,
    ):
        """
        Generate a text given text and the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a HTTP(s) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images.

            text (str, List[str], `List[Dict[str, Union[str, PIL.Image]]]`):
                The text to be used for generation. If a list of strings is passed, the length of the list should be the
                same as the number of images. Text can also follow the chat format: a list of dictionaries where each
                dictionary represents a message in a conversation. Each dictionary should have two keys: 'role' and
                'content'. 'role' should be one of 'user', 'system' or 'assistant'. 'content' should be a dictionary
                containing the text of the message and the type of the message. The type of the message can be either
                'text' or 'image'. If the type is 'image', no text is needed.

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following key:

            - **generated_text** (`str`) -- The generated text.
            - **input_text** (`str`) -- The input text.
        """
        batch_size = kwargs.get("batch_size", 1)

        if not isinstance(images, (list, tuple)):
            images = [images]

        if isinstance(text, (list, tuple, KeyDataset) if is_torch_available() else (list, tuple)) and isinstance(
            text[0], (list, tuple, dict)
        ):
            # We have one or more prompts in list-of-dicts format, so this is chat mode
            if isinstance(text[0], dict):
                return super().__call__(Chat(text, images), **kwargs)
            else:
                chats = [Chat(chat, image) for chat, image in zip(text, images)]  # 🐈 🐈 🐈
                return super().__call__(chats, **kwargs)

        if isinstance(text, str):
            text = [text] * len(images)
        if not isinstance(text[0], str):
            raise ValueError("The pipeline does not support nested lists of prompts.")

        # Check number of IMAGE_TOKEN token in each text
        num_images_in_text = [text_single.count(IMAGE_TOKEN) for text_single in text]
        if sum(num_images_in_text) > 0:
            if any(num > 1 for num in num_images_in_text) and batch_size > 1:
                raise ValueError(
                    "The pipeline does not support multiple images for a single prompt with batch_size > 1."
                )
            # Check if already nested images and consistency
            if isinstance(images[0], (list, tuple)):
                if len(images) != len(text):
                    raise ValueError("The number of nested image groups and prompts should be the same.")
                num_images_in_images = [len(image) for image in images]
                if num_images_in_text != num_images_in_images:
                    raise ValueError(
                        f"The number of images in each nested image group should be the same as the number of {IMAGE_TOKEN} tokens in the corresponding prompt."
                    )
            elif sum(num_images_in_text) != len(images):
                raise ValueError(
                    f"The total number of {IMAGE_TOKEN} tokens in the prompts should be the same as the number of images passed."
                )
            else:
                # Reorganize the images to match the prompts
                images_reorganized = []
                for num_images in num_images_in_text:
                    images_reorganized.append(images[:num_images])
                    images = images[num_images:]
                images = images_reorganized
        # After reorganizing, these should be the same
        if len(images) != len(text):
            raise ValueError("The number of images and text should be the same.")

        return super().__call__([ImageText(image, text_single) for image, text_single in zip(images, text)], **kwargs)

    def preprocess(self, inputs=None, truncation=None, padding=False, max_length=None, timeout=None):
        kwargs = {
            "legacy": False,
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
        }
        images = inputs.images

        if isinstance(inputs, Chat):
            # kwargs["chats"] = inputs.messages
            text = self.processor.apply_chat_template(
                inputs.messages,
                add_generation_prompt=True,
                return_tensors=self.framework,
                **kwargs,
            )
        else:
            text = inputs.text
        if not isinstance(images, (list, tuple)):
            images = load_image(images, timeout=timeout)
        else:
            images = [load_image(image, timeout=timeout) for image in images]

        try:
            model_inputs = self.processor(images=images, text=text, return_tensors=self.framework, **kwargs)
        except TypeError:
            kwargs.pop("legacy", None)
            model_inputs = self.processor(images=images, text=text, return_tensors=self.framework, **kwargs)

        model_inputs["text"] = text

        return model_inputs

    def _forward(self, model_inputs, generate_kwargs=None):
        if generate_kwargs is None:
            generate_kwargs = {}
        input_text = model_inputs.pop("text")
        input_ids = (
            model_inputs["input_ids"] if "input_ids" in model_inputs else model_inputs["decoder_input_ids"]
        )  # for decoder-only models
        model_outputs = self.model.generate(**model_inputs, **generate_kwargs)
        return {"outputs": model_outputs, "input_text": input_text, "input_ids": input_ids}

    def postprocess(self, model_outputs):
        input_text = model_outputs["input_text"]
        input_text = [input_text] if isinstance(input_text, str) else input_text
        outputs = model_outputs["outputs"]
        inputs_id = model_outputs["input_ids"]

        # Decode inputs and outputs the same way to remove input text from generated text if present
        generated_texts = self.processor.post_process_image_text_to_text(outputs)
        decoded_inputs = self.processor.post_process_image_text_to_text(inputs_id)
        generated_texts = [text.strip() for text in generated_texts]
        decoded_inputs = [text.strip() for text in decoded_inputs]
        # Remove the input text from the generated text if the generated text starts with the input text
        generated_texts = [
            text_generated[len(decoded_inputs[i]) :].strip()
            if text_generated.startswith(decoded_inputs[i])
            else text_generated
            for i, text_generated in enumerate(generated_texts)
        ]

        records = [
            {"input_text": input_text[i], "generated_text": generated_text}
            for i, generated_text in enumerate(generated_texts)
        ]

        return records
