# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""
Processor class for Qwen2-VL.
"""

import base64
from io import BytesIO
from typing import Dict, List, Optional, Union

import requests
import torch
from PIL import Image
from torchvision import io

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from ...utils import TensorType


class Qwen2VLProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2-VL processor which wraps a Qwen2-VL image processor and a Qwen2 tokenizer into a single processor.

    [`Qwen2VLProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2VLProcessor.__call__`] and [`~Qwen2VLProcessor.decode`] for more information.

    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Qwen2VLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.vision_token_id = self.tokenizer("<|vision_pad|>")["input_ids"][0]

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        vision_infos: List[Dict] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: int = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            vision_infos (`List[Dict]`):
                The list of vision info dict. Each vision info dict has the path of the image or the video. support url, local path, base64.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `vision_infos` is not `None`.
            - **vision_grid_thw** -- List of 3D temporal grid in vision encoder. Returned when `imagvision_infoses` is not `None`.
        """
        if len(vision_infos) > 0:
            merge_vision_infos = []
            if isinstance(vision_infos[0], list):
                for vision_info in vision_infos:
                    merge_vision_infos.extend(vision_info)
            else:
                merge_vision_infos = vision_infos
            vision_infos = merge_vision_infos

        ## Read images or videos
        vision_pixel_inputs = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                images = [self.fetch_image(vision_info)]
            elif "video" in vision_info:
                images = self.fetch_video(vision_info, nframe_factor=self.image_processor.temporal_patch_size)
            else:
                raise ValueError("image, image_url or video should in content.")
            vision_pixel_inputs.append(images)

        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        if len(vision_infos) > 0:
            vision_inputs = self.image_processor(
                vision_pixel_inputs, vision_infos=vision_infos, return_tensors=return_tensors
            )
        else:
            vision_inputs = {}

        return BatchFeature(data={**text_inputs, **vision_inputs})

    def fetch_image(self, ele: Dict):
        if "image" in ele:
            image = ele["image"]
        elif "image_url" in ele:
            image = ele["image_url"]
        image_obj = None
        if isinstance(image, Image.Image):
            image_obj = image
        if image.startswith("http://") or image.startswith("https://"):
            image_obj = Image.open(requests.get(image, stream=True).raw)
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            data = image.split(";", 1)[1]
            if data.startswith("base64,"):
                data = base64.b64decode(data[7:])
                image_obj = Image.open(BytesIO(data))
        if image_obj is None:
            raise ValueError(
                "Unrecognized image input, support local path, http url, base64 and " "PIL.Image, got {image}"
            )
        image_obj = image_obj.convert("RGB")
        return image_obj

    def fetch_video(self, ele: Dict, nframe_factor=2):
        if isinstance(ele["video"], str):
            # TODO: support http url
            def round_by_factor(number: int, factor: int) -> int:
                return round(number / factor) * factor

            video = ele["video"]
            if video.startswith("file://"):
                video = video[7:]

            video, _, info = io.read_video(
                video,
                start_pts=ele.get("video_start", 0.0),
                end_pts=ele.get("video_end", None),
                pts_unit="sec",
                output_format="TCHW",
            )
            assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
            if "nframes" in ele:
                nframes = round_by_factor(ele["nframes"], nframe_factor)
            else:
                fps = ele.get("fps", 1.0)
                nframes = round_by_factor(video.size(0) / info["video_fps"] * fps, nframe_factor)
            idx = torch.linspace(0, video.size(0) - 1, nframes, dtype=torch.int64)
            return video[idx]
        else:
            assert isinstance(ele["video"], (list, tuple))
            assert len(ele["video"]) % nframe_factor == 0
            images = [self.fetch_image({"image": ele}) for ele in ele["video"]]
            return images

    def extract_vision_info(self, conversation):
        vision_infos = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] in ("image", "image_url", "video"):
                        vision_infos.append(ele)
        return vision_infos

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def apply_chat_template(
        self,
        conversation: List[Dict[str, Union[str, List]]],
        chat_template: Optional[str] = None,
        tokenize: bool = False,
        **kwargs,
    ) -> tuple[str, List]:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        Args:
            conversation (`List[Dict, str, str]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the default chat template
                is used.
            tokenize (`bool`, *optional*, defaults to `False`):
                Whether to tokenize the output or not.
            **kwargs:
                Additional keyword arguments
        """

        if chat_template is None:
            if self.chat_template is not None:
                chat_template = self.chat_template
            elif getattr(self, "default_chat_template", None) is not None:
                chat_template = self.default_chat_template
            else:
                raise ValueError(
                    "No chat template is set for this processor. Please either set the `chat_template` attribute, "
                    "or provide a chat template as an argument. See "
                    "https://huggingface.co/docs/transformers/main/en/chat_templating for more information."
                )
        return self.tokenizer.apply_chat_template(
            conversation, chat_template=chat_template, tokenize=tokenize, **kwargs
        ), self.extract_vision_info(conversation)

    @property
    def default_chat_template(self):
        """
        This default vicuna template formats inputs in the form of a chat history. For each message in the chat history:
        * the template will output the role of the speaker followed by the content of the message.
        * content is a list of strings and images.
        * If the content element is an image, the template will output a sequence of <image> or <video> tokens

        Example:

        ```python
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": [{"type": "image", "image_url": "https://example.com/image2.jpg"},{"text":"I'm doing well, thank you for asking. How can I assist you today?"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Can you describe these images and video?"},
                {"type": "image", "image_url": "https://example.com/image1.jpg"},
                {"type": "image", "image_url": "https://example.com/image2.jpg"},
                {"type": "video", "video_url": "https://example.com/video1.mp4"},
                {"type": "text", "text": "These are from my vacation."}
            ]},
            {"role": "assistant", "content": "I'd be happy to describe the images and video for you. Could you please provide more context about your vacation?"},
            {"role": "user", "content": "It was a trip to the mountains. Can you see the details in the images and video?"},
        ]

        result_with_id = template.render(messages=messages, add_generation_prompt=True, add_vision_id=True)
        result_without_id = template.render(messages=messages, add_generation_prompt=True, add_vision_id=False)
        ```
        """
        # fmt: off
        return (
            "{% set image_count = namespace(value=0) %}"
            "{% set video_count = namespace(value=0) %}"
            "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "{% endif %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}<|im_end|>\n"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if 'image' in content or 'image_url' in content %}"
                            "{% set image_count.value = image_count.value + 1 %}"
                            "{% if add_vision_id %}"
                                "Picture {{ image_count.value }}: "
                            "{% endif %}"
                            "<|vision_start|><|vision_pad|><|vision_end|>"
                        "{% elif 'video' in content %}"
                            "{% set video_count.value = video_count.value + 1 %}"
                            "{% if add_vision_id %}"
                                "Video {{ video_count.value }}: "
                            "{% endif %}"
                            "<|vision_start|><|vision_pad|><|vision_end|>"
                        "{% elif 'text' in content %}"
                            "{{ content['text'] }}"
                        "{% endif %}"
                    "{% endfor %}"
                    "<|im_end|>\n"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
            "{% endif %}"
        )
        # fmt: on
