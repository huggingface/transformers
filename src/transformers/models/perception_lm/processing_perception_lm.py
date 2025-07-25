# coding=utf-8
# Copyright 2025 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
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
Processor class for PerceptionLM.
"""

from typing import Iterable, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, get_image_size, to_numpy_array
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging
from ...video_utils import VideoInput


logger = logging.get_logger(__name__)


class PerceptionLMProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


class PerceptionLMProcessor(ProcessorMixin):
    r"""
    Constructs a PerceptionLM processor which wraps a PerceptionLM image processor, a PerceptionLM video processor, and a tokenizer into a single processor.

    [`PerceptionLMProcessor`] offers all the functionalities of [`PerceptionLMImageProcessorFast`], [`PerceptionLMVideoProcessor`], and the tokenizer (e.g. [`LlamaTokenizerFast`]). See the
    [`~PerceptionLMProcessor.__call__`] and [`~PerceptionLMProcessor.decode`] for more information.

    Args:
        video_processor ([`PerceptionLMVideoProcessor`], *optional*):
            The video processor to process video inputs.
        image_processor ([`PerceptionLMImageProcessorFast`], *optional*):
            The image processor to process image inputs.
        tokenizer ([`LlamaTokenizerFast`] or similar, *optional*):
            The tokenizer to process text inputs.
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.
        pooling_ratio (`int`, *optional*, defaults to 2):
            Pooling ratio for vision tokens. If not 1, 2D adaptive pooling is applied over projected vision tokens.
    """

    attributes = ["video_processor", "image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        video_processor=None,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        chat_template=None,
        pooling_ratio=2,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.pooling_ratio = pooling_ratio
        self.image_token = tokenizer.image_token
        self.video_token = tokenizer.video_token
        self.image_token_id = tokenizer.image_token_id
        self.video_token_id = tokenizer.video_token_id
        super().__init__(video_processor, image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        audio=None,
        videos: VideoInput = None,
        **kwargs: Unpack[PerceptionLMProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepares a batch containing one or more sequences of text and/or images and/or videos.

        If `text` is provided, it is tokenized using the tokenizer.
        If `images` is provided, they are processed using the image processor.
        If `videos` is provided, they are processed using the video processor.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The image or batch of images to be processed. Each image can be a PIL image, NumPy array, or PyTorch tensor.
                Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, *optional*):
                The sequence or batch of sequences to be tokenized. Each sequence can be a string.
            videos (`Any`, *optional*):
                The video or batch of videos to be processed.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is provided.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is provided).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is provided.
            - **pixel_values_videos** -- Video pixel values to be fed to a model. Returned when `videos` is provided.
        """
        if text is None:
            raise ValueError(
                "You have to specify at least `text` input. Optionally, you can also specify `images` or `videos`."
            )

        output_kwargs = self._merge_kwargs(
            PerceptionLMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if videos is not None:
            videos_inputs = self.video_processor(videos, **output_kwargs["videos_kwargs"])
        else:
            videos_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = []

        pixel_values = iter(image_inputs.get("pixel_values", []))
        pixel_values_videos = iter(videos_inputs.get("pixel_values_videos", []))
        for sample in text:
            # Replace the media token with the expanded media token sequence
            sample = self._expand_media_tokens(sample, self.tokenizer.image_token, pixel_values)
            sample = self._expand_media_tokens(sample, self.tokenizer.video_token, pixel_values_videos)
            prompt_strings.append(sample)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _expand_media_tokens(self, sample, media_token: str, media_iter: Iterable):
        media_count = sample.count(media_token)
        if media_count > 0:
            media_list = [next(media_iter) for _ in range(media_count)]
            sample_splits = sample.split(media_token)
            media_token_list = []
            for media in media_list:
                height, width = get_image_size(to_numpy_array(media))
                num_tiles = media.shape[0]
                num_media_tokens = (
                    (height // self.patch_size // self.pooling_ratio)
                    * (width // self.patch_size // self.pooling_ratio)
                    * num_tiles
                )
                media_token_list.append(num_media_tokens)
            sample = ""
            for i, num_media_tokens in enumerate(media_token_list):
                sample += sample_splits[i]
                sample += media_token * num_media_tokens
            sample += sample_splits[-1]
        return sample

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = PerceptionLMProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            tile_size = images_kwargs.get("tile_size", None) or self.image_processor.tile_size

            num_image_tokens = []
            num_image_patches = []
            for height, width in image_sizes:
                if self.image_processor.vision_input_type == "thumb+tile":
                    aspect_ratio = self.image_processor._fit_image_to_canvas(
                        img_width=width, img_height=height, tile_size=tile_size
                    )
                    if aspect_ratio is None:
                        aspect_ratio = self.image_processor._find_closest_aspect_ratio(
                            img_width=width, img_height=height, tile_size=tile_size
                        )
                    num_tiles = aspect_ratio[0] * aspect_ratio[1] + 1  # base image and tiles
                else:
                    num_tiles = 1

                num_image_tokens.append(
                    (tile_size // self.patch_size // self.pooling_ratio)
                    * (tile_size // self.patch_size // self.pooling_ratio)
                    * num_tiles
                )
                num_image_patches.append(num_tiles)

            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})
        return MultiModalData(**vision_data)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PerceptionLMTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PerceptionLMTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["PerceptionLMProcessor"]
