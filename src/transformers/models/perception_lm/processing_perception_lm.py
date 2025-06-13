# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
Processor class for PerceptionLM.
"""

import torch
from typing import List, Union, Iterable

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, get_image_size, to_numpy_array, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from ...video_utils import VideoInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, _validate_images_text_input_order
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging

logger = logging.get_logger(__name__)


class PerceptionLMProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "do_resize": True,
            "do_rescale": False,
            "do_normalize": True,
            "size": 448,
            "resample": PILImageResampling.BICUBIC,
            "image_mean": IMAGENET_STANDARD_MEAN,
            "image_std": IMAGENET_STANDARD_STD,
        },
    }


class PerceptionLMProcessor(ProcessorMixin):
    r"""
    Constructs a PerceptionLM processor which wraps a PerceptionLM image processor and a LLaMa tokenizer into a single processor.

    [`PerceptionLMProcessor`] offers all the functionalities of [`PerceptionLMImageProcessor`] and [`PerceptionLMTokenizerFast`]. See the
    [`~PerceptionLMProcessor.__call__`] and [`~PerceptionLMProcessor.decode`] for more information.

    Args:
        image_processor ([`PerceptionLMImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PerceptionLMTokenizerFast`], *optional*):
            The tokenizer is a required input.
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Shoudl be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        num_additional_image_tokens (`int`, *optional*, defaults to 0):
            Number of additional tokens added to the image embeddings, such as CLS (+1). If the backbone has no CLS or other
            extra tokens appended, no need to set this arg.
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
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos: VideoInput = None,
        **kwargs: Unpack[PerceptionLMProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to PerceptionLMTokenizerFast's [`~PerceptionLMTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
    of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
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
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

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
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image", "video"])
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)
    
    def _expand_media_tokens(self, sample, media_token: str, media_iter: Iterable):
        media_count = sample.count(media_token)
        if media_count > 0:
            media_list = [next(media_iter) for _ in range(media_count)]
            sample_splits = sample.split(media_token)
            media_token_list = []
            for media in media_list:
                height, width = get_image_size(to_numpy_array(media))
                num_tiles = media.shape[0]
                num_media_tokens = (height // self.patch_size // self.pooling_ratio) * (
                    width // self.patch_size // self.pooling_ratio
                ) * num_tiles
                media_token_list.append(num_media_tokens)
            sample = ""
            for i, num_media_tokens in enumerate(media_token_list):
                sample += sample_splits[i]
                sample += media_token * num_media_tokens
            sample += sample_splits[-1]
        return sample

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
