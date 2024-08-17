# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for VideoLlava.
"""

import sys
from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, get_image_size, to_numpy_array
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


logger = logging.get_logger(__name__)


class VideoLlavaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


class VideoLlavaProcessor(ProcessorMixin):
    r"""
    Constructs a VideoLlava processor which wraps a VideoLlava image processor and a Llava tokenizer into a single processor.

    [`VideoLlavaProcessor`] offers all the functionalities of [`VideoLlavaImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~VideoLlavaProcessor.__call__`] and [`~VideoLlavaProcessor.decode`] for more information.

    Args:
        image_processor ([`VideoLlavaImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Shoudl be same as in model's config
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "patch_size", "vision_feature_select_strategy", "image_token", "video_token"]
    image_processor_class = "VideoLlavaImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        video_token="<video>",
        chat_template=None,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token
        self.video_token = video_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        images: Optional[ImageInput] = None,
        videos: Optional[ImageInput] = None,
        audio=None,
        **kwargs: Unpack[VideoLlavaProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        VideoLlavaImageProcessor's [`~VideoLlavaImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                Video frames to preprocess. Expects a single or batch of video frames in NumPy array or PyTorch
                tensor. Each video should be of shape (T, C, H, W), where T is number of frames, C is
                number of channels, H and W are image height and width.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            VideoLlavaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # Temporary fix for "paddding_side" in init_kwargs
        _ = output_kwargs["text_kwargs"].pop("padding_side", None)

        data = {}
        if images is not None or videos is not None:
            encoded_images = self.image_processor(images=images, videos=videos, **output_kwargs["images_kwargs"])
            data.update(encoded_images)

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        if encoded_images is not None and self.patch_size is None or self.vision_feature_select_strategy is None:
            prompt_strings = text
            logger.warning_once(
                "Expanding inputs for image tokens in Video-LLaVa should be done in processing. "
                "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.44."
            )
        elif encoded_images is not None:
            # Replace the image token with the expanded image token sequence
            if "pixel_values" in encoded_images:
                height, width = get_image_size(to_numpy_array(encoded_images.get("pixel_values")[0]))
                num_frames = 1
            else:
                one_video = to_numpy_array(encoded_images.get("pixel_values_videos")[0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]  # frame dim is always after batch dim

            num_image_tokens = (height // self.patch_size) * (width // self.patch_size) + 1
            num_video_tokens = num_image_tokens * num_frames
            if self.vision_feature_select_strategy == "default":
                num_image_tokens -= 1

            prompt_strings = []
            for sample in text:
                sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
                sample = sample.replace(self.video_token, self.video_token * num_video_tokens)
                prompt_strings.append(sample)

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
        data.update(text_inputs)

        return BatchFeature(data=data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
