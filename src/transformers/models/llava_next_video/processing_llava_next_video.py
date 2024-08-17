# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
Processor class for LLaVa-NeXT-Video.
"""

import sys
from typing import TYPE_CHECKING, List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, VideoInput, get_image_size, to_numpy_array
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)


class LlavaNextVideoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


class LlavaNextVideoProcessor(ProcessorMixin):
    r"""
    Constructs a LLaVa-NeXT-Video processor which wraps a LLaVa-NeXT image processor, LLaVa-NeXT-Video video processor and
    a LLaMa tokenizer into a single processor.

    [`LlavaNextVideoProcessor`] offers all the functionalities of [`LlavaNextImageProcessor`], [`LlavaNextVideoImageProcessor`] and
    [`LlamaTokenizerFast`]. See the [`~LlavaNextVideoProcessor.__call__`] and [`~LlavaNextVideoProcessor.decode`] for more information.

    Args:
        video_processor ([`LlavaNextVideoImageProcessor`], *optional*):
            The video processor is a required input.
        image_processor ([`LlavaNextImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*):
            Jinja chat template that will be used in tokenizer's `apply_chat_template`
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Shoudl be same as in model's config
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
    """

    # video and image processor share same args, but have different processing logic
    # only image processor config is saved in the hub
    attributes = ["video_processor", "image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "patch_size", "vision_feature_select_strategy", "image_token", "video_token"]
    image_processor_class = "LlavaNextImageProcessor"
    video_processor_class = "LlavaNextVideoImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __init__(
        self,
        video_processor=None,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        video_token="<video>",
        image_token="<image>",
        **kwargs,
    ):
        self.patch_size = patch_size
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token
        self.video_token = video_token
        super().__init__(video_processor, image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Optional[ImageInput] = None,
        videos: Optional[VideoInput] = None,
        audio=None,
        **kwargs: Unpack[LlavaNextVideoProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. To prepare the video(s),
        this method forwards the `videos` and `kwrags` arguments to LlavaNextVideoImageProcessor's
        [`~LlavaNextVideoImageProcessor.__call__`] if `videos` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            LlavaNextVideoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
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

        print(self.patch_size, self.vision_feature_select_strategy, image_inputs, videos_inputs.keys())

        if self.patch_size is None or self.vision_feature_select_strategy is None:
            prompt_strings = text
            logger.warning_once(
                "Expanding inputs for image/video tokens in LLaVa-NeXT-Video should be done in processing. "
                "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
            )
        # cannot infer image expansion length if no images/videos are found
        elif not image_inputs and not videos_inputs:
            prompt_strings = text
        else:
            # images expand taking into account num_of_patches in each image
            if image_inputs:
                image_sizes = image_inputs["image_sizes"]
                height, width = get_image_size(to_numpy_array(image_inputs["pixel_values"][0][0]))
                prompt_strings = []
                for image_size, sample in zip(image_sizes, text):
                    # Replace the image token with the expanded image token sequence
                    orig_height, orig_width = image_size
                    num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)
                    if self.vision_feature_select_strategy == "default":
                        num_image_tokens -= 1

                    sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
                    prompt_strings.append(sample)
                text = prompt_strings

            # videos are easier, simply get frames and multiply
            if videos_inputs:
                one_video = to_numpy_array(videos_inputs.get("pixel_values_videos")[0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]  # frame dim is always after batch dim
                num_image_tokens = (height // self.patch_size) * (width // self.patch_size)
                num_video_tokens = num_image_tokens // 4 * num_frames  # divide by 4 needed for avg pooling layer

                prompt_strings = []
                for sample in text:
                    sample = sample.replace(self.video_token, self.video_token * num_video_tokens)
                    prompt_strings.append(sample)

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

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
