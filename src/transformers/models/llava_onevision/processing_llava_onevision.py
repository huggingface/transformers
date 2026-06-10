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
Processor class for LLaVa-Onevision.
"""

import math

from ...image_processing_utils import select_best_resolution
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class LlavaOnevisionProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


@auto_docstring
class LlavaOnevisionProcessor(ProcessorMixin):
    valid_processor_kwargs = LlavaOnevisionProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        num_image_tokens=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        video_token="<video>",
        vision_aspect_ratio="anyres_max_9",
        **kwargs,
    ):
        r"""
        num_image_tokens (`int`, *optional*):
            Number of image tokens for one imagethat will be returned by vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
        vision_aspect_ratio (`str`, *optional*, defaults to `"anyres_max_9"`):
            Aspect ratio used when processong image features. The default value is "anyres_max_9".
        """
        self.num_image_tokens = num_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.video_token = tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.vision_aspect_ratio = vision_aspect_ratio
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        batch_num_images = image_inputs["batch_num_images"]
        cumulative = 0
        is_multi_image = False
        for n in batch_num_images:
            if image_idx < cumulative + n:
                is_multi_image = n != 1
                break
            cumulative += n

        if is_multi_image:
            num_image_tokens = self.num_image_tokens + 1  # +1 for image_newline
        else:
            image_size = image_inputs["image_sizes"][image_idx]
            if not isinstance(image_size, (list, tuple)):
                image_size = image_size.tolist()
            orig_height, orig_width = image_size
            height, width = image_inputs["pixel_values"][0][0].shape[-2:]
            num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)

        if self.vision_feature_select_strategy == "default":
            num_image_tokens -= 1
        return self.image_token * num_image_tokens

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        processed_video = video_inputs["pixel_values_videos"][video_idx]
        num_frames = len(processed_video) if isinstance(processed_video, (list, tuple)) else processed_video.shape[0]
        patches_height_width = int(math.sqrt(self.num_image_tokens))
        pooled_height_width = math.ceil(patches_height_width / 2)
        num_video_tokens = (num_frames * pooled_height_width * pooled_height_width) + 1  # +1 for newline token
        return self.video_token * num_video_tokens

    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        patches_height = patches_width = int(math.sqrt(self.num_image_tokens))
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )

        # The base patch covers the entire image (no CLS for SigLIP)
        base_features = self.num_image_tokens
        num_image_tokens = unpadded_features + newline_features + base_features
        return num_image_tokens

    # Adapted from transformers.models.llava_next.processing_llava_next.LlavaNextProcessor._get_unpadded_features
    def _get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
        """
        Get number of features for a given image with height/width. LLaVA-NeXT is different from LLaVA
        because it divided each image into patches depending on its resolution. Therefore we need to calculate how many
        patches an image is divided into and get the number of features from that.
        """
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(round(height * (current_width / width), 7))
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(round(width * (current_height / height), 7))
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height

        max_num_patches = int(self.vision_aspect_ratio.strip("anyres_max_"))
        ratio = math.sqrt(current_height * current_width / (max_num_patches * patches_height**2))
        if ratio > 1.1:
            unpadded_features = int(current_height // ratio) * int(current_width // ratio)
            newline_features = int(current_height // ratio)

        return (unpadded_features, newline_features)

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (list[list[str]], *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (list[list[str]], *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
            audio_lengths (list[int], *optional*):
                The input length formatted as per each audio.
        Returns:
            dict[str, list[int]]: A dictionary mapping each modality ("image", "video", "audio")
            to a list containing the number of placeholder tokens required. If the model doesn't accept
            a certain modality or no input sizes are provided, the dict value is set to an empty list.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = LlavaOnevisionProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            size = images_kwargs.get("size", None) or self.image_processor.size
            size = (
                (size["shortest_edge"], size["shortest_edge"])
                if "shortest_edge" in size
                else (min(size["height"], size["width"]), min(size["height"], size["width"]))
            )
            processed_height, processed_width = size

            batch_num_image_tokens = []
            num_image_patches = [1] * len(image_sizes)  # llava-ov doesn't batch pixels as Idefics, thus `1` patch`
            for image_size in image_sizes:
                orig_height, orig_width = image_size
                num_image_tokens = self._get_number_of_features(
                    orig_height, orig_width, processed_height, processed_width
                )
                if self.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1
                batch_num_image_tokens.append(num_image_tokens)
            vision_data.update({"num_image_tokens": batch_num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)


__all__ = ["LlavaOnevisionProcessor"]
