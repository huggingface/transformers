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

import numpy as np

from ...image_processing_utils import select_best_resolution
from ...image_utils import get_image_size, to_numpy_array
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class LlavaNextVideoProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


@auto_docstring
class LlavaNextVideoProcessor(ProcessorMixin):
    valid_processor_kwargs = LlavaNextVideoProcessorKwargs

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
        num_additional_image_tokens=0,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        num_additional_image_tokens (`int`, *optional*, defaults to 0):
            Number of additional tokens added to the image embeddings, such as CLS (+1). If the backbone has no CLS or other
            extra tokens appended, no need to set this arg.
        """
        self.patch_size = patch_size
        self.num_additional_image_tokens = num_additional_image_tokens
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
        super().__init__(video_processor, image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict | None = None, image_idx: int = 0) -> str:
        image_size = image_inputs["image_sizes"][image_idx]
        pixel_values = [pixel_values for sub_list in image_inputs["pixel_values"] for pixel_values in sub_list]
        height, width = get_image_size(to_numpy_array(pixel_values[image_idx]))
        if not isinstance(image_size, (list, tuple)):
            # cast to list to avoid numerical precision errors when calculating unpadding
            image_size = image_size.tolist()
        orig_height, orig_width = image_size
        num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)
        if self.vision_feature_select_strategy == "default":
            num_image_tokens -= 1
        return self.image_token * num_image_tokens

    def replace_video_token(self, video_inputs: dict | None = None, video_idx: int = 0) -> str:
        processed_video = video_inputs["pixel_values_videos"][video_idx]
        if isinstance(processed_video, (list, tuple)):
            processed_video = np.array(processed_video)
        else:
            processed_video = to_numpy_array(processed_video)
        height, width = get_image_size(processed_video[0])
        num_frames = processed_video.shape[0]  # frame dim is always after batch dim

        # no `self.num_additional_image_tokens` added because video always has a default feature selection strategy
        num_image_tokens = (height // self.patch_size) * (width // self.patch_size)
        num_video_tokens = num_image_tokens // 4 * num_frames  # divide by 4 needed for avg pooling layer
        return self.video_token * num_video_tokens

    # Copied from transformers.models.llava_next.processing_llava_next.LlavaNextProcessor._get_number_of_features
    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        patches_height = height // self.patch_size
        patches_width = width // self.patch_size
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )
        # The base patch covers the entire image (+1 for the CLS)
        base_features = patches_height * patches_width + self.num_additional_image_tokens
        num_image_tokens = unpadded_features + newline_features + base_features
        return num_image_tokens

    # Copied from transformers.models.llava_next.processing_llava_next.LlavaNextProcessor._get_unpadded_features
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
        return (unpadded_features, newline_features)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (list[list[str]], *optional*):
                The input sizes formatted as (height, width) per each image.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = LlavaNextVideoProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            size = images_kwargs.get("size", None) or self.image_processor.size
            size = (
                (size["shortest_edge"], size["shortest_edge"])
                if "shortest_edge" in size
                else (min(size["height"], size["width"]), min(size["height"], size["width"]))
            )
            processed_height, processed_width = size

            batch_num_image_tokens = []
            num_image_patches = [1] * len(image_sizes)  # llava-next doesn't batch pixels as Idefics, thus `1` patch`
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


__all__ = ["LlavaNextVideoProcessor"]
