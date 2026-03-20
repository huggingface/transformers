# Copyright 2026 The HuggingFace Inc. team.
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
Processor class for Granite 4 Vision.
"""

from fractions import Fraction

from ...image_processing_utils import select_best_resolution
from ...utils import logging
from ..llava_next.processing_llava_next import LlavaNextProcessor


logger = logging.get_logger(__name__)


class Granite4VisionProcessor(LlavaNextProcessor):
    r"""
    Constructs a Granite4Vision processor which wraps a Granite4Vision image processor and a Granite tokenizer into a single processor.

    [`Granite4VisionProcessor`] offers all the functionalities of [`LlavaNextImageProcessor`] and [`AutoTokenizer`]. See the
    [`~Granite4VisionProcessor.__call__`] and [`~Granite4VisionProcessor.decode`] for more information.

    Args:
        image_processor ([`LlavaNextImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`], *optional*):
            The tokenizer is a required input.
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        num_additional_image_tokens (`int`, *optional*, defaults to 0):
            Number of additional tokens added to the image embeddings, such as CLS (+1). If the backbone has no CLS or other
            extra tokens appended, no need to set this arg.
        downsample_rate (`str`, *optional*):
            The downsample rate for the vision features, specified as "query_side/window_side".
    """

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        num_additional_image_tokens=0,
        downsample_rate=None,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=patch_size,
            vision_feature_select_strategy=vision_feature_select_strategy,
            chat_template=chat_template,
            image_token=image_token,
            num_additional_image_tokens=num_additional_image_tokens,
        )
        self.downsample_rate = downsample_rate

    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        patches_height = height // self.patch_size
        patches_width = width // self.patch_size
        if self.downsample_rate is not None:
            ds_rate = Fraction(self.downsample_rate)
            patches_height = int(patches_height * ds_rate)
            patches_width = int(patches_width * ds_rate)

        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )
        # The base patch covers the entire image (+1 for the CLS)
        base_features = patches_height * patches_width + self.num_additional_image_tokens
        num_image_tokens = unpadded_features + newline_features + base_features
        return num_image_tokens


__all__ = ["Granite4VisionProcessor"]

# Made with Bob
