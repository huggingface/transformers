# Copyright 2023 The Intel AIA Team Authors, and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License=, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing=, software
# distributed under the License is distributed on an "AS IS" BASIS=,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND=, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TVP model configuration"""

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Intel/tvp-base")
class TvpConfig(PreTrainedConfig):
    r"""
    distance_loss_weight (`float`, *optional*, defaults to 1.0):
        The weight of distance loss.
    duration_loss_weight (`float`, *optional*, defaults to 0.1):
        The weight of duration loss.
    visual_prompter_type (`str`, *optional*, defaults to `"framepad"`):
        Visual prompt type. The type of padding. Framepad means padding on each frame. Should be one of "framepad"
        or "framedownpad"
    visual_prompter_apply (`str`, *optional*, defaults to `"replace"`):
        The way of applying visual prompt. Replace means use the value of prompt to change the original value in
        visual inputs. Should be one of "replace", or "add", or "remove".
    visual_prompt_size (`int`, *optional*, defaults to 96):
        The size of visual prompt.
    max_img_size (`int`, *optional*, defaults to 448):
        The maximum size of frame.
    num_frames (`int`, *optional*, defaults to 48):
        The number of frames extracted from a video.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    max_grid_col_position_embeddings (`int`, *optional*, defaults to 100):
        The largest number of horizontal patches from a video frame.
    max_grid_row_position_embeddings (`int`, *optional*, defaults to 100):
        The largest number of vertical patches from a video frame.
    """

    model_type = "tvp"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config=None,
        distance_loss_weight=1.0,
        duration_loss_weight=0.1,
        visual_prompter_type="framepad",
        visual_prompter_apply="replace",
        visual_prompt_size=96,
        max_img_size=448,
        num_frames=48,
        vocab_size=30522,
        type_vocab_size=2,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=512,
        max_grid_col_position_embeddings=100,
        max_grid_row_position_embeddings=100,
        hidden_dropout_prob=0.1,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        attention_probs_dropout_prob=0.1,
        pad_token_id=None,
        **kwargs,
    ):
        backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=backbone_config,
            default_config_type="resnet",
            default_config_kwargs={"out_features": ["stage4"]},
            **kwargs,
        )

        self.backbone_config = backbone_config
        self.distance_loss_weight = distance_loss_weight
        self.duration_loss_weight = duration_loss_weight
        self.visual_prompter_type = visual_prompter_type
        self.visual_prompter_apply = visual_prompter_apply
        self.visual_prompt_size = visual_prompt_size
        self.max_img_size = max_img_size
        self.num_frames = num_frames
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.max_grid_col_position_embeddings = max_grid_col_position_embeddings
        self.max_grid_row_position_embeddings = max_grid_row_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.pad_token_id = pad_token_id

        super().__init__(**kwargs)


__all__ = ["TvpConfig"]
