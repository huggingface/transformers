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

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="Intel/tvp-base")
@strict
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

    backbone_config: dict | PreTrainedConfig | None = None
    distance_loss_weight: float = 1.0
    duration_loss_weight: float = 0.1
    visual_prompter_type: str = "framepad"
    visual_prompter_apply: str = "replace"
    visual_prompt_size: int = 96
    max_img_size: int = 448
    num_frames: int = 48
    vocab_size: int = 30522
    type_vocab_size: int = 2
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    max_grid_col_position_embeddings: int = 100
    max_grid_row_position_embeddings: int = 100
    hidden_dropout_prob: float = 0.1
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    attention_probs_dropout_prob: float = 0.1
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="resnet",
            default_config_kwargs={"out_features": ["stage4"]},
            **kwargs,
        )

        super().__post_init__(**kwargs)


__all__ = ["TvpConfig"]
