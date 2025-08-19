# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
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

"""Ernie model configuration"""
from transformers import PretrainedConfig

from ...modeling_rope_utils import rope_config_validation


class Ernie4_5_VLVisionConfig(PretrainedConfig):
    model_type = "ernie4_5_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=1280,
        hidden_act="quick_gelu",
        intermediate_size=4*1280,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        text_hidden_size=2560,
        spatial_conv_size=2,
        temporal_conv_size=2,
        rms_norm_eps=1e-5,
        vision_rms_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # vision projection
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size

        # resampler
        self.text_hidden_size = text_hidden_size
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.rms_norm_eps = rms_norm_eps
        self.vision_rms_norm_eps = vision_rms_norm_eps

        self.initializer_range = initializer_range


class Ernie4_5_VLTextConfig(PretrainedConfig):
    model_type = "ernie4_5_vl_text"
    base_config_key = "text_config"

    def __init__(
        self,
        hidden_size=2560,
        hidden_act="silu",
        intermediate_size=12288,
        max_position_embeddings=131072,
        moe_intermediate_size=[1536, 512],
        moe_k=6,
        moe_layer_end_index=29,
        moe_layer_interval=1,
        moe_layer_start_index=1,
        moe_norm_min=1e-12,
        moe_num_experts=64,
        moe_num_shared_experts=2,
        num_attention_heads=20,
        num_hidden_layers=28,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        rope_theta=500_000.0,
        vocab_size=103424,
        tie_word_embeddings=True,
        use_cache=True,
        use_bias=False,
        freq_allocation=20,
        rope_scaling=None,
        initializer_range=0.02,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_k = moe_k
        self.moe_layer_end_index = moe_layer_end_index
        self.moe_layer_interval = moe_layer_interval
        self.moe_layer_start_index = moe_layer_start_index
        self.moe_norm_min = moe_norm_min
        self.moe_num_experts = moe_num_experts
        self.moe_num_shared_experts = moe_num_shared_experts
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.use_bias = use_bias
        self.freq_allocation = freq_allocation
        self.rope_scaling = rope_scaling
        if rope_scaling is None:
            self.rope_scaling = {"rope_type": "ernie_3d", "freq_allocation": freq_allocation}
        rope_config_validation(self)
        self.initializer_range = initializer_range

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Ernie4_5_VLConfig(PretrainedConfig):
    model_type = "ernie4_5_vl"
    sub_configs = {"vision_config": Ernie4_5_VLVisionConfig, "text_config": Ernie4_5_VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=100295,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        # TODO: preprocessing
        #self.video_token_id = video_token_id

        super().__init__(**kwargs)


__all__ = [
    "Ernie4_5_VLConfig",
    "Ernie4_5_VLTextConfig",
    "Ernie4_5_VLVisionConfig",
]
