# coding=utf-8
# Copyright 2025-present, the HuggingFace Inc. Team and AIRAS Inc. Team. All rights reserved.
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
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig

logger = logging.get_logger(__name__)

SAPNOUS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Sapnous-AI/Sapnous-6B": "https://huggingface.co/Sapnous-AI/Sapnous-6B/resolve/main/config.json",
}

class SapnousT1Config(PretrainedConfig):
    """Configuration class for Sapnous-T1 model with vision-language capabilities.
    
    This configuration class handles both text and vision modalities, supporting multimodal
    tasks like image understanding, video processing, and vision-language reasoning.
    """
    
    model_type = "sapnous_t1"

    def __init__(
        self,
        # Text model parameters
        vocab_size=151936,
        hidden_size=5120,
        intermediate_size=20480,
        num_hidden_layers=36,
        num_attention_heads=40,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=151643,
        eos_token_id=151645,
        tie_word_embeddings=True,
        
        # Vision model parameters
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        vision_token_id=151654,
        image_token_id=151655,
        video_token_id=151656,
        vision_config=None,
        patch_size=14,
        image_size=224,
        num_channels=3,
        vision_layers=24,
        vision_heads=16,
        vision_hidden_size=1024,
        vision_intermediate_size=4096,
        vision_act="gelu",
        vision_layer_norm_eps=1e-5,
        vision_dropout=0.0,
        vision_attention_dropout=0.0,
        vision_embedding_dropout=0.0,
        
        # Cross-attention parameters
        num_cross_attention_layers=12,
        cross_attention_heads=16,
        cross_attention_dropout=0.0,
        use_cross_attention=True,
        
        # Positional encoding and attention parameters
        rope_theta=1000000.0,
        sliding_window=32768,
        use_sliding_window=False,
        max_window_layers=70,
        attention_dropout=0.0,
        rope_scaling=None,
        scoring_func="softmax",
        
        # Training parameters
        aux_loss_alpha=0.001,
        seq_aux=True,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.vision_token_id = vision_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_config = vision_config
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux

    model_type = "sapnous_t1"
    keys_to_ignore_at_inference = ["past_key_values"]

AutoConfig.register("sapnous_t1", SapnousT1Config)