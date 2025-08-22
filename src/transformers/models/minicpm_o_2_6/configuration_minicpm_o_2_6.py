# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
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

import os
from typing import Union

from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...modeling_rope_utils import rope_config_validation
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers import Qwen2Config, WhisperConfig
from ...utils import logging

logger = logging.get_logger(__name__)


class MiniCPMVSliceConfig(PretrainedConfig):
    model_type = "minicpmv"

    def __init__(
        self,
        patch_size=14,
        max_slice_nums=9,
        scale_resolution=448,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "minicpmv":
            config_dict = config_dict["slice_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class MiniCPMConditionalTTSConfig(PretrainedConfig):
    model_type = "conditional_chattts"

    def __init__(
        self,
        llm_dim: int = 2560,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 20,
        max_position_embeddings: int = 4096,
        num_audio_tokens: int = 626,
        num_text_tokens: int = 21178,
        num_mel_bins: int = 100,
        num_vq: int = 4,
        use_speaker_embedding: bool = True,
        use_llm_hidden_state: bool = False,
        spk_emb_token_id: int = 21143,
        num_spk_embs: int = 1,
        audio_bos_token_id: int = 21132,
        text_eos_token_id: int = 21133,
        use_text: bool = True,
        streaming: bool = True,
        streaming_text_chunk_size: int = 10,
        streaming_text_reserved_len: int = 300,
        streaming_audio_chunk_size: int = 50,
        attn_implementation: str = "sdpa",
        use_mlp: bool = True,
        aug_loss_weight: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.llm_dim = llm_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.num_audio_tokens = num_audio_tokens
        self.num_text_tokens = num_text_tokens
        self.num_mel_bins = num_mel_bins
        self.num_vq = num_vq
        self.use_speaker_embedding = use_speaker_embedding
        self.use_llm_hidden_state = use_llm_hidden_state
        self.spk_emb_token_id = spk_emb_token_id
        self.num_spk_embs = num_spk_embs
        self.audio_bos_token_id = audio_bos_token_id
        self.text_eos_token_id = text_eos_token_id
        self.use_text = use_text
        self.streaming = streaming
        self.streaming_text_chunk_size = streaming_text_chunk_size
        self.streaming_text_reserved_len = streaming_text_reserved_len
        self.streaming_audio_chunk_size = streaming_audio_chunk_size
        self.attn_implementation = attn_implementation
        self.use_mlp = use_mlp
        self.aug_loss_weight = aug_loss_weight


class MiniCPM_o_2_6Config(PretrainedConfig):
    model_type = "minicpmo"
    keys_to_ignore_at_inference = ["past_key_values"]

    default_vision_config = {
        "hidden_size": 1152,
        "image_size": 980,
        "intermediate_size": 4304,
        "model_type": "siglip",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        use_cache=True,
        query_num=64,
        image_size=448,
        drop_vision_last_layer=True,
        batch_vision_input=True,
        slice_config=None,
        vision_config=None,
        audio_config=None,
        tts_config=None,
        use_image_id=True,
        vision_batch_size=16,
        audio_pool_step=2,
        audio_chunk_length=1.0,
        stream_input=False,
        init_vision=True,
        init_audio=True,
        init_tts=True,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.use_cache = use_cache
        self.query_num = query_num
        self.image_size = image_size
        self.drop_vision_last_layer = drop_vision_last_layer
        self.batch_vision_input = batch_vision_input
        self.use_image_id = use_image_id
        self.vision_batch_size = vision_batch_size
        self.audio_pool_step = audio_pool_step
        self.audio_chunk_length = audio_chunk_length
        self.stream_input = stream_input
        self.init_vision = init_vision
        self.init_audio = init_audio
        self.init_tts = init_tts

        if slice_config is None:
            self.slice_config = MiniCPMVSliceConfig(max_slice_nums=1)
        else:
            self.slice_config = MiniCPMVSliceConfig(**slice_config)
        self.slice_mode = True

        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if vision_config is None:
            self.vision_config = SiglipVisionConfig(
                **self.default_vision_config)
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = SiglipVisionConfig(**vision_config)
        elif isinstance(vision_config, SiglipVisionConfig):
            self.vision_config = vision_config

        # same as openai/whisper-medium add use_cache
        if audio_config is None:
            self.audio_config = WhisperConfig()
        elif isinstance(audio_config, dict):
            self.audio_config = WhisperConfig(**audio_config)
        elif isinstance(audio_config, WhisperConfig):
            self.audio_config = audio_config

        if tts_config is None:
            self.tts_config = MiniCPMConditionalTTSConfig()
        elif isinstance(tts_config, dict):
            self.tts_config = MiniCPMConditionalTTSConfig(**tts_config)
        elif isinstance(tts_config, MiniCPMConditionalTTSConfig):
            self.tts_config = tts_config

        self.patch_size = self.vision_config.patch_size

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["MiniCPM_o_2_6Config"]
