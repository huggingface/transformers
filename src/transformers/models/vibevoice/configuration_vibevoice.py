# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
""" VibeVoice model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..qwen2.configuration_qwen2 import Qwen2Config
from ..vibevoice_acoustic_tokenizer import VibeVoiceAcousticTokenizerConfig
from ..vibevoice_semantic_tokenizer import VibeVoiceSemanticTokenizerConfig


logger = logging.get_logger(__name__)


class VibeVoiceDiffusionHeadConfig(PretrainedConfig):
    model_type = "vibevoice_diffusion_head"

    def __init__(
        self,
        hidden_size=768,
        num_head_layers=4,
        head_ffn_ratio=3,
        rms_norm_eps=1e-5,
        # TODO (ebezzam) `latent_size` same as `acoustic_tokenizer_config.hidden_size`
        latent_size=64,
        prediction_type="v_prediction",
        ddpm_num_steps=1000,
        ddpm_num_inference_steps=20,
        ddpm_beta_schedule="squaredcos_cap_v2",
        hidden_act="silu",
        frequency_embedding_size=256,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_head_layers = num_head_layers
        self.head_ffn_ratio = head_ffn_ratio
        self.rms_norm_eps = rms_norm_eps
        self.latent_size = latent_size
        self.prediction_type = prediction_type
        self.ddpm_num_steps = ddpm_num_steps
        self.ddpm_num_inference_steps = ddpm_num_inference_steps
        self.ddpm_beta_schedule = ddpm_beta_schedule
        self.hidden_act = hidden_act
        self.frequency_embedding_size = frequency_embedding_size

        # NOTE (ebezzam) to use LlamaMLP via modular
        self.mlp_bias = False

        super().__init__(**kwargs)

    @property
    def intermediate_size(self) -> int:
        return self.hidden_size * self.head_ffn_ratio


class VibeVoiceConfig(PretrainedConfig):
    model_type = "vibevoice"
    is_composition = True

    # TODO: change to AutoConfig and "decoder_config" to "text_config"
    # -- see Llava
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig,
        "semantic_tokenizer_config": VibeVoiceSemanticTokenizerConfig,
        "text_config": Qwen2Config,
        "diffusion_head_config": VibeVoiceDiffusionHeadConfig,
    }
    # keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `Qwen2`
    # TODO (ebezzam) can be copied over with modular
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        acoustic_tokenizer_config=None,
        semantic_tokenizer_config=None,
        text_config=None,
        diffusion_head_config=None,
        **kwargs
    ):

        # kwargs["_attn_implementation"] = "flash_attention_2"
        kwargs["_attn_implementation_autoset"] = False

        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"]()
        elif isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = "vibevoice_acoustic_tokenizer"
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer_config"](**acoustic_tokenizer_config)
        elif isinstance(acoustic_tokenizer_config, VibeVoiceAcousticTokenizerConfig):
            # If an instance of the config class is provided
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if semantic_tokenizer_config is None:
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"]()
        elif isinstance(semantic_tokenizer_config, dict):
            semantic_tokenizer_config["model_type"] = "vibevoice_semantic_tokenizer"
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer_config"](**semantic_tokenizer_config)
        elif isinstance(semantic_tokenizer_config, VibeVoiceSemanticTokenizerConfig):
            # If an instance of the config class is provided
            self.semantic_tokenizer_config = semantic_tokenizer_config

        if text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        elif isinstance(text_config, dict):
            # If a dictionary is provided, instantiate the config class with it
            # self.text_config = self.sub_configs["text_config"](**text_config)
            if text_config.get("model_type", '') == "qwen2":
                self.text_config = Qwen2Config(**text_config)
            else:
                raise ValueError(f"Unsupported text model type: {text_config.get('model_type', '')}")
        elif isinstance(text_config, (Qwen2Config,)):
            # If an instance of the config class is provided
            self.text_config = text_config

        if diffusion_head_config is None:
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"]()
        elif isinstance(diffusion_head_config, dict):
            diffusion_head_config["model_type"] = "vibevoice_diffusion_head"
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"](**diffusion_head_config)
        elif isinstance(diffusion_head_config, VibeVoiceDiffusionHeadConfig):
            # If an instance of the config class is provided
            self.diffusion_head_config = diffusion_head_config

        # TODO (ebezzam) move to top? Llava has at bottom
        super().__init__(**kwargs)

    @property
    def acoustic_hidden_size(self) -> int:
        return self.acoustic_tokenizer_config.hidden_size

    @property
    def semantic_hidden_size(self) -> int:
        return self.semantic_tokenizer_config.hidden_size


__all__ = [
    "VibeVoiceDiffusionHeadConfig",
    "VibeVoiceConfig"
]
