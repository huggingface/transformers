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
""" VibeVoice_AcousticTokenizer model configuration"""

from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..qwen2.configuration_qwen2 import Qwen2Config


logger = logging.get_logger(__name__)

class VibeVoiceAcousticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_acoustic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0.5,
        std_dist_type: str = 'gaussian',
        # common
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        # encoder specific
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[list[int]] = [8,5,5,4,2,2],
        encoder_depths: str = "3-3-3-3-3-3-8",
        # decoder specific
        decoder_n_filters: int = 32,
        decoder_ratios: Optional[list[int]] = None, # if None, same as encoder
        decoder_depths: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type

        # common parameters
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        # encoder specific parameters
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths

        # decoder specific parameters
        self.decoder_ratios = decoder_ratios if decoder_ratios is not None else encoder_ratios
        self.decoder_n_filters = decoder_n_filters
        self.decoder_depths = decoder_depths


class VibeVoiceSemanticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_semantic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0,
        std_dist_type: str = 'none',
        # common
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        # encoder specific
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[list[int]] = [8,5,5,4,2,2],
        encoder_depths: str = "3-3-3-3-3-3-8",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type

        # common parameters
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        # encoder specific parameters
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths


class VibeVoiceDiffusionHeadConfig(PretrainedConfig):
    model_type = "vibevoice_diffusion_head"

    def __init__(
        self,
        hidden_size=768,
        head_layers=4,
        head_ffn_ratio=3.0,
        rms_norm_eps=1e-5,
        latent_size=64,
        speech_vae_dim=None,
        prediction_type="v_prediction",
        diffusion_type="ddpm",
        ddpm_num_steps=1000,
        ddpm_num_inference_steps=20,
        ddpm_beta_schedule="cosine",
        ddpm_batch_mul=4,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.head_layers = head_layers
        self.head_ffn_ratio = head_ffn_ratio
        self.rms_norm_eps = rms_norm_eps
        self.latent_size = latent_size
        self.speech_vae_dim = speech_vae_dim
        self.prediction_type = prediction_type
        self.diffusion_type = diffusion_type
        self.ddpm_num_steps = ddpm_num_steps
        self.ddpm_num_inference_steps = ddpm_num_inference_steps
        self.ddpm_beta_schedule = ddpm_beta_schedule
        self.ddpm_batch_mul = ddpm_batch_mul

        super().__init__(**kwargs)

class VibeVoiceConfig(PretrainedConfig):
    model_type = "vibevoice"
    is_composition = True
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig,
        "semantic_tokenizer_config": VibeVoiceSemanticTokenizerConfig,
        "decoder_config": Qwen2Config,
        "diffusion_head_config": VibeVoiceDiffusionHeadConfig,
    }
    # keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `Qwen2`
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
        decoder_config=None,
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

        if decoder_config is None:
            self.decoder_config = self.sub_configs["decoder_config"]()
        elif isinstance(decoder_config, dict):
            # If a dictionary is provided, instantiate the config class with it
            # self.decoder_config = self.sub_configs["decoder_config"](**decoder_config)
            if decoder_config.get("model_type", '') == "qwen2":
                self.decoder_config = Qwen2Config(**decoder_config)
            else:
                raise ValueError(f"Unsupported decoder model type: {decoder_config.get('model_type', '')}")
        elif isinstance(decoder_config, (Qwen2Config,)):
            # If an instance of the config class is provided
            self.decoder_config = decoder_config

        if diffusion_head_config is None:
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"]()
        elif isinstance(diffusion_head_config, dict):
            diffusion_head_config["model_type"] = "vibevoice_diffusion_head"
            self.diffusion_head_config = self.sub_configs["diffusion_head_config"](**diffusion_head_config)
        elif isinstance(diffusion_head_config, VibeVoiceDiffusionHeadConfig):
            # If an instance of the config class is provided
            self.diffusion_head_config = diffusion_head_config

        # other parameters
        self.acoustic_vae_dim = getattr(self.acoustic_tokenizer_config, 'vae_dim', 64)
        self.semantic_vae_dim = getattr(self.semantic_tokenizer_config, 'vae_dim', 128)
        self.num_hidden_layers = self.decoder_config.num_hidden_layers

        super().__init__(**kwargs)

__all__ = [
    "VibeVoiceAcousticTokenizerConfig",
    "VibeVoiceSemanticTokenizerConfig",
    "VibeVoiceDiffusionHeadConfig",
    "VibeVoiceConfig"
]
