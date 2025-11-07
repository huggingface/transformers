# coding=utf-8
# Copyright 2025 The Microsoft Team and The HuggingFace Inc. team. All rights reserved.
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
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class VibeVoiceDiffusionHeadConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceDiffusionHead`]. It is used to instantiate a
    VibeVoice Diffusion Head according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the VibeVoice
    architecture.

    e.g. [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the model's hidden states.
        num_head_layers (`int`, *optional*, defaults to 4):
            Number of layers in the diffusion head.
        head_ffn_ratio (`int`, *optional*, defaults to 3):
            The ratio of the hidden size to the intermediate size in the diffusion head.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the RMSNorm layers.
        latent_size (`int`, *optional*, defaults to 64):
            Dimensionality of the latent representation.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The activation function used by the diffusion head.
        frequency_embedding_size (`int`, *optional*, defaults to 256):
            The size of the frequency embedding.
        ddpm_num_steps (`int`, *optional*, defaults to 1000):
            The number of diffusion steps used during training.
        prediction_type (`str`, *optional*, defaults to `"v_prediction"`):
            The prediction type of the diffusion model. Default is `"v_prediction"`.
        ddpm_num_inference_steps (`int`, *optional*, defaults to 20):
            The number of diffusion steps used during inference.
        ddpm_beta_schedule (`str`, *optional*, defaults to `"squaredcos_cap_v2"`):
            The beta schedule used by the diffusion model.

    ```python
    >>> from transformers import VibeVoiceDiffusionHeadConfig, VibeVoiceDiffusionHead

    >>> # Initializing a VibeVoiceDiffusionHeadConfig
    >>> configuration = VibeVoiceDiffusionHeadConfig()

    >>> # Initializing a VibeVoiceDiffusionHead (with random weights)
    >>> model = VibeVoiceDiffusionHead(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice_diffusion_head"

    def __init__(
        self,
        hidden_size=768,
        num_head_layers=4,
        head_ffn_ratio=3,
        rms_norm_eps=1e-5,
        latent_size=64,
        hidden_act="silu",
        frequency_embedding_size=256,
        ddpm_num_steps=1000,
        prediction_type="v_prediction",
        ddpm_num_inference_steps=20,
        ddpm_beta_schedule="squaredcos_cap_v2",
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
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceForConditionalGeneration`]. It is used to instantiate an
    VibeVoice model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VibeVoice-1.5B.

    e.g. [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        acoustic_tokenizer_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the acoustic tokenizer.
        semantic_tokenizer_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the semantic tokenizer.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text model.
        diffusion_head_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the diffusion head.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*, defaults to 151643):
            The token ID for padding.
        eos_token_id (`int`, *optional*, defaults to 151643):
            The token ID for the end of sequence.
        speech_start_id (`int`, *optional*, defaults to 151652):
            The token ID indicating the start of speech tokens.
        speech_end_id (`int`, *optional*, defaults to 151653):
            The token ID indicating the end of speech tokens.
        speech_diffusion_id (`int`, *optional*, defaults to 151654):
            The token ID indicating the start of speech diffusion tokens.

    ```python
    >>> from transformers import VoxtralForConditionalGeneration, VoxtralConfig

    >>> # Initializing a Voxtral configuration
    >>> configuration = VoxtralConfig(audio_token_id=24, projector_hidden_act="gelu")

    >>> # Initializing a 3B model with random weights
    >>> model = VoxtralForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice"
    is_composition = True

    sub_configs = {
        "acoustic_tokenizer_config": AutoConfig,
        "semantic_tokenizer_config": AutoConfig,
        "text_config": AutoConfig,
        "diffusion_head_config": AutoConfig,
    }

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
        text_config=None,
        diffusion_head_config=None,
        use_cache=True,
        pad_token_id=151643,
        eos_token_id=151643,
        speech_start_id=151652,
        speech_end_id=151653,
        speech_diffusion_id=151654,
        **kwargs
    ):

        # TODO (ebezzam) check this setting
        kwargs["_attn_implementation_autoset"] = False

        if isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = acoustic_tokenizer_config.get("model_type", "vibevoice_acoustic_tokenizer")
            acoustic_tokenizer_config = CONFIG_MAPPING[acoustic_tokenizer_config["model_type"]](**acoustic_tokenizer_config)
        elif acoustic_tokenizer_config is None:
            acoustic_tokenizer_config = CONFIG_MAPPING["vibevoice_acoustic_tokenizer"]()
        self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if isinstance(semantic_tokenizer_config, dict):
            semantic_tokenizer_config["model_type"] = semantic_tokenizer_config.get("model_type", "vibevoice_semantic_tokenizer")
            semantic_tokenizer_config = CONFIG_MAPPING[semantic_tokenizer_config["model_type"]](**semantic_tokenizer_config)
        elif semantic_tokenizer_config is None:
            semantic_tokenizer_config = CONFIG_MAPPING["vibevoice_semantic_tokenizer"]()
        self.semantic_tokenizer_config = semantic_tokenizer_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()
        self.text_config = text_config

        if isinstance(diffusion_head_config, dict):
            diffusion_head_config["model_type"] = diffusion_head_config.get("model_type", "vibevoice_diffusion_head")
            diffusion_head_config = CONFIG_MAPPING[diffusion_head_config["model_type"]](**diffusion_head_config)
        elif diffusion_head_config is None:
            diffusion_head_config = CONFIG_MAPPING["vibevoice_diffusion_head"]()
        self.diffusion_head_config = diffusion_head_config
        if diffusion_head_config.latent_size != self.acoustic_tokenizer_config.hidden_size:
            raise ValueError(
                f"diffusion_head_config.latent_size ({diffusion_head_config.latent_size}) must match "
                f"acoustic_tokenizer_config.hidden_size ({self.acoustic_tokenizer_config.hidden_size})"
            )

        self.use_cache = use_cache
        self.speech_start_id = speech_start_id
        self.speech_end_id = speech_end_id
        self.speech_diffusion_id = speech_diffusion_id

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

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
