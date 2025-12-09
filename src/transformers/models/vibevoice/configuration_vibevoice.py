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

from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


class VibeVoiceConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceForConditionalGeneration`]. It is used to instantiate an
    VibeVoice model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults similar to that of [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        acoustic_tokenizer_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the acoustic tokenizer.
        semantic_tokenizer_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the semantic tokenizer.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text model.
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
        num_head_layers (`int`, *optional*, defaults to 4):
            Number of layers in the diffusion head.
        head_ffn_ratio (`int`, *optional*, defaults to 3):
            The ratio of the language model hidden size to the intermediate size in the diffusion head.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the RMSNorm layers.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The activation function used by the diffusion head.
        frequency_embedding_size (`int`, *optional*, defaults to 256):
            The size of the frequency embedding.

    ```python
    >>> from transformers import VibeVoiceForConditionalGeneration, VibeVoiceConfig

    >>> # Initializing a VibeVoice configuration
    >>> configuration = VibeVoiceConfig()

    >>> # Initializing a 1.5B model with random weights
    >>> model = VibeVoiceForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice"
    is_composition = True

    sub_configs = {
        "acoustic_tokenizer_config": AutoConfig,
        "semantic_tokenizer_config": AutoConfig,
        "text_config": AutoConfig,
    }

    # Default tensor parallel plan for base model `Qwen2`
    base_model_tp_plan = {
        "language_model.layers.*.self_attn.q_proj": "colwise",
        "language_model.layers.*.self_attn.k_proj": "colwise",
        "language_model.layers.*.self_attn.v_proj": "colwise",
        "language_model.layers.*.self_attn.o_proj": "rowwise",
        "language_model.layers.*.mlp.gate_proj": "colwise",
        "language_model.layers.*.mlp.up_proj": "colwise",
        "language_model.layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        acoustic_tokenizer_config=None,
        semantic_tokenizer_config=None,
        text_config=None,
        use_cache=True,
        pad_token_id=151643,
        eos_token_id=151643,
        speech_start_id=151652,
        speech_end_id=151653,
        speech_diffusion_id=151654,
        num_head_layers=4,
        head_ffn_ratio=3,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        frequency_embedding_size=256,
        **kwargs,
    ):
        if isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = acoustic_tokenizer_config.get(
                "model_type", "vibevoice_acoustic_tokenizer"
            )
            acoustic_tokenizer_config = CONFIG_MAPPING[acoustic_tokenizer_config["model_type"]](
                **acoustic_tokenizer_config
            )
        elif acoustic_tokenizer_config is None:
            acoustic_tokenizer_config = CONFIG_MAPPING["vibevoice_acoustic_tokenizer"]()
        self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if isinstance(semantic_tokenizer_config, dict):
            semantic_tokenizer_config["model_type"] = semantic_tokenizer_config.get(
                "model_type", "vibevoice_semantic_tokenizer"
            )
            semantic_tokenizer_config = CONFIG_MAPPING[semantic_tokenizer_config["model_type"]](
                **semantic_tokenizer_config
            )
        elif semantic_tokenizer_config is None:
            semantic_tokenizer_config = CONFIG_MAPPING["vibevoice_semantic_tokenizer"]()
        self.semantic_tokenizer_config = semantic_tokenizer_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()
        self.text_config = text_config

        self.vocab_size = text_config.vocab_size
        self.use_cache = use_cache
        self.speech_start_id = speech_start_id
        self.speech_end_id = speech_end_id
        self.speech_diffusion_id = speech_diffusion_id
        self.num_head_layers = num_head_layers
        self.head_ffn_ratio = head_ffn_ratio
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.frequency_embedding_size = frequency_embedding_size

        # NOTE (ebezzam) to use LlamaMLP via modular
        self.mlp_bias = False

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @property
    def intermediate_size(self) -> int:
        return self.hidden_size * self.head_ffn_ratio

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def acoustic_hidden_size(self) -> int:
        return self.acoustic_tokenizer_config.hidden_size

    @property
    def semantic_hidden_size(self) -> int:
        return self.semantic_tokenizer_config.hidden_size


__all__ = ["VibeVoiceConfig"]
