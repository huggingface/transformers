# Copyright 2026 The Microsoft Team and The HuggingFace Inc. team. All rights reserved.
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


class VibeVoiceSemanticTokenizerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceSemanticTokenizerModel`]. It is used to
    instantiate a VibeVoice semantic tokenizer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    semantic tokenizer of [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B).

    Args:
        channels (`int`, *optional*, defaults to 1):
            Number of input channels.
        hidden_size (`int`, *optional*, defaults to 128):
            Dimensionality of latent representations.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for convolutional layers.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon value for RMSNorm layers.
        bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in convolution and feed-forward layers.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-06):
            Initial value for layer scaling.
        weight_init_value (`float`, *optional*, defaults to 0.01):
            Standard deviation for weight initialization.
        n_filters (`int`, *optional*, defaults to 32):
            Number of filters in initial convolutional layer, and doubles after each downsampling.
        downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 2, 4, 5, 5, 8]`):
            Downsampling ratios for each layer.
        depths (`List[int]`, *optional*, defaults to `[3, 3, 3, 3, 3, 3, 8]`):
            Number of ConvNeXt blocks at each stage.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function to use.
        ffn_expansion (`int`, *optional*, defaults to 4):
            Expansion factor for feed-forward networks.
    Example:

    ```python
    >>> from transformers import VibeVoiceSemanticTokenizerModel, VibeVoiceSemanticTokenizerConfig

    >>> # Initializing a VibeVoice Semantic Tokenizer configuration
    >>> configuration = VibeVoiceSemanticTokenizerConfig()

    >>> # Initializing a model (with random weights)
    >>> model = VibeVoiceSemanticTokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice_semantic_tokenizer"

    def __init__(
        self,
        channels=1,
        hidden_size=128,
        kernel_size=7,
        rms_norm_eps=1e-5,
        bias=True,
        layer_scale_init_value=1e-6,
        weight_init_value=1e-2,
        n_filters=32,
        downsampling_ratios=[2, 2, 4, 5, 5, 8],
        depths=[3, 3, 3, 3, 3, 3, 8],
        hidden_act="gelu",
        ffn_expansion=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.kernel_size = kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.bias = bias
        self.layer_scale_init_value = layer_scale_init_value
        self.ffn_expansion = ffn_expansion
        self.weight_init_value = weight_init_value
        self.n_filters = n_filters
        self.downsampling_ratios = downsampling_ratios
        self.depths = depths


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
        pad_token_id (`int`, *optional*, defaults to 151643):
            The token ID for padding.
        eos_token_id (`int`, *optional*, defaults to 151643):
            The token ID for the end of sequence.
        audio_bos_token_id (`int`, *optional*, defaults to 151652):
            The token ID indicating the start of audio tokens.
        audio_eos_token_id (`int`, *optional*, defaults to 151653):
            The token ID indicating the end of audio tokens.
        audio_diffusion_token_id (`int`, *optional*, defaults to 151654):
            The token ID indicating the start of audio diffusion tokens.
        num_head_layers (`int`, *optional*, defaults to 4):
            Number of layers in the diffusion head.
        intermediate_size (`int`, *optional*, defaults to 4608):
            The intermediate size of the feed-forward layers.
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
        pad_token_id=151643,
        eos_token_id=151643,
        audio_bos_token_id=151652,
        audio_eos_token_id=151653,
        audio_diffusion_token_id=151654,
        num_head_layers=4,
        intermediate_size=4608,
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
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_diffusion_token_id = audio_diffusion_token_id
        self.num_head_layers = num_head_layers
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.frequency_embedding_size = frequency_embedding_size

        # NOTE (ebezzam) to use LlamaMLP via modular
        self.mlp_bias = False
        self.intermediate_size = intermediate_size

        kwargs.pop("tie_word_embeddings", None)  # remove if present to take priority from text_config
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=getattr(text_config, "tie_word_embeddings", False),
            **kwargs,
        )

    @property
    def initializer_range(self) -> float:
        return self.acoustic_tokenizer_config.initializer_range

    @property
    def layer_scale_init_value(self) -> float:
        return self.acoustic_tokenizer_config.layer_scale_init_value

    # NOTE (ebezzam) for modular usage of `LlamaMLP`
    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size


__all__ = ["VibeVoiceConfig", "VibeVoiceSemanticTokenizerConfig"]
