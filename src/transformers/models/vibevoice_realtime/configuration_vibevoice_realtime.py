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



class VibeVoiceRealTimeAcousticDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceRealTimeAcousticDecoder`]. It is used to
    instantiate a VibeVoice real-time acoustic decoder according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    real-time decoder of [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B).

    Args:
        channels (`int`, *optional*, defaults to 1):
            Number of input channels.
        hidden_size (`int`, *optional*, defaults to 64):
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
        upsampling_ratios (`List[int]`, *optional*, defaults to `[8, 5, 5, 4, 2, 2]`):
            Upsampling ratios for each layer.
        decoder_depths (`List[int]`, *optional*, defaults to `[8, 3, 3, 3, 3, 3, 3]`):
            Number of ConvNeXt blocks at each stage.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function to use.
        ffn_expansion (`int`, *optional*, defaults to 4):
            Expansion factor for feed-forward networks.
    Example:

    ```python
    >>> from transformers import VibeVoiceRealTimeAcousticDecoder, VibeVoiceRealTimeAcousticDecoderConfig

    >>> # Initializing a VibeVoice real-time decoder configuration
    >>> configuration = VibeVoiceRealTimeAcousticDecoderConfig()

    >>> # Initializing a model (with random weights)
    >>> model = VibeVoiceRealTimeAcousticDecoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice_realtime_acoustic_decoder"

    def __init__(
        self,
        channels=1,
        hidden_size=64,
        kernel_size=7,
        rms_norm_eps=1e-5,
        bias=True,
        layer_scale_init_value=1e-6,
        weight_init_value=1e-2,
        n_filters=32,
        upsampling_ratios=[8, 5, 5, 4, 2, 2],
        decoder_depths=[8, 3, 3, 3, 3, 3, 3],
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
        self.upsampling_ratios = upsampling_ratios
        self.decoder_depths = decoder_depths


class VibeVoiceRealTimeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VibeVoiceRealTimeForConditionalGeneration`]. It is
    used to instantiate an VibeVoice model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults similar to that of
    [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        acoustic_tokenizer_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the acoustic tokenizer.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text model.
        tts_backbone_num_hidden_layers (`int`, *optional*, defaults to 20):
            Two language models are created according to `text_config`. `tts_backbone_num_hidden_layers` specifies the
            number of upper Transformer layers used for encoding text and generating speech. The remaining lower
            Transformer layers are only used for encoding text.
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
    >>> from transformers import VibeVoiceRealTimeForConditionalGeneration, VibeVoiceRealTimeConfig

    >>> # Initializing a VibeVoiceRealTime configuration
    >>> configuration = VibeVoiceRealTimeConfig(audio_token_id=24, projector_hidden_act="gelu")

    >>> # Initializing a 0.5B model with random weights
    >>> model = VibeVoiceRealTimeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vibevoice_realtime"
    is_composition = True

    sub_configs = {
        "acoustic_tokenizer_config": AutoConfig,
        "text_config": AutoConfig,
        "tts_text_config": AutoConfig,
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
        text_config=None,
        tts_text_config=None,
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
                "model_type", "vibevoice_realtime_acoustic_decoder"
            )
            acoustic_tokenizer_config = CONFIG_MAPPING[acoustic_tokenizer_config["model_type"]](**acoustic_tokenizer_config)
        elif acoustic_tokenizer_config is None:
            acoustic_tokenizer_config = CONFIG_MAPPING["vibevoice_realtime_acoustic_decoder"]()
        self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            # Qwen2.5-0.5B but with 4 hidden layers
            text_config = CONFIG_MAPPING["qwen2"](
                hidden_size=896,
                intermediate_size=4864,
                num_hidden_layers=4,
                num_attention_heads=14,
                num_key_value_heads=2,
                max_position_embeddings=8192,
                max_window_layers=24,
                sliding_window=None,
                rope_parameters={"rope_theta": 1_000_000.0},
            )
        self.text_config = text_config
        
        if isinstance(tts_text_config, dict):
            tts_text_config["model_type"] = tts_text_config.get("model_type", "qwen2")
            tts_text_config = CONFIG_MAPPING[tts_text_config["model_type"]](**tts_text_config)
        elif tts_text_config is None:
            # Qwen2.5-0.5B but with 20 hidden layers
            tts_text_config = CONFIG_MAPPING["qwen2"](
                hidden_size=896,
                intermediate_size=4864,
                num_hidden_layers=20,
                num_attention_heads=14,
                num_key_value_heads=2,
                max_position_embeddings=8192,
                max_window_layers=24,
                sliding_window=None,
                rope_parameters={"rope_theta": 1_000_000.0},
            )
        self.tts_text_config = tts_text_config

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
    def weight_init_value(self) -> float:
        return self.acoustic_tokenizer_config.weight_init_value

    @property
    def layer_scale_init_value(self) -> float:
        return self.acoustic_tokenizer_config.layer_scale_init_value

    @property
    def intermediate_size(self) -> int:
        return self.hidden_size * self.head_ffn_ratio

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def acoustic_hidden_size(self) -> int:
        return self.acoustic_tokenizer_config.hidden_size


__all__ = ["VibeVoiceRealTimeConfig", "VibeVoiceRealTimeAcousticDecoderConfig"]
