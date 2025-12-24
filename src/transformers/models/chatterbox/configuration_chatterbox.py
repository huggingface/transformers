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
"""Chatterbox model configuration"""

from ...configuration_utils import PretrainedConfig


# ============================================================================
# T3 Configuration
# ============================================================================

# Llama 520M configuration for T3 backbone
LLAMA_520M_CONFIG_DICT = {
    # Arbitrary small number that won't cause problems when loading.
    # These params are unused due to custom input layers.
    "vocab_size": 8,
    # Default params needed for loading most pretrained 1B weights
    "max_position_embeddings": 131072,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_hidden_layers": 30,
    "num_attention_heads": 16,
    "attn_implementation": "sdpa",
    "head_dim": 64,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "mlp_bias": False,
    "model_type": "llama",
    "num_key_value_heads": 16,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "torch_dtype": "bfloat16",
    "use_cache": True,
}


class T3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a T3 component. It is used internally by
    ChatterboxModel for the T3 (Token-To-Token) TTS component.

    T3 (Token-To-Token) is a TTS model that uses a LLaMA transformer backbone to generate speech tokens from text tokens.
    The speech tokens can then be decoded by S3Gen to produce mel spectrograms and finally waveforms.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_tokens_dict_size (`int`, *optional*, defaults to 704):
            Size of the text token vocabulary. Use 704 for English-only, 2454 for multilingual.
        speech_tokens_dict_size (`int`, *optional*, defaults to 8194):
            Size of the speech token vocabulary. Includes special tokens (start: 6561, stop: 6562).
        start_text_token (`int`, *optional*, defaults to 255):
            Token ID for start-of-text marker.
        stop_text_token (`int`, *optional*, defaults to 0):
            Token ID for end-of-text marker.
        start_speech_token (`int`, *optional*, defaults to 6561):
            Token ID for start-of-speech marker.
        stop_speech_token (`int`, *optional*, defaults to 6562):
            Token ID for end-of-speech marker.
        max_text_tokens (`int`, *optional*, defaults to 2048):
            Maximum number of text tokens in a sequence.
        max_speech_tokens (`int`, *optional*, defaults to 4096):
            Maximum number of speech tokens in a sequence.
        llama_config_name (`str`, *optional*, defaults to `"Llama_520M"`):
            Name of the LLaMA configuration to use as backbone.
        hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size of the transformer backbone (from LLaMA config).
        input_pos_emb (`str`, *optional*, defaults to `"learned"`):
            Type of positional embeddings. Currently only "learned" is supported.
        speech_cond_prompt_len (`int`, *optional*, defaults to 150):
            Length of speech conditioning prompt tokens.
        encoder_type (`str`, *optional*, defaults to `"voice_encoder"`):
            Type of speaker encoder to use.
        speaker_embed_size (`int`, *optional*, defaults to 256):
            Dimension of speaker embeddings from the voice encoder.
        use_perceiver_resampler (`bool`, *optional*, defaults to `True`):
            Whether to use perceiver resampler for conditioning prompts.
        perceiver_num_latents (`int`, *optional*, defaults to 32):
            Number of latent query tokens in the perceiver resampler.
        perceiver_latent_dim (`int`, *optional*, defaults to 1024):
            Dimension of latent tokens in the perceiver resampler.
        perceiver_num_heads (`int`, *optional*, defaults to 4):
            Number of attention heads in the perceiver resampler.
        emotion_adv (`bool`, *optional*, defaults to `True`):
            Whether to use emotion/exaggeration conditioning.
        use_alignment_analyzer (`bool`, *optional*):
            Whether to use alignment stream analyzer for multilingual models. If None, automatically enabled for multilingual.
        alignment_layer_idx (`int`, *optional*, defaults to 9):
            Layer index to use for attention-based alignment analysis in multilingual models.
    """

    model_type = "t3"

    def __init__(
        self,
        # Token vocabulary sizes
        text_tokens_dict_size=704,
        speech_tokens_dict_size=8194,
        # Special tokens
        start_text_token=255,
        stop_text_token=0,
        start_speech_token=6561,
        stop_speech_token=6562,
        # Sequence lengths
        max_text_tokens=2048,
        max_speech_tokens=4096,
        # LLaMA backbone config
        llama_config_name="Llama_520M",
        hidden_size=1024,
        # Positional embeddings
        input_pos_emb="learned",
        # Conditioning
        speech_cond_prompt_len=150,
        encoder_type="voice_encoder",
        speaker_embed_size=256,
        use_perceiver_resampler=True,
        perceiver_num_latents=32,
        perceiver_latent_dim=1024,
        perceiver_num_heads=4,
        emotion_adv=True,
        # Multilingual support
        use_alignment_analyzer=None,
        alignment_layer_idx=9,
        **kwargs,
    ):
        self.text_tokens_dict_size = text_tokens_dict_size
        self.speech_tokens_dict_size = speech_tokens_dict_size
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.start_speech_token = start_speech_token
        self.stop_speech_token = stop_speech_token
        self.max_text_tokens = max_text_tokens
        self.max_speech_tokens = max_speech_tokens
        self.llama_config_name = llama_config_name
        self.hidden_size = hidden_size
        self.input_pos_emb = input_pos_emb
        self.speech_cond_prompt_len = speech_cond_prompt_len
        self.encoder_type = encoder_type
        self.speaker_embed_size = speaker_embed_size
        self.use_perceiver_resampler = use_perceiver_resampler
        self.perceiver_num_latents = perceiver_num_latents
        self.perceiver_latent_dim = perceiver_latent_dim
        self.perceiver_num_heads = perceiver_num_heads
        self.emotion_adv = emotion_adv
        self.alignment_layer_idx = alignment_layer_idx

        # Auto-detect multilingual based on vocab size if not explicitly set
        if use_alignment_analyzer is None:
            self.use_alignment_analyzer = self.is_multilingual
        else:
            self.use_alignment_analyzer = use_alignment_analyzer

        # Store LLaMA config dict
        self.llama_config_dict = LLAMA_520M_CONFIG_DICT.copy()
        self.llama_config_dict["hidden_size"] = hidden_size

        super().__init__(**kwargs)

    @property
    def is_multilingual(self):
        """Check if this is a multilingual configuration based on vocab size."""
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls):
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704)

    @classmethod
    def multilingual(cls):
        """Create configuration for multilingual TTS model."""
        return cls(text_tokens_dict_size=2454)


# ============================================================================
# Chatterbox Configuration
# ============================================================================


class ChatterboxConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ChatterboxModel`]. It is used to instantiate a
    Chatterbox model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [ResembleAI/chatterbox-hf](https://huggingface.co/ResembleAI/chatterbox-hf).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Chatterbox is a complete TTS pipeline that combines T3, S3Gen, and HiFTNet models.

    Args:
        t3_config (`dict` or `T3Config`, *optional*):
            Dictionary or config object for T3 model. If not provided, uses English-only defaults.
        s3gen_config (`dict` or `S3GenConfig`, *optional*):
            Dictionary or config object for S3Gen model. If not provided, uses defaults.
        hiftnet_config (`dict` or `HiFTNetConfig`, *optional*):
            Dictionary or config object for HiFTNet model. If not provided, uses defaults.
        is_multilingual (`bool`, *optional*, defaults to `False`):
            Whether to use multilingual configuration.

    ```python
    >>> from transformers import ChatterboxConfig, ChatterboxModel

    >>> # Initializing a Chatterbox configuration
    >>> configuration = ChatterboxConfig()

    >>> # Initializing a model from the configuration
    >>> model = ChatterboxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "chatterbox"
    is_composition = True

    def __init__(
        self,
        t3_config=None,
        s3gen_config=None,
        hiftnet_config=None,
        is_multilingual=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        from ...models.s3gen.configuration_s3gen import HiFTNetConfig, S3GenConfig

        # Initialize sub-model configs
        # Handle both dict and Config object inputs
        if t3_config is None:
            if is_multilingual:
                self.t3_config = T3Config.multilingual()
            else:
                self.t3_config = T3Config.english_only()
        elif isinstance(t3_config, dict):
            self.t3_config = T3Config(**t3_config)
        else:
            self.t3_config = t3_config

        if s3gen_config is None:
            self.s3gen_config = S3GenConfig()
        elif isinstance(s3gen_config, dict):
            self.s3gen_config = S3GenConfig(**s3gen_config)
        else:
            self.s3gen_config = s3gen_config

        if hiftnet_config is None:
            self.hiftnet_config = HiFTNetConfig()
        elif isinstance(hiftnet_config, dict):
            self.hiftnet_config = HiFTNetConfig(**hiftnet_config)
        else:
            self.hiftnet_config = hiftnet_config

        self.is_multilingual = is_multilingual

    @classmethod
    def english_only(cls):
        """Create English-only configuration."""
        return cls(is_multilingual=False)

    @classmethod
    def multilingual(cls):
        """Create multilingual configuration."""
        return cls(is_multilingual=True)

    def to_dict(self):
        """Serialize to dict."""
        output = super().to_dict()
        output["t3_config"] = self.t3_config.to_dict()
        output["s3gen_config"] = self.s3gen_config.to_dict()
        output["hiftnet_config"] = self.hiftnet_config.to_dict()
        return output


__all__ = ["ChatterboxConfig"]
