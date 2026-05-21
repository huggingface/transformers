# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""Config classes for Granite Speech NAR (Non-Autoregressive ASR)."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-nar")
@strict
class GraniteSpeechNarEncoderConfig(PreTrainedConfig):
    r"""
    Configuration for the conformer encoder component of GraniteSpeechNar.

    feedforward_mult (`int`, *optional*, defaults to 4):
        Multiplier for the feedforward layers; intermediate dim = `hidden_dim * feedforward_mult`.
    output_dim (`int`, *optional*, defaults to 348):
        Output dimension of the mid-layer CTC prediction head.
    context_size (`int`, *optional*, defaults to 200):
        Context size for block-wise conformer attention.
    max_pos_emb (`int`, *optional*, defaults to 512):
        Maximum relative positional embedding index (Shaw's relative positional encoding).
    pred_dropout (`float`, *optional*, defaults to 0.25):
        Dropout applied to encoder hidden states before prediction heads.
    conv_expansion_factor (`int`, *optional*, defaults to 2):
        Expansion factor for conformer convolution module.
    self_conditioning_layer (`int`, *optional*):
        Layer index at which self-conditioning (mid-layer CTC feedback) is applied.
        Defaults to `num_layers // 2`.
    bpe_output_dim (`int`, *optional*):
        Vocabulary size for the BPE CTC head (same as LLM vocab, blank reuses eos_token_id). If None, BPE head is disabled.
    bpe_pooling_window (`int`, *optional*, defaults to 4):
        Window size for posterior-weighted pooling before the BPE CTC head.
    blank_token_id (`int`, *optional*):
        Token ID used as the CTC blank symbol. Defaults to the language model's `eos_token_id` if not set.

    Example:

    ```python
    >>> from transformers import GraniteSpeechNarEncoderConfig

    >>> configuration = GraniteSpeechNarEncoderConfig()
    >>> print(configuration.hidden_dim)
    1024
    ```"""

    model_type = "granite_speech_nar_encoder"
    attribute_map = {
        "hidden_size": "hidden_dim",
        "num_hidden_layers": "num_layers",
        "num_attention_heads": "num_heads",
        "num_mel_bins": "input_dim",
    }

    input_dim: int = 160
    num_layers: int = 16
    hidden_dim: int = 1024
    feedforward_mult: int = 4
    num_heads: int = 8
    dim_head: int | None = None
    output_dim: int = 348
    context_size: int = 200
    max_pos_emb: int = 512
    dropout: float = 0.1
    pred_dropout: float = 0.25
    conv_kernel_size: int = 15
    conv_expansion_factor: int = 2
    self_conditioning_layer: int | None = None
    bpe_output_dim: int | None = None
    bpe_pooling_window: int = 4
    blank_token_id: int | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.dim_head is None:
            self.dim_head = self.hidden_dim // self.num_heads
        if self.self_conditioning_layer is None:
            self.self_conditioning_layer = self.num_layers // 2


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-nar")
@strict
class GraniteSpeechNarProjectorConfig(PreTrainedConfig):
    r"""
    Configuration for the QFormer-based audio projector in GraniteSpeechNar.

    encoder_dim (`int`, *optional*, defaults to 1024):
        Hidden dimension of the encoder (per layer).
    llm_dim (`int`, *optional*, defaults to 2048):
        Hidden dimension of the language model.
    downsample_rate (`int`, *optional*, defaults to 5):
        Temporal downsampling rate within each window block.
    num_encoder_layers (`int`, *optional*, defaults to 4):
        Number of encoder layers concatenated as projector input.
    block_size (`int`, *optional*, defaults to 15):
        Window size for blocked cross-attention in the projector.
    layernorm_eps (`float`, *optional*, defaults to 1e-6):
        Epsilon for layer normalization.
    attn_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in attention projections.

    Example:

    ```python
    >>> from transformers import GraniteSpeechNarProjectorConfig

    >>> configuration = GraniteSpeechNarProjectorConfig()
    >>> print(configuration.hidden_size)
    2048
    ```"""

    model_type = "granite_speech_nar_projector"

    encoder_dim: int = 1024
    llm_dim: int = 2048
    downsample_rate: int = 5
    num_encoder_layers: int = 4
    hidden_size: int = 2048
    num_heads: int = 32
    num_layers: int = 2
    dropout_prob: float = 0.1
    block_size: int = 15
    mlp_ratio: int = 2
    layernorm_eps: float = 1e-6
    attn_bias: bool = True
    mlp_bias: bool = True


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-nar")
@strict
class GraniteSpeechNarConfig(PreTrainedConfig):
    r"""
    Configuration for the GraniteSpeechNar non-autoregressive ASR model.

    This model uses a conformer encoder with BPE CTC head, a QFormer-based projector,
    and a bidirectional Granite LLM backbone for single-pass speech recognition.

    projector_config (`GraniteSpeechNarProjectorConfig` or `dict`, *optional*):
        Configuration for the QFormer-based audio projector.
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether the LLM's input and output word embeddings should be tied.
    encoder_layer_indices (`list[int]`, *optional*, defaults to `[4, 8, 12, -1]`):
        Indices of encoder layers whose hidden states are concatenated as projector input.
    scale_projected_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to divide projected audio embeddings by the LLM's embedding multiplier.
    blank_token_id (`int`, *optional*):
        Token ID used as the CTC blank symbol. Defaults to `text_config.eos_token_id`.
    min_edit_sequence_length (`int`, *optional*, defaults to 8):
        Minimum length of the edit sequence (CTC tokens + insertion slots) fed to the LLM.
    ce_loss_lambda (`float`, *optional*, defaults to 0.0):
        Weight for auxiliary cross-entropy loss on the LLM output.
    encoder_ctc_loss_lambda (`float`, *optional*, defaults to 0.0):
        Weight for auxiliary encoder BPE CTC loss.

    Example:

    ```python
    >>> from transformers import GraniteSpeechNarConfig, GraniteSpeechNarForASR

    >>> configuration = GraniteSpeechNarConfig()
    >>> model = GraniteSpeechNarForASR(configuration)
    >>> print(configuration.model_type)
    'granite_speech_nar'
    ```"""

    model_type = "granite_speech_nar"
    sub_configs = {
        "encoder_config": GraniteSpeechNarEncoderConfig,
        "projector_config": GraniteSpeechNarProjectorConfig,
        "text_config": "AutoConfig",
    }

    encoder_config: dict | PreTrainedConfig | None = None
    projector_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    tie_word_embeddings: bool = True
    encoder_layer_indices: list[int] | None = None
    scale_projected_embeddings: bool = True
    blank_token_id: int | None = None
    min_edit_sequence_length: int = 8
    ce_loss_lambda: float = 0.0
    encoder_ctc_loss_lambda: float = 0.0

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "granite")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["granite"]()

        if not isinstance(self.encoder_config, GraniteSpeechNarEncoderConfig):
            self.encoder_config = GraniteSpeechNarEncoderConfig(**(self.encoder_config or {}))

        if not isinstance(self.projector_config, GraniteSpeechNarProjectorConfig):
            self.projector_config = GraniteSpeechNarProjectorConfig(**(self.projector_config or {}))

        if self.encoder_layer_indices is None:
            self.encoder_layer_indices = [4, 8, 12, -1]

        if self.blank_token_id is None:
            self.blank_token_id = self.text_config.eos_token_id

        # Propagate blank_token_id to encoder config
        self.encoder_config.blank_token_id = self.blank_token_id

        super().__post_init__(**kwargs)


__all__ = ["GraniteSpeechNarEncoderConfig", "GraniteSpeechNarProjectorConfig", "GraniteSpeechNarConfig"]
