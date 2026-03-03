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
"""Blt model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import logging


logger = logging.get_logger(__name__)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class BltLocalEncoderConfig(PreTrainedConfig):
    """
    Configuration class for the Blt Local Encoder component.
    """

    model_type = "blt_local_encoder"
    default_theta = 500000.0

    vocab_size: int = 260
    cross_attn_all_layers: bool | None = False
    cross_attn_k: int | None = 2
    hidden_size_global: int | None = 2048
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    num_hidden_layers: int = 1
    rms_norm_eps: float = 1e-5
    dropout: float | int | None = 0.0
    max_position_embeddings: int = 24576
    rope_parameters: RopeParameters | dict | None = None
    hidden_act: str = "silu"
    intermediate_size: int | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.intermediate_size = self.intermediate_size or int(8 * self.hidden_size / 3)
        self.tie_word_embeddings = False
        super().__post_init__(**kwargs)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class BltLocalDecoderConfig(PreTrainedConfig):
    """
    Configuration class for the Blt Local Decoder component.
    """

    model_type = "blt_local_decoder"
    default_theta = 500000.0

    vocab_size: int = 260
    cross_attn_all_layers: bool | None = True
    cross_attn_k: int | None = 2
    hidden_size_global: int | None = 2048
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    num_hidden_layers: int = 9
    rms_norm_eps: float = 1e-5
    dropout: float | int | None = 0.0
    max_position_embeddings: int = 24576
    rope_parameters: RopeParameters | dict | None = None
    hidden_act: str = "silu"
    intermediate_size: int = 2816
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = self.intermediate_size or int(8 * self.hidden_size / 3)
        self.tie_word_embeddings = False  # Force-set to False for BC
        super().__post_init__(**kwargs)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class BltGlobalTransformerConfig(PreTrainedConfig):
    """
    Configuration class for the Blt Global Transformer component.
    """

    model_type = "blt_global_transformer"
    default_theta = 500000.0

    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    num_hidden_layers: int = 25
    rms_norm_eps: float = 1e-5
    dropout: float | int | None = 0.0
    max_position_embeddings: int = 4096
    rope_parameters: RopeParameters | dict | None = None
    hidden_act: str = "silu"
    intermediate_size: int = 5632
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_key_value_heads or self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = self.intermediate_size or int(8 * self.hidden_size / 3)
        self.tie_word_embeddings = False

        super().__post_init__(**kwargs)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class BltPatcherConfig(PreTrainedConfig):
    r"""
    Configuration class for the Blt Patcher/Entropy model component.

    Args:
        vocab_size (`int`, *optional*, defaults to 260):
            Vocabulary size of the Blt patcher model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling the patcher model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 14):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP representations.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "blt_patcher"

    vocab_size: int = 260
    hidden_size: int = 768
    num_hidden_layers: int = 14
    num_attention_heads: int = 12
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    dropout: float | int | None = 0.0
    intermediate_size: int = 2048
    rope_parameters: RopeParameters | dict | None = None
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_key_value_heads or self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = self.intermediate_size or int(8 * self.hidden_size / 3)
        self.tie_word_embeddings = False
        self.hidden_act = "silu"  # Blt uses silu activation

        super().__post_init__(**kwargs)


@strict(accept_kwargs=True)
@dataclass(repr=False)
class BltConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BltModel`]. It is used to instantiate a
    Blt model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 260):
            Vocabulary size of the Blt model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BltModel`].
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        patch_in_forward (`bool`, *optional*, defaults to `True`):
            Whether to perform patching during the forward pass.
        patch_size (`int`, *optional*, defaults to 4):
            Size of the patches used in the patching mechanism.
        patching_mode (`str`, *optional*, defaults to `"entropy"`):
            The mode used for patching, such as entropy-based patching.
        patching_threshold (`float`, *optional*, defaults to 1.34):
            Threshold value used for determining when to apply patches.
        patching_batch_size (`int`, *optional*, defaults to 1):
            Batch size used during the patching process.
        max_patch_length (`int`, *optional*):
            Maximum length of patches that can be generated.
        cross_attn_k (`int`, *optional*, defaults to 2):
            Number of cross-attention heads used in the model.
        encoder_hash_byte_group_size (`list`, *optional*):
            List of byte group sizes used in the encoder hash function.
        encoder_hash_byte_group_vocab (`int`, *optional*, defaults to 500002):
            Vocabulary size for the encoder hash byte groups.
        encoder_hash_byte_group_nb_functions (`int`, *optional*, defaults to 1):
            Number of hash functions used in the encoder byte grouping.
        patcher_config (`BltPatcherConfig`, *optional*):
            Configuration for the patcher component of the model.
        encoder_config (`BltLocalEncoderConfig`, *optional*):
            Configuration for the local encoder component of the model.
        decoder_config (`BltLocalDecoderConfig`, *optional*):
            Configuration for the local decoder component of the model.
        global_config (`BltGlobalTransformerConfig`, *optional*):
            Configuration for the global transformer component of the model.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.

    ```python
    >>> from transformers import BltModel, BltConfig

    >>> # Initializing a Blt configuration
    >>> configuration = BltConfig()

    >>> # Initializing a model from the configuration
    >>> model = BltModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    Checkpoint: [facebook/blt](https://huggingface.co/facebook/blt)
    """

    model_type = "blt"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0
    sub_configs = {
        "patcher_config": BltPatcherConfig,
        "encoder_config": BltLocalEncoderConfig,
        "decoder_config": BltLocalDecoderConfig,
        "global_config": BltGlobalTransformerConfig,
    }

    vocab_size: int = 260
    max_position_embeddings: int = 4096
    patch_in_forward: bool | None = True
    patch_size: int | None = 4
    patching_mode: str | None = "entropy"
    patching_threshold: float | None = 1.335442066192627
    patching_batch_size: int | None = 1
    max_patch_length: int | None = None
    cross_attn_k: int | None = 2
    encoder_hash_byte_group_size: list[int] | None = None
    encoder_hash_byte_group_vocab: int | None = 500002
    encoder_hash_byte_group_nb_functions: int | None = 1
    patcher_config: dict | PreTrainedConfig | None = None
    encoder_config: dict | PreTrainedConfig | None = None
    decoder_config: dict | PreTrainedConfig | None = None
    global_config: dict | PreTrainedConfig | None = None
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    initializer_range: float = 0.02
    rope_parameters: RopeParameters | dict | None = None

    def __post_init__(self, **kwargs):
        self.encoder_hash_byte_group_size = self.encoder_hash_byte_group_size or [3, 4, 5, 6, 7, 8]

        # Initialize component configurations
        if self.patcher_config is None:
            self.patcher_config = BltPatcherConfig(initializer_range=self.initializer_range)
            logger.info("patcher_config is None, using default Blt patcher config")
        elif isinstance(self.patcher_config, dict):
            self.patcher_config.setdefault("initializer_range", self.initializer_range)
            self.patcher_config = BltPatcherConfig(**self.patcher_config)

        if self.encoder_config is None:
            self.encoder_config = BltLocalEncoderConfig(initializer_range=self.initializer_range)
            logger.info("encoder_config is None, using default Blt encoder config")
        elif isinstance(self.encoder_config, dict):
            self.encoder_config.setdefault("initializer_range", self.initializer_range)
            self.encoder_config = BltLocalEncoderConfig(**self.encoder_config)

        if self.decoder_config is None:
            self.decoder_config = BltLocalDecoderConfig(initializer_range=self.initializer_range)
            logger.info("decoder_config is None, using default Blt decoder config")
        elif isinstance(self.decoder_config, dict):
            self.decoder_config.setdefault("initializer_range", self.initializer_range)
            self.decoder_config = BltLocalDecoderConfig(**self.decoder_config)

        if self.global_config is None:
            self.global_config = BltGlobalTransformerConfig(initializer_range=self.initializer_range)
            logger.info("global_config is None, using default Blt global config")
        elif isinstance(self.global_config, dict):
            self.global_config.setdefault("initializer_range", self.initializer_range)
            self.global_config = BltGlobalTransformerConfig(**self.global_config)

        # Determine if token embedding projection is needed based on dimension mismatch (7b)
        encoder_cross_output_size = self.encoder_config.hidden_size * self.cross_attn_k
        self.global_config.encoder_cross_output_size = (
            encoder_cross_output_size if encoder_cross_output_size != self.global_config.hidden_size else None
        )

        super().__post_init__(**kwargs)


__all__ = [
    "BltConfig",
    "BltPatcherConfig",
    "BltLocalEncoderConfig",
    "BltLocalDecoderConfig",
    "BltGlobalTransformerConfig",
]
