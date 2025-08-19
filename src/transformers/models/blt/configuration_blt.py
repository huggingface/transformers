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
"""Blt model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class BltLocalEncoderConfig(PretrainedConfig):
    """
    Configuration class for the Blt Local Encoder component.
    """

    model_type = "blt_local_encoder"

    def __init__(
        self,
        vocab_size=260,
        cross_attn_all_layers=False,
        cross_attn_k=2,
        hidden_size_global=2048,
        pm_size=0,
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=None,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
        dropout=0.0,
        max_position_embeddings=24576,
        rope_theta=500000.0,
        rope_scaling=None,
        hidden_act="silu",
        intermediate_size=2816,
        initializer_range=0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.cross_attn_all_layers = cross_attn_all_layers
        self.cross_attn_k = cross_attn_k
        self.hidden_size_global = hidden_size_global
        self.pm_size = pm_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size or int(8 * hidden_size / 3)
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


class BltLocalDecoderConfig(PretrainedConfig):
    """
    Configuration class for the Blt Local Decoder component.
    """

    model_type = "blt_local_decoder"

    def __init__(
        self,
        vocab_size=260,
        cross_attn_all_layers=True,
        cross_attn_k=2,
        hidden_size_global=2048,
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=None,
        num_hidden_layers=9,
        rms_norm_eps=1e-5,
        dropout=0.0,
        max_position_embeddings=24576,
        rope_theta=500000.0,
        rope_scaling=None,
        hidden_act="silu",
        intermediate_size=2816,
        initializer_range=0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.cross_attn_all_layers = cross_attn_all_layers
        self.cross_attn_k = cross_attn_k
        self.hidden_size_global = hidden_size_global
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size or int(8 * hidden_size / 3)
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


class BltGlobalTransformerConfig(PretrainedConfig):
    """
    Configuration class for the Blt Global Transformer component.
    """

    model_type = "blt_global_transformer"

    def __init__(
        self,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=None,
        num_hidden_layers=25,
        rms_norm_eps=1e-5,
        dropout=0.0,
        max_position_embeddings=4096,
        rope_theta=500000.0,
        rope_scaling=None,
        hidden_act="silu",
        intermediate_size=5632,
        initializer_range=0.02,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size or int(8 * hidden_size / 3)
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


class BltPatcherConfig(PretrainedConfig):
    r"""
    Configuration class for the Blt Patcher/Entropy model component.

    Args:
            vocab_size (`<fill_type>`, *optional*, defaults to 260): <fill_docstring>
            hidden_size (`<fill_type>`, *optional*, defaults to 768): <fill_docstring>
            num_hidden_layers (`<fill_type>`, *optional*, defaults to 14): <fill_docstring>
            num_attention_heads (`<fill_type>`, *optional*, defaults to 12): <fill_docstring>
            num_key_value_heads (`<fill_type>`, *optional*): <fill_docstring>
            max_position_embeddings (`<fill_type>`, *optional*, defaults to 8192): <fill_docstring>
            rms_norm_eps (`<fill_type>`, *optional*, defaults to 1e-05): <fill_docstring>
            dropout (`<fill_type>`, *optional*, defaults to 0.0): <fill_docstring>
            rope_theta (`<fill_type>`, *optional*, defaults to 10000.0): <fill_docstring>
            attn_bias_type (`<fill_type>`, *optional*, defaults to `"local_block_causal"`): <fill_docstring>
            intermediate_size (`<fill_type>`, *optional*, defaults to 2048): <fill_docstring>
            rope_scaling (`<fill_type>`, *optional*): <fill_docstring>
            initializer_range (`<fill_type>`, *optional*, defaults to 0.02): <fill_docstring>
    """

    model_type = "blt_patcher"

    def __init__(
        self,
        vocab_size=260,
        hidden_size=768,
        num_hidden_layers=14,
        num_attention_heads=12,
        num_key_value_heads=None,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        dropout=0.0,
        rope_theta=10000.0,
        attn_bias_type="local_block_causal",
        intermediate_size=2048,
        rope_scaling=None,
        initializer_range=0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.attn_bias_type = attn_bias_type
        self.hidden_act = "silu"  # Blt uses silu activation
        self.intermediate_size = intermediate_size or int(8 * self.hidden_size / 3)
        self.rope_scaling = rope_scaling
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


class BltConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BltModel`]. It is used to instantiate a
    Blt model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
            vocab_size (`<fill_type>`, *optional*, defaults to 260): <fill_docstring>
            max_position_embeddings (`<fill_type>`, *optional*, defaults to 4096): <fill_docstring>
            patch_in_forward (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
            patch_size (`<fill_type>`, *optional*, defaults to 4): <fill_docstring>
            patching_mode (`<fill_type>`, *optional*, defaults to `"entropy"`): <fill_docstring>
            patching_threshold (`<fill_type>`, *optional*, defaults to 1.34): <fill_docstring>
            patching_batch_size (`<fill_type>`, *optional*, defaults to 1): <fill_docstring>
            max_patch_length (`<fill_type>`, *optional*): <fill_docstring>
            cross_attn_k (`<fill_type>`, *optional*, defaults to 2): <fill_docstring>
            encoder_hash_byte_group_size (`<fill_type>`, *optional*): <fill_docstring>
            encoder_hash_byte_group_vocab (`<fill_type>`, *optional*, defaults to 500002): <fill_docstring>
            encoder_hash_byte_group_nb_functions (`<fill_type>`, *optional*, defaults to 1): <fill_docstring>
            patcher_config (`<fill_type>`, *optional*): <fill_docstring>
            encoder_config (`<fill_type>`, *optional*): <fill_docstring>
            decoder_config (`<fill_type>`, *optional*): <fill_docstring>
            global_config (`<fill_type>`, *optional*): <fill_docstring>
            tie_word_embeddings (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
            initializer_range (`<fill_type>`, *optional*, defaults to 0.02): <fill_docstring>
            rope_theta (`<fill_type>`, *optional*, defaults to 500000.0): <fill_docstring>
            rope_scaling (`<fill_type>`, *optional*): <fill_docstring>

    ```python
    >>> from transformers import BltModel, BltConfig

    >>> # Initializing a Blt configuration
    >>> configuration = BltConfig()

    >>> # Initializing a model from the configuration
    >>> model = BltModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blt"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "patcher_config": BltPatcherConfig,
        "encoder_config": BltLocalEncoderConfig,
        "decoder_config": BltLocalDecoderConfig,
        "global_config": BltGlobalTransformerConfig,
    }

    def __init__(
        self,
        vocab_size=260,
        max_position_embeddings=4096,
        patch_in_forward=True,
        patch_size=4,
        patching_mode="entropy",
        patching_threshold=1.335442066192627,
        patching_batch_size=1,
        max_patch_length=None,
        cross_attn_k=2,
        encoder_hash_byte_group_size=None,
        encoder_hash_byte_group_vocab=500002,
        encoder_hash_byte_group_nb_functions=1,
        patcher_config=None,
        encoder_config=None,
        decoder_config=None,
        global_config=None,
        tie_word_embeddings=False,
        initializer_range=0.02,
        rope_theta=500000.0,
        rope_scaling=None,
        **kwargs,
    ):
        # Basic model configuration
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # Patching configuration
        self.patch_in_forward = patch_in_forward
        self.patch_size = patch_size
        self.patching_mode = patching_mode
        self.patching_threshold = patching_threshold
        self.patching_batch_size = patching_batch_size
        self.max_patch_length = max_patch_length
        self.patching_device = kwargs.get("patching_device", "cuda")
        self.realtime_patching = kwargs.get("realtime_patching", True)
        self.patching_threshold_add = kwargs.get("patching_threshold_add")
        self.monotonicity = kwargs.get("monotonicity", False)
        self.pm_size = kwargs.get("pm_size", 0)

        # Cross attention configurations
        self.cross_attn_k = cross_attn_k

        # Encoder configurations
        self.encoder_hash_byte_group_size = encoder_hash_byte_group_size or [3, 4, 5, 6, 7, 8]
        self.encoder_hash_byte_group_vocab = encoder_hash_byte_group_vocab
        self.encoder_hash_byte_group_nb_functions = encoder_hash_byte_group_nb_functions

        # Initialize component configurations
        if patcher_config is None:
            self.patcher_config = BltPatcherConfig(initializer_range=initializer_range)
            logger.info("patcher_config is None, using default Blt patcher config")
        elif isinstance(patcher_config, dict):
            patcher_config.setdefault("initializer_range", initializer_range)
            self.patcher_config = BltPatcherConfig(**patcher_config)
        elif isinstance(patcher_config, BltPatcherConfig):
            self.patcher_config = patcher_config

        if encoder_config is None:
            self.encoder_config = BltLocalEncoderConfig(initializer_range=initializer_range)
            logger.info("encoder_config is None, using default Blt encoder config")
        elif isinstance(encoder_config, dict):
            encoder_config.setdefault("initializer_range", initializer_range)
            self.encoder_config = BltLocalEncoderConfig(**encoder_config)
        elif isinstance(encoder_config, BltLocalEncoderConfig):
            self.encoder_config = encoder_config

        if decoder_config is None:
            self.decoder_config = BltLocalDecoderConfig(initializer_range=initializer_range)
            logger.info("decoder_config is None, using default Blt decoder config")
        elif isinstance(decoder_config, dict):
            decoder_config.setdefault("initializer_range", initializer_range)
            self.decoder_config = BltLocalDecoderConfig(**decoder_config)
        elif isinstance(decoder_config, BltLocalDecoderConfig):
            self.decoder_config = decoder_config

        if global_config is None:
            self.global_config = BltGlobalTransformerConfig(initializer_range=initializer_range)
            logger.info("global_config is None, using default Blt global config")
        elif isinstance(global_config, dict):
            global_config.setdefault("initializer_range", initializer_range)
            self.global_config = BltGlobalTransformerConfig(**global_config)
        elif isinstance(global_config, BltGlobalTransformerConfig):
            self.global_config = global_config

        # Determine if token embedding projection is needed based on dimension mismatch (7b)
        encoder_cross_output_size = self.encoder_config.hidden_size * self.cross_attn_k
        self.global_config.encoder_cross_output_size = (
            encoder_cross_output_size if encoder_cross_output_size != self.global_config.hidden_size else None
        )

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = [
    "BltConfig",
    "BltPatcherConfig",
    "BltLocalEncoderConfig",
    "BltLocalDecoderConfig",
    "BltGlobalTransformerConfig",
]
