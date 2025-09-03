# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2025 the HuggingFace Team. All rights reserved.

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class PanguEmbeddedConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PanguEmbeddedConfig`]. It is used to instantiate an
    openPangu-Embedded model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [openPangu-Embedded-7B](https://ai.gitcode.com/ascend-tribe/openpangu-embedded-7b-model).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "openpangu_dense"
    _auto_class = "AutoConfig"

    def __init__(
        self,
        vocab_size=153376,
        hidden_size=4096,
        intermediate_size=12800,
        num_hidden_layers=34,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=45892,
        tie_word_embeddings=False,
        rope_theta=16000000.0,
        bias=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.bias = bias
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["PanguEmbeddedConfig"]
