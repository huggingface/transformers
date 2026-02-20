# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)


class CircuitGptConfig(PretrainedConfig):
    model_type = "circuit_gpt"

    def __init__(
        self,
        vocab_size=50257,
        n_embd=768,
        n_layer=12,
        n_head=12,
        sparsity=0.0,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.sparsity = sparsity
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon

        super().__init__(**kwargs)
