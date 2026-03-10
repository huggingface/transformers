# Copyright 2021 The I-BERT Authors (Sehoon Kim, Amir Gholami, Zhewei Yao,
# Michael Mahoney, Kurt Keutzer - UC Berkeley) and The HuggingFace Inc. team.
# Copyright (c) 20121, NVIDIA CORPORATION.  All rights reserved.
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
"""I-BERT configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="kssteven/ibert-roberta-base")
class IBertConfig(PreTrainedConfig):
    r"""
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling [`IBertModel`]
    quant_mode (`bool`, *optional*, defaults to `False`):
        Whether to quantize the model or not.
    force_dequant (`str`, *optional*, defaults to `"none"`):
        Force dequantize specific nonlinear layer. Dequantized layers are then executed with full precision.
        `"none"`, `"gelu"`, `"softmax"`, `"layernorm"` and `"nonlinear"` are supported. As default, it is set as
        `"none"`, which does not dequantize any layers. Please specify `"gelu"`, `"softmax"`, or `"layernorm"` to
        dequantize GELU, Softmax, or LayerNorm, respectively. `"nonlinear"` will dequantize all nonlinear layers,
        i.e., GELU, Softmax, and LayerNorm.
    """

    model_type = "ibert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        quant_mode=False,
        force_dequant="none",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.quant_mode = quant_mode
        self.force_dequant = force_dequant


__all__ = ["IBertConfig"]
