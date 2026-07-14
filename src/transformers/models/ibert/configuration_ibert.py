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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="kssteven/ibert-roberta-base")
@strict
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

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    quant_mode: bool = False
    force_dequant: str = "none"
    tie_word_embeddings: bool = True


__all__ = ["IBertConfig"]
