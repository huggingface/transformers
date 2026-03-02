# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
"""HunYuanMoEV1 model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="tencent/Hunyuan-A13B-Instruct")
class HunYuanMoEV1Config(PreTrainedConfig):
    r"""
    eod_token_id (int, *optional*, defaults to 3):
        Token ID representing the end-of-document marker. Used to indicate the termination of a text sequence.
        For Example, in multi-document processing, this token helps the model distinguish between separate documents.
    moe_topk (`int | list`, *optional*, defaults to 1):
        Number of experts selected per token (Top-K routing). List form enables layer-wise customization.
    """

    model_type = "hunyuan_v1_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_experts_per_tok": "moe_topk",
        "num_local_experts": "num_experts",
    }

    def __init__(
        self,
        vocab_size: int | None = 290943,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 11008,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 2048,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        eod_token_id: int | None = 3,
        sep_token_id: int | None = 4,
        pretraining_tp: int | None = 1,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        num_experts: int | list = 1,
        moe_topk: int | list = 1,
        head_dim: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_experts = num_experts
        self.moe_topk = moe_topk

        self.head_dim = head_dim
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)

    def _rope_parameters_validation(self):
        """
        Validate the `rope_parameters` configuration.
        """
        if self.rope_parameters is None:
            return

        if not isinstance(self.rope_parameters, dict) or len(self.rope_parameters) != 2:
            raise ValueError(
                "`rope_parameters` must be a dictionary with two fields, `type` and `factor` or `type` and `alpha`,"
                f"got {self.rope_parameters}"
            )
        rope_parameters_type = self.rope_parameters.get("type", None)
        rope_parameters_factor = self.rope_parameters.get("factor", None)
        rope_parameters_alpha = self.rope_parameters.get("alpha", None)
        if rope_parameters_type is None or rope_parameters_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_parameters`'s type field must be one of ['linear', 'dynamic'], got {rope_parameters_type}"
            )
        if rope_parameters_factor is None and rope_parameters_alpha is None:
            raise ValueError("`rope_parameters`'s factor or alpha field must be have one, got both of none")
        if rope_parameters_factor is not None:
            if not isinstance(rope_parameters_factor, float) or rope_parameters_factor <= 1.0:
                raise ValueError(
                    f"`rope_parameters`'s factor field must be a float > 1.0, got {rope_parameters_factor}"
                )
        if rope_parameters_alpha is not None:
            if not isinstance(rope_parameters_alpha, float) or rope_parameters_alpha <= 1.0:
                raise ValueError(f"`rope_parameters`'s alpha field must be a float > 1.0, got {rope_parameters_alpha}")


__all__ = ["HunYuanMoEV1Config"]
