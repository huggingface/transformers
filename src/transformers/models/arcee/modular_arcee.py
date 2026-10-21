# Copyright 2025 Arcee AI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Arcee model."""

from huggingface_hub.dataclasses import strict

from transformers.utils import auto_docstring, logging

from ...modeling_rope_utils import RopeParameters
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
)
from ..nemotron.modeling_nemotron import NemotronMLP


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="arcee-ai/AFM-4.5B")
@strict
class ArceeConfig(LlamaConfig):
    r"""
    ```python
    >>> from transformers import ArceeModel, ArceeConfig

    >>> # Initializing an Arcee AFM-4.5B-Base style configuration
    >>> configuration = ArceeConfig()

    >>> # Initializing a model from the AFM-4.5B-Base style configuration
    >>> model = ArceeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "arcee"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    vocab_size: int = 32000
    hidden_size: int = 2560
    intermediate_size: int = 18432
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "relu2"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 128000
    eos_token_id: int | list[int] | None = 128001
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    mlp_bias: bool = False
    head_dim: int | None = None

    pretraining_tp = AttributeError()


class ArceeMLP(NemotronMLP):
    pass


@auto_docstring(checkpoint="arcee-ai/AFM-4.5B")
class ArceeForCausalLM(LlamaForCausalLM):
    pass


@auto_docstring(checkpoint="arcee-ai/AFM-4.5B")
class ArceeForSequenceClassification(LlamaForSequenceClassification):
    pass


@auto_docstring(checkpoint="arcee-ai/AFM-4.5B")
class ArceeForQuestionAnswering(LlamaForQuestionAnswering):
    pass


@auto_docstring(checkpoint="arcee-ai/AFM-4.5B")
class ArceeForTokenClassification(LlamaForTokenClassification):
    pass


__all__ = [
    "ArceeConfig",
    "ArceeForCausalLM",
    "ArceeForQuestionAnswering",
    "ArceeForSequenceClassification",
    "ArceeForTokenClassification",
    "ArceeModel",  # noqa: F822
    "ArceePreTrainedModel",  # noqa: F822
]
