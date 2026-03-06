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
"""OLMoE model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="allenai/OLMoE-1B-7B-0924")
class OlmoeConfig(PreTrainedConfig):
    r"""
    clip_qkv (`float`, *optional*):
        If not `None`, elements of query, key and value attention states are clipped so that their
        absolute value does not exceed this value.

    ```python
    >>> from transformers import OlmoeModel, OlmoeConfig

    >>> # Initializing a OLMoE 7B A1B style configuration
    >>> configuration = OlmoeConfig()

    >>> # Initializing a model from the OLMoE 7B A1B style configuration
    >>> model = OlmoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "olmoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "num_experts"}

    def __init__(
        self,
        vocab_size: int | None = 50304,
        hidden_size: int | None = 2048,
        intermediate_size: int | None = 2048,
        num_hidden_layers: int | None = 16,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 4096,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-05,
        use_cache: bool | None = True,
        pad_token_id: int | None = 1,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 50279,
        tie_word_embeddings: int | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        clip_qkv: bool | None = None,
        num_experts_per_tok: int | None = 8,
        num_experts: int | None = 64,
        output_router_logits: bool | None = False,
        router_aux_loss_coef: float | None = 0.01,
        norm_topk_prob: bool | None = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.clip_qkv = clip_qkv
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.norm_topk_prob = norm_topk_prob
        self.rope_parameters = rope_parameters

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)


__all__ = ["OlmoeConfig"]
