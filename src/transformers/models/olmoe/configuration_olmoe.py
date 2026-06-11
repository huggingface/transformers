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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="allenai/OLMoE-1B-7B-0924")
@strict
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
    ```
    """

    model_type = "olmoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "num_experts"}

    # Default tensor parallel plan for base model `Olmoe`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",  # due to the norm, we have to gather
        "layers.*.self_attn.k_proj": "colwise_gather_output",  # due to the norm, we have to gather
        "layers.*.self_attn.v_proj": "colwise_gather_output",  # due to the norm, we have to gather
        "layers.*.self_attn.o_proj": "rowwise_split_input",  # due to the norm, we have to gather
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    vocab_size: int = 50304
    hidden_size: int = 2048
    intermediate_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 50279
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    clip_qkv: float | None = None
    num_experts_per_tok: int = 8
    num_experts: int = 64
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.01
    norm_topk_prob: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


__all__ = ["OlmoeConfig"]
