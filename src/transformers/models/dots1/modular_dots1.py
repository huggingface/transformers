# Copyright 2025 The rednote-hilab team and the HuggingFace Inc. team. All rights reserved.
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
import torch

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
    DeepseekV3TopkRouter,
)
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    TransformersKwargs,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="rednote-hilab/dots.llm1.base")
class Dots1Config(PreTrainedConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    first_k_dense_replace (`int`, *optional*, defaults to 0):
        Number of dense layers at the beginning of the model before the first MoE layer.

    Examples:

    ```python
    >>> from transformers import Dots1Model, Dots1Config
    >>> # Initializing a Dots1 style configuration
    >>> configuration = Dots1Config()
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "dots1"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size: int | None = 152064,
        hidden_size: int | None = 4608,
        intermediate_size: int | None = 10944,
        moe_intermediate_size: int | None = 1408,
        num_hidden_layers: int | None = 62,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 32,
        n_shared_experts: int | None = None,
        n_routed_experts: int | None = None,
        n_group: int | None = 1,
        topk_group: int | None = 1,
        num_experts_per_tok: int | None = None,
        first_k_dense_replace: int | None = 0,
        norm_topk_prob: bool | None = False,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 2048,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        routed_scaling_factor: float | None = 1.0,
        sliding_window: int | None = 4096,
        max_window_layers: int | None = 62,
        layer_types: list[str] | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.routed_scaling_factor = routed_scaling_factor
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


class Dots1RMSNorm(Qwen3RMSNorm):
    pass


class Dots1RotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class Dots1Attention(Qwen3Attention):
    pass


class Dots1MLP(DeepseekV3MLP):
    pass


class Dots1TopkRouter(DeepseekV3TopkRouter):
    pass


class Dots1MoE(DeepseekV3MoE):
    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()  # main diff with deepseekv3
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class Dots1DecoderLayer(DeepseekV3DecoderLayer):
    pass


class Dots1PreTrainedModel(DeepseekV3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = None


class Dots1Model(Qwen3Model):
    pass


class Dots1ForCausalLM(Qwen3ForCausalLM):
    def forward(
        self,
        **super_kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Dots1ForCausalLM

        >>> model = Dots1ForCausalLM.from_pretrained("rednote-hilab/dots1.llm1.inst")
        >>> tokenizer = AutoTokenizer.from_pretrained("rednote-hilab/dots1.llm1.inst")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "Dots1Config",
    "Dots1PreTrainedModel",
    "Dots1Model",
    "Dots1ForCausalLM",
]
