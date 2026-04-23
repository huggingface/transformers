# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

import math

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2RotaryEmbedding,
)
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="openbmb/MiniCPM3-4B")
@strict
class MiniCPM3Config(LlamaConfig):
    r"""
    kv_lora_rank (`int`, *optional*, defaults to 256):
        Rank of the low-rank KV projection in multi-head latent attention.
    q_lora_rank (`int`, *optional*, defaults to 768):
        Rank of the low-rank query projection in multi-head latent attention.
    qk_nope_head_dim (`int`, *optional*, defaults to 64):
        Dimension of the non-RoPE part of each query/key head.
    qk_rope_head_dim (`int`, *optional*, defaults to 32):
        Dimension of the RoPE part of each query/key head.
    v_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each value head.
    scale_emb (`int`, *optional*, defaults to 1):
        Scaling factor applied to input embeddings.
    scale_depth (`float`, *optional*, defaults to 1.0):
        Scaling factor for residual connections, applied as `scale_depth / sqrt(num_hidden_layers)`.
    dim_model_base (`int`, *optional*, defaults to 1):
        Base model dimension used to scale logits before the language model head.

    Example:

    ```python
    >>> from transformers import MiniCPM3Model, MiniCPM3Config
    >>> configuration = MiniCPM3Config()
    >>> model = MiniCPM3Model(configuration)
    >>> print(model.config)
    ```
    """

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    model_type = "minicpm3"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 73448
    hidden_size: int = 2560
    intermediate_size: int = 6400
    num_hidden_layers: int = 62
    num_attention_heads: int = 40
    num_key_value_heads: int | None = 40
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.1
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | None = 0.0
    mlp_bias: bool = False
    kv_lora_rank: int = 256
    q_lora_rank: int | None = 768
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 128
    scale_emb: int = 1
    scale_depth: float = 1.0
    dim_model_base: int = 1

    def __post_init__(self, **kwargs):
        self.head_dim = self.qk_rope_head_dim
        super().__post_init__(**kwargs)


class MiniCPM3RMSNorm(LlamaRMSNorm):
    pass


class MiniCPM3RotaryEmbedding(DeepseekV2RotaryEmbedding):
    pass


class MiniCPM3Attention(DeepseekV2Attention):
    pass


class MiniCPM3MLP(LlamaMLP):
    pass


class MiniCPM3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MiniCPM3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MiniCPM3Attention(config=config, layer_idx=layer_idx)
        self.mlp = MiniCPM3MLP(config)
        self.input_layernorm = MiniCPM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniCPM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        return hidden_states


class MiniCPM3PreTrainedModel(LlamaPreTrainedModel):
    pass


@auto_docstring
class MiniCPM3Model(LlamaModel):
    def __init__(self, config: MiniCPM3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniCPM3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MiniCPM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MiniCPM3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids) * self.config.scale_emb

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class MiniCPM3ForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPM3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniCPM3ForCausalLM

        >>> model = MiniCPM3ForCausalLM.from_pretrained("openbmb/MiniCPM3-4B")
        >>> tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM3-4B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(
            hidden_states[:, slice_indices, :] / (self.config.hidden_size / self.config.dim_model_base)
        )

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MiniCPM3ForSequenceClassification(LlamaForSequenceClassification):
    pass


__all__ = [
    "MiniCPM3PreTrainedModel",
    "MiniCPM3Model",
    "MiniCPM3ForCausalLM",
    "MiniCPM3ForSequenceClassification",
    "MiniCPM3Config",
]
