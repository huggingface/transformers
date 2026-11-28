# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..clip.modeling_clip import CLIPMLP
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import LlamaPreTrainedModel
from ..vit.modeling_vit import ViTAttention


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="noanabeshima/tiny_model")
@strict
class TinyModelConfig(LlamaConfig):
    r"""
    attention_output_bias (`bool`, *optional*, defaults to `True`):
        Whether to use a bias in the attention output projection.
    mlp_bias (`bool`, *optional*, defaults to `True`):
        Whether to use biases in the feed-forward projections.
    lm_head_bias (`bool`, *optional*, defaults to `True`):
        Whether to use a bias in the language-modeling head.
    embedding_initializer_range (`float`, *optional*, defaults to 0.0001):
        Standard deviation used to initialize token and position embeddings.
    """

    model_type = "tiny_model"

    vocab_size: int = 10_000
    hidden_size: int = 768
    intermediate_size: int = 3_072
    num_hidden_layers: int = 4
    num_attention_heads: int = 16
    max_position_embeddings: int = 256
    hidden_act: str = "relu"
    attention_output_bias: bool = True
    mlp_bias: bool = True
    lm_head_bias: bool = True
    initializer_range: float = 0.02
    embedding_initializer_range: float = 1e-4
    bos_token_id: int | None = 9_996
    eos_token_id: int | list[int] | None = 9_997
    pad_token_id: int | None = 9_998

    # TinyModel has no rotary embeddings, normalization, KV cache, or parallelism plans.
    num_key_value_heads = AttributeError()
    rms_norm_eps = AttributeError()
    use_cache = AttributeError()
    pretraining_tp = AttributeError()
    rope_parameters = AttributeError()
    attention_dropout = AttributeError()
    head_dim = AttributeError()
    keys_to_ignore_at_inference = AttributeError()
    base_model_tp_plan = AttributeError()
    base_model_pp_plan = AttributeError()

    def __post_init__(self, **kwargs):
        raise AttributeError("TinyModel does not need Llama's derived head and KV-head settings.")


class TinyModelAttention(ViTAttention):
    # ViTAttention performs plain self-attention on sequence tensors without image operations,
    # rotary embeddings, or a KV cache. Its forward matches TinyModel when configured with
    # causal attention, zero dropout, and the checkpoint's Q/K/V and output projection biases.
    def __init__(self, config: TinyModelConfig):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dropout = 0.0
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_output_bias)


class TinyModelMLP(CLIPMLP):
    def __init__(self, config: TinyModelConfig):
        super().__init__(config)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)


class TinyModelDecoderLayer(GradientCheckpointingLayer):
    # Jais2 has a similar residual layout, but TinyModel replaces its attention/MLP
    # and omits both normalization layers. Define this small block directly because
    # inheriting Jais2 would still require a complete forward override.
    def __init__(self, config: TinyModelConfig):
        super().__init__()
        self.self_attn = TinyModelAttention(config)
        self.mlp = TinyModelMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@auto_docstring
class TinyModelPreTrainedModel(LlamaPreTrainedModel):
    _skip_keys_device_placement = AttributeError()
    _supports_flash_attn = AttributeError()
    _supports_flex_attn = AttributeError()
    _can_compile_fullgraph = AttributeError()

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=self.config.embedding_initializer_range)


@auto_docstring
class TinyModel(TinyModelPreTrainedModel):
    def __init__(self, config: TinyModelConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([TinyModelDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds + self.embed_positions(position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(hidden_states, attention_mask=causal_mask, **kwargs)

        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class TinyModelForCausalLM(TinyModelPreTrainedModel, GenerationMixin):
    def __init__(self, config: TinyModelConfig):
        super().__init__(config)
        self.model = TinyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        self.post_init()
        self.generation_config.use_cache = False

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        if kwargs.get("inputs_embeds") is not None:
            raise ValueError("TinyModel cannot generate from `inputs_embeds` without key/value caching.")
        return super().prepare_inputs_for_generation(input_ids, **kwargs)

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
    ) -> CausalLMOutput:
        if use_cache:
            raise ValueError("TinyModel does not support key/value caching; set `use_cache=False`.")
        if past_key_values is not None:
            raise ValueError("TinyModel does not support `past_key_values`.")

        outputs: BaseModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :]).contiguous()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["TinyModelConfig", "TinyModel", "TinyModelForCausalLM", "TinyModelPreTrainedModel"]
