# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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
import copy
import inspect
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, TypedDict, Union

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple, logging
from .configuration_step3p7 import Step3p7Config, Step3p7TextConfig
from .vision_encoder import StepRoboticsVisionEncoder


logger = logging.get_logger(__name__)
_MASK_INPUT_EMBEDS_ARG = (
    "inputs_embeds"
    if "inputs_embeds" in inspect.signature(create_causal_mask).parameters
    else "input_embeds"
)

__all__ = [
    "Step3p7Model",
]


class StepVLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    patch_pixel_values: Optional[torch.Tensor]
    num_patches: list[int]


class StepVLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: torch.Tensor


StepVLImageInputs = Union[StepVLImagePixelInputs, StepVLImageEmbeddingInputs]


def _flatten_embeddings(embeddings) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))

def _embedding_count_expression(embeddings) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(_embedding_count_expression(inner) for inner in embeddings)


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    is_multimodal: torch.Tensor,
    multimodal_embeddings,
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.
    Note:
        This updates ``inputs_embeds`` in place.
    """
    num_expected_tokens = is_multimodal.sum().item()
    assert isinstance(num_expected_tokens, int)

    flattened = _flatten_embeddings(multimodal_embeddings)
    if flattened.shape[0] != num_expected_tokens:
        expr = _embedding_count_expression(multimodal_embeddings)
        raise ValueError(
            f"Attempted to assign {expr} = {flattened.shape[0]} "
            f"multimodal tokens to {num_expected_tokens} placeholders"
        )

    is_multimodal = is_multimodal.to(inputs_embeds.device)
    flattened = flattened.to(inputs_embeds.device)
    inputs_embeds[is_multimodal] = flattened
    return inputs_embeds

def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings,
    placeholder_token_id: Union[int, list[int]],
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.
    
    ``placeholder_token_id`` can be a list of token ids (e.g, token ids 
    of img_start, img_break, and img_end tokens) when needed: This means 
    the order of these tokens in the ``input_ids`` MUST MATCH the order of 
    their embeddings in ``multimodal_embeddings`` since we need to 
    slice-merge instead of individually scattering.
    For example, if input_ids is "TTTTTSIIIBIIIBIIIETTT", where
    - T is text token
    - S is image start token
    - I is image embedding token
    - B is image break token
    - E is image end token.
    
    Then the image embeddings (that correspond to I's) from vision encoder 
    must be padded with embeddings of S, B, and E in the same order of 
    input_ids for a correct embedding merge.
    Note:
        This updates ``inputs_embeds`` in place.
    """
    if isinstance(placeholder_token_id, list):
        placeholder_token_id = torch.tensor(
            placeholder_token_id, device=input_ids.device
        )
        return _merge_multimodal_embeddings(
            inputs_embeds,
            torch.isin(input_ids, placeholder_token_id),
            multimodal_embeddings,
        )

    return _merge_multimodal_embeddings(
        inputs_embeds,
        (input_ids == placeholder_token_id),
        multimodal_embeddings,
    )


class Step3p7PreTrainedModel(PreTrainedModel):
    # Link this model family to its configuration class so PreTrainedModel.from_pretrained
    # can load the config instead of failing with a NoneType error.
    config_class = Step3p7Config
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _keys_to_ignore_on_load_unexpected = [
        r"model\.layers\.45\.*",
        r"model\.layers\.46\.*",
        r"model\.layers\.47\.*",
    ]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_static_cache = True
    _supports_attention_backend = True

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args, **kwargs
    ):
        key_mapping = getattr(cls, "_checkpoint_conversion_mapping", None)
        if key_mapping is not None and kwargs.get("key_mapping") is None:
            # Transformers only applies checkpoint renaming when key_mapping is
            # passed explicitly; inheriting the class attribute alone is not enough.
            kwargs["key_mapping"] = copy.deepcopy(key_mapping)
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )


class Step3p7RotaryEmbedding(nn.Module):
    def __init__(self, config: Step3p7TextConfig, device=None, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        rope_theta = config.rope_theta
        if isinstance(rope_theta, list):
            rope_theta = rope_theta[0 if layer_idx is None else layer_idx]

        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        partial_rotary_factors = getattr(config, "partial_rotary_factors", None)
        if partial_rotary_factors is not None:
            partial_rotary_factor = partial_rotary_factors[
                0 if layer_idx is None else layer_idx
            ]

        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor

        self.config = copy.copy(config)
        self.config.rope_theta = rope_theta
        self.config.partial_rotary_factor = partial_rotary_factor

        if config.rope_parameters is not None:
            self.config.rope_parameters = copy.deepcopy(config.rope_parameters)
            self.config.rope_parameters["rope_theta"] = rope_theta
            self.config.rope_parameters["partial_rotary_factor"] = (
                partial_rotary_factor
            )
            self.rope_type = self.config.rope_parameters.get(
                "rope_type", self.config.rope_parameters.get("type")
            )
        else:
            self.rope_type = "default"

        self.rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float().to(x.device)

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(
            device_type=device_type, enabled=False
        ):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def compute_default_rope_parameters(
        config: Step3p7TextConfig | None = None,
        device: Optional["torch.device"] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_theta
        partial_rotary_factor = getattr(
            config, "partial_rotary_factor", 1.0
        )
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )


# Adapted from transformers.models.llama.modeling_llama.eager_attention_forward.
# Llama4 does not cast attention weights to fp32 here.
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    # breakpoint()
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@dataclass
class Step3p7CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class Step3p7MLP(nn.Module):
    def __init__(self, config, intermediate_size=None, swiglu_limit=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else config.intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size,
                                   self.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(self.hidden_size,
                                 self.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(self.intermediate_size,
                                   self.hidden_size,
                                   bias=False)
        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.act_fn(self.gate_proj(x))
        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)

        return self.down_proj(gate * up)


def sigmoid_routing_function(gating_output: torch.Tensor, topk: int,
                             renormalize: bool):
    gating_output = gating_output.float()
    gate_prob = torch.sigmoid(gating_output)
    gate_prob = gate_prob / gate_prob.sum(dim=-1, keepdim=True)
    topk_prob, indices = torch.topk(gate_prob, k=topk, dim=1)
    expert_topk_weight = topk_prob
    if renormalize:
        expert_topk_weight = expert_topk_weight / torch.sum(
            expert_topk_weight, dim=-1, keepdim=True)
    return expert_topk_weight, indices


def softmax_routing_function(gating_output: torch.Tensor, top_k: int,
                             renormalize: bool):
    gating_output = gating_output.float()
    gate_prob = torch.softmax(gating_output, dim=-1)
    gate_prob = gate_prob / gate_prob.sum(dim=-1, keepdim=True)
    topk_prob, indices = torch.topk(gate_prob, k=top_k, dim=1)
    expert_topk_weight = topk_prob
    if renormalize:
        expert_topk_weight = expert_topk_weight / torch.sum(
            expert_topk_weight, dim=-1, keepdim=True)
    return expert_topk_weight, indices.to(torch.int32)


class MoELinear(nn.Module):

    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(num_experts, out_features, in_features))

    def forward(self, x, expert_id):
        x = F.linear(x.float(), self.weight[expert_id].float())
        return x


class Step3p7MoEMLP(nn.Module):

    def __init__(self, config, swiglu_limit=None):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        self.use_moe_router_bias = config.use_moe_router_bias
        if self.use_moe_router_bias:
            self.router_bias = nn.Parameter(torch.zeros(config.moe_num_experts,
                                                        dtype=torch.float32),
                                            requires_grad=False)
            self.custom_routing_function = self.router_bias_func
        elif config.moe_router_activation == "sigmoid":
            self.custom_routing_function = sigmoid_routing_function
        else:
            self.custom_routing_function = None
        self.need_fp32_gate = config.need_fp32_gate
        self.routed_scaling_factor = getattr(config,
                                             "moe_router_scaling_factor", 1.0)
        
        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
            
        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit

        self.up_proj = MoELinear(self.num_experts, self.hidden_size,
                                 self.moe_intermediate_size)
        self.gate_proj = MoELinear(self.num_experts, self.hidden_size,
                                   self.moe_intermediate_size)
        self.down_proj = MoELinear(self.num_experts,
                                   self.moe_intermediate_size,
                                   self.hidden_size)

    def router_bias_func(self, gating_output: torch.Tensor, topk: int,
                         renormalize: bool):
        gate_prob = torch.sigmoid(gating_output.float())
        gate_prob_with_bias = gate_prob + self.router_bias.unsqueeze(0)
        _, indices = torch.topk(gate_prob_with_bias, k=topk, dim=1)
        topk_prob = torch.gather(gate_prob, 1, indices)
        expert_topk_weight = topk_prob
        if renormalize:
            expert_topk_weight = expert_topk_weight / (
                torch.sum(expert_topk_weight, dim=-1, keepdim=True) + 1e-20)
        return expert_topk_weight, indices

    def get_expert_output(self, inputs: torch.Tensor, expert_id):
        #if self.limit is None:
        up = self.up_proj(inputs, expert_id)
        gate = self.act_fn(self.gate_proj(inputs, expert_id))
        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)

        return self.down_proj(gate * up, expert_id)

    def forward(self, hidden_states):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.need_fp32_gate:
            router_logits = torch.matmul(
                hidden_states.to(torch.float32),
                self.gate.weight.t().to(torch.float32),
            )
        else:
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

        if self.custom_routing_function:
            routing_weights, selected_experts = self.custom_routing_function(
                router_logits, self.top_k, renormalize=True)
        else:
            routing_weights = F.softmax(router_logits,
                                        dim=1,
                                        dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights,
                                                           self.top_k,
                                                           dim=-1)

        routing_weights = routing_weights * self.routed_scaling_factor

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                self.get_expert_output(current_state, expert_idx) *
                routing_weights[top_x, idx, None])

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class Step3p7RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.variance_epsilon)
        normed = normed * (self.weight.float() + 1)
        return normed.to(dtype)
class Step3p7Attention(nn.Module):

    def __init__(self, config: Step3p7TextConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_groups

        layer_types = getattr(config, "layer_types", [])
        if layer_types:
            enable_sliding_window = layer_types[
                self.layer_idx] == "sliding_attention"
        else:
            enable_sliding_window = self.layer_idx % 2 == 0
        
        yarn_only_types = getattr(config, "yarn_only_types", None)
        if yarn_only_types and layer_types[
                self.layer_idx] not in yarn_only_types:
            config.rope_parameters = None
        else:
            config.rope_parameters = getattr(config, "rope_scaling", None)

        self.sliding_window = config.sliding_window
        if enable_sliding_window:
            self.num_attention_heads = config.attention_other_setting[
                "num_attention_heads"]
            self.num_key_value_heads = config.attention_other_setting[
                "num_attention_groups"]

        if self.sliding_window is not None and enable_sliding_window:
            self.sliding_window = (self.sliding_window)
        else:
            self.sliding_window = None
        self.head_dim = getattr(config, "head_dim",
                        config.hidden_size // self.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.rotary_emb = Step3p7RotaryEmbedding(config, layer_idx=layer_idx)

        self.q_size = self.num_attention_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.q_norm = Step3p7RMSNorm(self.head_dim,
                                    eps=config.rms_norm_eps)
        self.k_norm = Step3p7RMSNorm(self.head_dim,
                                    eps=config.rms_norm_eps)

        self.use_head_wise_attn_gate = config.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = nn.Linear(config.hidden_size,
                                    self.num_attention_heads,
                                    bias=False)

        self.use_rope = True
        use_rope_layers = getattr(config, "use_rope_layers", None)
        if use_rope_layers:
            self.use_rope = use_rope_layers[self.layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(
            1, 2)
        if self.use_head_wise_attn_gate:
            gate_states = self.g_proj(hidden_states)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        # query_states, key_states = apply_rotary_pos_emb(query_norm_states, key_norm_states, cos, sin)
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        # TODO: considering FP8；
        # RuntimeError: Expected attn_mask dtype to be bool or float or to match query dtype,
        # but got attn_mask.dtype: long int and  query.dtype: c10::BFloat16 instead.
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1)
        if self.use_head_wise_attn_gate:
            output = attn_output.view(
                *attn_output.shape[:-1], self.num_attention_heads,
                self.head_dim) * gate_states.unsqueeze(-1).sigmoid()
            attn_output = output.view(*attn_output.shape)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Step3p7DecoderLayer(GradientCheckpointingLayer):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Step3p7Attention(config, layer_idx)
        layer_types = getattr(config, "layer_types", None) or []
        if layer_types:
            self.attention_type = layer_types[layer_idx]
        else:
            self.attention_type = (
                "sliding_attention" if layer_idx % 2 == 0 else "full_attention"
            )

        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            if isinstance(moe_layers_enum, str):
                moe_layers_idx = [
                    int(i) for i in moe_layers_enum.split(',') if i.strip()
                ]
            else:
                moe_layers_idx = [int(i) for i in moe_layers_enum]
        else:
            moe_layers_idx = [i for i in range(1, config.num_hidden_layers)]
        self.is_moe_layer = layer_idx in moe_layers_idx
        self.use_moe = False

        if config.swiglu_limits_shared and config.swiglu_limits_shared[
                layer_idx] is not None and config.swiglu_limits_shared[
                    layer_idx] != 0:
            swiglu_limit_shared = config.swiglu_limits_shared[layer_idx]
        else:
            swiglu_limit_shared = None
        if config.swiglu_limits and config.swiglu_limits[
                layer_idx] is not None and config.swiglu_limits[layer_idx] != 0:
            swiglu_limit = config.swiglu_limits[layer_idx]
        else:
            swiglu_limit = None
        if self.is_moe_layer:
            self.moe = Step3p7MoEMLP(config, swiglu_limit=swiglu_limit)  #
            self.share_expert = Step3p7MLP(
                config,
                intermediate_size=config.share_expert_dim,
                swiglu_limit=swiglu_limit_shared)
            self.use_moe = True
        else:
            self.mlp = Step3p7MLP(config,
                                 intermediate_size=config.intermediate_size,
                                 swiglu_limit=swiglu_limit_shared)

        self.input_layernorm = Step3p7RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step3p7RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_moe:
            share_output = self.share_expert(hidden_states)
            moe_output = self.moe(hidden_states)
            ffn_output = moe_output + share_output
        else:
            ffn_output = self.mlp(hidden_states)
        if isinstance(ffn_output, tuple):
            hidden_states, _ = ffn_output
        else:
            hidden_states = ffn_output

        hidden_states = residual + hidden_states
        return hidden_states


class Step3p7TextPreTrainedModel(PreTrainedModel):
    # Link this model family to its configuration class so PreTrainedModel.from_pretrained
    # can load the config instead of failing with a NoneType error.
    config_class = Step3p7TextConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _keys_to_ignore_on_load_unexpected = [
        r"model\.layers\.45\.*",
        r"model\.layers\.46\.*",
        r"model\.layers\.47\.*",
    ]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_static_cache = True
    _supports_attention_backend = True


class Step3p7TextModel(Step3p7TextPreTrainedModel, GenerationMixin):
    _no_split_modules = ["Step3p7DecoderLayer"]
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    config: Step3p7TextConfig

    def __init__(self, config: Step3p7TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList([
            Step3p7DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        layer_types = self.config.layer_types or []
        self.has_sliding_layers = (not layer_types or
                                   "sliding_attention" in layer_types)

        # Initialize weights and apply final processing
        self.post_init()

    
    def get_input_embeddings(self, input_ids):
        return self.embed_tokens(input_ids)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "return_dict", True)
        )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(
                input_ids.to(self.embed_tokens.weight.device))

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(
            ) if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens,
                                          past_seen_tokens +
                                          inputs_embeds.shape[1],
                                          device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            _mask_sig = inspect.signature(create_causal_mask).parameters
            mask_kwargs = {
                "config": self.config,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            if "cache_position" in _mask_sig:
                mask_kwargs["cache_position"] = cache_position
            mask_kwargs[_MASK_INPUT_EMBEDS_ARG] = inputs_embeds
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping[
                    "sliding_attention"] = create_sliding_window_causal_mask(
                        **mask_kwargs)

        # # create position embeddings to be shared across the decoder layers
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[
                    decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Step3p7Model(Step3p7PreTrainedModel, GenerationMixin):
    config: Step3p7Config
    _tied_weights_keys = ["lm_head.weight"]
    base_model_prefix = ""

    def __init__(self, config: Step3p7Config):
        super().__init__(config)
        self.vision_model = StepRoboticsVisionEncoder(config.vision_config)
        self.language_model = Step3p7TextModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.vit_large_projector = nn.Linear(
                config.vision_config.width * 4,
                config.text_config.hidden_size,                
                bias=config.projector_bias) 
        self.image_placeholder_token_id = config.image_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings  = None,
    ) -> torch.Tensor:
        # breakpoint()
        input_ids = input_ids.squeeze(0)
        if multimodal_embeddings is None:
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        else:
            is_text = input_ids != self.config.image_token_id
            text_ids = input_ids[is_text]
            text_embeds = self.language_model.get_input_embeddings(text_ids)
                     
            inputs_embeds = torch.empty(input_ids.shape[0],
                                        text_embeds.shape[-1],
                                        dtype=text_embeds.dtype,
                                        device=text_embeds.device)
            inputs_embeds[is_text] = text_embeds
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.image_token_id)
        inputs_embeds = inputs_embeds.unsqueeze(0)
        return inputs_embeds
       

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model
    
    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[StepVLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        patch_pixel_values = kwargs.pop("patch_pixel_values", None)
        num_patches = kwargs.pop("num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            # pixel_values = flatten_bn(pixel_values, concat=True)
            if pixel_values.dim() >= 3:
                pixel_values = pixel_values.view(-1, *pixel_values.shape[-3:])
            if patch_pixel_values is not None:
                # patch_pixel_values = flatten_bn(patch_pixel_values,
                #                                 concat=True)
                patch_pixel_values = patch_pixel_values.view(
                    -1, *patch_pixel_values.shape[-3:])
                # Handle empty patch_pixel_values by setting to None
                if patch_pixel_values.shape[0] == 0:
                    patch_pixel_values = None

            return StepVLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values.to(self.dtype).to(self.device),
                patch_pixel_values=patch_pixel_values.to(self.dtype).to(
                    self.device) if patch_pixel_values is not None else None,
                num_patches=num_patches,
            )

        if image_embeds is not None:
            if image_embeds.dim() == 2 or image_embeds.dim() >= 3:
                image_embeds = image_embeds.view(-1, image_embeds.shape[-1])
            else:
                raise ValueError(
                    f"Unexpected shape for image_embeds: {image_embeds.shape}")

            return StepVLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds.to(self.dtype).to(self.device),
            )
        return None
    
    def _process_image_features(self,
                                image_features: torch.Tensor) -> torch.Tensor:
        B, P = image_features.shape[:2]
        HW = int(P ** 0.5)
        image_features = image_features.permute(0, 2, 1).view(B, -1, HW, HW)
        image_features = self.vision_model.vit_downsampler1(image_features)
        image_features = self.vision_model.vit_downsampler2(image_features)

        B, C, HW, HW = image_features.shape
        image_features = image_features.view(B, -1, HW * HW).permute(0, 2, 1)
        image_features = self.vit_large_projector(image_features)
        return image_features

    def _get_vision_model_output(self,
                                 input_tensor: torch.Tensor) -> torch.Tensor:
        return self.vision_model(input_tensor)

    def _process_image_input(
            self, image_input: StepVLImageInputs) -> tuple[torch.Tensor, ...]:

        if image_input["type"] == "image_embeds":
            image_features = image_input["image_embeds"]
        else:
            image_features = self._get_vision_model_output(
                image_input["pixel_values"])
            patch_image_features = self._get_vision_model_output(
                image_input["patch_pixel_values"]
            ) if image_input["patch_pixel_values"] is not None else None
            num_patches = image_input["num_patches"]

        image_features = self._process_image_features(image_features)
        patch_image_features = self._process_image_features(
            patch_image_features) if patch_image_features is not None else None

        merged_image_features = []
        cur_patch_idx = 0
        for i, num_patch in enumerate(num_patches):
            cur_feature = []
            if num_patch > 0:
                patch_slice = patch_image_features[
                    cur_patch_idx:cur_patch_idx + num_patch]
                cur_feature.append(patch_slice.view(-1, patch_slice.shape[-1]))
            cur_feature.append(image_features[i].view(
                -1, image_features.shape[-1]))
            cur_patch_idx += num_patch
            merged_image_features.append(
                torch.cat(cur_feature) if len(cur_feature) >
                1 else cur_feature[0])
    
        return merged_image_features
    
    def get_multimodal_embeddings(self, **kwargs):
        # breakpoint()
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        images: Optional[list[Image.Image]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Step3p7CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Example:
        ```python
        >>> from transformers import AutoTokenizer, Llama4ForCausalLM
        >>> model = Llama4ForCausalLM.from_pretrained("meta-llama4/Llama4-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama4/Llama4-2-7b-hf")
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            input_ids = input_ids
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Step3p7CausalLMOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            attentions=outputs.attentions,
        )
        return output if return_dict else output.to_tuple()


class Step3p7ForConditionalGeneration(Step3p7PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^vision_model": "model.vision_model",
        r"^model(?!\.(language_model|vision_model))": "model.language_model",
        "^vit_large_projector": "model.vit_large_projector",
    }
    _tied_weights_keys = ["lm_head.weight"]
    config: Step3p7Config

    def __init__(self, config: Step3p7Config):
        super().__init__(config)
        self.model = Step3p7Model(config)
        self.lm_head = nn.Linear(config.hidden_size,
                                config.text_config.vocab_size,
                                bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.vision_model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.Tensor] = None,
        num_patches=None,
        patch_pixel_values=None,
        patch_newline_mask=None,
        image_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Step3p7CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            num_patches=num_patches,
            patch_pixel_values=patch_pixel_values,
            patch_newline_mask=patch_newline_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        los = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size
            )

        return Step3p7CausalLMOutputWithPast(
            logits=logits,
        )


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        patch_pixel_values=None,
        num_patches=None,
        image_embeds=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        generation_cache_position = model_inputs.get("cache_position", cache_position)
        is_prefill = past_key_values is None
        if generation_cache_position is not None and generation_cache_position.numel() > 0:
            is_prefill = generation_cache_position[0].item() == 0

        if is_prefill:
            # During cached decoding, input ids no longer contain image tokens,
            # so pixel values should only be passed at the first step.
            model_inputs["pixel_values"] = pixel_values

        return model_inputs
    
    def _fix_state_dict_key_on_load(self, key: str) -> tuple[str, bool]:
        if key.startswith("language_model."):
            return key[len("language_model.") :], True
        
        return key, False
