
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Callable

import torch
import torch.nn as nn

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask, create_masks_for_generate
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ...utils.deprecation import deprecate_kwarg

logger = logging.get_logger(__name__)

# ===== Outputs =====
@dataclass
class ReferenceVLMOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: Optional[torch.Tensor] = None
    audio_hidden_states: Optional[torch.Tensor] = None

@dataclass
class ReferenceVLMCausalLMOutputWithPast(CausalLMOutputWithPast):
    image_hidden_states: Optional[torch.Tensor] = None
    audio_hidden_states: Optional[torch.Tensor] = None

# ===== Small utils =====
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# ===== Attention Placeholder =====

def eager_attention_forward(module, q, k, v, attention_mask, dropout=0.0, scaling=None, sliding_window=None, **kwargs):
    if scaling is None:
        scaling = module.head_dim ** -0.5
    k = repeat_kv(k, module.num_key_value_groups)
    v = repeat_kv(v, module.num_key_value_groups)
    attn_weights = torch.matmul(q, k.transpose(2,3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : k.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, v).transpose(1,2).contiguous()
    return attn_output, attn_weights

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, n_kv, s, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hidden_states.reshape(b, n_kv * n_rep, s, d)

# ===== Image scatter helper (pure functional) =====

def scatter_image_embeddings(inputs_embeds, image_features, special_image_mask):
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    return inputs_embeds.masked_scatter(special_image_mask, image_features)

# ===== Token-type OR-mask (Gemma3 style) =====

def token_type_ids_mask_function(token_type_ids: Optional[torch.Tensor], image_group_ids: Optional[torch.Tensor], tokens_per_image: int) -> Optional[Callable]:
    if token_type_ids is None:
        return None
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        safe_idx = torch.where(kv_idx < token_type_ids.shape[1], kv_idx, 0)
        kv_tt = torch.where(kv_idx < token_type_ids.shape[1], token_type_ids[batch_idx, safe_idx], 0)
        kv_gid = torch.where(kv_idx < image_group_ids.shape[1], image_group_ids[batch_idx, safe_idx], -1)
        is_image_block = (token_type_ids[batch_idx, q_idx] == 1) & (kv_tt == 1)
        same_image_block = image_group_ids[batch_idx, q_idx] == kv_gid
        return is_image_block & same_image_block
    return inner_mask

# ===== Core Modules =====
class ReferenceAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_sliding = getattr(config.layer_types, '__getitem__', lambda i: 'full_attention')(layer_idx) == 'sliding_attention'
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.attn_logit_softcapping = getattr(config, 'attn_logit_softcapping', None)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_value: Optional[Cache]=None, cache_position=None, **kwargs: Unpack[FlashAttentionKwargs]):
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        q = self.q_norm(q); k = self.k_norm(k)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
        attn_impl = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation] if self.config._attn_implementation != "eager" else eager_attention_forward
        attn_out, attn_weights = attn_impl(self, q, k, v, attention_mask, dropout=self.attention_dropout if self.training else 0.0, scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        attn_out = attn_out.reshape(bsz, q_len, -1).contiguous()
        attn_out = self.o_proj(attn_out)
        return attn_out, attn_weights

class ReferenceDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx] if hasattr(config, 'layer_types') else 'full_attention'
        self.self_attn = ReferenceAttention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ffn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_ffn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
        )

    @deprecate_kwarg("last_cache_position", version="4.53.0")
    def forward(self, hidden_states, position_embeddings_global, position_embeddings_local, attention_mask=None, position_ids=None, past_key_value: Optional[Cache]=None, output_attentions=False, use_cache=False, cache_position=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        pos_emb = position_embeddings_local if self.self_attn.is_sliding else position_embeddings_global
        hidden_states, attn_w = self.self_attn(hidden_states, pos_emb, attention_mask, past_key_value, cache_position, **kwargs)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_ffn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_ffn_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_w,)
        return outputs

# ===== TEXT BACKBONE WRAPPER =====
class ReferenceTextModel(PreTrainedModel):
    config_class = None
    supports_gradient_checkpointing = True
    _no_split_modules = ["ReferenceDecoderLayer"]
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([ReferenceDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # rotary emb placeholders (global & local like Gemma3)
        self.rotary_emb = nn.Identity()
        self.rotary_emb_local = nn.Identity()
        self.gradient_checkpointing = False
        self.post_init()

    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values: Optional[Cache]=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, cache_position=None, **kwargs):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        # causal masks
        if not isinstance(attention_mask, dict):
            mask_kwargs = dict(config=self.config, input_embeds=inputs_embeds, attention_mask=attention_mask, cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids)
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        else:
            causal_mask_mapping = attention_mask
        hidden_states = inputs_embeds
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids) if callable(getattr(self.rotary_emb, '__call__', None)) else (None,None)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids) if callable(getattr(self.rotary_emb_local, '__call__', None)) else (None,None)
        all_h = () if output_hidden_states else None
        all_a = () if output_attentions else None
        for layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_h = all_h + (hidden_states,)
            layer_out = layer(hidden_states, position_embeddings_global, position_embeddings_local, attention_mask=causal_mask_mapping[layer.attention_type], position_ids=position_ids, past_key_value=past_key_values, output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position, **kwargs)
            hidden_states = layer_out[0]
            if output_attentions:
                all_a = all_a + (layer_out[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_h = all_h + (hidden_states,)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values, hidden_states=all_h, attentions=all_a)

# ===== Causal LM head (text only) =====
class ReferenceForCausalLM(ReferenceTextModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    base_model_prefix = "language_model"
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values: Optional[Cache]=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, cache_position=None, logits_to_keep: Union[int, torch.Tensor]=0, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, cache_position=cache_position, **kwargs)
        hs = outputs.last_hidden_state
        slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep,int) else logits_to_keep
        logits = self.lm_head(hs[:, slice_idx, :])
        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1]:].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask != 0].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1).to(shift_logits.device))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

# ===== Multimodal (text+vision(+audio)) =====
class ReferenceVLMModel(PreTrainedModel):
    accepts_loss_kwargs = False
    def __init__(self, config):
        super().__init__(config)
        # plug real backbones
        self.vision_tower = getattr(config, 'vision_model', None)
        self.language_model = getattr(config, 'text_model', None)
        self.multi_modal_projector = MultiModalProjector(config.vision_config.hidden_size, config.text_config.hidden_size)
        self.vocab_size = config.text_config.vocab_size
        self.post_init()
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)
    def get_image_features(self, pixel_values: torch.Tensor):
        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        return self.multi_modal_projector(vision_outputs)
    @can_return_tuple
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, position_ids=None, past_key_values: Optional[Cache]=None, token_type_ids=None, cache_position=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, **lm_kwargs):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone(); llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
        image_features = None
        special_image_mask = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device))
                special_image_mask = special_image_mask.all(-1)
            else:
                special_image_mask = input_ids == self.config.image_token_id
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                raise ValueError("Mismatch between special image tokens and provided image features")
            inputs_embeds = scatter_image_embeddings(inputs_embeds, image_features, special_image_mask)
        if not isinstance(attention_mask, dict):
            mask_kwargs = dict(config=self.config.get_text_config(), input_embeds=inputs_embeds, attention_mask=attention_mask, cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids)
            if token_type_ids is not None and inputs_embeds.shape[1] != 1:
                is_image = (token_type_ids == 1).to(cache_position.device)
                new_image_start = is_image & ~nn.functional.pad(is_image, (1,0), value=0)[:, :-1]
                image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
                image_group_ids = torch.where(is_image, image_group_ids, torch.full_like(token_type_ids, -1))
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(token_type_ids.to(cache_position.device), image_group_ids, self.config.mm_tokens_per_image)
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs), "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs)}
        else:
            causal_mask_mapping = attention_mask
        outputs = self.language_model(attention_mask=causal_mask_mapping, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True, cache_position=cache_position, **lm_kwargs)
        return ReferenceVLMOutputWithPast(last_hidden_state=outputs.last_hidden_state, past_key_values=outputs.past_key_values if use_cache else None, hidden_states=outputs.hidden_states, attentions=outputs.attentions, image_hidden_states=image_features)

class ReferenceVLMForConditionalGeneration(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = ReferenceVLMModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, position_ids=None, past_key_values: Optional[Cache]=None, token_type_ids=None, cache_position=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, logits_to_keep: Union[int, torch.Tensor]=0, **lm_kwargs):
        outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, labels=labels, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, cache_position=cache_position, **lm_kwargs)
        hs = outputs.last_hidden_state
        slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep,int) else logits_to_keep
        logits = self.lm_head(hs[:, slice_idx, :])
        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                mask2d = attention_mask[:, -shift_logits.shape[1]:].to(logits.device)
                shift_logits = shift_logits[mask2d != 0].contiguous()
                shift_labels = shift_labels[mask2d != 0].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1).to(shift_logits.device))
        return ReferenceVLMCausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, image_hidden_states=outputs.image_hidden_states)
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, cache_position=None, position_ids=None, pixel_values=None, attention_mask=None, token_type_ids=None, use_cache=True, logits_to_keep=None, labels=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, cache_position=cache_position, use_cache=use_cache, logits_to_keep=logits_to_keep, token_type_ids=token_type_ids, **kwargs)
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        return model_inputs
    @staticmethod
    def create_masks_for_generate(config: PretrainedConfig, input_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor], cache_position: torch.Tensor, past_key_values: Optional[Cache], position_ids: Optional[torch.Tensor], token_type_ids: Optional[torch.Tensor]=None, **kwargs) -> dict:
        mask_kwargs = dict(config=config.get_text_config(), input_embeds=input_embeds, attention_mask=attention_mask, cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids)
        if token_type_ids is not None and input_embeds.shape[1] != 1:
            is_image = (token_type_ids == 1).to(cache_position.device)
            new_image_start = is_image & ~nn.functional.pad(is_image, (1,0), value=0)[:, :-1]
            image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
            image_group_ids = torch.where(is_image, image_group_ids, torch.full_like(token_type_ids, -1))
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(token_type_ids.to(cache_position.device), image_group_ids, config.mm_tokens_per_image)
        return create_masks_for_generate(**mask_kwargs)

__all__ = [
    "ReferenceVLMOutputWithPast",
    "ReferenceVLMCausalLMOutputWithPast",
    "ReferenceAttention",
    "ReferenceDecoderLayer",
    "ReferenceTextModel",
    "ReferenceForCausalLM",
    "ReferenceVLMModel",
    "ReferenceVLMForConditionalGeneration",
]