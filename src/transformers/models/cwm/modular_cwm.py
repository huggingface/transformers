# coding=utf-8
# Copyright 2025
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

from typing import Optional, List, Tuple

import torch

from ...utils import logging
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
)

logger = logging.get_logger(__name__)


# -----------------------------------------------------------------------------
# Config (Llama-compatible weights; model_type='cwm' for modular converter routing)
# -----------------------------------------------------------------------------
class CwmTextConfig(LlamaConfig):
    """
    Llama3-compatible configuration with layer-interleaved sliding-window attention.

    Behavior:
      - Layers marked "sliding_attention" use local causal window = `sliding_window`.
      - Layers marked "full_attention" use pure causal by default; if `global_window` is set,
        they instead use a capped local causal window of size `global_window`.

    Keep weights Llama-shaped; expose model_type='cwm' so modular tooling targets this file.
    """

    model_type = "llama"  # important for vLLM + HF compatibility


    def __init__(
        self,
        # Llama fields
        vocab_size: int = 128256,
        hidden_size: int = 6144,
        intermediate_size: int = 21504,
        num_hidden_layers: int = 64,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id=(128001, 128008, 128009),
        bos_token_id: int = 128000,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1_000_000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        pretraining_tp: int = 1,
        mlp_bias: bool = False,
        rope_scaling: Optional[dict] = None,
        # CWM interleaved SWA fields
        sliding_window: int = 8192,
        layer_types: Optional[List[str]] = None,  # ["full_attention"|"sliding_attention"] per layer
        window_pattern: Optional[int] = None,     # convenience: 4 => every 4th layer "full"
        global_window: Optional[int] = None,      # cap for "full" layers; None => pure causal
        **kwargs,
    ):
        if rope_scaling is None:
            rope_scaling = {
                "factor": 16.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            }

        if layer_types is None:
            if window_pattern is None or window_pattern <= 0:
                window_pattern = 4
            layer_types = [
                ("full_attention" if (i % window_pattern == 0) else "sliding_attention")
                for i in range(num_hidden_layers)
            ]

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=list(eos_token_id),
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rope_scaling=rope_scaling,
            pretraining_tp=pretraining_tp,
            mlp_bias=mlp_bias,
            **kwargs,
        )

        self.sliding_window = int(sliding_window)
        self.layer_types = list(layer_types)
        self.window_pattern = int(window_pattern) if window_pattern is not None else None
        self.global_window = None if global_window is None else int(global_window)

        # Prefer SDPA when sliding is active (dense additive masks)
        try:
            if any(t == "sliding_attention" for t in self.layer_types) and self.sliding_window > 0:
                self._attn_implementation = "sdpa"
        except Exception:
            pass


class CwmConfig(CwmTextConfig):
    pass


# -----------------------------------------------------------------------------
# Mask helpers (build additive masks compatible with SDPA)
# -----------------------------------------------------------------------------
def _infer_past_len(past_key_value, layer_idx: int) -> int:
    if past_key_value is None:
        return 0
    if hasattr(past_key_value, "get_seq_length"):
        try:
            return int(past_key_value.get_seq_length(layer_idx))
        except Exception:
            pass
    if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
        k0 = past_key_value[0]
        if torch.is_tensor(k0) and k0.dim() >= 3:
            return int(k0.size(-2))
    return 0


def _additive_mask_local(
    position_ids: torch.LongTensor,  # [B,Q] absolute positions
    kv_len: int,                     # total keys (past + current)
    window: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Local-causal additive mask [B,1,Q,K]: allow iff k <= q and k >= q-(window-1).
    Returns 0 where allowed, -inf otherwise.
    """
    B, Q = position_ids.shape
    K = kv_len
    last = position_ids[:, -1]
    offset = last - (K - 1)
    key_abs = offset[:, None] + torch.arange(K, device=device, dtype=position_ids.dtype)[None, :]
    q_abs = position_ids[:, :, None]
    causal_ok = key_abs[:, None, :] <= q_abs
    local_ok = key_abs[:, None, :] >= (q_abs - (window - 1)) if window > 0 else torch.zeros_like(causal_ok, dtype=torch.bool)
    allowed = causal_ok & local_ok
    finfo = torch.finfo(dtype if dtype.is_floating_point else torch.float32)
    neg_inf = torch.tensor(finfo.min, device=device, dtype=dtype if dtype.is_floating_point else torch.float32)
    add = torch.where(allowed, torch.zeros((), device=device, dtype=dtype), neg_inf)
    return add[:, None, :, :]


def _additive_mask_causal(
    position_ids: torch.LongTensor,  # [B,Q]
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Pure causal additive mask [B,1,Q,K]: allow iff k <= q; 0 where allowed, -inf otherwise.
    """
    B, Q = position_ids.shape
    K = kv_len
    last = position_ids[:, -1]
    offset = last - (K - 1)
    key_abs = offset[:, None] + torch.arange(K, device=device, dtype=position_ids.dtype)[None, :]
    q_abs = position_ids[:, :, None]
    allowed = key_abs[:, None, :] <= q_abs
    finfo = torch.finfo(dtype if dtype.is_floating_point else torch.float32)
    neg_inf = torch.tensor(finfo.min, device=device, dtype=dtype if dtype.is_floating_point else torch.float32)
    add = torch.where(allowed, torch.zeros((), device=device, dtype=dtype), neg_inf)
    return add[:, None, :, :]


# -----------------------------------------------------------------------------
# Attention registry (modular converter expects these symbols)
# We keep stock Llama classes; masking happens in CwmDecoderLayer.forward.
# -----------------------------------------------------------------------------
ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


# -----------------------------------------------------------------------------
# Decoder with mask injection, Model, LM Head
# -----------------------------------------------------------------------------
class CwmDecoderLayer(LlamaDecoderLayer):
    """
    Same as LlamaDecoderLayer, but we inject an additive mask (local or causal) per layer
    based on config.layer_types / sliding_window / global_window *before* calling attention.
    """

    def __init__(self, config: CwmTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self._cwm_layer_types = getattr(config, "layer_types", None)
        self._cwm_W_local = int(getattr(config, "sliding_window", 0))
        self._cwm_W_global = getattr(config, "global_window", None)
        if self._cwm_W_global is not None:
            self._cwm_W_global = int(self._cwm_W_global)

    def _cwm_build_mask(
        self,
        position_ids: torch.LongTensor,   # [B,Q]
        past_key_value,                   # cache for this layer
        hidden_states_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # Determine KV length (past + current)
        kv_len = _infer_past_len(past_key_value, self.layer_idx) + position_ids.size(1)

        # Choose policy for this layer
        layer_type = "full_attention"
        if self._cwm_layer_types is not None:
            layer_type = self._cwm_layer_types[self.layer_idx]

        if layer_type == "sliding_attention":
            return _additive_mask_local(position_ids, kv_len, self._cwm_W_local, hidden_states_dtype, device)
        else:
            if self._cwm_W_global is None:
                return _additive_mask_causal(position_ids, kv_len, hidden_states_dtype, device)
            else:
                return _additive_mask_local(position_ids, kv_len, self._cwm_W_global, hidden_states_dtype, device)

    # Mirror LlamaDecoderLayer.forward but only to inject our mask before attention call.
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Build our additive mask (requires position_ids); if not available, skip.
        if position_ids is not None:
            add = self._cwm_build_mask(
                position_ids=position_ids,
                past_key_value=past_key_value,
                hidden_states_dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            attention_mask = add if attention_mask is None else (attention_mask + add)

        # Call the selected attention impl (stock Llama classes via ATTENTION_CLASSES)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs

        return outputs


class CwmMLP(LlamaMLP):
    pass


class CwmRMSNorm(LlamaRMSNorm):
    pass


class CwmRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class CwmPreTrainedModel(LlamaPreTrainedModel):
    config_class = CwmTextConfig
    base_model_prefix = "model"


class CwmModel(LlamaModel):
    """
    Llama backbone; decoder layers are CwmDecoderLayer which inject additive masks per layer.
    """
    config_class = CwmTextConfig

    def __init__(self, config: CwmTextConfig):
        # Favor SDPA when sliding is active
        try:
            if any(t == "sliding_attention" for t in getattr(config, "layer_types", [])) and config.sliding_window > 0:
                config._attn_implementation = "sdpa"
        except Exception:
            pass

        super().__init__(config)

        # Recreate layers using our decoder layer and port submodule weights
        new_layers = torch.nn.ModuleList()
        for i, layer in enumerate(self.layers):
            cwm_layer = CwmDecoderLayer(config, i)
            try:
                cwm_layer.input_layernorm.load_state_dict(layer.input_layernorm.state_dict())
                cwm_layer.post_attention_layernorm.load_state_dict(layer.post_attention_layernorm.state_dict())
                cwm_layer.mlp.load_state_dict(layer.mlp.state_dict())
                # Also copy attention weights from the original (same class & params)
                cwm_layer.self_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
            except Exception as e:
                logger.warning(f"CWM: could not port weights at layer {i}: {e}")
            new_layers.append(cwm_layer)
        self.layers = new_layers


class CwmForCausalLM(LlamaForCausalLM):
    """
    Causal LM head using CwmModel. One Llama-shaped checkpoint; no weight remap required.
    """
    config_class = CwmTextConfig

    def __init__(self, config: CwmTextConfig):
        super().__init__(config)
        self.model = CwmModel(config)


__all__ = [
    "CwmTextConfig",
    "CwmConfig",
    "CwmPreTrainedModel",
    "CwmModel",
    "CwmForCausalLM",
    "CwmMLP",
    "CwmRMSNorm",
    "CwmRotaryEmbedding",
    "CwmDecoderLayer",
    "ATTENTION_CLASSES",
]
