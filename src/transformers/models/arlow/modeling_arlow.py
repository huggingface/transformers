# modeling_arlow.py

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

# Import your config. Adjust the path as needed:
from .configuration_arlow import ArlowConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Arlow/arlow-base"
_CONFIG_FOR_DOC = "ArlowConfig"


# -----------------------------------------------------------------------------
# Rotary Embedding Helpers
# -----------------------------------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Splits x into two halves and rotates them. If x = [x1, x2], then
    rotate_half(x) = [-x2, x1].
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary position embedding to query and key.
    q, k => [batch, heads, seq, head_dim]
    cos, sin => [batch, seq, head_dim], typically broadcast over heads dimension.
    """
    cos = cos.unsqueeze(1)  # shape => (batch, 1, seq, head_dim)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ArlowRotaryEmbedding(nn.Module):
    """
    Basic Rotary Embedding. If you want advanced or dynamic rope, expand here.
    """
    def __init__(self, config: ArlowConfig):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_base = config.rope_theta  # e.g. 100000.0

        half_dim = self.head_dim
        inv_freq = 1.0 / (
            self.rope_base ** (torch.arange(0, half_dim, 2).float() / half_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return cos, sin each shape [batch_size, seq_len, head_dim].
        """
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # [seq_len, half_dim/2]
        cos = torch.cat([freqs, freqs], dim=-1).cos()  # [seq_len, half_dim]
        sin = torch.cat([freqs, freqs], dim=-1).sin()  # [seq_len, half_dim]

        cos = cos[None, :, :].expand(batch_size, seq_len, -1).to(dtype=dtype)
        sin = sin[None, :, :].expand(batch_size, seq_len, -1).to(dtype=dtype)
        return cos, sin


# -----------------------------------------------------------------------------
# RMSNorm
# -----------------------------------------------------------------------------
class ArlowRMSNorm(nn.Module):
    """
    Root Mean Square Layer Norm, used in LLaMA-like models.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        return hidden_states.to(orig_dtype)


# -----------------------------------------------------------------------------
# Flash Attention Integration (Grouped Query, etc.)
# -----------------------------------------------------------------------------
try:
    from flash_attn.modules.mha import MHA
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    MHA = None
    FLASH_ATTN_AVAILABLE = False
    logger.warning("flash_attn is not installed. Will not use MHA-based grouped query attention.")


class ArlowGroupedQueryAttention(nn.Module):
    """
    Minimal wrapper around flash_attn's MHA for causal or cross-attention.
    If cross-attn is provided, pass in `encoder_hidden_states`.
    """
    def __init__(self, config: ArlowConfig):
        super().__init__()
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("flash_attn is not installed. Can't instantiate ArlowGroupedQueryAttention.")

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        dropout = getattr(config, "attention_dropout", 0.0)
        # For causal LMs, set 'causal=True'; cross_attn=True allows MHA to do cross-attention if given an encoder hidden state
        self.mha = MHA(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            num_heads_kv=self.num_key_value_heads,
            dropout=dropout,
            causal=True,
            cross_attn=True,
            use_flash_attn=True
        )
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x => [batch_size, seq_len, hidden_size]
        If `encoder_hidden_states` is not None => cross-attn (K/V from encoder).
        For the attention_mask, flash-attn expects `key_padding_mask`, shape [batch, seq_len],
        with 1 for "padding" (ignore) or 0 for "active" tokens. Some data might invert that.
        """
        key_padding_mask = None
        if attention_mask is not None and attention_mask.dim() == 2:
            # If attention_mask has 1 for real tokens and 0 for pads, we must invert it for flash-attn:
            # flash-attn wants 1 for "mask" and 0 for "keep."
            key_padding_mask = (1 - attention_mask).bool()

        if encoder_hidden_states is not None:
            # Cross-attn
            attn_out = self.mha(x, key_value=encoder_hidden_states, key_padding_mask=key_padding_mask)[0]
        else:
            # Self-attn
            attn_out = self.mha(x, key_padding_mask=key_padding_mask)[0]

        attn_out = self.out_proj(attn_out)
        return attn_out


# -----------------------------------------------------------------------------
# One Transformer Layer (FlashAttention-based)
# -----------------------------------------------------------------------------
class ArlowFlashAttentionTransformerLayer(nn.Module):
    """
    A single decoder block that uses:
      - Self-attention via flash-attn MHA.
      - Optional cross-attention (if config.cross_attention=True).
      - RMSNorm, then feed-forward.
    """
    def __init__(self, config: ArlowConfig):
        super().__init__()
        self.self_attn = ArlowGroupedQueryAttention(config)

        # If cross_attention is enabled, define a second attention block
        self.use_cross_attention = config.use_cross_attention
        if config.cross_attention:
            self.cross_attn = ArlowGroupedQueryAttention(config)
            self.norm_cross_attn = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.cross_attn = None

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = nn.SiLU() if config.hidden_act == "silu" else nn.ReLU()

        self.norm1 = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.fc_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc_out = nn.Linear(config.intermediate_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1) Self-Attn
        residual = hidden_states
        hidden_states_normed = self.norm1(hidden_states)
        self_attn_out = self.self_attn(
            hidden_states_normed,
            attention_mask=attention_mask,
            encoder_hidden_states=None  # self-attn => no enc states
        )
        hidden_states = residual + self_attn_out

        # 2) Optional Cross-Attn
        if self.cross_attn is not None and self.use_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            cross_normed = self.norm_cross_attn(hidden_states)
            cross_out = self.cross_attn(
                cross_normed,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states
            )
            hidden_states = residual + cross_out

        # 3) Feed-Forward
        residual = hidden_states
        hidden_states_normed2 = self.norm2(hidden_states)
        ff_out = self.fc_out(self.dropout(self.act_fn(self.fc_in(hidden_states_normed2))))
        hidden_states = residual + ff_out

        return hidden_states


# -----------------------------------------------------------------------------
# ArlowPreTrainedModel
# -----------------------------------------------------------------------------
class ArlowPreTrainedModel(PreTrainedModel):
    """
    Base class for all Arlow-based models.
    Handles config, weight init, and hooking into HF save/load.
    """
    config_class = ArlowConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ArlowModel):
            module.gradient_checkpointing = value


# -----------------------------------------------------------------------------
# ArlowModel (the Backbone)
# -----------------------------------------------------------------------------
class ArlowModel(ArlowPreTrainedModel):
    r"""
    This is a decoder-only Transformer model.

    Examples:
    ```python
    >>> from transformers import AutoModel, ArlowConfig

    >>> # Using a placeholder name "Arlow/arlow-base"
    >>> model = AutoModel.from_pretrained("Arlow/arlow-base")
    ```
    """

    def __init__(self, config: ArlowConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )

        # Optional: If you want to apply rotary embeddings yourself, define a separate logic or call ArlowRotaryEmbedding
        self.rotary_emb = ArlowRotaryEmbedding(config)

        self.layers = nn.ModuleList(
            [ArlowFlashAttentionTransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = ArlowRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,       # not used by default
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **unused
    ) -> Union[BaseModelOutputWithPast, Tuple[torch.Tensor]]:
        # 1) Validate Inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot provide both input_ids and inputs_embeds")

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            hidden_states = self.embed_tokens(input_ids)
        else:
            batch_size, seq_len, _ = inputs_embeds.shape
            hidden_states = inputs_embeds

        # 2) (Optional) If you truly want to apply RoPE, do so per-head.
        #    Typically you'd do Q/K split in the attention.
        #    We'll just define cos, sin here if you want to pass them in:
        # cos, sin = self.rotary_emb(batch_size, seq_len, hidden_states.device, hidden_states.dtype)
        # Then you'd integrate them inside the layer's self-attn logic if needed.

        all_hidden_states = () if output_hidden_states else None

        # For gradient checkpointing
        def custom_forward(layer_module, hidden_states, attention_mask, enc_h, enc_mask):
            return layer_module(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=enc_h,
                encoder_attention_mask=enc_mask
            )

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    layer,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask
                )

        # Final RMS norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return (hidden_states, all_hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=None,  # Not returning individual layer attentions here
        )


# -----------------------------------------------------------------------------
# ArlowForCausalLM
# -----------------------------------------------------------------------------
class ArlowForCausalLM(ArlowPreTrainedModel, GenerationMixin):
    """
    Wraps ArlowModel with a language modeling head to produce vocabulary logits.
    """

    def __init__(self, config: ArlowConfig):
        super().__init__(config)
        self.model = ArlowModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings if asked
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.model.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **unused
    ) -> Union[CausalLMOutputWithPast, Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through the base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Standard causal LM shift:
            # tokens < n predict nth token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not return_dict:
            out = (logits,) + outputs[1:]
            return ((loss,) + out) if loss is not None else out

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # Not implemented
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def tie_weights(self):
        """
        Ties the input embeddings and output embeddings if
        config.tie_word_embeddings=True.
        """
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight


__all__ = [
    "ArlowPreTrainedModel",
    "ArlowModel",
    "ArlowForCausalLM",
#   "ArlowFlashAttentionTransformerLayer",
#   "ArlowGroupedQueryAttention",
#   "ArlowRMSNorm",
#   "ArlowRotaryEmbedding",
#   "apply_rotary_pos_emb",
]
