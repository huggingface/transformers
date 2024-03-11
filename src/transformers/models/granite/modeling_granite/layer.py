from typing import Tuple, Union

import torch
import torch.nn as nn

from .attention import get_attention_module
from ..configuration_granite import GraniteConfig
from .enums import AttentionHeadType, AttentionImplementation, NormalizationImplementation
from .mlp import MLP
from .normalization import get_normalization_function


class GraniteBlock(nn.Module):
    """
    Layer implementation for the transformer block
    """

    def __init__(
        self,
        config: GraniteConfig,
        attention_implementation: AttentionImplementation,
        normalization_implementation: NormalizationImplementation,
        layer_idx: int = None,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.layer_idx = layer_idx

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.attn = get_attention_module(
            config, True, layer_idx=layer_idx, attention_implementation=attention_implementation
        )
        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.mlp = MLP(
            config.hidden_size, self.inner_dim, config.activation_function, config.add_bias, config.resid_pdrop
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        alibi_bias: torch.Tensor = None,
        rope_cos_sin: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> Union[
        Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        if self.apply_residual_connection_post_layernorm:
            hidden_states = self.ln_1(hidden_states)
            residual = hidden_states
        else:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            alibi_bias=alibi_bias,
            rope_cos_sin=rope_cos_sin,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if self.apply_residual_connection_post_layernorm:
            hidden_states = self.ln_2(hidden_states)
            residual = hidden_states
        else:
            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs
