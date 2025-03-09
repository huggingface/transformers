# coding=utf-8
# Copyright 2025 Apple Inc. and The HuggingFace Team. All rights reserved.
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

"""Pytorch implementation of AIMv2 Model"""

from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

from ...activations import ACT2FN
from ...utils import (
    logging,
)
from ..llama.modeling_llama import LlamaRMSNorm
from ..siglip.modeling_siglip import SiglipEncoder
from .configuration_aimv2 import AIMv2Config


logger = logging.get_logger(__name__)


class AIMv2RMSNorm(LlamaRMSNorm):
    pass


class AIMv2SwiGLUFFN(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        in_features = config.hidden_size
        out_features = config.intermediate_size
        self.act_fn = config.hidden_act

        self.fc1 = nn.Linear(in_features, out_features, bias=config.use_bias)
        self.fc2 = nn.Linear(out_features, in_features, bias=config.use_bias)
        self.fc3 = nn.Linear(in_features, out_features, bias=config.use_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        fc3_out = self.fc3(hidden_states)
        fc1_out = self.fc1(hidden_states)
        hidden_states = ACT2FN[self.act_fn](fc1_out) * fc3_out
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class AIMv2Embeddings(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.rms_norm = AIMv2RMSNorm(config.hidden_size, config.rms_norm_eps)

        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embeddings = nn.Embedding(num_patches, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)), persistent=False)

    @staticmethod
    def build_2d_sincos_position_embedding(height, width, embed_dim):
        pass

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        hidden_states = self.rms_norm(hidden_states)

        _, num_patches, _ = hidden_states.size()

        # added logic for native in build s2d sincos pos embed
        hidden_states = hidden_states + self.position_embeddings(self.position_ids)

        return hidden_states

# Replace atttn_mask with head mask
def eager_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Only apply attention dropout during training.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class AIMv2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.attention_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.attention_bias)
        self.proj_out = nn.Linear(self.embed_dim, self.embed_dim, bias=config.attention_bias)
        self.proj_drop = nn.Dropout(config.projection_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scale,
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.proj_out(attn_output)
        attn_output = self.proj_drop(attn_output)

        output = (attn_output, attn_weights) if output_attentions else (attn_output, None)

        return output


class AIMv2EncoderLayer(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.attention = AIMv2Attention(config)
        self.ffn = AIMv2SwiGLUFFN(config)
        self.rms_norm1 = AIMv2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rms_norm2 = AIMv2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask, output_attentions: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states = self.rms_norm1(hidden_states)
        attn_output, attn_wights = self.attention(
            hidden_states=norm_hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )

        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.rms_norm2(hidden_states)
        mlp_output = self.ffn(norm_hidden_states)

        hidden_states = hidden_states + mlp_output
        return (hidden_states, attn_wights) if output_attentions else (hidden_states, None)


class AIMv2Encoder(SiglipEncoder):
    pass


class AIMv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AIMv2Config
    base_model_prefix = "aimv2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AIMv2SwiGLUFFN"]
    _supports_sdpa = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d]) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, AIMv2Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)


class AIMv2Model(AIMv2PreTrainedModel):
    def __init__(self, config: AIMv2Config):
        super().__init__(config)
        self.config = config
        self.embeddings = AIMv2Embeddings(config)
        self.encoder = AIMv2Encoder(config)
        self.rms_norm = AIMv2RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.rms_norm(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


__all__ = ["AIMv2Model"]
