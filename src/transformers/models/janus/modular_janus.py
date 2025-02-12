import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
    torch_int,
)

from .configuration_janus import JanusVisionEncoderConfig
from ..vit.modeling_vit import ViTPatchEmbeddings
from ..dinov2_with_registers.modeling_dinov2_with_registers  import Dinov2WithRegistersLayerScale, Dinov2WithRegistersDropPath
from ..siglip.modeling_siglip import SiglipEncoder, SiglipVisionTransformer, SiglipVisionModel, SiglipMultiheadAttentionPoolingHead

class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """
    return_indices: torch.jit.Final[bool]

    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
            return_indices: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        self.return_indices = return_indices

    def forward(self, x) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if not self.training or self.prob == 0.:
            if self.return_indices:
                return x, None
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1. - self.prob)))
        keep_indices = torch.argsort(torch.randn(B, L, device=x.device), dim=-1)[:, :num_keep]
        if self.ordered:
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices = keep_indices.sort(dim=-1)[0]
        x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x

class JanusVisionEncoderPatchEmbeddings(ViTPatchEmbeddings):
    pass


class JanusVisionEncoderEmbeddings(nn.Module):
    def __init__(self, config:JanusVisionEncoderConfig, use_special_tokens: bool,):
        super().__init__()

        self.use_special_tokens = use_special_tokens
        if use_special_tokens:
            self.cls_token = nn.Parameter(torch.rand(1,1,config.hidden_size))
            self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))

        # Currently using hidden_drop_rate instead of positional_dropout_rate, is it necessary?
        self.dropout = nn.Dropout(config.hidden_dropout_rate)
        self.patch_embeddings = JanusVisionEncoderPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches

        num_prefix_tokens = config.num_register_tokens + 1
        pos_embed_len = self.num_patches + num_prefix_tokens if use_special_tokens else self.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, pos_embed_len, config.hidden_size) * 0.02)

        # Used to reduce computationality.
        self.patch_dropout = PatchDropout(config.drop_path_rate, num_prefix_tokens) if config.drop_path_rate else nn.Identity()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        # Add CLS and Register token embeddings.
        special_token_embeddings = []
        if self.use_special_tokens:
            cls_token_embeddings = self.cls_token.expand((batch_size, -1,-1))
            special_token_embeddings.append(cls_token_embeddings)

            if self.register_tokens.shape[1]:
                register_token_embeddings = self.register_tokens.expand((batch_size, -1,-1))
                special_token_embeddings.append(register_token_embeddings)

        if self.use_special_tokens:
            embeddings = embeddings + self.position_embeddings
            embeddings = torch.cat(special_token_embeddings+[embeddings], dim=1)
        else:
            # embeddings = torch.cat(special_token_embeddings+[embeddings], dim=1)
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)


        # Perform Patch dropout
        embeddings = self.patch_dropout(embeddings)
        return embeddings

# Todo: introduce compatiability for cache
class JanusVisionEncoderAttention(nn.Module):
    """Attention Class for Janus Vision Encoder """
    def __init__(self, config: JanusVisionEncoderConfig):
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
        self.attention_dropout = config.attention_dropout
        proj_dropout = config.projection_dropout
        qk_norm = config.use_qk_norm

        # Split the weights manually and checkif getting correct output or not
        self.qkv = nn.Linear(self.embed_dim, 3*self.embed_dim, bias=config.qkv_bias)
        self.projection_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        self.query_norm = nn.LayerNorm(self.embed_dim) if qk_norm else nn.Identity()
        self.key_norm = nn.LayerNorm(self.embed_dim) if qk_norm else nn.Identity()

    def forward(self,hidden_states:torch.Tensor,attention_mask: Optional[torch.Tensor] = None, output_attentions: Optional[torch.Tensor] = None):
        batch_size , seq_len, _ = hidden_states.size()

        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = qkv.unbind(0)

        query_states = self.query_norm(query_states)
        key_states = self.key_norm(key_states)

        # Is it a bug or deliberate change?
        query_states = query_states * self.scale

        # batch, num head,seqlen,seqlen
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Only apply attention dropout during training.
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (batch_size,1, seq_len, self.head_dim):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, seq_len, self.head_dim)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.reshape( batch_size, seq_len, self.embed_dim )

        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)

        outputs = (output, attn_weights) if output_attentions else (output, None)
        return outputs

class JanusVisionEncoderLayerScale(Dinov2WithRegistersLayerScale):
    pass
class JanusVisionEncoderDropPath(Dinov2WithRegistersDropPath):
    pass

class JanusVisionEncoderMLP(nn.Module):
    def __init__(self, config:JanusVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act] # Gelu act
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_rate)
        self.dropout2 = nn.Dropout(config.hidden_dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        return hidden_states

class JanusVisionEncoderLayer(nn.Module):
    def __init__(self,config: JanusVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = JanusVisionEncoderAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)

        self.layer_scale1 = JanusVisionEncoderLayerScale(config) if config.layerscale_value else nn.Identity()
        self.layer_scale2 = JanusVisionEncoderLayerScale(config) if config.layerscale_value else nn.Identity()
        self.drop_path1 = (
            JanusVisionEncoderDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            JanusVisionEncoderDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )
        self.mlp = JanusVisionEncoderMLP(config)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # Pre-Norm before attention .
        norm_hidden_states = self.layer_norm1(hidden_states)
        attn_output, attn_weights = self.self_attn(
            norm_hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )

        scaled_attn_output = self.layer_scale1(attn_output)
        dropped_attn_output = self.drop_path1(scaled_attn_output)
        hidden_states = hidden_states + dropped_attn_output

        norm_hidden_states = self.layer_norm2(hidden_states)

        mlp_output = self.mlp(norm_hidden_states)

        scaled_mlp_output = self.layer_scale2(mlp_output)
        dropped_mlp_output = self.drop_path2(scaled_mlp_output)
        hidden_states = hidden_states + dropped_mlp_output

        return (hidden_states, attn_weights if output_attentions else None)

# copied from SiglipMultiheadAttentionPoolingHead
# class JanusAttentionPoolLatent(SiglipMultiheadAttentionPoolingHead):
#     pass

class JanusAttentionPoolLatent(nn.Module):
    """ Hugging Face-compatible Attention Pooling with Manual QKV """

    def __init__(self, config: JanusVisionEncoderConfig):
        super().__init__()

        self.latent_len = getattr(config, "latent_len", 1)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.mlp_ratio = getattr(config, "mlp_ratio", 4.0)
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention

        # Learnable latent query (probe)
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, self.hidden_size))

        # Linear layers for QKV projection
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.kv_proj = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Normalization & MLP
        self.norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = JanusVisionEncoderMLP(config)

        self.proj_drop = nn.Dropout(getattr(config, "dropout", 0.0))

    def forward(self, hidden_state):
        batch_size, seq_len, _ = hidden_state.shape

        # Expand learnable latent tokens for batch
        q_latent = self.latent.expand(batch_size, -1, -1)  # (B, latent_len, hidden_size)

        # Compute Q projection from latent tokens
        q = self.q_proj(q_latent)  # (B, latent_len, hidden_size)

        # Compute combined KV projection
        kv = self.kv_proj(hidden_state)  # (B, seq_len, 2 * hidden_size)
        k, v = kv.view(batch_size, seq_len, 2, self.num_heads, self.head_dim).unbind(2)
        # Batch_sisxe, num_heads, seq_len, head_dim
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # Reshape Q for multi-head attention (B, N, H) → (B, num_heads, N, head_dim)
        q = q.view(batch_size, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, latent_len, head_dim)

        # Scaled dot-product attention (without `torch.nn.MultiheadAttention`)
        attn_weights = (q @ k.transpose(2,3)) * self.scale  # (B, num_heads, latent_len, seq_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        x = attn_weights @ v  # (B, num_heads, latent_len, head_dim)

        x = x.transpose(1, 2).reshape(batch_size, self.latent_len, self.hidden_size)

        x = self.proj(x)
        x = self.proj_drop(x)

        # Residual connection + MLP
        residual = x
        x = self.norm(x)
        x = residual + self.mlp(x)

        return x[:, 0]


# Copied from siglip encoder
class JanusVisionEncoder(nn.Module):

    def __init__(self,config:JanusVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([JanusVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class JanusPreTrainedModel():
     """An abstract class to load pretrained weigths"""
     pass

# Copied from siglip vision transformer
class JanusVisionEncoderTransformer(nn.Module):
    def __init__(self, config: JanusVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embeddings = JanusVisionEncoderEmbeddings(config,use_special_tokens=False)
        self.encoder = JanusVisionEncoder(config)
        self.layenorm = nn.LayerNorm(config.hidden_size)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = JanusAttentionPoolLatent(config)
            # Won't be using as a standalone classifier head hence no num classes

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.layenorm(last_hidden_state)
        pooler_output = self.head(hidden_states)


        if not return_dict:
            return (last_hidden_state) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )