from dataclasses import dataclass
import copy
import math
import re
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from typing import Optional, List, Tuple
from transformers import CONFIG_MAPPING
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.utils.generic import ModelOutput


# Get this from the original kosmos-2 demo
original_kosmos2_checkpoint_only_2_layers = "kosmos-2-state-dict-num-layers-2.bin"
dog_sample_file = "sample.bin"

# ==============================================================================================================
# Config class

from transformers.models.kosmos2.configuration_kosmos2 import Kosmos2Config, Kosmos2TextConfig, Kosmos2VisionConfig


# ==============================================================================================================
# Vision model class
# (completely copied from CLIPVision)

# -----------------------------------------------------------------------
# Fake values to make it work


from typing import Union
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
KOSMOS2_VISION_INPUTS_DOCSTRING = ""
KOSMOS2_START_DOCSTRING = ""
from transformers import add_start_docstrings
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings



# -----------------------------------------------------------------------

class Kosmos2VisionEmbeddings(nn.Module):
    def __init__(self, config: Kosmos2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class Kosmos2VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
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
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class Kosmos2VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Kosmos2VisionEncoderLayer(nn.Module):
    def __init__(self, config: Kosmos2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Kosmos2VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Kosmos2VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Kosmos2VisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Kosmos2VisionEncoderLayer`].

    Args:
        config: Kosmos2VisionConfig
    """

    def __init__(self, config: Kosmos2VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Kosmos2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
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
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

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
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
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


class Kosmos2VisionTransformer(nn.Module):
    def __init__(self, config: Kosmos2VisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Kosmos2VisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = Kosmos2VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(KOSMOS2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Kosmos2VisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# ==============================================================================================================
# Model class


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx



class Kosmos2TextSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0
    ):
        if input_ids is not None:
            bsz, seq_len = input_ids.size()
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
                input_ids.device
            )
        else:
            bsz, seq_len = inputs_embeds.size()[:-1]
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()

    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length


class KosmosTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        add_inner_attn_layernorm=False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.inner_attn_ln = None
        if add_inner_attn_layernorm:
            self.inner_attn_ln = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        # new
        if self.inner_attn_ln is not None:
            attn_output = self.inner_attn_ln(attn_output)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class Kosmos2TextFFN(nn.Module):
    def __init__(self, config: Kosmos2TextConfig):
        super().__init__()

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.fc1 = nn.Linear(config.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.embed_dim)

        self.ffn_layernorm = nn.LayerNorm(config.ffn_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.ffn_layernorm(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        return hidden_states


class Kosmos2TextBlock(nn.Module):
    def __init__(self, config: Kosmos2TextConfig):
        super().__init__()
        self.embed_dim = config.embed_dim

        self.self_attn = KosmosTextAttention(
            config,
            embed_dim=self.embed_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            add_inner_attn_layernorm=True,
        )

        self.dropout = config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        if config.add_cross_attention:
            self.encoder_attn = KosmosTextAttention(
                config,
                self.embed_dim,
                config.attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                add_inner_attn_layernorm=False,
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.ffn = Kosmos2TextFFN(config)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        hidden_states = self.self_attn_layer_norm(hidden_states)

        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            if not hasattr(self, "encoder_attn"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            residual = hidden_states

            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states

        hidden_states = self.final_layer_norm(hidden_states)

        # FFN
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Kosmos2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Kosmos2Config
    # base_model_prefix = "kosmos_2_text"


class Kosmos2TextTransformer(nn.Module):

    def __init__(self, config: Kosmos2TextConfig):
        super().__init__()
        self.config = config

        self.dropout = config.dropout

        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)

        self.embed_positions = Kosmos2TextSinusoidalPositionalEmbedding(
            num_positions=config.max_position_embeddings,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_token_id
        )

        self.embed_scale = math.sqrt(config.embed_dim) if config.scale_embedding else 1.0

        self.layers = nn.ModuleList([Kosmos2TextBlock(config) for _ in range(config.layers)])

        self.layer_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)

        self.gradient_checkpointing = False

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward_embedding(self, input_ids, inputs_embeds=None, img_features=None, img_input_mask=None, past_key_values_length: int = 0):

        # The argument `inputs_embeds` should be the one without being multiplied by `self.embed_scale`
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if img_features is not None:
            inputs_embeds[img_input_mask] = img_features

        inputs_embeds = inputs_embeds * self.embed_scale

        # We need to use inputs ids! otherwise we don't know where is the padding!
        # embed positions
        positions = self.embed_positions(input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        return hidden_states

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        img_features = None,
        img_attn_mask = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # We don't need img info is `past_key_values_length` > 0
        if past_key_values_length > 0:
            img_features = None
            img_attn_mask = None

        hidden_states = self.forward_embedding(
            input_ids=input_ids, inputs_embeds=inputs_embeds, img_features=img_features, img_input_mask=img_attn_mask, past_key_values_length=past_key_values_length,
        )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, hidden_states, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add final layer norm
        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class Kosmos2TextModel(Kosmos2PreTrainedModel):
    config_class = Kosmos2TextConfig

    def __init__(self, config: Kosmos2TextConfig):
        super().__init__(config)
        self.model = Kosmos2TextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        img_features = None,
        img_attn_mask = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            img_features=img_features,
            img_attn_mask=img_attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs


class Kosmos2TextForCausalLM(Kosmos2PreTrainedModel):
    config_class = Kosmos2TextConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Kosmos2TextConfig):
        super().__init__(config)

        self.model = Kosmos2TextTransformer(config)
        self.lm_head = nn.Linear(in_features=config.embed_dim, out_features=config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        img_features = None,
        img_attn_mask = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            img_features=img_features,
            img_attn_mask=img_attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, img_features, img_attn_mask, past_key_values=None, attention_mask=None, use_cache=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut input_ids if past_key_values is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            # the image info. is already encoded into the past keys/values
            img_features = None,
            img_attn_mask = None,
        elif img_attn_mask is not None:
            # appending `False` to `img_attn_mask` (because `input_ids` grows during generation)
            batch_size, seq_len = input_ids.size()
            mask_len = img_attn_mask.size()[-1]
            img_attn_mask = torch.cat((img_attn_mask, torch.zeros(size=(batch_size, seq_len - mask_len), dtype=torch.bool)), dim=1)

        return {
            "input_ids": input_ids,
            "img_features": img_features,
            "img_attn_mask": img_attn_mask,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }


# ==============================================================================================================
# Vision model class
# (completely copied from CLIPVision)

@add_start_docstrings(
    """The vision model from KOSMOS2 without any head or projection on top.""",
    KOSMOS2_START_DOCSTRING,
)
class Kosmos2VisionModel(Kosmos2PreTrainedModel):
    config_class = Kosmos2VisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Kosmos2VisionConfig):
        super().__init__(config)
        self.vision_model = Kosmos2VisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(KOSMOS2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Kosmos2VisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Kosmos2VisionModel

        >>> model = Kosmos2VisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# ==============================================================================================================


class KosmosConnector(nn.Module):

    def __init__(self, config: Kosmos2Config):
        super().__init__()
        self.dense = nn.Linear(config.vision_config.hidden_size, config.text_config.embed_dim)
        self.latent_query = nn.Parameter(torch.randn(config.latent_query_num, config.text_config.embed_dim))

        self.x_attn = KosmosTextAttention(
            config.text_config,
            config.text_config.embed_dim,
            config.text_config.attention_heads,
            # shared with text,
            dropout=config.text_config.attention_dropout,
            is_decoder=False,
            add_inner_attn_layernorm=False,
        )

    def forward(self, features):

        hidden_states = self.dense(features)

        # [batch, latent_query_num, h_dim]
        latent_query = self.latent_query.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
        key_value_states = torch.cat([hidden_states, latent_query], dim=1)

        hidden_states, attn_weights, _ = self.x_attn(
            hidden_states=latent_query,
            key_value_states=key_value_states,
            past_key_value=None,
            attention_mask=None,
            output_attentions=None,
        )

        return hidden_states, attn_weights


@dataclass
class Kosmos2ModelOutput(ModelOutput):
    """
    """

    last_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_features: Optional[torch.FloatTensor] = None
    image_connector_attention: Optional[torch.FloatTensor] = None
    vision_model_output: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Kosmos2ForConditionalGenerationModelOutput(ModelOutput):
    """
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_features: Optional[torch.FloatTensor] = None
    image_connector_attention: Optional[torch.FloatTensor] = None
    vision_model_output: Optional[Tuple[torch.FloatTensor]] = None


class Kosmos2Model(Kosmos2PreTrainedModel):
    config_class = Kosmos2Config

    def __init__(
        self,
        config: Kosmos2Config,
    ):
        super().__init__(config)

        self.text_model = Kosmos2TextModel(config.text_config)
        self.vision_model = Kosmos2VisionTransformer(config.vision_config)

        self.img_connector = KosmosConnector(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.text_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.text_model.model.embed_tokens = value

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        img_attn_mask = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask = None,
        head_mask: Optional[torch.Tensor] = None,
        img_features: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_model_output = None
        image_connector_attention = None
        if img_features is None:

            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            vision_model_output = self.vision_model(pixel_values)
            # HF CLIP has `last_hidden_state` without through `post_layernorm`
            # Here we want to pass the whole `last_hidden_state` instead of `pooled_output` from clip model
            img_features = self.vision_model.post_layernorm(vision_model_output.last_hidden_state)
            # normalized
            img_features = nn.functional.normalize(img_features, dim=-1)
            img_features, image_connector_attention = self.img_connector(img_features)

        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            img_features=img_features,
            img_attn_mask=img_attn_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            outputs = outputs + (img_features, image_connector_attention, vision_model_output)
            return tuple(output for output in outputs if output is not None)

        return Kosmos2ModelOutput(
            last_hidden_states=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_features=img_features,
            image_connector_attention=image_connector_attention,
            vision_model_output=vision_model_output,
        )


class Kosmos2ForConditionalGeneration(Kosmos2PreTrainedModel):
    config_class = Kosmos2Config
    _tied_weights_keys = ["text_model.lm_head.weight"]

    def __init__(
        self,
        config: Kosmos2Config,
    ):
        super().__init__(config)

        self.text_model = Kosmos2TextForCausalLM(config.text_config)
        self.vision_model = Kosmos2VisionTransformer(config.vision_config)

        self.img_connector = KosmosConnector(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.text_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.text_model.model.embed_tokens = value

    # We can't have this (with `config.tie_word_embeddings=True`) if we don't implement `get_input_embeddings` above
    def get_output_embeddings(self):
        return self.text_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.text_model.set_output_embeddings(new_embeddings)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        img_attn_mask = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask = None,
        head_mask: Optional[torch.Tensor] = None,
        img_features: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_model_output = None
        image_connector_attention = None
        if img_features is None:

            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            vision_model_output = self.vision_model(pixel_values)
            # HF CLIP has `last_hidden_state` without through `post_layernorm`
            # Here we want to pass the whole `last_hidden_state` instead of `pooled_output` from clip model
            img_features = self.vision_model.post_layernorm(vision_model_output.last_hidden_state)
            # normalized
            img_features = nn.functional.normalize(img_features, dim=-1)
            img_features, image_connector_attention = self.img_connector(img_features)

        lm_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            img_features=img_features,
            img_attn_mask=img_attn_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            outputs = lm_outputs + (img_features, image_connector_attention, vision_model_output)
            return tuple(output for output in outputs if output is not None)

        return Kosmos2ForConditionalGenerationModelOutput(
            loss=lm_outputs.loss,
            logits=lm_outputs.logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
            image_features=img_features,
            image_connector_attention=image_connector_attention,
            vision_model_output=vision_model_output,
        )

    def generate(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        img_features=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # to allow `inputs` argument
        inputs = kwargs.pop("inputs", None)
        if pixel_values is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs} were passed alongside `pixel_values` which is not allowed."
                f"Make sure to either pass `inputs` or pixel_values=..."
            )
        if pixel_values is None:
            if inputs is not None:
                pixel_values = inputs

        if img_features is None:

            vision_model_output = self.vision_model(pixel_values)
            # HF CLIP has `last_hidden_state` without through `post_layernorm`
            # Here we want to pass the whole `last_hidden_state` instead of `pooled_output` from clip model
            img_features = self.vision_model.post_layernorm(vision_model_output.last_hidden_state)
            # normalized
            img_features = nn.functional.normalize(img_features, dim=-1)
            img_features, image_connector_attention = self.img_connector(img_features)

        output = self.text_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            img_features=img_features,
            input_embeds=inputs_embeds,
            **kwargs,
        )

        return output

# ==============================================================================================================
# conversion


def rename_vision_key(key):

    # text decoder
    key = re.sub(r"img_model.visual\.", "vision_model.", key)

    key = re.sub(r"\.class_embedding$", ".embeddings.class_embedding", key)
    key = re.sub(r"\.positional_embedding$", ".embeddings.position_embedding.weight", key)
    key = re.sub(r"\.conv1.weight$", ".embeddings.patch_embedding.weight", key)

    key = re.sub(r"\.ln_pre\.", ".pre_layrnorm.", key)


    key = re.sub(r"\.transformer.resblocks\.", ".encoder.layers.", key)

    key = re.sub(r"\.ts_attn\.", ".self_attn.", key)

    key = re.sub(r"\.ln_1\.", ".layer_norm1.", key)
    key = re.sub(r"\.ln_2\.", ".layer_norm2.", key)

    key = re.sub(r"\.c_fc\.", ".fc1.", key)
    key = re.sub(r"\.c_proj\.", ".fc2.", key)

    key = re.sub(r"\.ln_post\.", ".post_layernorm.", key)

    return key


def rename_key(key):
    # text decoder
    key = re.sub(r"gpt_model.decoder\.", "text_model.", key)
    # text decode: `embed_tokens`
    key = re.sub(r"\.embed_tokens\.", ".model.embed_tokens.", key)

    # text decode: `embed_positions` (no weight)
    # key: gpt_model.decoder.embed_positions._float_tensor
    # renamed_key: text_model.embed_positions._float_tensor

    key = re.sub(r"\.layers\.", ".model.layers.", key)

    key = re.sub(r"^text_model.layer_norm\.", "text_model.model.layer_norm.", key)

    key = re.sub(r"^text_model.output_projection\.", "text_model.lm_head.", key)

    # not used in forward!
    # self.self_attn_sope

    key = rename_vision_key(key)

    return key


# ==============================================================================================================
# Original model topology


"""
UniGPTmodel(
  (gpt_model): TransformerLanguageModel(
    (decoder): LMDecoder(
      (dropout_module): Dropout(p=0.1, inplace=True)
      (embed_tokens): Embedding(65037, 2048, padding_idx=1)
      (embed_positions): SinusoidalPositionalEmbedding()place=True)
      (output_projection): Linear(in_features=2048, out_features=65037, bias=False)
      (layers): ModuleList(features=2048, out_features=8192, bias=True)
        (0-23): 24 x DecoderLayer(s=8192, out_features=2048, bias=True)
          (dropout_module): Dropout(p=0.1, inplace=True)92]), eps=1e-05, elementwise_affine=True)
          (self_attn): MultiheadAttention(
            (k_proj): Linear(in_features=2048, out_features=2048, bias=True)ementwise_affine=True)
            (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
            (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
            (out_proj): Linear(in_features=2048, out_features=2048, bias=True)fine=True)
            (inner_attn_ln): FusedLayerNorm(torch.Size([2048]), eps=1e-05, elementwise_affine=True)
            (dropout_module): Dropout(p=0.1, inplace=True)
          )
          (self_attn_layer_norm): FusedLayerNorm(torch.Size([2048]), eps=1e-05, elementwise_affine=True)
          (ffn): FeedForwardNetwork(
            (activation_dropout_module): Dropout(p=0.0, inplace=True)
            (dropout_module): Dropout(p=0.1, inplace=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (ffn_layernorm): FusedLayerNorm(torch.Size([8192]), eps=1e-05, elementwise_affine=True)
          )
          (final_layer_norm): FusedLayerNorm(torch.Size([2048]), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): FusedLayerNorm(torch.Size([2048]), eps=1e-05, elementwise_affine=True)
      (self_attn_sope): SoPE()
    )
  )
  (img_model): ClipVisualOnly(
    (visual): VisualTransformer4Seq2Seq(
      (conv1): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
      (ln_pre): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (transformer): Transformer(
        (resblocks): ModuleList(
          (0-23): 24 x ResidualAttentionBlock(
            (attn): None
            (ts_attn): MultiheadAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout_module): Dropout(p=0.0, inplace=True)
            )
            (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (ln_post): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
"""


# ==============================================================================================================
# Vision


# load `kosmos-2` img_model to HF clip vision
# (with `model.img_model` --> `clip.vision_model`)


def load_and_check_vision_model():

    # ================================================================================
    from transformers import AutoModel
    hf_clip_model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")

    # ================================================================================
    # load (small) state dict
    state_dict = torch.load(original_kosmos2_checkpoint_only_2_layers)
    state_dict_keys = list(state_dict.keys())

    # ================================================================================
    # rename key

    renamed_keys = []
    for key in state_dict_keys:
        if key.startswith("img_model"):
            renamed_key = rename_vision_key(key)
            renamed_keys.append(renamed_key)

    # ================================================================================
    # keep only 2 layers in HF clip model

    hf_clip_vision_keys = []
    for key in hf_clip_model.state_dict().keys():
        if key.startswith("vision_model") and not any(f".layers.{x}." in key for x in range(2, 25)):
            hf_clip_vision_keys.append(key)

    # ================================================================================
    # check renamed keys

    print(set(hf_clip_vision_keys).difference(renamed_keys))
    assert(set(hf_clip_vision_keys).issubset(renamed_keys))


def load_and_check_model(model, ckpt_path):

    # ================================================================================
    # load (small) state dict
    state_dict = torch.load(ckpt_path)
    state_dict_keys = list(state_dict.keys())

    # ================================================================================
    # rename key

    renamed_state_dict_keys = [rename_key(k) for k in state_dict_keys]

    # ================================================================================
    # check renamed keys

    model_state_dict_keys = list(model.state_dict().keys())
    diff_keys = set(model_state_dict_keys).difference(renamed_state_dict_keys)
    print(diff_keys)

    # all HF model keys should be in the renamed keys from the original checkpoint
    assert set(model_state_dict_keys).issubset(renamed_state_dict_keys)

    # ================================================================================
    # Create new model state dict

    loaded_model_state_dict = {}
    for key in state_dict_keys:
        renamed_key = rename_key(key)
        loaded_model_state_dict[renamed_key] = state_dict[key]

    # ================================================================================
    # check weight loading

    model.load_state_dict(loaded_model_state_dict, strict=False)


def check_model_with_dummy_inputs(model):
    """
    The `original model` here is the original `kosmos-2` model with the first 2 layers in both its text and vision
    components.
    """

    # ================================================================================
    # check loaded text model outputs

    # --------------------------------------------------------------------
    # For original kosmos-2

    # dummy_input_ids = torch.arange(0, 0 + 71, device="cuda").unsqueeze(0).clone()
    # # dummy_input_ids = torch.arange(2, 2 + 71, device="cuda").unsqueeze(0).clone()

    # original_text_outputs = model.gpt_model.decoder(dummy_input_ids, features_only=False)

    # # (original_text_outputs[0] is `logits`)
    # print(original_text_outputs[0].shape)
    # print(original_text_outputs[0])
    """
    torch.Size([1, 71, 65037])

    tensor(
        [
            [
                [15.6060, -5.1565,  8.0064,  ..., -2.2804, -2.0610, -1.0114],
                [ 9.7421, -4.9860,  4.9630,  ..., -1.5206, -1.4377, -0.5475],
                [11.6581, -5.1816, 18.9927,  ..., -2.3973, -1.9231, -1.3065],
                ...,
                [10.6616, -4.7872,  5.0161,  ..., -2.1092, -1.6931, -1.6256],
                [10.9655, -4.8194,  5.6438,  ..., -1.5778, -0.9324, -0.3715],
                [ 9.8335, -4.9696,  4.6688,  ..., -2.2745, -1.7485, -1.7921],
            ]
        ],
    )
    """

    # --------------------------------------------------------------------
    # Ours

    # Including the padding token id `1` in `input_ids` to make sure everything work
    # (especially the positional embedding)
    dummy_input_ids = torch.arange(0, 0 + 71, device="cpu").unsqueeze(0).clone()
    # dummy_input_ids = torch.arange(2, 2 + 71, device="cpu").unsqueeze(0).clone()

    hf_outputs = model.text_model(dummy_input_ids)

    print(hf_outputs.logits.shape)
    print(hf_outputs.logits)

    # --------------------------------------------------------------------
    # sanity check

    assert list(hf_outputs.logits.shape) == [1, 71, 65037]

    expected_block_1 = torch.tensor(
        [
            [15.605998,  -5.156513,   8.00637  ],
            [ 9.742102,  -4.9859715,  4.9629903],
            [11.658097,  -5.1816287, 18.992657 ],
        ],
    )
    expected_block_2 = torch.tensor(
        [
            [-2.2804384,  -2.0610392,  -1.0114111],
            [-1.5205702,  -1.4377496,  -0.54751855],
            [-2.39729,    -1.9231234,  -1.3065333],
        ],
    )
    expected_block_3 = torch.tensor(
        [
            [10.661624,  -4.7872415,  5.016076 ],
            [10.965461,  -4.819447,   5.6437974],
            [ 9.833488,  -4.9696445,  4.668755 ],
        ],
    )
    expected_block_4 = torch.tensor(
        [
            [-2.1091897, -1.693118,  -1.6255622],
            [-1.5777528, -0.9324262, -0.3714557],
            [-2.2744524, -1.7484771, -1.7920786],
        ],
    )

    diff_1 = torch.max(torch.abs(hf_outputs.logits[0, :+3, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(hf_outputs.logits[0, :+3, -3:] - expected_block_2))
    diff_3 = torch.max(torch.abs(hf_outputs.logits[0, -3:, :+3] - expected_block_3))
    diff_4 = torch.max(torch.abs(hf_outputs.logits[0, -3:, -3:] - expected_block_4))

    max_diff = torch.max(torch.tensor([diff_1, diff_2, diff_3, diff_4]))
    assert max_diff < 3e-5

    # ================================================================================
    # check loaded vision model outputs

    # --------------------------------------------------------------------
    # For original kosmos-2

    # img_size, num_channels, batch_size = 224, 3, 1

    # # vision 2 layers
    # model.img_model.visual.transformer.resblocks = model.img_model.visual.transformer.resblocks[:2]

    # dummy_pixel_values = torch.ones(size=(batch_size, num_channels, img_size, img_size), device="cuda")

    # # The original shape is `torch.Size([257, 1, 1024])`
    # original_vision_output = model.img_model.visual(dummy_pixel_values)

    # # Make the shape being `torch.Size([1, 257, 1024])`
    # original_vision_output = original_vision_output.transpose(1, 0)

    # print(original_vision_output.shape)
    # print(original_vision_output)

    # # pooled (wth the post layer norm)
    # print(original_vision_output[:, 0, :])
    """
    torch.Size([1, 257, 1024])

    tensor(
        [
            [
                [-0.0115,  0.0596, -0.1132,  ...,  0.1799, -0.3139,  0.2753],
                [ 0.0846,  0.6236, -0.4391,  ...,  1.1525, -0.1509,  0.4326],
                [ 0.1050,  0.5756, -0.4778,  ...,  0.6579, -0.2205,  0.3997],
                ...,
                [ 0.1787,  0.5295, -0.6168,  ..., -0.9372, -0.3680,  0.2211],
                [ 0.1823,  0.5258, -0.5524,  ..., -0.8929, -0.3346,  0.2515],
                [ 0.0861,  0.5844, -0.6572,  ..., -0.7107, -0.2946,  0.3093],
            ],
        ],
    )
    
    tensor([[-0.0115,  0.0596, -0.1132,  ...,  0.1799, -0.3139,  0.2753]])
    """

    # --------------------------------------------------------------------
    # Ours

    img_size = model.config.vision_config.image_size  # 224
    num_channels = model.config.vision_config.num_channels  # 3

    batch_size = 1

    dummy_pixel_values = torch.ones(size=(batch_size, num_channels, img_size, img_size), device="cpu")

    hf_vision_output = model.vision_model(dummy_pixel_values)
    # HF CLIP has `last_hidden_state` without through `post_layernorm`
    hf_vision_output = model.vision_model.post_layernorm(hf_vision_output.last_hidden_state)

    print(hf_vision_output.shape)
    print(hf_vision_output[:, 0, :])

    # --------------------------------------------------------------------
    # sanity check

    assert list(hf_vision_output.shape) == [1, 257, 1024]

    expected_block_1 = torch.tensor(
        [
            [-0.01148908,  0.05956455, -0.11317716],
            [0.08458844,  0.6235921,  -0.43905595],
            [ 0.10498603,  0.57555795, -0.47783917],
        ],
    )
    expected_block_2 = torch.tensor(
        [
            [0.17986113, -0.31393886, 0.2753428],
            [1.1525147,  -0.15090114, 0.43260202],
            [0.6578805,  -0.22051974, 0.39973533],
        ],
    )
    expected_block_3 = torch.tensor(
        [
            [ 0.17873019,  0.5295236,  -0.6167609],
            [ 0.18225193,  0.52584666, -0.55239016],
            [ 0.08613532,  0.58441633, -0.6572151],
        ],
    )
    expected_block_4 = torch.tensor(
        [
            [-0.93716234, -0.36800045, 0.22114123],
            [-0.8929372,  -0.3345873, 0.2515392 ],
            [-0.7106602,  -0.2945692, 0.30925298],
        ],
    )

    diff_1 = torch.max(torch.abs(hf_vision_output[0, :+3, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(hf_vision_output[0, :+3, -3:] - expected_block_2))
    diff_3 = torch.max(torch.abs(hf_vision_output[0, -3:, :+3] - expected_block_3))
    diff_4 = torch.max(torch.abs(hf_vision_output[0, -3:, -3:] - expected_block_4))

    max_diff = torch.max(torch.tensor([diff_1, diff_2, diff_3, diff_4]))
    assert max_diff < 1e-5

    # ================================================================================
    # check the whole model

    # --------------------------------------------------------------------
    # For original kosmos-2

    # dummy_img_attn_mask = torch.cat((torch.ones(size=(1, 64)), torch.zeros(size=(1, 7))), dim=-1).to("cuda").bool()

    # # pass only text
    # original_model_output_only_text = model(src_tokens=dummy_input_ids, img_src_tokens=None)
    # print(original_model_output_only_text[0].shape)
    # print(original_model_output_only_text[0])
    """
    torch.Size([1, 71, 65037])

    tensor(
        [
            [
                [15.6060, -5.1565,  8.0064,  ..., -2.2804, -2.0610, -1.0114],
                [ 9.7421, -4.9860,  4.9630,  ..., -1.5206, -1.4377, -0.5475],
                [11.6581, -5.1816, 18.9927,  ..., -2.3973, -1.9231, -1.3065],
                ...,
                [10.6616, -4.7872,  5.0161,  ..., -2.1092, -1.6931, -1.6256],
                [10.9655, -4.8194,  5.6438,  ..., -1.5778, -0.9324, -0.3715],
                [ 9.8335, -4.9696,  4.6688,  ..., -2.2745, -1.7485, -1.7921],
            ]
        ],
    )
    """

    # # pass text and vision
    # original_model_output = model(src_tokens=dummy_input_ids, img_src_tokens=dummy_pixel_values, img_gpt_input_mask=dummy_img_attn_mask)
    # print(original_model_output[0].shape)
    # print(original_model_output[0])
    """
    torch.Size([1, 71, 65037])

    tensor(
        [
            [
                [ 4.8882, -4.3499,  5.4597,  ..., -2.2055, -1.6321, -1.0148],
                [ 4.4254, -4.2447,  5.7366,  ..., -1.8535, -1.4237, -0.7096],
                [ 4.4483, -4.2894,  5.5115,  ..., -2.3162, -1.6573, -1.1387],
                ...,
                [ 8.1921, -5.0712,  5.3592,  ..., -2.5887, -2.0496, -1.8316],
                [ 8.4758, -5.1724,  5.9626,  ..., -1.7432, -1.1267, -0.5763],
                [ 7.6652, -5.2538,  5.4017,  ..., -2.4623, -1.9893, -1.9341],
            ]
        ],
    )
    """

    # --------------------------------------------------------------------
    # Ours

    dummy_img_attn_mask = torch.cat((torch.ones(size=(1, 64)), torch.zeros(size=(1, 7))), dim=-1).to("cpu").bool()

    # pass only text
    model_output_only_text = model.text_model(
        # pixel_values=None,
        input_ids=dummy_input_ids,
        img_attn_mask=None,
    )
    logits_only_text = model_output_only_text.logits

    print(logits_only_text)

    # pass text and vision
    model_output = model(
        pixel_values=dummy_pixel_values,
        input_ids=dummy_input_ids,
        img_attn_mask=dummy_img_attn_mask
    )
    logits = model_output.logits
    img_features = model_output.image_features

    print(logits.shape)
    print(logits)
    print(img_features.shape)
    print(img_features)

    # --------------------------------------------------------------------
    # sanity check: text input only

    assert list(logits_only_text.shape) == [1, 71, 65037]

    expected_block_1 = torch.tensor(
        [
            [15.605998,   -5.156513,    8.00637],
            [ 9.742102,   -4.9859715,   4.9629903],
            [11.658097,   -5.1816287,  18.992657],
        ],
    )
    expected_block_2 = torch.tensor(
        [
            [-2.2804384,  -2.0610392, -1.0114111],
            [-1.5205702,  -1.4377496, -0.54751855],
            [-2.39729,    -1.9231234, -1.3065333 ],
        ],
    )
    expected_block_3 = torch.tensor(
        [
            [10.661624,   -4.7872415,   5.016076],
            [10.965461,   -4.819447,    5.6437974],
            [ 9.833488,   -4.9696445,   4.668755],
        ],
    )
    expected_block_4 = torch.tensor(
        [
            [-2.1091897,  -1.693118, -1.6255622 ],
            [-1.5777528,  -0.9324262, -0.3714557],
            [-2.2744524,  -1.7484771, -1.7920786 ],
        ],
    )

    diff_1 = torch.max(torch.abs(logits_only_text[0, :+3, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(logits_only_text[0, :+3, -3:] - expected_block_2))
    diff_3 = torch.max(torch.abs(logits_only_text[0, -3:, :+3] - expected_block_3))
    diff_4 = torch.max(torch.abs(logits_only_text[0, -3:, -3:] - expected_block_4))

    max_diff = torch.max(torch.tensor([diff_1, diff_2, diff_3, diff_4]))
    assert max_diff < 3e-5

    # --------------------------------------------------------------------
    # sanity check: text + image inputs

    assert list(logits_only_text.shape) == [1, 71, 65037]

    expected_block_1 = torch.tensor(
        [
            [4.888153,  -4.3498607,  5.4596553],
            [ 4.4253945, -4.244659,   5.736647],
            [4.448264,  -4.289385,  5.5114775],
        ],
    )
    expected_block_2 = torch.tensor(
        [
            [-2.2055285, -1.6320639, -1.0147916],
            [-1.8535209, -1.4236742, -0.7096378],
            [-2.3161755, -1.6573074, -1.1387042],
        ],
    )
    expected_block_3 = torch.tensor(
        [
            [8.192094,  -5.0711703,  5.3592353],
            [8.475775,  -5.172369,   5.9625816],
            [7.6652,    -5.2538114,  5.4017296],
        ],
    )
    expected_block_4 = torch.tensor(
        [
            [-2.5886784, -2.0495563, -1.831612],
            [-1.7432401, -1.1266646, -0.5763364],
            [-2.4622574, -1.9892663, -1.9341019],
        ],
    )

    diff_1 = torch.max(torch.abs(logits[0, :+3, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(logits[0, :+3, -3:] - expected_block_2))
    diff_3 = torch.max(torch.abs(logits[0, -3:, :+3] - expected_block_3))
    diff_4 = torch.max(torch.abs(logits[0, -3:, -3:] - expected_block_4))

    max_diff = torch.max(torch.tensor([diff_1, diff_2, diff_3, diff_4]))
    assert max_diff < 3e-5

# ==============================================================================================================


def check_model_with_dog_sample(model):

    # Attention! On the original kosmos-2 `demo`, keep only the first 2 layers in the vision model:
    #   self.model.models[0].img_model.visual.transformer.resblocks = self.model.models[0].img_model.visual.transformer.resblocks[:2]

    # --------------------------------------------------------------------
    # real input: (dog)

    sample = torch.load(dog_sample_file, map_location=torch.device('cpu'))

    pixel_values = sample["net_input"]["img_src_tokens"]
    # It's of shape [1, 1, 3, 224, 224]. Change it to `[1, 3, 224, 224]`
    pixel_values = pixel_values[0]

    input_ids = sample["net_input"]["src_tokens"]
    img_attn_mask = sample["net_input"]["img_gpt_input_mask"]
    # We need a `bool` value
    img_attn_mask = img_attn_mask.bool()

    # --------------------------------------------------------------------
    # `use_cache=False`

    model_output_no_cache = model(pixel_values=pixel_values, input_ids=input_ids, img_attn_mask=img_attn_mask, use_cache=False)

    logits_no_cache = model_output_no_cache.logits
    past_key_values_no_cache = model_output_no_cache.past_key_values
    image_features_no_cache = model_output_no_cache.image_features

    # --------------------------------------------------------------------
    # `use_cache=True` to get the initial `past_key_values`

    model_output = model(pixel_values=pixel_values, input_ids=input_ids, img_attn_mask=img_attn_mask, use_cache=True)

    logits = model_output.logits
    past_key_values = model_output.past_key_values
    image_features = model_output.image_features

    # --------------------------------------------------------------------
    # verify the results between with/without using `cache`

    assert past_key_values_no_cache is None
    assert past_key_values is not None

    assert torch.max(torch.abs(image_features - image_features_no_cache)) < 1e-12
    assert torch.max(torch.abs(logits - logits_no_cache)) < 1e-12

    # --------------------------------------------------------------------
    # check with the original kosmos-2 output: initial step (step 71 -> step 72)

    assert list(logits.shape) == [1, 71, 65037]

    expected_block_1 = torch.tensor(
        [
            [15.605998  , -5.156513  ,  8.00637],
            [ 8.577738  , -4.9635577 ,  7.6196694],
            [ 5.5543556 , -4.5773745 ,  4.523568],
        ],
    )
    expected_block_2 = torch.tensor(
        [
            [-2.2804384 , -2.0610392 , -1.0114111],
            [-2.2657313 , -1.9836413 , -1.3702303],
            [-1.2256985 , -1.2151622 , -1.9965916],
        ],
    )
    expected_block_3 = torch.tensor(
        [
            [ 7.4827657 , -5.6471753 ,  5.3313484],
            [ 6.3412886 , -4.821356  ,  5.9151964],
            [ 7.3028603 , -5.5100656 ,  6.581722],
        ],
    )
    expected_block_4 = torch.tensor(
        [
            [-2.835022  , -2.887678  , -1.3593428 ],
            [-1.830313  , -1.4463289 , -1.2882515 ],
            [-2.29154   , -1.9426216 , -0.93513656],
        ],
    )

    diff_1 = torch.max(torch.abs(logits[0, :+3, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(logits[0, :+3, -3:] - expected_block_2))
    diff_3 = torch.max(torch.abs(logits[0, -3:, :+3] - expected_block_3))
    diff_4 = torch.max(torch.abs(logits[0, -3:, -3:] - expected_block_4))

    max_diff = torch.max(torch.tensor([diff_1, diff_2, diff_3, diff_4]))
    assert max_diff < 3e-5

    # --------------------------------------------------------------------
    # next step: without `past_key_values`

    new_input_ids = torch.cat((input_ids, torch.tensor([[9]], dtype=torch.long, device="cpu")), dim=1)
    new_img_attn_mask = torch.cat((img_attn_mask, torch.tensor([[False]], dtype=torch.bool, device="cpu")), dim=1)
    new_model_output = model(pixel_values=pixel_values, input_ids=new_input_ids, img_attn_mask=new_img_attn_mask)

    new_logits = new_model_output.logits
    new_past_key_values = new_model_output.past_key_values

    assert new_past_key_values is None

    print(new_logits[:, -1, :])

    # --------------------------------------------------------------------
    # next step: with `past_key_values`

    next_input_ids = torch.tensor([[9]], dtype=torch.long, device="cpu")
    # (no need to pass `pixel_values`) -> need to specify it or `image_features`
    next_pixel_values = None
    next_image_features = image_features
    next_img_attn_mask = None
    next_model_output = model(pixel_values=next_pixel_values, img_features=next_image_features, input_ids=next_input_ids, img_attn_mask=next_img_attn_mask, past_key_values=past_key_values, use_cache=True)

    next_logits = next_model_output.logits
    next_past_key_values = next_model_output.past_key_values

    assert next_past_key_values is not None

    print(next_logits[:, -1, :])

    # --------------------------------------------------------------------
    # verify the results between with/without using `past_key_values`

    max_diff = torch.max(torch.abs(new_logits[:, -1, :] - next_logits[:, -1, :]))
    print(max_diff)
    assert max_diff < torch.tensor(3e-5)

    # --------------------------------------------------------------------
    # check with the original kosmos-2 output: next step (step 72 -> step 73)

    assert list(next_logits.shape) == [1, 1, 65037]

    expected_block_1 = torch.tensor([[[ 7.6893177 , -5.576222  ,  6.5033607]]])
    expected_block_2 = torch.tensor([[[ -2.398699  , -2.1435356 , -0.98740137]]])

    diff_1 = torch.max(torch.abs(next_logits[0, 0, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(next_logits[0, 0, -3:] - expected_block_2))

    max_diff = torch.max(torch.tensor([diff_1, diff_2]))
    assert max_diff < 3e-5

    # --------------------------------------------------------------------
    # generation

    expected_generation = [
        9,     9,     5,     5,     5,    10,    10,    10,     5,
        5,   106,   106,   106,     6,     6,     6,     8,     8,     8,
        6,     6,   106,   106,    10,    10,    42,    42,    42,    10,
        10,   106,   106,    19,    19,    19,     6,     6,    12,    12,
        12,    20,    20,    20,    12,    12,    10,    10,    12,    12,
        106,   106,    43,    43,    43,  2115,  2115,  2115,    43,    43,
        106,   106,    12,    12,
    ]

    # use `text_model` directly
    # with `past_key_values` being passed as the initialized
    # no need to pass `img_features` (`pixel_values`) and `img_attn_mask`
    generated_output = model.text_model.generate(
        # we need to provide the full `input_ids` not just the trailing one!
        input_ids=new_input_ids,
        use_cache=True,
        past_key_values=past_key_values,
        img_features=None,
        img_attn_mask=None,
        # we already generated the first token (step 71 -> 72)
        max_new_tokens=len(expected_generation) - 1,
    )

    # --------------------------------------------------------------------
    # check with the original kosmos-2 output generation output: (step 72 -> step X)

    assert generated_output[0, 71:].tolist() == expected_generation


def check_real_model_with_dog_sample(model):

    # --------------------------------------------------------------------
    # real input: (dog)

    sample = torch.load(dog_sample_file, map_location=torch.device('cpu'))

    pixel_values = sample["net_input"]["img_src_tokens"]
    # It's of shape [1, 1, 3, 224, 224]. Change it to `[1, 3, 224, 224]`
    pixel_values = pixel_values[0]

    input_ids = sample["net_input"]["src_tokens"]
    img_attn_mask = sample["net_input"]["img_gpt_input_mask"]
    # We need a `bool` value
    img_attn_mask = img_attn_mask.bool()

    # --------------------------------------------------------------------
    # `use_cache=False`

    model_output_no_cache = model(pixel_values=pixel_values, input_ids=input_ids, img_attn_mask=img_attn_mask, use_cache=False)

    logits_no_cache = model_output_no_cache.logits
    past_key_values_no_cache = model_output_no_cache.past_key_values
    image_features_no_cache = model_output_no_cache.image_features

    # --------------------------------------------------------------------
    # `use_cache=True` to get the initial `past_key_values`

    model_output = model(pixel_values=pixel_values, input_ids=input_ids, img_attn_mask=img_attn_mask, use_cache=True)

    logits = model_output.logits
    past_key_values = model_output.past_key_values
    image_features = model_output.image_features

    # --------------------------------------------------------------------
    # verify the results between with/without using `cache`

    assert past_key_values_no_cache is None
    assert past_key_values is not None

    assert torch.max(torch.abs(image_features - image_features_no_cache)) < 1e-12
    assert torch.max(torch.abs(logits - logits_no_cache)) < 1e-12

    # --------------------------------------------------------------------
    # check with the original kosmos-2 output: initial step (step 71 -> step 72)

    assert list(logits.shape) == [1, 71, 65037]

    expected_block_1 = torch.tensor(
        [
            [2.920926332473755, -5.4380574226379395, 2.8645224571228027],
            [0.004667307715862989, -5.0997819900512695, 4.338554382324219],
            [-0.5761765837669373, -4.547626972198486, 3.8142454624176025],
        ],
    )
    expected_block_2 = torch.tensor(
        [
            [-2.61974835395813, -2.6742029190063477, -1.6856958866119385],
            [-2.251966714859009, -2.242988348007202, -1.5341331958770752],
            [-2.3858885765075684, -1.5038200616836548, -1.013083577156067],
        ],
    )
    expected_block_3 = torch.tensor(
        [
            [-1.3929418325424194, -4.623406410217285, 3.7545101642608643],
            [0.522249698638916, -4.5460662841796875, 7.236062526702881],
            [-1.7789695262908936, -5.221266746520996, 3.770735740661621],
        ],
    )
    expected_block_4 = torch.tensor(
        [
            [-2.3952505588531494, -2.878037452697754, -1.3662471771240234],
            [-3.3000922203063965, -3.0199999809265137, -0.24584506452083588],
            [-2.8502795696258545, -3.096112012863159, -0.771698534488678],
        ],
    )

    diff_1 = torch.max(torch.abs(logits[0, :+3, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(logits[0, :+3, -3:] - expected_block_2))
    diff_3 = torch.max(torch.abs(logits[0, -3:, :+3] - expected_block_3))
    diff_4 = torch.max(torch.abs(logits[0, -3:, -3:] - expected_block_4))

    max_diff = torch.max(torch.tensor([diff_1, diff_2, diff_3, diff_4]))
    assert max_diff < 3e-5

    expected_next_token = 64007
    predicted_next_token = torch.argmax(logits[0, -1, :]).detach().to("cpu").numpy().tolist()

    assert predicted_next_token == expected_next_token

    # --------------------------------------------------------------------
    # next step: without `past_key_values`

    next_token = expected_next_token

    new_input_ids = torch.cat((input_ids, torch.tensor([[next_token]], dtype=torch.long, device="cpu")), dim=1)
    new_img_attn_mask = torch.cat((img_attn_mask, torch.tensor([[False]], dtype=torch.bool, device="cpu")), dim=1)
    new_model_output = model(pixel_values=pixel_values, input_ids=new_input_ids, img_attn_mask=new_img_attn_mask)

    new_logits = new_model_output.logits
    new_past_key_values = new_model_output.past_key_values

    assert new_past_key_values is None

    print(new_logits[:, -1, :])

    # --------------------------------------------------------------------
    # next step: with `past_key_values`

    next_input_ids = torch.tensor([[next_token]], dtype=torch.long, device="cpu")
    # (no need to pass `pixel_values`) -> need to specify it or `image_features`
    next_pixel_values = None
    next_image_features = image_features
    next_img_attn_mask = None
    next_model_output = model(pixel_values=next_pixel_values, img_features=next_image_features, input_ids=next_input_ids, img_attn_mask=next_img_attn_mask, past_key_values=past_key_values, use_cache=True)

    next_logits = next_model_output.logits
    next_past_key_values = next_model_output.past_key_values

    assert next_past_key_values is not None

    print(next_logits[:, -1, :])

    # --------------------------------------------------------------------
    # verify the results between with/without using `past_key_values`

    max_diff = torch.max(torch.abs(new_logits[:, -1, :] - next_logits[:, -1, :]))
    print(max_diff)
    assert max_diff < torch.tensor(3e-5)

    # --------------------------------------------------------------------
    # check with the original kosmos-2 output: next step (step 72 -> step 73)

    assert list(next_logits.shape) == [1, 1, 65037]

    expected_block_1 = torch.tensor([[[-1.3323104, -5.1079516,  5.359114 ]]])
    expected_block_2 = torch.tensor([[[-2.8319776, -3.5213413, -1.8274367]]])

    diff_1 = torch.max(torch.abs(next_logits[0, 0, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(next_logits[0, 0, -3:] - expected_block_2))

    max_diff = torch.max(torch.tensor([diff_1, diff_2]))
    assert max_diff < 3e-5

    expected_next_token = 94
    predicted_next_token = torch.argmax(next_logits[0, -1, :]).detach().to("cpu").numpy().tolist()

    assert predicted_next_token == expected_next_token

    # --------------------------------------------------------------------
    # repeat the above check with more extra steps (step 73 -> step 74)
    # check with the original kosmos-2 output: next step

    # steps: 73 -> 74 -> .. -> 83 (-> 84)
    steps = list(range(73, 83 + 1))

    next_tokens = [94, 17772, 64008, 64009, 64092, 65029, 64011, 64148, 65021, 64010, 1280, 12]

    expected_blocks = [
        ([-1.5333264, -5.0365257,  5.595204 ], [-2.1252668, -2.9195867, -1.3610152]),
        ([-1.14558  , -4.6416078,  8.611397 ], [-1.9524179, -2.3943331, -1.2364707]),
        ([ 2.1540604, -2.713409 ,  1.8866036], [ 1.631276 ,  0.8916559, -0.4697148]),
        ([-1.1833401, -3.1272492, -1.4443989], [2.75421  , 2.1421206, 1.2756062]),
        ([-1.429662 , -4.2857313,  1.123333 ], [13.215454, 13.476381, 14.000856]),
        ([-0.10423064, -4.0805306 ,  7.669438  ], [1.3264995 , 0.37444258, 2.872366  ]),
        ([-1.9969933, -4.391607 , -3.4535604], [ 0.15055317,  0.05899912, -0.0650674 ]),
        ([-2.1537013, -4.2108035,  2.163306 ], [8.790059, 8.622845, 9.70795 ]),
        ([-0.10536906, -2.7584782 ,  5.857536  ], [4.7097054, 3.5752287, 6.4874005]),
        ([-0.40761316, -4.65115   , 16.127958  ], [-3.0172224, -3.2040298, -2.283117 ]),
        ([ 0.4004581, -4.4891667, 14.7836075], [-0.9358875, -1.006671 , -0.1364981]),
    ]

    for (step, next_token, expected_next_token, (expected_block_1, expected_block_2)) in zip(steps, next_tokens[:-1], next_tokens[1:], expected_blocks):

        print(f"step: {step}")

        new_input_ids = torch.cat((new_input_ids, torch.tensor([[next_token]], dtype=torch.long, device="cpu")), dim=1)
        new_img_attn_mask = torch.cat((new_img_attn_mask, torch.tensor([[False]], dtype=torch.bool, device="cpu")), dim=1)
        new_model_output = model(pixel_values=pixel_values, input_ids=new_input_ids, img_attn_mask=new_img_attn_mask)

        new_logits = new_model_output.logits
        new_past_key_values = new_model_output.past_key_values

        assert new_past_key_values is None

        print(new_logits[:, -1, :])

        # --------------------------------------------------------------------
        # next step: with `past_key_values`

        next_input_ids = torch.tensor([[next_token]], dtype=torch.long, device="cpu")
        # (no need to pass `pixel_values`) -> need to specify it or `image_features`
        next_pixel_values = None
        next_image_features = image_features
        next_img_attn_mask = None
        next_model_output = model(pixel_values=next_pixel_values, img_features=next_image_features, input_ids=next_input_ids, img_attn_mask=next_img_attn_mask, past_key_values=next_past_key_values, use_cache=True)

        next_logits = next_model_output.logits
        next_past_key_values = next_model_output.past_key_values

        assert next_past_key_values is not None

        print(next_logits[:, -1, :])

        # --------------------------------------------------------------------
        # verify the results between with/without using `past_key_values`

        max_diff = torch.max(torch.abs(new_logits[:, -1, :] - next_logits[:, -1, :]))
        # step 75 has a slightly bigger diff
        allowed_max_diff = 3e-5 if step != 75 else 5e-5

        assert max_diff < torch.tensor(allowed_max_diff)

        # --------------------------------------------------------------------
        # check with the original kosmos-2 output: next step

        assert list(next_logits.shape) == [1, 1, 65037]

        expected_block_1 = torch.tensor([[expected_block_1]])
        expected_block_2 = torch.tensor([[expected_block_2]])

        diff_1 = torch.max(torch.abs(next_logits[0, 0, :+3] - expected_block_1))
        diff_2 = torch.max(torch.abs(next_logits[0, 0, -3:] - expected_block_2))

        max_diff = torch.max(torch.tensor([diff_1, diff_2]))
        allowed_max_diff = 3e-5

        assert max_diff < allowed_max_diff

        predicted_next_token = torch.argmax(next_logits[0, -1, :]).detach().to("cpu").numpy().tolist()

        assert predicted_next_token == expected_next_token

    # --------------------------------------------------------------------
    # generation

    new_input_ids = torch.cat((new_input_ids, torch.tensor([[predicted_next_token]], dtype=torch.long, device="cpu")), dim=1)
    new_img_attn_mask = torch.cat((new_img_attn_mask, torch.tensor([[False]], dtype=torch.bool, device="cpu")), dim=1)

    expected_generation = [
         64007,    94, 17772, 64008, 64009, 64092, 65029, 64011, 64148,
         65021, 64010,  1280,    12, 64007,     5,  4464, 64008, 64009, 64013,
         65036, 64010, 2
    ]

    # use `text_model` directly
    # with `past_key_values` being passed as the initialized
    # no need to pass `img_features` (`pixel_values`) and `img_attn_mask`
    generated_output = model.text_model.generate(
        input_ids=new_input_ids,
        use_cache=True,
        past_key_values=next_past_key_values,
        img_features=None,
        img_attn_mask=None,
        # we already generated 13 tokens: from `64007` (step 71 -> 72) to `12` (step 83 -> 84)
        max_new_tokens=len(expected_generation) - 13,
    )

    # --------------------------------------------------------------------
    # check with the original kosmos-2 output generation output: (step 84 -> step X)

    assert generated_output[0, 71:].tolist() == expected_generation

    # --------------------------------------------------------------------
    # Use `eos_token_id`

    # or we can specify `eos_token_id` to stop earlier.
    generated_output = model.text_model.generate(
        input_ids=new_input_ids,
        use_cache=True,
        past_key_values=next_past_key_values,
        img_features=None,
        img_attn_mask=None,
        # we already generated 13 tokens: from `64007` (step 71 -> 72) to `12` (step 83 -> 84)
        # use `eos_token_id=2` to stop earlier
        # TODO: specify this in the config file
        # we still need to specify this: so we get long enough generations
        max_new_tokens=len(expected_generation),
        eos_token_id=2,
    )
    assert generated_output[0, 71:].tolist() == expected_generation

    # --------------------------------------------------------------------
    # generation without `use_cache` (from step 84)

    # use `text_model` directly
    # with `use_cache=False` and `past_key_values=None`
    # need to pass `img_features` and `img_attn_mask` (for the `correctness`)
    generated_output = model.text_model.generate(
        input_ids=new_input_ids,
        use_cache=False,
        past_key_values=None,
        img_features=image_features,
        img_attn_mask=new_img_attn_mask,
        # we already generated 13 tokens: from `64007` (step 71 -> 72) to `12` (step 83 -> 84)
        max_new_tokens=len(expected_generation) - 13,
    )
    assert generated_output[0, 71:].tolist() == expected_generation

    # --------------------------------------------------------------------
    # generation without `use_cache` (from the start)

    # use `text_model` directly
    # with`use_cache=False` (from the start --> `past_key_values=None`)
    # need to pass `img_features` and `img_attn_mask` (for the `correctness`)
    generated_output = model.text_model.generate(
        input_ids=input_ids,
        use_cache=False,
        past_key_values=None,
        img_features=image_features,
        img_attn_mask=img_attn_mask,
        max_new_tokens=len(expected_generation),
        output_scores=True,
        return_dict_in_generate=True,
    )

    assert generated_output.sequences[0, 71:].tolist() == expected_generation

    for score, expected_block in zip(generated_output.scores[2:12], expected_blocks):
        assert torch.max(torch.abs(score[0, :+3] - torch.tensor(expected_block[0]))) < 5e-5
        assert torch.max(torch.abs(score[0, -3:] - torch.tensor(expected_block[1]))) < 5e-5

    # --------------------------------------------------------------------
    # generation with `use_cache` (from the start)

    # use `text_model` directly
    # with `use_cache=True` (from the start --> `past_key_values=None`)
    # need to pass `img_features` and `img_attn_mask` (for the `correctness`)
    generated_output = model.text_model.generate(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=None,
        img_features=image_features,
        img_attn_mask=img_attn_mask,
        max_new_tokens=len(expected_generation),
        output_scores=True,
        return_dict_in_generate=True,
    )

    assert generated_output.sequences[0, 71:].tolist() == expected_generation

    for score, expected_block in zip(generated_output.scores[2:12], expected_blocks):
        assert torch.max(torch.abs(score[0, :+3] - torch.tensor(expected_block[0]))) < 3e-5
        assert torch.max(torch.abs(score[0, -3:] - torch.tensor(expected_block[1]))) < 3e-5

    # --------------------------------------------------------------------
    # generation without `use_cache` (from the start)

    # use `model`
    # with`use_cache=False` (from the start --> `past_key_values=None`)
    generated_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        use_cache=False,
        past_key_values=None,
        # we can specify `None` here.
        img_features=image_features,
        img_attn_mask=img_attn_mask,
        max_new_tokens=len(expected_generation),
        output_scores=True,
        return_dict_in_generate=True,
    )

    assert generated_output.sequences[0, 71:].tolist() == expected_generation

    for score, expected_block in zip(generated_output.scores[2:12], expected_blocks):
        assert torch.max(torch.abs(score[0, :+3] - torch.tensor(expected_block[0]))) < 5e-5
        assert torch.max(torch.abs(score[0, -3:] - torch.tensor(expected_block[1]))) < 5e-5

    # --------------------------------------------------------------------
    # generation with `use_cache` (from the start)

    # use `model`
    # with `use_cache=True` (from the start --> `past_key_values=None`)
    generated_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        use_cache=True,
        past_key_values=None,
        img_features=None,
        img_attn_mask=img_attn_mask,
        max_new_tokens=len(expected_generation),
        output_scores=True,
        return_dict_in_generate=True,
    )

    assert generated_output.sequences[0, 71:].tolist() == expected_generation

    for score, expected_block in zip(generated_output.scores[2:12], expected_blocks):
        assert torch.max(torch.abs(score[0, :+3] - torch.tensor(expected_block[0]))) < 3e-5
        assert torch.max(torch.abs(score[0, -3:] - torch.tensor(expected_block[1]))) < 3e-5

    # --------------------------------------------------------------------
    # generation with `use_cache` (from the start)

    # use `model`
    # with `use_cache=True` (from the start --> `past_key_values=None`)
    generated_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        use_cache=True,
        past_key_values=None,
        img_features=image_features,
        img_attn_mask=img_attn_mask,
        max_new_tokens=len(expected_generation),
        output_scores=True,
        return_dict_in_generate=True,
    )

    assert generated_output.sequences[0, 71:].tolist() == expected_generation

    for score, expected_block in zip(generated_output.scores[2:12], expected_blocks):
        assert torch.max(torch.abs(score[0, :+3] - torch.tensor(expected_block[0]))) < 3e-5
        assert torch.max(torch.abs(score[0, -3:] - torch.tensor(expected_block[1]))) < 3e-5

    # --------------------------------------------------------------------
    # generation with `use_cache` (from step 84)

    # use `model`
    # with `use_cache=True`
    generated_output = model.generate(
        pixel_values=pixel_values,
        input_ids=new_input_ids,
        use_cache=True,
        past_key_values=next_past_key_values,
        img_features=None,
        img_attn_mask=new_img_attn_mask,
        # we already generated 13 tokens: from `64007` (step 71 -> 72) to `12` (step 83 -> 84)
        max_new_tokens=len(expected_generation) - 13,
    )
    assert generated_output[0, 71:].tolist() == expected_generation

    # --------------------------------------------------------------------
    # generation with `use_cache` (from step 84)

    # use `model`
    # with `use_cache=True`
    generated_output = model.generate(
        pixel_values=pixel_values,
        input_ids=new_input_ids,
        use_cache=True,
        past_key_values=next_past_key_values,
        img_features=image_features,
        img_attn_mask=new_img_attn_mask,
        # we already generated 13 tokens: from `64007` (step 71 -> 72) to `12` (step 83 -> 84)
        max_new_tokens=len(expected_generation) - 13,
    )
    assert generated_output[0, 71:].tolist() == expected_generation


def check_head_base_model_loading(config):

    model = Kosmos2ForConditionalGeneration(config=config)
    ckpt = "Kosmos2ForConditionalGeneration"
    model.save_pretrained(ckpt)

    loaded_config = Kosmos2Config.from_pretrained(ckpt)
    loaded_config.architectures = ["Kosmos2Model"]
    loaded_config.save_pretrained(ckpt)
    loaded_config = Kosmos2Config.from_pretrained(ckpt)
    print(loaded_config.architectures)

    _ = Kosmos2Model.from_pretrained(ckpt)


    base_model = Kosmos2Model(config=config)
    ckpt = "Kosmos2Model"
    base_model.save_pretrained(ckpt)

    loaded_config = Kosmos2Config.from_pretrained(ckpt)
    loaded_config.architectures = ["Kosmos2ForConditionalGeneration"]
    loaded_config.save_pretrained(ckpt)
    loaded_config = Kosmos2Config.from_pretrained(ckpt)
    print(loaded_config.architectures)

    _ = Kosmos2ForConditionalGeneration.from_pretrained(ckpt)


def create_model(num_layers=2):

    text_config = {
        "use_cache": False,
        "scale_embedding": True,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_function": "gelu",
        "activation_dropout": 0.0,
        "add_cross_attention": False,
        "attention_heads": 32,
        "ffn_dim": 8192,
        "embed_dim": 2048,
        "layers": num_layers,
        "layer_norm_eps": 1e-5,
        "gradient_checkpointing": False,
        # to match the demo
        "no_repeat_ngram_size": 3,
        # "use_cache": True,

    }
    vision_config = Kosmos2VisionConfig()
    #  2 layers
    vision_config.num_hidden_layers = num_layers
    vision_config = vision_config.to_dict()

    latent_query_num = 64

    config = Kosmos2Config(text_config=text_config, vision_config=vision_config, latent_query_num=latent_query_num)
    model = Kosmos2ForConditionalGeneration(config=config)
    model.eval()

    print(model)

    return model


if __name__ == "__main__":

    # ================================================================================
    # config & model creation

    dummy_model = create_model(num_layers=2)

    # ================================================================================
    # check the head model's checkpoint could be loaded into the base model and vice-versa

    check_head_base_model_loading(dummy_model.config)

    # ================================================================================
    # check model keys and loading

    load_and_check_vision_model()
    load_and_check_model(dummy_model, ckpt_path=original_kosmos2_checkpoint_only_2_layers)

    # ================================================================================
    # check loaded text model outputs

    # Tip:
    # We need to pass `attention mask` if we want to call decoder layers directly!
    # (use `_prepare_decoder_attention_mask`)

    # Tip
    # Including the padding token id `1`  in `input_ids` to make sure everything work
    # (especially the positional embedding)

    check_model_with_dummy_inputs(dummy_model)
    check_model_with_dog_sample(dummy_model)

    # ================================================================================
    # real config & model creation

    real_model = create_model(num_layers=24)

    # need to create this checkpoint
    load_and_check_model(real_model, ckpt_path="kosmos2_state_dict.bin")
    real_model.save_pretrained("HF_Kosmos2")

    # check we can load
    real_model = Kosmos2ForConditionalGeneration.from_pretrained("HF_Kosmos2")

    # # If we want to push to the Hub
    # # repo_id = "ydshieh/kosmos-2"
    # # real_model.save_pretrained("HF_Kosmos2", push_to_hub=True, repo_id=repo_id, use_auth_token="XXX")
    #
    # # check we can load from the Hub
    # # real_model = Kosmos2ForConditionalGeneration.from_pretrained(repo_id)

    # repo_id = "ydshieh/kosmos-2"
    #
    # # check we can load from the Hub
    # real_model = Kosmos2ForConditionalGeneration.from_pretrained(repo_id)

    # ================================================================================

    check_real_model_with_dog_sample(real_model)

    # ================================================================================
