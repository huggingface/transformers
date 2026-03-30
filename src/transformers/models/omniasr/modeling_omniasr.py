# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Callable
from typing import Optional, Union
import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ..auto import AutoModel, AutoModelForCausalLM
from ...generation import GenerationMixin
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...cache_utils import Cache
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    CausalLMOutputWithPast,
    Wav2Vec2BaseModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from .configuration_omniasr import OmniASRCTCConfig, OmniASRLLMConfig, OmniASREncoderConfig



class OmniASRLayerNormConvLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


# TODO use pos_encoder_depth? or always set to 1?
# NOTE: overwrite comapared to `Wav2Vec2PositionalConvEmbedding` as it doesnt weight norm with in the component
class OmniASRPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        # NOTE original: https://github.com/facebookresearch/fairseq2/blob/a1f0c565a99d3cd3e3157678b5c48653e3d439f4/src/fairseq2/models/wav2vec2/position_encoder.py#L64
        self.remove_pad = config.num_conv_pos_embeddings % 2 == 0
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        if self.remove_pad:
            hidden_states = hidden_states[:, :, :-1]
        # hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states + residual


# NOTE: modular from Wav2Vec2FeatureEncoder?
class OmniASRFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__()

        conv_layers = [
            OmniASRLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
        ]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def forward(self, input_values):
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        hidden_states = self.layer_norm(hidden_states.transpose(1, 2))
        return hidden_states


# NOTE: modular from Wav2Vec2FeatureProjection (with self.layer_norm removed)?
class OmniASRFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)    # removed compared to `Wav2Vec2FeatureProjection`
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class OmniASRAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[OmniASREncoderConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        # TODO: we need a refactor so that the different attention modules can get their specific kwargs
        # ATM, we have mixed things encoder, decoder, and encoder-decoder attn
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]
        src_len = key_value_states.shape[1] if is_cross_attention else tgt_len

        q_input_shape = (bsz, tgt_len, -1, self.head_dim)
        kv_input_shape = (bsz, src_len, -1, self.head_dim)

        # get query proj
        query_states = self.q_proj(hidden_states).view(*q_input_shape).transpose(1, 2)

        current_states = key_value_states if is_cross_attention else hidden_states
        key_states = self.k_proj(current_states).view(*kv_input_shape).transpose(1, 2)
        value_states = self.v_proj(current_states).view(*kv_input_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            output_attentions=output_attentions,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, None


class OmniASREncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.self_attn = OmniASRAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            config=config,
        )

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = OmniASRFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # NOTE: original: https://github.com/facebookresearch/fairseq2/blob/a1f0c565a99d3cd3e3157678b5c48653e3d439f4/src/fairseq2/models/transformer/encoder_layer.py#L141
    # can see `Wav2Vec2BertEncoderLayer` from src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py
    # but here we have no convolution layer, do we need it?
    def forward(
        self, 
        hidden_states,
        attention_mask=None, 
        output_attentions=False,
    ):
        # Self-attention block with pre-norm (layer_norm_pre=True in config)
        attn_residual = hidden_states
        # NOTE (ebezzam) add pre-norm flag? So leave option to do pre- or post norm?
        hidden_states = self.layer_norm(hidden_states)  # Pre-norm: normalize BEFORE attention
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states, 
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states  # Add residual

        # FFN block with pre-norm
        ffn_residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)  # Pre-norm: normalize BEFORE FFN
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = ffn_residual + hidden_states  # Add residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# NOTE: modular from Wav2Vec2Encoder?
class OmniASREncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # NOTE in fairseq2 (embed_positions, layer_norm, dropout) are in "Frontend" instead of Encoder: https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/models/wav2vec2/frontend.py#L209-L216
        self.pos_conv_embed = OmniASRPositionalConvEmbedding(config)  # Wav2vec2
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # Wav2vec2 applies here, while Wav2Vec2Bert applies within layers?
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([OmniASREncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )
        
        # NOTE (ebezzam) equivalent code? https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/models/wav2vec2/frontend.py#L209-L213
        hidden_states = self.pos_conv_embed(hidden_states)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])
            skip_the_layer = self.training and dropout_probability < self.config.layerdrop
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.layer_norm(hidden_states)
        if self.training:
            hidden_states = self.dropout(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@auto_docstring
class OmniASRPreTrainedModel(PreTrainedModel):
    config: OmniASREncoderConfig
    base_model_prefix = "omniasr"
    main_input_name = "input_values"
    input_modalities = "audio"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, OmniASRPositionalConvEmbedding):
            init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            init.constant_(module.conv.bias, 0)
        elif isinstance(module, OmniASRFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            init.uniform_(module.projection.weight, a=-k, b=k)
            init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                init.uniform_(module.bias, a=-k, b=k)

    
    def apply_weight_norm(self, legacy=True):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm") and not legacy:
            weight_norm = nn.utils.parametrizations.weight_norm

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.encoder.encoder.pos_conv_embed.conv.weight, modifier_rank=0):
                weight_norm(self.encoder.encoder.pos_conv_embed.conv, name="weight", dim=2)
            if hasattr(self.encoder.encoder.pos_conv_embed.conv, "parametrizations"):
                weight_g = self.encoder.encoder.pos_conv_embed.conv.parametrizations.weight.original0
                weight_v = self.encoder.encoder.pos_conv_embed.conv.parametrizations.weight.original1
            else:
                weight_g = self.encoder.encoder.pos_conv_embed.conv.weight_g
                weight_v = self.encoder.encoder.pos_conv_embed.conv.weight_v
            deepspeed.zero.register_external_parameter(self.encoder.encoder.pos_conv_embed, weight_v)
            deepspeed.zero.register_external_parameter(self.encoder.encoder.pos_conv_embed, weight_g)
        else:
            weight_norm(self.encoder.encoder.pos_conv_embed.conv, name="weight", dim=2)


    def remove_weight_norm(self, legacy=True):
        remove_weight_norm = nn.utils.remove_weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm") and not legacy:
            remove_weight_norm = torch.nn.utils.parametrize.remove_parametrizations

        # TODO deepspeed zero3 case
        remove_weight_norm(self.encoder.encoder.pos_conv_embed.conv, name="weight")
    

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask


class OmniASRFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# TODO rename this to OmniASREncoder and unwrap .encoder?
# NOTE: modular from Wav2Vec2Model? or directly AutoMode?
@auto_docstring
class OmniASRModel(OmniASRPreTrainedModel):
    def __init__(self, config: OmniASREncoderConfig):
        # see Wav2Vec2BertModel
        super().__init__(config)
        self.config = config
        self.feature_extractor = OmniASRFeatureEncoder(config)
        self.feature_projection = OmniASRFeatureProjection(config)
        self.encoder = OmniASREncoder(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.post_init()


    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, Wav2Vec2BaseModelOutput]:
        r"""
        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.feature_extractor(input_values)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask, add_adapter=False
            )
        hidden_states = self.feature_projection(hidden_states)


        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = encoder_outputs[0]

        if self.training:
            hidden_states = self.dropout(hidden_states)

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# NOTE: see Wav2Vec2ForCTC
@auto_docstring(
    custom_intro="""
    OmniASR Model with a head for Connectionist Temporal Classification (CTC).
    """
)
class OmniASRForCTC(OmniASRPreTrainedModel):
    config: OmniASRCTCConfig

    def __init__(self, config: OmniASRCTCConfig):
        super().__init__(config)
        self.encoder = OmniASRModel(config.encoder_config)
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.ctc_head(hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


@auto_docstring(
    custom_intro="""
    OmniASR model, which consists of a Wav2Vec2, a multi-modal projector and a LLama language model.
    """
)
class OmniASRForConditionalGeneration(OmniASRPreTrainedModel, GenerationMixin):
    config: OmniASRLLMConfig
    main_input_name = "input_ids"  # generative model; audio is passed as extra input_values
    # TODO keep below from Voxtral?
    # _keep_in_fp32_modules_strict = ["embed_positions"]

    # TODO encoder_stacking used by Zero-Shot variant
    # https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/model.py#L1024

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.language_token_id = config.language_token_id
        if config.num_special_tokens > 0:
            reserved_language_token_id = config.vocab_size - config.num_special_tokens
            # LID syntax uses a reserved special token appended after the base
            # vocabulary. Some converted checkpoints may carry a stale in-vocab
            # ID; remap to the reserved slot to match original syntax.
            if self.language_token_id < reserved_language_token_id:
                self.language_token_id = reserved_language_token_id

        self.encoder = AutoModel.from_config(config.encoder_config)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        self.multi_modal_projector = nn.Linear(
            config.encoder_config.hidden_size * config.encoder_stacking,
            config.text_config.hidden_size,
            bias=True,
        )

        self.lang_embeddings = nn.Embedding(config.num_language_embeddings, config.text_config.hidden_size)

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_audio_features(self, input_features: torch.FloatTensor):
        audio_outputs = self.encoder(input_features)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_embeds = self.multi_modal_projector(audio_hidden_states)
        return audio_embeds

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_values: Optional[torch.Tensor] = None,
        language_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        Original: https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/model.py#L141
        Input syntax: audio | lid_marker | lang_id | bos | [target_text | eos]
        """

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_values is not None:
            # First step: build full audio context (audio | lid_marker | lang_id | bos)
            batch_size = input_values.size(0)
            device = input_values.device

            audio_embeds = self.get_audio_features(input_values)
            dtype = audio_embeds.dtype

            language_id_token_batch = torch.full((batch_size, 1), self.language_token_id, dtype=torch.long, device=device)
            bos_batch = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=device)

            if language_ids is not None:
                language_id_batch = language_ids.to(device)
            else:
                # Default to 0 = "no language" embedding, matching the original
                # omnilingual-asr behavior where index 0 is a language-agnostic
                # representation learned during training via language dropout.
                language_id_batch = torch.zeros(batch_size, dtype=torch.long, device=device)

            if self.training and self.config.language_embedding_probability > 0.0:
                dropout_mask = torch.rand(batch_size, device=device) < (1 - self.config.language_embedding_probability)
                language_id_batch = language_id_batch.clone()
                language_id_batch[dropout_mask] = 0

            text_embed_fn = self.get_input_embeddings()
            lid_marker_embeds = text_embed_fn(language_id_token_batch).to(dtype)
            bos_embeds = text_embed_fn(bos_batch).to(dtype)
            lang_id_embeds = self.lang_embeddings(language_id_batch.unsqueeze(-1)).to(dtype)

            inputs_embeds = torch.cat([audio_embeds, lid_marker_embeds, lang_id_embeds, bos_embeds], dim=1)

            # Override attention_mask and cache_position to match actual context length,
            # since GenerationMixin computes them from input_ids length (1 BOS token),
            # not the full audio context length.
            seq_len = inputs_embeds.shape[1]
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
            cache_position = torch.arange(seq_len, device=device)

        if labels is not None:
            # Training: append target_text + eos embeddings for teacher forcing
            batch_size = input_values.size(0)
            device = input_values.device
            dtype = inputs_embeds.dtype
            text_embed_fn = self.get_input_embeddings()
            eos_batch = torch.full((batch_size, 1), self.config.eos_token_id, dtype=torch.long, device=device)
            target_embeds = text_embed_fn(labels).to(dtype)
            eos_embeds = text_embed_fn(eos_batch).to(dtype)
            inputs_embeds = torch.cat([inputs_embeds, target_embeds, eos_embeds], dim=1)

        # Build attention mask if not provided
        if attention_mask is None and past_key_values is None:
            batch_size = inputs_embeds.size(0)
            seq_len = inputs_embeds.size(1)
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=inputs_embeds.device)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        logits = outputs.logits
        loss = None
        if labels is not None:
            # Context length = audio_seq_len + 3 (lid_marker + lang_id + bos)
            context_seq_len = inputs_embeds.size(1) - labels.size(1) - 1  # subtract target + eos
            target_len = labels.size(1) + 1  # +1 for EOS
            target_logits = logits[:, context_seq_len - 1 : context_seq_len - 1 + target_len, :]

            batch_size = labels.size(0)
            device = labels.device
            eos_batch = torch.full((batch_size, 1), self.config.eos_token_id, dtype=torch.long, device=device)
            targets = torch.cat([labels, eos_batch], dim=1)

            loss = nn.functional.cross_entropy(
                input=target_logits.reshape(-1, target_logits.size(-1)),
                target=targets.reshape(-1),
                ignore_index=self.config.pad_token_id,
                reduction="mean",
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(self, input_values=None, language_ids=None, **kwargs):
        """Generate token sequences from audio input.

        Builds ``inputs_embeds`` from the audio context (audio | lid_marker |
        lang_id | bos) and passes them to ``GenerationMixin.generate`` so that it
        tracks the correct sequence length from the start.  On subsequent
        decoding steps the cached KV states are reused and only the newly
        generated ``input_ids`` are forwarded to the language model.

        When ``language_ids`` is not provided, the model defaults to language
        ID 0 (a language-agnostic embedding learned during training).  Providing
        language codes is recommended for better transcription quality.

        Returns only newly generated tokens (the context prefix is stripped).
        """
        # Allow callers to pass **inputs from the processor directly.
        if input_values is None:
            input_values = kwargs.pop("input_values", None)
        if language_ids is None:
            language_ids = kwargs.pop("language_ids", None)

        if input_values is not None:
            batch_size = input_values.size(0)
            device = input_values.device

            # Build the full audio context as inputs_embeds
            audio_embeds = self.get_audio_features(input_values)
            dtype = audio_embeds.dtype
            text_embed_fn = self.get_input_embeddings()

            lid_marker_ids = torch.full((batch_size, 1), self.language_token_id, dtype=torch.long, device=device)
            bos_ids = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=device)

            lid_marker_embeds = text_embed_fn(lid_marker_ids).to(dtype)
            bos_embeds = text_embed_fn(bos_ids).to(dtype)

            if language_ids is not None:
                language_id_batch = language_ids.to(device)
            else:
                language_id_batch = torch.zeros(batch_size, dtype=torch.long, device=device)

            lang_id_embeds = self.lang_embeddings(language_id_batch.unsqueeze(-1)).to(dtype)

            inputs_embeds = torch.cat([audio_embeds, lid_marker_embeds, lang_id_embeds, bos_embeds], dim=1)
            # Drop the feature-extractor attention_mask (audio-length, wrong shape).
            kwargs.pop("attention_mask", None)

            # Let GenerationMixin initialize empty input_ids for decoder-only
            # generation when inputs_embeds are provided. This avoids creating a
            # synthetic prefix history and keeps cache/attention bookkeeping
            # aligned with the real multimodal context.
            return super().generate(inputs_embeds=inputs_embeds, **kwargs)

        return super().generate(**kwargs)


__all__ = [
    "OmniASRForCTC",
    "OmniASRForConditionalGeneration",
    "OmniASRModel",
    "OmniASRPreTrainedModel",
]