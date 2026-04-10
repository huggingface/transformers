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
"""PhoneticXeus model: E-Branchformer encoder for multilingual phone recognition."""

import math

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ..wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder
from .configuration_phoneticxeus import PhoneticXeusConfig


_HIDDEN_STATES_START_POSITION = 1


class PhoneticXeusFeatureEncoder(Wav2Vec2FeatureEncoder):
    pass


class PhoneticXeusFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class PhoneticXeusPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        if config.conv_pos_weight_norm:
            weight_norm = nn.utils.weight_norm
            if hasattr(nn.utils.parametrizations, "weight_norm"):
                weight_norm = nn.utils.parametrizations.weight_norm

            if is_deepspeed_zero3_enabled():
                import deepspeed

                with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                    self.conv = weight_norm(self.conv, name="weight", dim=2)
                if hasattr(self.conv, "parametrizations"):
                    weight_g = self.conv.parametrizations.weight.original0
                    weight_v = self.conv.parametrizations.weight.original1
                else:
                    weight_g = self.conv.weight_g
                    weight_v = self.conv.weight_v
                deepspeed.zero.register_external_parameter(self, weight_v)
                deepspeed.zero.register_external_parameter(self, weight_g)
            else:
                self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.num_pad_remove = 1 if config.num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class PhoneticXeusSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = hidden_states.size()

        query = self.linear_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        key = self.linear_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        value = self.linear_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, -1)
        hidden_states = self.linear_out(hidden_states)

        return hidden_states, attn_weights if output_attentions else None


class PhoneticXeusFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.w_2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.w_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_2(hidden_states)
        return hidden_states


class PhoneticXeusConvolutionalSpatialGatingUnit(nn.Module):
    """CSGU: splits input, applies depthwise conv on gate half, then element-wise multiply."""

    def __init__(self, config):
        super().__init__()
        n_channels = config.cgmlp_linear_units // 2
        self.norm = nn.LayerNorm(n_channels)
        self.conv = nn.Conv1d(
            n_channels,
            n_channels,
            config.cgmlp_conv_kernel,
            stride=1,
            padding=(config.cgmlp_conv_kernel - 1) // 2,
            groups=n_channels,
        )
        self.linear = nn.Linear(n_channels, n_channels) if config.use_linear_after_conv else None

        if config.gate_activation == "identity":
            self.gate_activation = nn.Identity()
        else:
            self.gate_activation = ACT2FN[config.gate_activation]

        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        x_r, x_g = hidden_states.chunk(2, dim=-1)
        x_g = self.norm(x_g)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)
        if self.linear is not None:
            x_g = self.linear(x_g)
        x_g = self.gate_activation(x_g)
        return self.dropout(x_r * x_g)


class PhoneticXeusConvolutionalGatingMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channel_proj1 = nn.Sequential(
            nn.Linear(config.hidden_size, config.cgmlp_linear_units),
            nn.GELU(),
        )
        self.csgu = PhoneticXeusConvolutionalSpatialGatingUnit(config)
        self.channel_proj2 = nn.Linear(config.cgmlp_linear_units // 2, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.channel_proj1(hidden_states)
        hidden_states = self.csgu(hidden_states)
        hidden_states = self.channel_proj2(hidden_states)
        return hidden_states


class PhoneticXeusEBranchformerEncoderLayer(GradientCheckpointingLayer):
    """E-Branchformer layer: parallel self-attention and cgMLP branches merged via depthwise conv."""

    def __init__(self, config):
        super().__init__()
        self.self_attn = PhoneticXeusSelfAttention(config)
        self.cgmlp = PhoneticXeusConvolutionalGatingMLP(config)

        self.norm_mha = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_mlp = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.depthwise_conv_fusion = nn.Conv1d(
            2 * config.hidden_size,
            2 * config.hidden_size,
            kernel_size=config.merge_conv_kernel,
            stride=1,
            padding=(config.merge_conv_kernel - 1) // 2,
            groups=2 * config.hidden_size,
            bias=True,
        )
        self.merge_proj = nn.Linear(2 * config.hidden_size, config.hidden_size)

        self.ff_scale = 1.0
        if config.use_ffn and config.macaron_ffn:
            self.feed_forward_macaron = PhoneticXeusFeedForward(config)
            self.norm_ff_macaron = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.ff_scale = 0.5
        else:
            self.feed_forward_macaron = None

        if config.use_ffn:
            self.feed_forward = PhoneticXeusFeedForward(config)
            self.norm_ff = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.feed_forward = None

        self.norm_final = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.feed_forward_macaron is not None:
            residual = hidden_states
            hidden_states = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(self.norm_ff_macaron(hidden_states))
            )

        x1 = self.norm_mha(hidden_states)
        x1, attn_weights = self.self_attn(x1, attention_mask=attention_mask, output_attentions=output_attentions)
        x1 = self.dropout(x1)

        x2 = self.norm_mlp(hidden_states)
        x2 = self.dropout(self.cgmlp(x2))

        x_concat = torch.cat([x1, x2], dim=-1)
        x_tmp = self.depthwise_conv_fusion(x_concat.transpose(1, 2)).transpose(1, 2)
        hidden_states = hidden_states + self.dropout(self.merge_proj(x_concat + x_tmp))

        if self.feed_forward is not None:
            residual = hidden_states
            hidden_states = residual + self.ff_scale * self.dropout(self.feed_forward(self.norm_ff(hidden_states)))

        hidden_states = self.norm_final(hidden_states)
        return hidden_states, attn_weights


class PhoneticXeusEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = PhoneticXeusPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [PhoneticXeusEBranchformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.interctc_layer_idx = list(config.interctc_layer_idx) if config.interctc_layer_idx else []
        self.interctc_use_conditioning = config.interctc_use_conditioning
        if self.interctc_use_conditioning and self.interctc_layer_idx:
            self.conditioning_layer = nn.Linear(config.vocab_size, config.hidden_size)
        else:
            self.conditioning_layer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ctc_head: nn.Linear | None = None,
    ) -> tuple | BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0.0

            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        hidden_states = self.pos_conv_embed(hidden_states)
        hidden_states = self.dropout(hidden_states)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            dropout_probability = torch.rand([])
            skip_the_layer = self.training and dropout_probability < self.config.layerdrop

            if not skip_the_layer or synced_gpus:
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

            if (
                self.interctc_use_conditioning
                and self.conditioning_layer is not None
                and ctc_head is not None
                and (i + 1) in self.interctc_layer_idx
            ):
                ctc_out = torch.softmax(ctc_head(hidden_states), dim=-1)
                hidden_states = hidden_states + self.conditioning_layer(ctc_out)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@auto_docstring
class PhoneticXeusPreTrainedModel(PreTrainedModel):
    config: PhoneticXeusConfig
    base_model_prefix = "phonetic_xeus"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, PhoneticXeusPositionalConvEmbedding):
            std = math.sqrt(4.0 / (module.conv.kernel_size[0] * module.conv.in_channels))
            init.normal_(module.conv.weight, mean=0, std=std)
            init.constant_(module.conv.bias, 0)
        elif isinstance(module, PhoneticXeusFeatureProjection):
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

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor | int):
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.cumsum(dim=-1)[:, -1]).to(torch.long)
        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask[(torch.arange(batch_size, device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask


class PhoneticXeusModel(PhoneticXeusPreTrainedModel):
    def __init__(self, config: PhoneticXeusConfig):
        super().__init__(config)
        self.feature_extractor = PhoneticXeusFeatureEncoder(config)
        self.feature_projection = PhoneticXeusFeatureProjection(config)
        self.encoder = PhoneticXeusEncoder(config)
        self.post_init()

    def freeze_feature_encoder(self):
        self.feature_extractor._freeze_parameters()

    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if self.config.normalize_audio:
            input_values = torch.nn.functional.layer_norm(input_values, input_values.shape)

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            ctc_head=kwargs.get("ctc_head"),
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class PhoneticXeusForCTC(PhoneticXeusPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.phonetic_xeus = PhoneticXeusModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `PhoneticXeusForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def freeze_feature_encoder(self):
        self.phonetic_xeus.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        for param in self.phonetic_xeus.parameters():
            param.requires_grad = False

    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        outputs = self.phonetic_xeus(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            ctc_head=self.lm_head,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

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

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


__all__ = [
    "PhoneticXeusForCTC",
    "PhoneticXeusModel",
    "PhoneticXeusPreTrainedModel",
]
