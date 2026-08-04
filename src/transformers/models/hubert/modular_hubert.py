# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Hubert model."""

import torch
import torch.nn as nn

from ... import initialization as init
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Model,
    Wav2Vec2SamePadLayer,
)
from .configuration_hubert import HubertConfig


_HIDDEN_STATES_START_POSITION = 1


class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        self.batch_norm = None
        if getattr(config, "conv_pos_batch_norm", False):
            self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        else:
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

        self.padding = HubertSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def _masked_batch_norm(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_mask = attention_mask[:, None, :].bool()

        if not self.batch_norm.training and self.batch_norm.track_running_stats:
            hidden_states = self.batch_norm(hidden_states)
            return hidden_states.masked_fill(~attention_mask, 0.0)

        input_dtype = hidden_states.dtype
        if input_dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.float()

        mask = attention_mask.to(hidden_states.dtype)
        count = mask.sum().clamp_min(1.0)
        mean = (hidden_states * mask).sum(dim=(0, 2)) / count
        centered_hidden_states = hidden_states - mean[None, :, None]
        variance = (centered_hidden_states.square() * mask).sum(dim=(0, 2)) / count

        if self.batch_norm.training and self.batch_norm.track_running_stats:
            self.batch_norm.num_batches_tracked.add_(1)
            exponential_average_factor = self.batch_norm.momentum
            if exponential_average_factor is None:
                exponential_average_factor = self.batch_norm.num_batches_tracked.reciprocal()

            unbiased_variance = variance * count / (count - 1.0).clamp_min(1.0)
            with torch.no_grad():
                self.batch_norm.running_mean.lerp_(
                    mean.to(self.batch_norm.running_mean.dtype), exponential_average_factor
                )
                self.batch_norm.running_var.lerp_(
                    unbiased_variance.to(self.batch_norm.running_var.dtype), exponential_average_factor
                )

        hidden_states = centered_hidden_states * torch.rsqrt(variance[None, :, None] + self.batch_norm.eps)
        if self.batch_norm.affine:
            hidden_states = hidden_states * self.batch_norm.weight[None, :, None]
            hidden_states = hidden_states + self.batch_norm.bias[None, :, None]

        return hidden_states.to(input_dtype).masked_fill(~attention_mask, 0.0)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = hidden_states.transpose(1, 2)
        if self.batch_norm is not None:
            if attention_mask is None:
                hidden_states = self.batch_norm(hidden_states)
            else:
                hidden_states = self._masked_batch_norm(hidden_states, attention_mask)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class HubertSamePadLayer(Wav2Vec2SamePadLayer):
    pass


class HubertFeatureEncoder(Wav2Vec2FeatureEncoder):
    pass


class HubertFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class HubertEncoder(Wav2Vec2Encoder):
    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

        position_embeddings = self.pos_conv_embed(hidden_states, attention_mask=attention_mask)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + position_embeddings.to(hidden_states.device)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and dropout_probability < self.config.layerdrop
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class HubertEncoderStableLayerNorm(Wav2Vec2EncoderStableLayerNorm):
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
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

        position_embeddings = self.pos_conv_embed(hidden_states, attention_mask=attention_mask)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and dropout_probability < self.config.layerdrop
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

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
class HubertPreTrainedModel(PreTrainedModel):
    config: HubertConfig
    base_model_prefix = "hubert"
    main_input_name = "input_values"
    input_modalities = "audio"
    _no_split_modules = ["HubertEncoderLayer", "ParametrizedConv1d"]
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, nn.Conv1d):
            if is_deepspeed_zero3_enabled():
                import deepspeed

                if hasattr(module, "weight_v") and hasattr(module, "weight_g"):
                    with deepspeed.zero.GatheredParameters([module.weight_v, module.weight_g], modifier_rank=0):
                        init.kaiming_normal_(module.weight)
                else:
                    with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                        init.kaiming_normal_(module.weight)
            else:
                init.kaiming_normal_(module.weight)

            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, HubertModel):
            if hasattr(module, "masked_spec_embed"):
                init.uniform_(module.masked_spec_embed)
        elif isinstance(module, HubertForSequenceClassification):
            if hasattr(module, "layer_weights"):
                init.constant_(module.layer_weights, 1.0 / (self.config.num_hidden_layers + 1))

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor | int):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask


class HubertModel(Wav2Vec2Model, HubertPreTrainedModel):
    def __init__(self, config: HubertConfig):
        super().__init__(config)
        self.config = config
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = HubertFeatureProjection(config)

        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = HubertEncoderStableLayerNorm(config)
        else:
            self.encoder = HubertEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

        del self.adapter

    def freeze_feature_encoder(self):
        raise AttributeError("Not needed for Hubert")

    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        mask_time_indices: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutput:
        r"""
        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.

        Example:

        ```python
        >>> from transformers import AutoProcessor, HubertModel
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        >>> model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


        >>> def map_to_array(example):
        ...     example["speech"] = example["audio"]["array"]
        ...     return example


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
        >>> hidden_states = model(input_values).last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class HubertForCTC(Wav2Vec2ForCTC):
    pass


class HubertForSequenceClassification(Wav2Vec2ForSequenceClassification):
    pass


__all__ = ["HubertForCTC", "HubertForSequenceClassification", "HubertModel", "HubertPreTrainedModel"]
