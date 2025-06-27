# coding=utf-8
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
"""PyTorch UniSpeechSat model."""

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from ...modeling_outputs import ModelOutput, Wav2Vec2BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2ForXVector,
    Wav2Vec2GumbelVectorQuantizer,
    Wav2Vec2Model,
    Wav2Vec2PositionalConvEmbedding,
)
from .configuration_unispeech_sat import UniSpeechSatConfig


logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 2


@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`UniSpeechSatForPreTrainingOutput`], with potential hidden states and attentions.
    """
)
class UniSpeechSatForPreTrainingOutput(ModelOutput):
    r"""
    loss (*optional*, returned when model is in train mode, `torch.FloatTensor` of shape `(1,)`):
        Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
        paper](https://arxiv.org/pdf/2006.11477.pdf) . (classification) loss.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`, *optional*):
        Prediction scores of the contrastive loss model, i.e. the output of the model before the final softmax.
    projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
        Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
        projected quantized states.
    projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
        Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
        target vectors for contrastive loss.
    codevector_perplexity (`torch.FloatTensor` of shape `(1,)`):
        The perplexity of the codevector distribution, used to measure the diversity of the codebook.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    projected_states: Optional[torch.FloatTensor] = None
    projected_quantized_states: Optional[torch.FloatTensor] = None
    codevector_perplexity: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class UniSpeechSatPositionalConvEmbedding(Wav2Vec2PositionalConvEmbedding):
    pass


class UniSpeechSatFeatureEncoder(Wav2Vec2FeatureEncoder):
    pass


class UniSpeechSatFeatureProjection(Wav2Vec2FeatureProjection):
    pass


class UniSpeechSatEncoder(Wav2Vec2Encoder):
    pass


class UniSpeechSatEncoderStableLayerNorm(Wav2Vec2EncoderStableLayerNorm):
    pass


class UniSpeechSatGumbelVectorQuantizer(Wav2Vec2GumbelVectorQuantizer):
    def __init__(self, config):
        super().__init__()
        self.weight_proj = nn.Linear(config.hidden_size, self.num_groups * self.num_vars)

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        marginal_probs = probs.mean(dim=0)
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # sample code vector probs via gumbel in differentiateable way
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # compute perplexity
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist)
        else:
            # take argmax in non-differentiable way
            # comptute hard codevector distribution (one hot)
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # use probs to retrieve codevectors
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        return codevectors, perplexity


@auto_docstring
class UniSpeechSatPreTrainedModel(PreTrainedModel):
    config_class = UniSpeechSatConfig
    base_model_prefix = "unispeech_sat"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # gumbel softmax requires special init
        if isinstance(module, UniSpeechSatGumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        elif isinstance(module, UniSpeechSatPositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, UniSpeechSatFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
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
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths).to(torch.long)
        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask


UniSpeechSatBaseModelOutput = Wav2Vec2BaseModelOutput


class UniSpeechSatModel(UniSpeechSatPreTrainedModel, Wav2Vec2Model):
    def __init__(self, config: UniSpeechSatConfig):
        UniSpeechSatPreTrainedModel.__init__(config)
        self.config = config
        self.feature_extractor = UniSpeechSatFeatureEncoder(config)
        self.feature_projection = UniSpeechSatFeatureProjection(config)

        self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = UniSpeechSatEncoderStableLayerNorm(config)
        else:
            self.encoder = UniSpeechSatEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        raise AttributeError("Not needed for UniSpeechSat")

    def freeze_feature_encoder(self):
        raise AttributeError("Not needed for UniSpeechSat")

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, UniSpeechSatBaseModelOutput]:
        r"""
        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return UniSpeechSatBaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    UniSpeechSat Model with a vector-quantization module and ctc loss for pre-training.
    """
)
class UniSpeechSatForPreTraining(UniSpeechSatPreTrainedModel):
    def __init__(self, config: UniSpeechSatConfig):
        super().__init__(config)
        self.unispeech_sat = UniSpeechSatModel(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        self.quantizer = UniSpeechSatGumbelVectorQuantizer(config)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)
        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)

        self.dropout = nn.Dropout(config.final_dropout)

        self.speaker_proj = nn.Linear(config.hidden_size, config.codevector_dim)
        self.label_embeddings_concat = nn.Parameter(torch.FloatTensor(config.num_clusters, config.codevector_dim))
        self.label_embeddings_concat.data.zero_()

        self.layer_norm_for_extract = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if self.config.do_stable_layer_norm:
            self.layer_norm_for_extract.requires_grad = False

        # Initialize weights and apply final processing
        self.post_init()

    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.unispeech_sat.feature_extractor._freeze_parameters()

    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = 1,
    ):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        target_features = torch.cat([target_features, negative_features], dim=0)

        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1)
        logits = logits.type_as(target_features)

        # apply temperature
        logits = logits / temperature
        return logits

    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, UniSpeechSatForPreTrainingOutput]:
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, UniSpeechSatForPreTraining
        >>> from transformers.models.unispeech_sat.modeling_unispeech_sat import _compute_mask_indices

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-sat-base")
        >>> model = UniSpeechSatForPreTraining.from_pretrained("microsoft/unispeech-sat-base")
        >>> # TODO: Add full pretraining example
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        transformer_features = outputs[0]

        # quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])

        # TODO(PVP) - add pretraining logic and add to tests
        logits = extract_features
        loss = quantized_features = codevector_perplexity = None

        if not return_dict:
            if loss is not None:
                return (loss, logits, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (logits, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return UniSpeechSatForPreTrainingOutput(
            loss=loss,
            logits=logits,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class UniSpeechSatForCTC(Wav2Vec2ForCTC):
    pass


class UniSpeechSatForSequenceClassification(Wav2Vec2ForSequenceClassification):
    pass


class UniSpeechSatForAudioFrameClassification(Wav2Vec2ForAudioFrameClassification):
    pass


class UniSpeechSatForXVector(Wav2Vec2ForXVector):
    pass


__all__ = [
    "UniSpeechSatForAudioFrameClassification",
    "UniSpeechSatForCTC",
    "UniSpeechSatForPreTraining",
    "UniSpeechSatForSequenceClassification",
    "UniSpeechSatForXVector",
    "UniSpeechSatModel",
    "UniSpeechSatPreTrainedModel",
]
