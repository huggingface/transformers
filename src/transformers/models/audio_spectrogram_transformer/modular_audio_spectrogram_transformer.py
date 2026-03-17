# Copyright 2022 MIT and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Audio Spectrogram Transformer (AST) model."""

import torch
from torch import nn

from ... import initialization as init
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..vit.modeling_vit import (
    ViTAttention,
    ViTEncoder,
    ViTLayer,
    ViTMLP,
    ViTPreTrainedModel,
)
from .configuration_audio_spectrogram_transformer import ASTConfig


class ASTPatchEmbeddings(nn.Module):
    """
    This class turns `input_values` (mel spectrograms) of shape `(batch_size, max_length, num_mel_bins)` into the
    initial `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: ASTConfig):
        super().__init__()

        patch_size = config.patch_size
        frequency_stride = config.frequency_stride
        time_stride = config.time_stride

        self.projection = nn.Conv2d(
            1, config.hidden_size, kernel_size=(patch_size, patch_size), stride=(frequency_stride, time_stride)
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        input_values = input_values.unsqueeze(1)
        input_values = input_values.transpose(2, 3)
        embeddings = self.projection(input_values).flatten(2).transpose(1, 2)
        return embeddings


class ASTEmbeddings(nn.Module):
    """
    Construct the CLS token, distillation token, position and patch embeddings for audio spectrograms.
    """

    def __init__(self, config: ASTConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ASTPatchEmbeddings(config)

        frequency_out_dimension, time_out_dimension = self.get_shape(config)
        num_patches = frequency_out_dimension * time_out_dimension
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def get_shape(self, config):
        # see Karpathy's cs231n blog on how to calculate the output dimensions
        # https://cs231n.github.io/convolutional-networks/#conv
        frequency_out_dimension = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
        time_out_dimension = (config.max_length - config.patch_size) // config.time_stride + 1
        return frequency_out_dimension, time_out_dimension

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        batch_size = input_values.shape[0]
        embeddings = self.patch_embeddings(input_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class ASTAttention(ViTAttention):
    pass


class ASTMLP(ViTMLP):
    pass


class ASTLayer(ViTLayer):
    pass


class ASTEncoder(ViTEncoder):
    pass


class ASTPreTrainedModel(ViTPreTrainedModel):
    config: ASTConfig
    base_model_prefix = "audio_spectrogram_transformer"
    main_input_name = "input_values"
    input_modalities = ("audio",)
    _no_split_modules = ["ASTEmbeddings", "ASTLayer"]
    _can_record_outputs = {
        "hidden_states": ASTLayer,
        "attentions": ASTAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, ASTEmbeddings):
            init.zeros_(module.cls_token)
            init.zeros_(module.position_embeddings)
            init.zeros_(module.distillation_token)


@auto_docstring
class ASTModel(ASTPreTrainedModel):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = ASTEmbeddings(config)
        self.encoder = ASTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ASTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, max_length, num_mel_bins)`):
            Float values mel features extracted from the raw audio waveform. Raw audio waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a
            `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or the soundfile library
            (`pip install soundfile`).
            To prepare the array into `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the
            mel features, padding and conversion into a tensor of type `torch.FloatTensor`.
            See [`~ASTFeatureExtractor.__call__`]
        """

        embedding_output = self.embeddings(input_values)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )
        encoder_outputs: BaseModelOutput = self.encoder(embedding_output, attention_mask, **kwargs)

        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        pooled_output = (sequence_output[:, 0] + sequence_output[:, 1]) / 2

        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output)


class ASTMLPHead(nn.Module):
    def __init__(self, config: ASTConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

    def forward(self, hidden_state):
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.dense(hidden_state)
        return hidden_state


@auto_docstring(
    custom_intro="""
    Audio Spectrogram Transformer model with an audio classification head on top (a linear layer on top of the pooled
    output) e.g. for datasets like AudioSet, Speech Commands v2.
    """
)
class ASTForAudioClassification(ASTPreTrainedModel):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.audio_spectrogram_transformer = ASTModel(config)

        # Classifier head
        self.classifier = ASTMLPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, max_length, num_mel_bins)`):
            Float values mel features extracted from the raw audio waveform. Raw audio waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via
            the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
            To prepare the array into `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the
            mel features, padding and conversion into a tensor of type `torch.FloatTensor`.
            See [`~ASTFeatureExtractor.__call__`]
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the audio classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs: BaseModelOutputWithPooling = self.audio_spectrogram_transformer(
            input_values,
            attention_mask=attention_mask,
            **kwargs,
        )

        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["ASTForAudioClassification", "ASTModel", "ASTPreTrainedModel"]
