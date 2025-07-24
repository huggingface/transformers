# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""PyTorch Data2VecText model."""

import math

import torch
from torch import nn

from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import Wav2Vec2BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ..wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Adapter,
    Wav2Vec2Encoder,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2ForXVector,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2SamePadLayer,
)
from .configuration_data2vec_audio import Data2VecAudioConfig


class Data2VecAudioConvLayer(GradientCheckpointingLayer):
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


class Data2VecAudioPadLayer(Wav2Vec2SamePadLayer):
    pass


class Data2VecAudioPositionalConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_pos_kernel_size,
            padding=config.conv_pos_kernel_size // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        self.padding = Data2VecAudioPadLayer(config.conv_pos_kernel_size)
        self.activation = ACT2FN[config.feat_extract_activation]
        # no learnable parameters
        self.layer_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Data2VecAudioPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Data2VecAudioPositionalConvLayer(config) for _ in range(config.num_conv_pos_embeddings)]
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Data2VecAudioFeatureEncoder(Wav2Vec2FeatureEncoder, nn.Module):
    def __init__(self, config):
        nn.Module.__init__()
        self.conv_layers = nn.ModuleList(
            [Data2VecAudioConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        )
        self.gradient_checkpointing = False
        self._requires_grad = True


class Data2VecAudioFeatureProjection(Wav2Vec2FeatureProjection):
    pass


class Data2VecAudioEncoder(Wav2Vec2Encoder):
    pass


class Data2VecAudioAdapter(Wav2Vec2Adapter):
    pass


class Data2VecAudioPreTrainedModel(PreTrainedModel, Wav2Vec2PreTrainedModel):
    config: Data2VecAudioConfig
    base_model_prefix = "data2vec_audio"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Data2VecAudioFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, Data2VecAudioPositionalConvLayer):
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_adapters(self):
        raise AttributeError("Not needed for Data2VecAudio")

    def init_adapter_layers(self):
        raise AttributeError("Not needed for Data2VecAudio")

    def load_adapter(self):
        raise AttributeError("Not needed for Data2VecAudio")


Data2VecAudioBaseModelOutput = Wav2Vec2BaseModelOutput


class Data2VecAudioModel(Data2VecAudioPreTrainedModel, Wav2Vec2Model):
    def __init__(self, config: Data2VecAudioConfig):
        Data2VecAudioPreTrainedModel.__init__(config)
        self.config = config
        self.feature_extractor = Data2VecAudioFeatureEncoder(config)
        self.feature_projection = Data2VecAudioFeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

        self.encoder = Data2VecAudioEncoder(config)

        self.adapter = Data2VecAudioAdapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        raise AttributeError("Not needed for Data2VecAudio")

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    def forward(self, **super_kwargs):
        return super().forward(**super_kwargs)


class Data2VecAudioForCTC(Data2VecAudioPreTrainedModel, Wav2Vec2ForCTC):
    def __init__(self, config):
        Data2VecAudioPreTrainedModel.__init__(config)

        self.data2vec_audio = Data2VecAudioModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Data2VecAudioForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_base_model(self):
        raise AttributeError("Not needed for Data2VecAudio")

    def tie_weights(self):
        raise AttributeError("Not needed for Data2VecAudio")

    def forward(self, **super_kwargs):
        return super().forward(**super_kwargs)


class Data2VecAudioForSequenceClassification(Wav2Vec2ForSequenceClassification):
    pass


class Data2VecAudioForAudioFrameClassification(Wav2Vec2ForAudioFrameClassification):
    pass


class Data2VecAudioForXVector(Wav2Vec2ForXVector):
    pass


__all__ = [
    "Data2VecAudioForAudioFrameClassification",
    "Data2VecAudioForCTC",
    "Data2VecAudioForSequenceClassification",
    "Data2VecAudioForXVector",
    "Data2VecAudioModel",
    "Data2VecAudioPreTrainedModel",
]
