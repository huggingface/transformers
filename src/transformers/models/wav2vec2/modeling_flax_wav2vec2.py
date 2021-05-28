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
""" PyTorch Wav2Vec2 model. """

from dataclasses import dataclass
from re import M
from typing import Callable, Optional, Tuple

import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as jax_xla
from flax.core.frozen_dict import FrozenDict
from flax.linen import dot_product_attention
from jax import config, lax
from jax.dtypes import dtype
from jax.random import PRNGKey

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxNextSentencePredictorOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import logging
from .configuration_wav2vec2 import Wav2Vec2Config


logger = logging.get_logger(__name__)


class Wav2Vec2NoLayerNormConvLayer(nn.Module):
    config: Wav2Vec2Config
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_conv_dim = self.config.conv_dim[self.layer_id] if self.layer_id > 0 else 1
        self.out_conv_dim = self.config.conv_dim[self.layer_id]

        self.conv = nn.Conv(
            features=self.out_conv_dim,
            kernel_size=self.config.conv_kernel[self.layer_id],
            strides=(self.config.conv_stride[self.layer_id],),
            use_bias=self.config.conv_bias,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.feat_extract_activation]

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2LayerNormConvLayer(nn.Module):
    config: Wav2Vec2Config
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_conv_dim = self.config.conv_dim[self.layer_id] if self.layer_id > 0 else 1
        self.out_conv_dim = self.config.conv_dim[self.layer_id]

        self.conv = nn.Conv(
            features=self.out_conv_dim,
            kernel_size=self.config.conv_kernel[self.layer_id],
            use_bias=self.config.conv_bias,
            kernel_init=jax.nn.initializers(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.activation = ACT2FN[self.config.feat_extract_activation]

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2GroupNormConvLayer(nn.Module):
    config: Wav2Vec2Config
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_conv_dim = self.config.conv_dim[self.layer_id] if self.layer_id > 0 else 1
        self.out_conv_dim = self.config.conv_dim[self.layer_id]

        self.conv = nn.Conv(
            features=self.out_conv_dim,
            kernel_size=self.config.conv_kernel[self.layer_id],
            use_bias=self.config.conv_bias,
            kernel_init=jax.nn.initializers(self.config.initializer_range, self.dtype),
            dtype=self.dtype,
        )
        self.layer_norm = nn.GroupNorm(
            num_groups=self.out_conv_dim, group_size=self.out_conv_dim, epsilon=1e-5, dtype=self.dtype
        )
        self.activation = ACT2FN[self.config.feat_extract_activation]

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            features=self.config.hidden_size,
            kernel_size=self.config.num_conv_pos_embeddings,
            kernel_init=jax.nn.initializers(self.config.initializer_range, self.dtype),
            padding=(self.config.num_conv_pos_embeddings // 2,),
            feature_group_count=self.config.num_conv_pos_embeddings,
            dtype=self.dtype,
        )  # TODO (PS): weigh_norm ?
        self.activation = ACT2FN[self.config.feat_extract_activation]
        self.num_pad_remove = 1 if self.config.num_conv_pos_embeddings % 2 == 0 else 0

    def __call__(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2FeatureExtractor(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        config = self.config
        if config.feat_extract_norm == "group":
            self.conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0, dtype=self.dtype)] + [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1, dtype=self.dtype)
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            self.conv_layers = [
                Wav2Vec2LayerNormConvLayer(config, layer_id=i, dtype=self.dtype)
                for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )

    def __call__(self, input_values):
        hidden_states = input_values[:, None]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class Wav2Vec2FeatureProjection(nn.Module):
    pass


class Wav2Vec2Attention(nn.Module):
    pass


class Wav2Vec2FeedForward(nn.Module):
    pass


class Wav2Vec2Output(nn.Module):
    pass


class Wav2Vec2EncoderLayer(nn.Module):
    pass


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    pass


class Wav2Vec2Encoder(nn.Module):
    pass


class Wav2Vec2StableLayerNormEncoder(nn.Module):
    pass


class Wav2Vec2PreTrainedModel(FlaxPreTrainedModel):
    pass


class Wav2Vec2Model(Wav2Vec2PreTrainedModel):
    pass


class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    pass
