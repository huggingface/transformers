# coding=utf-8
# Copyright 2022 Intel Labs, OpenMMLab and The HuggingFace Inc. team. All rights reserved.
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
""" Flax DPT (Dense Prediction Transformers) model.

TThis implementation is heavily inspired by OpenMMLab's implementation, found here:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/dpt_head.py.

"""
import math
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxDepthEstimatorOutput,
    FlaxSemanticSegmenterOutput,
)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_dpt import DPTConfig


DPT_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models) This
    model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
    general usage and behavior. Finally, this model supports inherent JAX features such as:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs). This can be used to enable mixed-precision training or half-precision
            inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.** If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`]
            and [`~FlaxPreTrainedModel.to_bf16`].
"""

DPT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`DPTFeatureExtractor`]. See
            [`DPTFeatureExtractor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FlaxPatchEmbeddings(nn.Module):

    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.num_patches = num_patches
        self.projection = nn.Conv(
            self.config.hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(self, pixel_values):
        x = self.projection(pixel_values)
        batch_size, _, _, channels = x.shape
        return jnp.reshape(x, (batch_size, -1, channels))


class FlaxDPTEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.cls_token = self.param("cls_token", nn.initializers.zeros, (1, 1, self.config.hidden_size))
        self.patch_embeddings = FlaxPatchEmbeddings(self.config, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.param(
            "position_embeddings", nn.initializers.zeros, (1, num_patches + 1, self.config.hidden_size)
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, pixel_values, deterministic=True):
        batch_size = pixel_values.shape[0]

        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings


class FlaxViTSelfAttention(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`:"
                " {self.config.num_attention_heads}"
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            use_bias=self.config.qkv_bias,
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            use_bias=self.config.qkv_bias,
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            use_bias=self.config.qkv_bias,
        )

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxDPTViTOutput(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = hidden_states + attention_output
        return hidden_states


class FlaxDPTViTSelfOutput(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxDPTViTAttention(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxViTSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxDPTViTSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True, output_attentions: bool = False):
        attn_outputs = self.attention(hidden_states, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


class FlaxDPTViTIntermediate(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# DPT reassemble & Fusion
class FlaxDPTReassembleLayer(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32
    factor: int = 1
    channels: int = None

    def setup(self):
        # projection
        self.projection = nn.Conv(
            self.config.hidden_size,
            kernel_size=(1, 1),
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # up/down sampling depending on factor
        if self.factor > 1:
            self.resize = nn.ConvTranspose(
                self.channels, kernel_size=(self.factor, self.factor), strides=(self.factor, self.factor)
            )
        elif self.factor < 1:
            # so should downsample
            self.resize = nn.Conv(
                self.channels,
                kernel_size=(3, 3),
                strides=(int(1 / self.factor), int(1 / self.factor)),
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )

    def __call__(self, hidden_state):
        hidden_state = self.projection(hidden_state)
        if self.factor != 1:
            hidden_state = self.resize(hidden_state)
        return hidden_state


class FlaxDPTReassembleStage(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):

        self.layers = [
            FlaxDPTReassembleLayer(self.config, factor=factor, channels=self.config.neck_hidden_sizes[i])
            for i, factor in zip(range(len(self.config.neck_hidden_sizes)), self.config.reassemble_factors)
        ]

        if self.config.readout_type == "project":
            self.readout_projects = [
                nn.Sequential([nn.Dense(self.config.hidden_size), ACT2FN[self.config.hidden_act]])
                for _ in range(len(self.config.neck_hidden_sizes))
            ]

    def __call__(self, hidden_states):
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        out = []

        for i, hidden_state in enumerate(hidden_states):
            # reshape to (B, C, H, W)
            hidden_state, cls_token = hidden_state[:, 1:], hidden_state[:, 0]
            batch_size, sequence_length, num_channels = hidden_state.shape
            size = int(math.sqrt(sequence_length))
            hidden_state = jnp.reshape(hidden_state, (batch_size, size, size, num_channels))

            feature_shape = hidden_state.shape
            if self.config.readout_type == "project":
                # reshape to (B, H*W, C)
                hidden_state = jnp.reshape(hidden_state, (batch_size, size * size, num_channels))
                readout = jnp.expand_dims(cls_token, axis=1)
                readout = jnp.repeat(readout, size * size, axis=1)
                # concatenate the readout token to the hidden states and project
                hidden_state = self.readout_projects[i](jnp.concatenate((hidden_state, readout), axis=-1))
                # reshape back to (B, C, H, W)
                hidden_state = jnp.reshape(hidden_state, feature_shape)
            elif self.config.readout_type == "add":
                hidden_state = jnp.reshape(hidden_state, (batch_size, size * size, num_channels)) + jnp.expand_dims(
                    cls_token, axis=-1
                )
                hidden_state = jnp.reshape(hidden_state, feature_shape)
            hidden_state = self.layers[i](hidden_state)
            out.append(hidden_state)

        return out


class FlaxDPTFeatureFusionStage(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        super().__init__()
        self.layers = [FlaxDPTFeatureFusionLayer(self.config) for _ in range(len(self.config.neck_hidden_sizes))]

    def __call__(self, hidden_states):
        # reversing the hidden_states, we start from the last
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []
        # first layer only uses the last hidden_state
        fused_hidden_state = self.layers[0](hidden_states[0])
        fused_hidden_states.append(fused_hidden_state)
        # looping from the last layer to the second
        for hidden_state, layer in zip(hidden_states[1:], self.layers[1:]):
            fused_hidden_state = layer(fused_hidden_state, hidden_state)
            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class FlaxDPTPreActResidualLayer(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):

        self.use_batch_norm = self.config.use_batch_norm_in_fusion_residual
        self.activation1 = ACT2FN["relu"]
        self.convolution1 = nn.Conv(
            self.config.fusion_hidden_size,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            use_bias=not self.use_batch_norm,
        )

        self.activation2 = ACT2FN["relu"]
        self.convolution2 = nn.Conv(
            self.config.fusion_hidden_size,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            use_bias=not self.use_batch_norm,
        )

        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm(use_running_average=False)
            self.batch_norm2 = nn.BatchNorm(use_running_average=False)

    def __call__(self, hidden_state):
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)

        hidden_state = self.convolution1(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)

        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)

        return hidden_state + residual

    def __call__(self, hidden_state):
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)

        hidden_state = self.convolution1(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)

        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)

        return hidden_state + residual


class FlaxDPTFeatureFusionLayer(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32
    align_corners: bool = True

    def setup(self):
        self.projection = nn.Conv(self.config.fusion_hidden_size, kernel_size=(1, 1))  # , bias=True)

        self.residual_layer1 = FlaxDPTPreActResidualLayer(self.config)
        self.residual_layer2 = FlaxDPTPreActResidualLayer(self.config)
        self.upsample = FlaxDPTUpsample()

    def __call__(self, hidden_state, residual=None):
        if residual is not None:
            if hidden_state.shape != residual.shape:
                size = hidden_state.shape
                residual = self.upsample(residual, size)
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = self.upsample(hidden_state)
        hidden_state = self.projection(hidden_state)

        return hidden_state


class FlaxDPTViTLayer(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.attention = FlaxDPTViTAttention(self.config, dtype=self.dtype)
        self.intermediate = FlaxDPTViTIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxDPTViTOutput(self.config, dtype=self.dtype)
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

        # first residual connection
        attention_output = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(attention_output)

        hidden_states = self.intermediate(layer_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


class FlaxDPTViTLayerCollection(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxDPTViTLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class FlaxDPTViTPooler(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)


class FlaxDPTViTEncoder(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxDPTViTLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


DPT_START_DOCSTRING = r"""
    This model is a Flax [jax.nn.Module](https://jax.readthedocs.io/en/latest/jax.nn.html?highlight=nn.Module)
    subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general
    usage and behavior.

    Parameters:
        config ([`DPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DPT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`jax.numpy.array` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ViTFeatureExtractor`]. See
            [`ViTFeatureExtractor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


class FlaxDPTPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPTConfig
    base_model_prefix = "dpt"
    main_input_name = "pixel_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: DPTConfig,
        input_shape=None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, 3)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, pixel_values, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[jnp.ndarray] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            labels,
            rngs=rngs,
        )


class FlaxDPTModule(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxDPTEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxDPTViTEncoder(self.config, dtype=self.dtype)
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.pooler = FlaxDPTViTPooler(self.config, dtype=self.dtype) if self.add_pooling_layer else None

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        labels: Optional[jnp.ndarray] = None,
    ):

        hidden_states = self.embeddings(pixel_values, deterministic=deterministic)

        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.layernorm(hidden_states)
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    DPT Model with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    DPT_START_DOCSTRING,
)
class FlaxDPTModel(FlaxDPTPreTrainedModel):
    module_class = FlaxDPTModule


class FlaxDPTNeck(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # postprocessing
        self.reassemble_stage = FlaxDPTReassembleStage(self.config)
        self.conv_list = [
            nn.Conv(self.config.fusion_hidden_size, kernel_size=(3, 3), padding=1, use_bias=False)
            for i in range(len(self.config.neck_hidden_sizes))
        ]
        # fusion
        self.fusion_stage = FlaxDPTFeatureFusionStage(self.config)

    def __call__(self, hidden_states):
        if not isinstance(hidden_states, list):
            raise ValueError("hidden_states should be a list of tensors")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")

        # postprocess hidden states
        features = self.reassemble_stage(hidden_states)

        features = [self.conv_list[i](feature) for i, feature in enumerate(features)]

        # fusion blocks
        output = self.fusion_stage(features)

        return output


class FlaxDPTUpsample(nn.Module):
    scale: int = 2
    method: str = "bilinear"

    def setup(self):
        pass

    def __call__(self, x, output_size=None):
        if output_size is None:
            output_size = x.shape
            output_size = (output_size[0], output_size[1] * self.scale, output_size[2] * self.scale, output_size[3])
        return jax.image.resize(x, output_size, method="bilinear")


class FlaxDPTDepthEstimationHead(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):

        features = self.config.fusion_hidden_size
        self.head = nn.Sequential(
            [
                nn.Conv(features // 2, kernel_size=(3, 3), strides=(1, 1), padding=1),
                FlaxDPTUpsample(scale=2, method="bilinear"),
                nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding=1),
                ACT2FN["relu"],
                nn.Conv(1, kernel_size=(1, 1), strides=(1, 1), padding=0),
                ACT2FN["relu"],
            ]
        )

    def __call__(self, hidden_states):
        # use last features
        hidden_states = hidden_states[self.config.head_in_index]

        predicted_depth = self.head(hidden_states)

        predicted_depth = jnp.squeeze(predicted_depth, -1)

        return predicted_depth


class FlaxDPTForDepthEstimationModule(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):

        self.dpt = FlaxDPTModule(self.config, add_pooling_layer=False)

        # Neck
        self.neck = FlaxDPTNeck(self.config)

        # Depth estimation head
        self.head = FlaxDPTDepthEstimationHead(self.config)

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import DPTFeatureExtractor, DPTForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        >>> model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        >>> # prepare image for the model
        >>> inputs = feature_extractor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     predicted_depth = outputs.predicted_depth

        >>> # interpolate to original size
        >>> prediction = torch.nn.functional.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().cpu().numpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.dpt(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states if return_dict else outputs

        # only keep certain features based on config.backbone_out_indices
        # note that the hidden_states also include the initial embeddings
        if return_dict:
            hidden_states = [
                feature for idx, feature in enumerate(hidden_states) if idx in self.config.backbone_out_indices
            ]
        else:
            hidden_states = [
                feature for idx, feature in enumerate(hidden_states[1]) if idx in self.config.backbone_out_indices
            ]

        hidden_states = self.neck(hidden_states)

        predicted_depth = self.head(hidden_states)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return FlaxDepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    DPT Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
    """,
    DPT_START_DOCSTRING,
)
class FlaxDPTForDepthEstimation(FlaxDPTPreTrainedModel):
    module_class = FlaxDPTForDepthEstimationModule


class FlaxDPTSemanticSegmentationHead(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32

    # TODO: Change this to make sure BatchNorm + Dropout work / Put them outside a Sequential Module
    def setup(self):
        features = self.config.fusion_hidden_size
        self.head = nn.Sequential(
            [
                nn.Conv(features, kernel_size=(3, 3), padding=1),
                # nn.BatchNorm(use_running_average=False),
                ACT2FN["relu"],
                # nn.Dropout(self.config.semantic_classifier_dropout, deterministic=False),
                nn.Conv(self.config.num_labels, kernel_size=(1, 1)),
                FlaxDPTUpsample(scale=2, method="bilinear"),
            ]
        )

    def __call__(self, hidden_states):
        # use last features
        hidden_states = hidden_states[self.config.head_in_index]

        logits = self.head(hidden_states)

        return logits


class FlaxDPTAuxiliaryHead(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32

    # TODO: Change this to make sure BatchNorm + Dropout work / Put them outside a Sequential Module
    def setup(self):
        features = self.config.fusion_hidden_size
        self.head = nn.Sequential(
            [
                nn.Conv(features, kernel_size=(3, 3), padding=1, use_bias=False),  # bias=False
                # nn.BatchNorm(use_running_average=False),
                ACT2FN["relu"],
                # nn.Dropout(0.1, deterministic=False),
                nn.Conv(self.config.num_labels, kernel_size=(1, 1)),
            ]
        )

    def __call__(self, hidden_states):
        logits = self.head(hidden_states)

        return logits


class FlaxDPTForSemanticSegmentationModule(nn.Module):
    config: DPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):

        self.dpt = FlaxDPTModule(self.config, add_pooling_layer=False)

        # Neck
        self.neck = FlaxDPTNeck(self.config)

        # Segmentation head(s)
        self.head = FlaxDPTSemanticSegmentationHead(self.config)
        self.auxiliary_head = FlaxDPTAuxiliaryHead(self.config) if self.config.use_auxiliary_head else None

        self.upsample = FlaxDPTUpsample(scale=2, method="bilinear")

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:
        ```python
        >>> from transformers import DPTFeatureExtractor, DPTForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large-ade")
        >>> model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")

        >>> inputs = feature_extractor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.dpt(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states if return_dict else outputs

        # only keep certain features based on config.backbone_out_indices
        # note that the hidden_states also include the initial embeddings
        if return_dict:
            hidden_states = [
                feature for idx, feature in enumerate(hidden_states) if idx in self.config.backbone_out_indices
            ]
        else:
            hidden_states = [
                feature for idx, feature in enumerate(hidden_states[1]) if idx in self.config.backbone_out_indices
            ]
        hidden_states = self.neck(hidden_states)

        logits = self.head(hidden_states)

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(hidden_states[-1])

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # upsample logits to the images' original size
                output_shape = (logits.shape[0], labels.shape[1], labels.shape[2], logits.shape[3])
                upsampled_logits = self.upsample(logits, output_shape)

                if auxiliary_logits is not None:
                    upsampled_auxiliary_logits = self.upsample(auxiliary_logits, output_shape)
                # compute weighted loss
                # Copied from: https://flax.readthedocs.io/en/latest/notebooks/annotated_mnist.html
                labels_onehot = jax.nn.one_hot(labels, num_classes=self.config.num_labels)
                main_loss = optax.softmax_cross_entropy(logits=upsampled_logits, labels=labels_onehot).mean()
                auxiliary_loss = optax.softmax_cross_entropy(
                    logits=upsampled_auxiliary_logits, labels=labels_onehot
                ).mean()
                loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return FlaxSemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    DPT Model with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    DPT_START_DOCSTRING,
)
class FlaxDPTForSemanticSegmentation(FlaxDPTPreTrainedModel):
    module_class = FlaxDPTForSemanticSegmentationModule
