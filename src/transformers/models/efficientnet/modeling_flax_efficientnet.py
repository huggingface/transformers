# coding=utf-8
# Copyright 2023 Google Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
from functools import partial
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithNoAttention,
    FlaxBaseModelOutputWithPooling,
    FlaxBaseModelOutputWithPoolingAndNoAttention,
    FlaxImageClassifierOutputWithNoAttention,
)
from transformers.modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from transformers.models.efficientnet.configuration_efficientnet import (
    EfficientNetConfig,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)


EFFICIENTNET_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
    general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`EfficientNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

EFFICIENTNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`EfficientNetImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class Identity(nn.Module):
    """Identity function."""

    @nn.compact
    def __call__(self, x, deterministic=None):
        return x


# Copied from transformers.models.efficientnet.modeling_efficientnet.round_filters
def round_filters(config: EfficientNetConfig, num_channels: int):
    r"""
    Round number of filters based on depth multiplier.
    """
    divisor = config.depth_divisor
    num_channels *= config.width_coefficient
    new_dim = max(divisor, int(num_channels + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_dim < 0.9 * num_channels:
        new_dim += divisor

    return int(new_dim)


# Copied from transformers.models.efficientnet.modeling_efficientnet.correct_pad
def correct_pad(kernel_size: Union[int, Tuple], adjust: bool = True):
    r"""
    Utility function to get the tuple padding value for the depthwise convolution.

    Args:
        kernel_size (`int` or `tuple`):
            Kernel size of the convolution layers.
        adjust (`bool`, *optional*, defaults to `True`):
            Adjusts padding value to apply to right and bottom sides of the input.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    if adjust:
        return (correct[1] - 1, correct[1], correct[0] - 1, correct[0])
    else:
        return (correct[1], correct[1], correct[0], correct[0])


class FlaxEfficientNetEmbeddings(nn.Module):
    r"""
    A module that corresponds to the stem module of the original work.
    """

    config: EfficientNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        out_channels = round_filters(self.config, 32)
        self.padding = partial(jnp.pad, pad_width=((0, 0), (0, 1), (0, 1), (0, 0)))
        self.convolution = nn.Conv(
            out_channels,
            kernel_size=[3, 3],
            strides=[2, 2],
            padding="VALID",
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.batchnorm = nn.BatchNorm(
            epsilon=self.config.batch_norm_eps,
            momentum=1 - self.config.batch_norm_momentum,
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, pixel_values: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        features = self.padding(
            pixel_values,
        )
        features = self.convolution(features)
        features = self.batchnorm(features, use_running_average=deterministic)
        features = self.activation(features)

        return features


class FlaxEfficientNetExpansionLayer(nn.Module):
    r"""
    This corresponds to the expansion phase of each block in the original implementation.
    """

    config: EfficientNetConfig
    in_dim: int
    out_dim: int
    stride: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.expand_conv = nn.Conv(
            self.out_dim,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.expand_bn = nn.BatchNorm()
        self.expand_act = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Expand phase
        hidden_states = self.expand_conv(hidden_states)
        hidden_states = self.expand_bn(hidden_states, use_running_average=deterministic)
        hidden_states = self.expand_act(hidden_states)

        return hidden_states


class FlaxEfficientNetDepthwiseLayer(nn.Module):
    r"""
    This corresponds to the depthwise convolution phase of each block in the original implementation.
    """

    config: EfficientNetConfig
    in_dim: int
    stride: int
    kernel_size: int
    adjust_padding: bool
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        conv_pad = "VALID" if self.stride == 2 else "SAME"
        padding = correct_pad(self.kernel_size, adjust=self.adjust_padding)

        self.depthwise_conv_pad = partial(
            jnp.pad,
            pad_width=(
                (0, 0),
                (padding[0], padding[1]),
                (padding[2], padding[3]),
                (0, 0),
            ),
            mode="constant",
            constant_values=0,
        )

        self.depthwise_conv = nn.Conv(
            self.in_dim,
            kernel_size=[self.kernel_size, self.kernel_size],
            strides=self.stride,
            padding=conv_pad,
            kernel_dilation=1,
            feature_group_count=self.in_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.depthwise_norm = nn.BatchNorm(
            epsilon=self.config.batch_norm_eps,
            momentum=1 - self.config.batch_norm_momentum,
            dtype=self.dtype,
        )
        self.depthwise_act = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Depthwise convolution
        if self.stride == 2:
            hidden_states = self.depthwise_conv_pad(
                hidden_states,
            )

        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_norm(hidden_states, use_running_average=deterministic)
        hidden_states = self.depthwise_act(hidden_states)

        return hidden_states


class FlaxEfficientNetSqueezeExciteLayer(nn.Module):
    r"""
    This corresponds to the Squeeze and Excitement phase of each block in the original implementation.
    """

    config: EfficientNetConfig
    in_dim: int
    expand_dim: int
    do_expand: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dim = self.expand_dim if self.do_expand else self.in_dim
        self.dim_se = max(1, int(self.in_dim * self.config.squeeze_expansion_ratio))

        self.squeeze = partial(
            nn.avg_pool,
            padding=((0, 0), (0, 0)),
        )  # Adaptation of AdaptiveAvgPool2d (for output size 1) for Flax
        self.reduce = nn.Conv(
            self.dim_se,
            kernel_size=[1, 1],
            padding="SAME",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.expand = nn.Conv(
            self.dim,
            kernel_size=[1, 1],
            padding="SAME",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.act_reduce = ACT2FN[self.config.hidden_act]
        self.act_expand = nn.sigmoid

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        inputs = hidden_states

        hidden_states = self.squeeze(
            hidden_states,
            window_shape=(hidden_states.shape[1], hidden_states.shape[2]),
            strides=(hidden_states.shape[1], hidden_states.shape[2]),
        )

        hidden_states = self.reduce(hidden_states)
        hidden_states = self.act_reduce(hidden_states)

        hidden_states = self.expand(hidden_states)
        hidden_states = self.act_expand(hidden_states)
        hidden_states = jnp.multiply(inputs, hidden_states)

        return hidden_states


class FlaxEfficientNetFinalBlockLayer(nn.Module):
    r"""
    This corresponds to the final phase of each block in the original implementation.
    """

    config: EfficientNetConfig
    in_dim: int
    out_dim: int
    stride: int
    drop_rate: float
    id_skip: bool
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.apply_dropout = self.stride == 1 and not self.id_skip
        self.project_conv = nn.Conv(
            self.out_dim,
            kernel_size=[1, 1],
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.project_bn = nn.BatchNorm(
            epsilon=self.config.batch_norm_eps,
            momentum=1 - self.config.batch_norm_momentum,
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.drop_rate)

    def __call__(self, embeddings: jnp.ndarray, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_states = self.project_conv(hidden_states)
        hidden_states = self.project_bn(hidden_states, use_running_average=deterministic)

        if self.apply_dropout:
            hidden_states = self.dropout(hidden_states, deterministic=deterministic)
            hidden_states = hidden_states + embeddings

        return hidden_states


class FlaxEfficientNetBlock(nn.Module):
    r"""
    This corresponds to the expansion and depthwise convolution phase of each block in the original implementation.

    Args:
        config ([`EfficientNetConfig`]):
            Model configuration class.
        in_dim (`int`):
            Number of input channels.
        out_dim (`int`):
            Number of output channels.
        stride (`int`):
            Stride size to be used in convolution layers.
        expand_ratio (`int`):
            Expand ratio to set the output dimensions for the expansion and squeeze-excite layers.
        kernel_size (`int`):
            Kernel size for the depthwise convolution layer.
        drop_rate (`float`):
            Dropout rate to be used in the final phase of each block.
        id_skip (`bool`):
            Whether to apply dropout and sum the final hidden states with the input embeddings during the final phase
            of each block. Set to `True` for the first block of each stage.
        adjust_padding (`bool`):
            Whether to apply padding to only right and bottom side of the input kernel before the depthwise convolution
            operation, set to `True` for inputs with odd input sizes.
        dtype (`jax.numpy.dtype`):
            The dtype of the computation (default: `jax.numpy.float32`).
    """

    config: EfficientNetConfig
    in_dim: int
    out_dim: int
    stride: int
    expand_ratio: int
    kernel_size: int
    drop_rate: float
    id_skip: bool
    adjust_padding: bool
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.expand = True if self.expand_ratio != 1 else False
        expand_in_dim = self.in_dim * self.expand_ratio

        if self.expand:
            self.expansion = FlaxEfficientNetExpansionLayer(
                config=self.config,
                in_dim=self.in_dim,
                out_dim=expand_in_dim,
                stride=self.stride,
                dtype=self.dtype,
            )

        self.depthwise_conv = FlaxEfficientNetDepthwiseLayer(
            config=self.config,
            in_dim=expand_in_dim if self.expand else self.in_dim,
            stride=self.stride,
            kernel_size=self.kernel_size,
            adjust_padding=self.adjust_padding,
            dtype=self.dtype,
        )
        self.squeeze_excite = FlaxEfficientNetSqueezeExciteLayer(
            config=self.config,
            in_dim=self.in_dim,
            expand_dim=expand_in_dim,
            do_expand=self.expand,
            dtype=self.dtype,
        )
        self.projection = FlaxEfficientNetFinalBlockLayer(
            config=self.config,
            in_dim=expand_in_dim if self.expand else self.in_dim,
            out_dim=self.out_dim,
            stride=self.stride,
            drop_rate=self.drop_rate,
            id_skip=self.id_skip,
            dtype=self.dtype,
        )

    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        embeddings = hidden_states
        # Expansion and depthwise convolution phase
        if self.expand_ratio != 1:
            hidden_states = self.expansion(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)

        # Squeeze and excite phase
        hidden_states = self.squeeze_excite(hidden_states)
        hidden_states = self.projection(embeddings, hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxEfficientNetEncoderCollection(nn.Module):
    r"""
    This corresponds to the collection of EfficientNet blocks.

    Args:
        config ([`EfficientNetConfig`]):
            Model configuration class.
        dtype (`jax.numpy.dtype`):
            The dtype of the computation (default: `jax.numpy.float32`).
    """

    config: EfficientNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.depth_coefficient = self.config.depth_coefficient

        def round_repeats(repeats):
            # Round number of block repeats based on depth multiplier.
            return int(math.ceil(self.depth_coefficient * repeats))

        num_base_blocks = len(self.config.in_channels)
        num_blocks = sum(round_repeats(n) for n in self.config.num_block_repeats)

        curr_block_num = 0
        blocks = []
        for i in range(num_base_blocks):
            in_dim = round_filters(self.config, self.config.in_channels[i])
            out_dim = round_filters(self.config, self.config.out_channels[i])
            stride = self.config.strides[i]
            kernel_size = self.config.kernel_sizes[i]
            expand_ratio = self.config.expand_ratios[i]

            for j in range(round_repeats(self.config.num_block_repeats[i])):
                id_skip = True if j == 0 else False
                stride = 1 if j > 0 else stride
                in_dim = out_dim if j > 0 else in_dim
                adjust_padding = False if curr_block_num in self.config.depthwise_padding else True
                drop_rate = self.config.drop_connect_rate * curr_block_num / num_blocks

                block = FlaxEfficientNetBlock(
                    config=self.config,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    stride=stride,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    drop_rate=drop_rate,
                    id_skip=id_skip,
                    adjust_padding=adjust_padding,
                    dtype=self.dtype,
                    name=str(curr_block_num),
                )
                blocks.append(block)
                curr_block_num += 1

        self.blocks = blocks

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        all_hidden_states = (hidden_states.transpose(0, 3, 1, 2),) if output_hidden_states else None

        for block in self.blocks:
            hidden_states = block(hidden_states, deterministic=deterministic)
            if output_hidden_states:
                all_hidden_states += (hidden_states.transpose(0, 3, 1, 2),)

        return hidden_states, all_hidden_states


class FlaxEfficientNetEncoder(nn.Module):
    r"""
    Forward propogates the embeddings through each EfficientNet block.

    Args:
        config ([`EfficientNetConfig`]):
            Model configuration class.

        dtype (`jax.numpy.dtype`):
            The dtype of the computation (default: `jax.numpy.float32`).
    """

    config: EfficientNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.blocks = FlaxEfficientNetEncoderCollection(self.config)
        self.top_conv = nn.Conv(
            round_filters(self.config, 1280),
            kernel_size=[1, 1],
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.top_bn = nn.BatchNorm(
            epsilon=self.config.batch_norm_eps,
            momentum=1 - self.config.batch_norm_momentum,
            dtype=self.dtype,
        )
        self.top_activation = ACT2FN[self.config.hidden_act]

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        deterministic: bool = True,
    ) -> FlaxBaseModelOutputWithNoAttention:
        hidden_states, all_hidden_states = self.blocks(
            hidden_states=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = self.top_conv(hidden_states)
        hidden_states = self.top_bn(hidden_states, use_running_average=deterministic)
        hidden_states = self.top_activation(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return FlaxBaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class FlaxEfficientNetPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EfficientNetConfig
    base_model_prefix = "efficientnet"
    main_input_name = "pixel_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: EfficientNetConfig,
        input_shape=None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, pixel_values, return_dict=False)

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(EFFICIENTNET_INPUTS_DOCSTRING)
    def __call__(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
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
            {
                "params": params["params"] if params is not None else self.params["params"],
                "batch_stats": params["batch_stats"] if params is not None else self.params["batch_stats"],
            },
            pixel_values=jnp.array(pixel_values, dtype=jnp.float32),
            deterministic=not train,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
            mutable=["batch_stats"] if train else False,
        )


class FlaxEfficientNetModule(nn.Module):
    config: EfficientNetConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.embeddings = FlaxEfficientNetEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxEfficientNetEncoder(self.config, dtype=self.dtype)

        if self.config.pooling_type == "mean":
            self.pooler = nn.avg_pool  # NOTE : ceil_mode=True
        elif self.config.pooling_type == "max":
            self.pooler = nn.max_pool  # NOTE : ceil_mode=True
        else:
            raise ValueError(f"config.pooling must be one of ['mean', 'max'] got {self.config.pooling}")

    def __call__(
        self,
        pixel_values: jnp.ndarray = None,
        deterministic: bool = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FlaxBaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        # Apply pooling
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(
            last_hidden_state,
            window_shape=(last_hidden_state.shape[1], last_hidden_state.shape[2]),
        )
        # (batch_size, 1280, 1 , 1) -> (batch_size, 1280)
        pooled_output = pooled_output[:, 0, 0]

        last_hidden_state = last_hidden_state.transpose(0, 3, 1, 2)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return FlaxBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    "The bare EfficientNet model outputting raw features without any specific head on top.",
    EFFICIENTNET_START_DOCSTRING,
)
class FlaxEfficientNetModel(FlaxEfficientNetPreTrainedModel):
    module_class = FlaxEfficientNetModule


FLAX_VISION_MODEL_DOCSTRING = """
    Returns:

    Examples:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxEfficientNetModel
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
    >>> model = FlaxEfficientNetModel.from_pretrained("google/efficientnet-b7")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

overwrite_call_docstring(FlaxEfficientNetModel, FLAX_VISION_MODEL_DOCSTRING)
append_replace_return_docstrings(
    FlaxEfficientNetModel,
    output_type=FlaxBaseModelOutputWithPooling,
    config_class=EfficientNetConfig,
)


class FlaxEfficientNetForImageClassificationModule(nn.Module):
    config: EfficientNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.efficientnet = FlaxEfficientNetModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.dropout_rate)

        if self.config.num_labels > 0:
            self.classifier = nn.Dense(
                self.config.num_labels,
                dtype=self.dtype,
            )
        else:
            self.classifier = Identity()

        self.classifier_act = nn.softmax

    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.efficientnet(
            pixel_values,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)

        logits = self.classifier(pooled_output)

        logits = self.classifier_act(logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return FlaxImageClassifierOutputWithNoAttention(logits=logits, hidden_states=outputs.hidden_states)


@add_start_docstrings(
    """
    EfficientNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g.
    for ImageNet.
    """,
    EFFICIENTNET_START_DOCSTRING,
)
class FlaxEfficientNetForImageClassification(FlaxEfficientNetPreTrainedModel):
    module_class = FlaxEfficientNetForImageClassificationModule


FLAX_VISION_CLASSIF_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxEfficientNetForImageClassification
    >>> from PIL import Image
    >>> import jax
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
    >>> model = FlaxEfficientNetForImageClassification.from_pretrained("google/efficientnet-b7")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
    >>> print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
    ```
"""

overwrite_call_docstring(FlaxEfficientNetForImageClassification, FLAX_VISION_CLASSIF_DOCSTRING)
append_replace_return_docstrings(
    FlaxEfficientNetForImageClassification,
    output_type=FlaxImageClassifierOutputWithNoAttention,
    config_class=EfficientNetConfig,
)
