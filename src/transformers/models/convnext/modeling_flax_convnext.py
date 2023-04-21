# coding=utf-8
# Copyright 2023 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import random

from ...modeling_flax_outputs import (
    FlaxBaseModelOutputWithNoAttention,
    FlaxBaseModelOutputWithPoolingAndNoAttention,
    FlaxImageClassifierOutputWithNoAttention,
)
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_convnext import ConvNextConfig


# Flax compitable code initially copied from transformers.models.convnext.modeling_convnext.drop_path
class FlaxConvNextDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """

    drop_prob: Optional[float] = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        if self.drop_prob == 0.0 or deterministic:
            return hidden_states
        keep_prob = 1 - self.drop_prob
        shape = (hidden_states.shape[0],) + (1,) * (
            hidden_states.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + random.uniform(self.make_rng("dropout"), shape, dtype=self.dtype)
        random_tensor = jnp.floor(random_tensor)  # binarize
        output = jnp.divide(hidden_states, keep_prob) * random_tensor
        return output


# Copied from transformers.models.resnet.modeling_flax_resnet.Identity
class Identity(nn.Module):
    """Identity function."""

    @nn.compact
    def __call__(self, x, **kwargs):
        return x


class FlaxConvNextEmbeddings(nn.Module):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    config: ConvNextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.patch_embeddings = nn.Conv(
            self.config.hidden_sizes[0],
            kernel_size=(self.config.patch_size, self.config.patch_size),
            strides=(self.config.patch_size, self.config.patch_size),
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.layernorm = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)

    def __call__(self, pixel_values: jnp.ndarray) -> jnp.ndarray:
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings


class FlaxConvNextLayer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch. This Flax implementation is based on (1)

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
        dtype (`jax.numpy.dtype`): Data type of the computation. Default: jax.numpy.float32
    """

    config: ConvNextConfig
    dim: int
    drop_path: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dwconv = nn.Conv(
            self.dim,
            kernel_size=(7, 7),
            padding=((3, 3), (3, 3)),
            feature_group_count=self.dim,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )  # depthwise conv
        self.layernorm = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)
        self.pwconv1 = nn.Dense(4 * self.dim, dtype=self.dtype)  # pointwise/1x1 convs, implemented with linear layers
        self.act = ACT2FN[self.config.hidden_act]
        self.pwconv2 = nn.Dense(self.dim, dtype=self.dtype)

        layer_scale_init_value = self.config.layer_scale_init_value if self.config.layer_scale_init_value > 0 else 1
        layer_scale_parameter_init = jax.nn.initializers.constant(layer_scale_init_value)
        self.layer_scale_parameter = self.param("layer_scale_parameter", layer_scale_parameter_init, (self.dim))

        self.drop_path_func = (
            FlaxConvNextDropPath(self.drop_path, dtype=self.dtype) if self.drop_path > 0.0 else Identity()
        )

    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale_parameter * x

        x = input + self.drop_path_func(x, deterministic=deterministic)
        return x


class FlaxNextLayerCollection(nn.Module):
    config: ConvNextConfig
    out_channels: int
    depth: int = 2
    drop_path_rates: list = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        drop_path_rates = self.drop_path_rates or [0.0] * self.depth
        self.layers = [
            FlaxConvNextLayer(
                self.config, dim=self.out_channels, drop_path=drop_path_rates[j], name=str(j), dtype=self.dtype
            )
            for j in range(self.depth)
        ]

    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, deterministic=True)
        return hidden_states


class FlaxDownsamplingLayerCollection(nn.Module):
    config: ConvNextConfig
    out_channels: int
    kernel_size: int = 2
    stride: int = 2
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, name="0"),
            nn.Conv(
                self.out_channels,
                kernel_size=[self.kernel_size, self.kernel_size],
                strides=[self.stride, self.stride],
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                name="1",
            ),
        ]

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class FlaxConvNextStage(nn.Module):
    """ConvNeXT stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
        dtype (`jax.numpy.dtype`): Data type of the computation. Default: jax.numpy.float32
    """

    config: ConvNextConfig
    in_channels: int
    out_channels: int
    kernel_size: int = 2
    stride: int = 2
    depth: int = 2
    drop_path_rates: list = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.in_channels != self.out_channels or self.stride > 1:
            self.downsampling_layer = FlaxDownsamplingLayerCollection(
                config=self.config, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride
            )
        else:
            self.downsampling_layer = Identity()

        self.layers = FlaxNextLayerCollection(
            self.config,
            out_channels=self.out_channels,
            depth=self.depth,
            drop_path_rates=self.drop_path_rates,
            dtype=self.dtype,
        )

    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxStageCollection(nn.Module):
    config: ConvNextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # np.split requires list with entries indicating where along axis the array is split
        sections = (np.cumsum(self.config.depths) - self.config.depths)[1:]

        drop_path_rates = [
            x.tolist()
            for x in np.split(
                np.linspace(0, self.config.drop_path_rate, sum(self.config.depths), dtype=self.dtype),
                sections,
            )
        ]

        prev_channels = [
            self.config.hidden_sizes[0],
            *[self.config.hidden_sizes[i] for i in range(self.config.num_stages - 1)],
        ]

        self.stages = [
            FlaxConvNextStage(
                self.config,
                in_channels=prev_channels[i],
                out_channels=self.config.hidden_sizes[i],
                stride=2 if i > 0 else 1,
                depth=self.config.depths[i],
                drop_path_rates=drop_path_rates[i],
                name=str(i),
                dtype=self.dtype,
            )
            for i in range(self.config.num_stages)
        ]

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        output_hidden_states: Optional[bool] = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, Tuple]:
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.transpose(0, 3, 1, 2),)

            hidden_states = layer_module(hidden_states, deterministic=deterministic)

        return hidden_states, all_hidden_states


class FlaxConvNextEncoder(nn.Module):
    config: ConvNextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.stages = FlaxStageCollection(self.config, self.dtype)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        deterministic: bool = True,
    ) -> Union[Tuple, FlaxBaseModelOutputWithNoAttention]:
        hidden_states, all_hidden_states = self.stages(
            hidden_states, output_hidden_states, deterministic=deterministic
        )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.transpose(0, 3, 1, 2),)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return FlaxBaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class FlaxConvNextPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ConvNextConfig
    base_model_prefix = "convnext"
    main_input_name = "pixel_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: ConvNextConfig,
        input_shape=(1, 224, 224, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        rngs = {"params": rng}

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

    def __call__(
        self,
        pixel_values,
        params: dict = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # Handle any PRNG if needed
        rngs = {}

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=["batch_stats"] if train else False,  # Returing tuple with batch_stats only when train is True
        )


CONVNEXT_START_DOCSTRING = r"""

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
        config ([`ConvNextConfig`]): Model configuration class with all the parameters of the model.
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

CONVNEXT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`jax.numpy.float32` of shape `(batch_size, num_channels, height, width`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FlaxConvNextModule(nn.Module):
    config: ConvNextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embeddings = FlaxConvNextEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxConvNextEncoder(self.config, dtype=self.dtype)

        # final layernorm layer
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
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

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        last_hidden_state = encoder_outputs[0]

        # global average pooling, (N, C, H, W) -> (N, C)
        pooled_output = self.layernorm(last_hidden_state.mean([-3, -2]))

        last_hidden_state = last_hidden_state.transpose(0, 3, 1, 2)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return FlaxBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    "The bare ConvNext model outputting raw features without any specific head on top.",
    CONVNEXT_START_DOCSTRING,
)
class FlaxConvNextModel(FlaxConvNextPreTrainedModel):
    module_class = FlaxConvNextModule


FLAX_VISION_MODEL_DOCSTRING = """
    Returns:

    Examples:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxConvNextModel
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
    >>> model = FlaxConvNextModel.from_pretrained("facebook/convnext-tiny-224")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

overwrite_call_docstring(FlaxConvNextModel, FLAX_VISION_MODEL_DOCSTRING)
append_replace_return_docstrings(
    FlaxConvNextModel, output_type=FlaxBaseModelOutputWithPoolingAndNoAttention, config_class=ConvNextConfig
)


class FlaxConvNextForImageClassificationModule(nn.Module):
    config: ConvNextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.convnext = FlaxConvNextModule(config=self.config, dtype=self.dtype)

        # Classifier head
        if self.config.num_labels > 0:
            self.classifier = nn.Dense(
                self.config.num_labels,
                dtype=self.dtype,
            )
        else:
            self.classifier = Identity()

    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.convnext(
            pixel_values,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return FlaxImageClassifierOutputWithNoAttention(
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


@add_start_docstrings(
    """
    ConvNext Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    CONVNEXT_START_DOCSTRING,
)
class FlaxConvNextForImageClassification(FlaxConvNextPreTrainedModel):
    module_class = FlaxConvNextForImageClassificationModule


FLAX_VISION_CLASSIF_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoImageProcessor, FlaxConvNextForImageClassification
    >>> from PIL import Image
    >>> import jax
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
    >>> model = FlaxConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
    >>> print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
    ```
"""

overwrite_call_docstring(FlaxConvNextForImageClassification, FLAX_VISION_CLASSIF_DOCSTRING)
append_replace_return_docstrings(
    FlaxConvNextForImageClassification,
    output_type=FlaxImageClassifierOutputWithNoAttention,
    config_class=ConvNextConfig,
)
