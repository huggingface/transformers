from typing import Optional, List, Tuple, Callable, Union, Sequence

import jax.random as random
import jax.numpy as jnp
import jax.lax

import flax.linen as nn
from flax.core import FrozenDict

from ...file_utils import ModelOutput, add_start_docstrings
from ...modeling_flax_outputs import FlaxSequenceClassifierOutput
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    overwrite_call_docstring,
    append_replace_return_docstrings,
)
from . import ConvNextConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class FlaxConvNextEncoderOutput(ModelOutput):
    """
    Class for [`FlaxConvNextEncoder`]'s outputs, with potential hidden states (feature maps).

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the model.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
    """

    last_hidden_state: jnp.float32 = None
    hidden_states: Optional[Tuple[jnp.float32]] = None


class FlaxConvNextModelOutput(ModelOutput):
    """
    Class for [`FlaxConvNextModel`]'s outputs, with potential hidden states (feature maps).

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, config.dim[-1])`):
            Global average pooling of the last feature map followed by a layernorm.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
    """

    last_hidden_state: jnp.float32 = None
    pooler_output: Optional[jnp.float32] = None
    hidden_states: Optional[Tuple[jnp.float32]] = None


class FlaxConvNextClassifierOutput(ModelOutput):
    """
    Class for [`FlaxConvNextForImageClassification`]'s outputs, with potential hidden states (feature maps).

    Args:
        loss (`jnp.ndarray` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Unimplemented for Flax.
        logits (`jnp.ndarray` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
    """

    loss: Optional[jnp.float32] = None
    logits: jnp.float32 = None
    hidden_states: Optional[Tuple[jnp.float32]] = None


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Implementation referred from https://github.com/rwightman/pytorch-image-models
    """

    dropout_prob: float = 0.1

    @nn.compact
    def __call__(self, input, train=None):
        if not train or self.dropout_prob == 0.0:
            return input
        keep_prob = 1 - self.dropout_prob
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        rng = self.make_rng("droppath")
        random_tensor = keep_prob + random.uniform(rng, shape)
        random_tensor = jnp.floor(random_tensor)
        return jnp.divide(input, keep_prob) * random_tensor


class DepthwiseConv2D(nn.Module):
    kernel_shape: Union[int, Sequence[int]] = (1, 1)
    stride: Union[int, Sequence[int]] = (1, 1)
    padding: str or Sequence[Tuple[int, int]] = "SAME"
    channel_multiplier: int = 1
    use_bias: bool = True
    weights_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Optional[Callable] = nn.initializers.zeros

    @nn.compact
    def __call__(self, input):
        w = self.param(
            "kernel",
            self.weights_init,
            self.kernel_shape + (1, self.channel_multiplier * input.shape[-1]),
        )
        if self.use_bias:
            b = self.param(
                "bias", self.bias_init, (self.channel_multiplier * input.shape[-1],)
            )

        conv = jax.lax.conv_general_dilated(
            input,
            w,
            self.stride,
            self.padding,
            (1,) * len(self.kernel_shape),
            (1,) * len(self.kernel_shape),
            ("NHWC", "HWIO", "NHWC"),
            input.shape[-1],
        )
        if self.use_bias:
            bias = jnp.broadcast_to(b, conv.shape)
            return conv + bias
        else:
            return conv


class FlaxConvNextEmbeddings(nn.Module):
    config: ConvNextConfig

    def setup(self):
        self.patch_embeddings = nn.Conv(
            self.config.hidden_sizes[0],
            (self.config.patch_size, self.config.patch_size),
            strides=self.config.patch_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.layernorm = nn.LayerNorm(epsilon=1e-6)

    def __call__(self, pixel_values):
        embeddings = self.patch_embeddings(jnp.transpose(pixel_values, (0, 2, 3, 1)))
        embeddings = self.layernorm(embeddings)
        return embeddings


class FlaxConvNextLayer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.
    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back
    The authors used (2) as they find it slightly faster in PyTorch.
    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    config: ConvNextConfig
    dim: int = 256
    layer_scale_init_value: float = 1e-6
    droppath: float = 0.1

    def init_fn(self, key, shape, fill_value):
        return jnp.full(shape, fill_value)

    def setup(self):
        self.dwconv = DepthwiseConv2D(
            (7, 7),
            weights_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.norm = nn.LayerNorm()
        self.pwconv1 = nn.Dense(
            4 * self.dim,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.pwconv2 = nn.Dense(
            self.dim,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.layer_scale_parameter = (
            self.param("gamma", self.init_fn, (self.dim,), self.layer_scale_init_value)
            if self.layer_scale_init_value > 0
            else None
        )

        self.drop_path = DropPath(self.droppath)

        self.act = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states, train=False):
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x

        x = input + self.drop_path(x, train)
        return x


class FlaxConvNextStage(nn.Module):
    """ConvNeXT stage, consisting of an optional downsampling layer + multiple residual blocks.
    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
    """

    config: ConvNextConfig
    in_channels: int
    out_channels: int
    depth: int
    drop_path_rates: List[float] = None
    kernel_size: int = 2
    stride: int = 2
    depth: int = 2

    def setup(self):
        if self.in_channels != self.out_channels or self.stride > 1:
            self.downsampling_layer = [
                nn.LayerNorm(name="downsampling_layer.0"),
                nn.Conv(
                    self.out_channels,
                    (self.kernel_size, self.kernel_size),
                    self.stride,
                    kernel_init=jax.nn.initializers.normal(
                        self.config.initializer_range
                    ),
                    name="downsampling_layer.1",
                ),
            ]
        else:
            self.downsampling_layer = []

        self.layers = [
            FlaxConvNextLayer(
                config=self.config,
                dim=self.out_channels,
                name=f"layers.{j}",
                droppath=self.drop_path_rates[j],
            )
            for j in range(self.depth)
        ]

    def __call__(self, hidden_states, train=False):

        for layer in self.downsampling_layer:
            hidden_states = layer(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, train)
        return hidden_states


class FlaxConvNextEncoder(nn.Module):
    config: ConvNextConfig

    def setup(self):
        stages = []
        self.drop_path_rates = [
            x
            for x in jnp.linspace(
                0.0, self.config.drop_path_rate, sum(self.config.depths)
            )
        ]
        cur = 0
        prev_chs = self.config.hidden_sizes[0]

        for i in range(self.config.num_stages):
            out_chs = self.config.hidden_sizes[i]
            stage = FlaxConvNextStage(
                config=self.config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=self.config.depths[i],
                drop_path_rates=self.drop_path_rates[cur]
                or [0.0] * self.config.depths[i],
                name=f"stages.{i}",
            )
            stages.append(stage)
            cur += self.config.depths[i]
            prev_chs = out_chs

        self.stages = stages

    def __call__(
        self, hidden_states, output_hidden_states=False, return_dict=True, train=False
    ):
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(hidden_states, train=train)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return FlaxConvNextEncoderOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states
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
        self, config: ConvNextConfig, input_shape=None, seed: int = 0, **kwargs
    ):
        module = self.module_class(config=config, **kwargs)
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, 3)
        super().__init__(config, module, input_shape=input_shape, seed=seed)

    def init_weights(
        self,
        rng: jax.random.PRNGKey,
        input_shape: Tuple,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ) -> FrozenDict:
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)
        params_rng, dropout_rng = jax.random.split(rng)
        dropout_rng, droppath_rng = jax.random.split(dropout_rng)
        rngs = {"params": params_rng, "dropout": dropout_rng, "droppath": droppath_rng}
        return self.module.init(rngs, pixel_values, return_dict=False)["params"]

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
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            dropout_rng, droppath_rng = jax.random.split(dropout_rng)
            rngs["dropout"] = dropout_rng
            rngs["droppath"] = droppath_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


CONVNEXT_START_DOCSTRING = r"""
    This model is a Flax [nn.Module](https://flax.readthedocs.io/en/latest/flax.linen.html?highlight=Module#flax.linen.Module) subclass. Use it
    as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and
    behavior.
    Parameters:
        config ([`ConvNextConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
"""

CONVNEXT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~modeling_flax_outputs.FlaxModelOutput`] instead of a plain tuple.
    Returns:
"""


class FlaxConvNextModule(nn.Module):
    config: ConvNextConfig

    def setup(self):
        self.embeddings = FlaxConvNextEmbeddings(self.config)
        self.encoder = FlaxConvNextEncoder(self.config)
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps)

    def __call__(
        self,
        pixel_values=None,
        output_hidden_states=None,
        return_dict=None,
        train=False,
    ):
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            train=train,
        )
        last_hidden_state = encoder_outputs[0]

        pooled_output = self.layernorm(last_hidden_state.mean([1, 2]))

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return FlaxConvNextModelOutput(
            last_hidden_state=last_hidden_state,
            pooled_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    "The bare ConvNext Model transformer outputting raw hidden-states without any specific head on top.",
    CONVNEXT_START_DOCSTRING,
)
class FlaxConvNextModel(FlaxConvNextPreTrainedModel):
    module_class = FlaxConvNextModule


overwrite_call_docstring(FlaxConvNextModel, CONVNEXT_INPUTS_DOCSTRING)
append_replace_return_docstrings(
    FlaxConvNextModel, output_type=FlaxConvNextModelOutput, config_class=ConvNextConfig
)


CONVNEXT_FOR_IMAGE_CLASSIFICATION = r"""
    labels (`jnp.ndarray` of shape `(batch_size,)`, *optional*):
        Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:

    Examples:

    ```python
    >>> from transformers import ConvNextFeatureExtractor, FlaxConvNextForImageClassification
    >>> import jax.numpy as jnp
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-tiny-224")
    >>> model = FlaxConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

    >>> inputs = feature_extractor(images=image, return_tensors="tf")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
    >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
    ```"""


class FlaxConvNextForImageClassificationModule(nn.Module):
    config: ConvNextConfig

    def setup(self):
        self.num_labels = self.config.num_labels

        self.convnext = FlaxConvNextModel(self.config, name="convnext")

        # Classifier head
        self.classifier = nn.Dense(
            self.config.num_labels,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            bias_init=nn.zeros,
            name="classifier",
        )

    def __call__(
        self,
        pixel_values: jnp.ndarray = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: Optional[bool] = False,
        **kwargs,
    ) -> Union[FlaxSequenceClassifierOutput, Tuple[jnp.ndarray]]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.convnext(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            train=train,
        )
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)
        hidden_states = outputs.hidden_states if return_dict else outputs[-1]
        return FlaxConvNextClassifierOutput(
            loss=None, logits=logits, hidden_states=hidden_states
        )


class FlaxConvNextForImageClassification(FlaxConvNextPreTrainedModel):
    module_class = FlaxConvNextForImageClassificationModule


overwrite_call_docstring(
    FlaxConvNextForImageClassification, CONVNEXT_FOR_IMAGE_CLASSIFICATION
)
append_replace_return_docstrings(
    FlaxConvNextForImageClassification,
    output_type=FlaxSequenceClassifierOutput,
    config_class=ConvNextConfig,
)
