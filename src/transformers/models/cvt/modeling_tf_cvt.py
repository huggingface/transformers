""" TF 2.0 Cvt model."""


import collections.abc
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import tensorflow as tf

from ...modeling_outputs import ModelOutput
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_cvt import CvtConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "CvtConfig"


@dataclass
class TFBaseModelOutputWithCLSToken(ModelOutput):
    """
    Base class for model's outputs.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cls_token_value (`tf.Tensor` of shape `(batch_size, 1, num_channels)`):
            Classification token at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, num_channels, height, width)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
    """

    last_hidden_state: tf.Tensor = None
    cls_token_value: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None


class TFCvtDropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_prob: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x: tf.Tensor, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class TFCvtEmbeddings(tf.keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        patch_size: int,
        embed_dim: int,
        stride: int,
        padding: int,
        dropout_rate: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.convolution_embeddings = TFCvtConvEmbeddings(
            config,
            patch_size=patch_size,
            embed_dim=embed_dim,
            stride=stride,
            padding=padding,
            name="convolution_embeddings",
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        hidden_state = self.convolution_embeddings(pixel_values)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class TFCvtConvEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: CvtConfig, patch_size: int, embed_dim: int, stride: int, padding: int, **kwargs):
        super().__init__(**kwargs)
        self.pad_value = padding
        self.patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.proj = tf.keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=stride,
            padding="valid",
            data_format="channels_last",
            kernel_initializer=get_initializer(config.initializer_range),
            name="projection",
        )
        # Using the same default epsilon & momentum as PyTorch
        self.Normalization = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="normalization")

    def convolution(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # Custom padding to match the model implementation in PyTorch
        height_pad = width_pad = (self.pad_value, self.pad_value)
        hidden_state = tf.pad(hidden_state, [(0, 0), height_pad, width_pad, (0, 0)])
        hidden_state = self.proj(hidden_state)
        return hidden_state

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values["pixel_values"]

        # When running on CPU, `tf.keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        pixel_values = self.convolution(pixel_values)
        # rearrange "b h w c" -> b (h w) c"
        batch_size, height, width, num_channels = shape_list(pixel_values)
        hidden_size = height * width
        pixel_values = tf.reshape(pixel_values, shape=(batch_size, hidden_size, num_channels))
        if self.Normalization:
            pixel_values = self.Normalization(pixel_values)
        # rearrange "b (h w) c" -> b c h w"
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 1))
        pixel_values = tf.reshape(pixel_values, shape=(batch_size, num_channels, height, width))
        return pixel_values


class TFCvtSelfAttentionConvProjection(tf.keras.layers.Layer):
    def __init__(self, config: CvtConfig, kernel_size: int, stride: int, padding: int, **kwargs):
        super().__init__(**kwargs)
        self.pad_value = padding
        self.conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            kernel_initializer=get_initializer(config.initializer_range),
            padding="valid",
            strides=stride,
            use_bias=False,
            name="convolution",
        )
        # Using the same default epsilon & momentum as PyTorch
        self.Normalization = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name="normalization")

    def convolution(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # Custom padding to match the model implementation in PyTorch
        height_pad = width_pad = (self.pad_value, self.pad_value)
        hidden_state = tf.pad(hidden_state, [(0, 0), height_pad, width_pad, (0, 0)])
        hidden_state = self.conv(hidden_state)
        return hidden_state

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.Normalization(hidden_state, training=training)
        return hidden_state


class TFCvtSelfAttentionLinearProjection(tf.keras.layers.Layer):
    def call(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # rearrange " b c h w -> b (h w) c"
        batch_size, num_channels, height, width = shape_list(hidden_state)
        hidden_size = height * width
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, num_channels, hidden_size))
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 1))
        return hidden_state


class TFCvtSelfAttentionProjection(tf.keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        kernel_size: int,
        stride: int,
        padding: int,
        projection_method: str = "dw_bn",
        **kwargs
    ):
        super().__init__(**kwargs)
        if projection_method == "dw_bn":
            self.convolution_projection = TFCvtSelfAttentionConvProjection(
                config, kernel_size, stride, padding, name="convolution_projection"
            )
        self.linear_projection = TFCvtSelfAttentionLinearProjection()

    def call(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # When running on CPU, `tf.keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        hidden_state = tf.transpose(hidden_state, perm=(0, 3, 2, 1))
        hidden_state = self.convolution_projection(hidden_state)
        hidden_state = tf.transpose(hidden_state, perm=(0, 3, 2, 1))
        hidden_state = self.linear_projection(hidden_state)
        return hidden_state


class TFCvtSelfAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        attention_drop_rate: float,
        with_cls_token: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scale = embed_dim**-0.5
        self.with_cls_token = with_cls_token
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.convolution_projection_query = TFCvtSelfAttentionProjection(
            config,
            kernel_size,
            stride_q,
            padding_q,
            projection_method="linear" if qkv_projection_method == "avg" else qkv_projection_method,
            name="convolution_projection_query",
        )
        self.convolution_projection_key = TFCvtSelfAttentionProjection(
            config,
            kernel_size,
            stride_kv,
            padding_kv,
            projection_method=qkv_projection_method,
            name="convolution_projection_key",
        )
        self.convolution_projection_value = TFCvtSelfAttentionProjection(
            config,
            kernel_size,
            stride_kv,
            padding_kv,
            projection_method=qkv_projection_method,
            name="convolution_projection_value",
        )

        self.projection_query = tf.keras.layers.Dense(
            units=self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=True,
            bias_initializer="zeros",
            name="projection_query",
        )
        self.projection_key = tf.keras.layers.Dense(
            units=self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=True,
            bias_initializer="zeros",
            name="projection_key",
        )
        self.projection_value = tf.keras.layers.Dense(
            units=self.embed_dim,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=True,
            bias_initializer="zeros",
            name="projection_value",
        )
        self.dropout = tf.keras.layers.Dropout(attention_drop_rate)

    def rearrange_for_multi_head_attention(self, hidden_state: tf.Tensor) -> tf.Tensor:
        batch_size, hidden_size, _ = shape_list(hidden_state)
        head_dim = self.embed_dim // self.num_heads
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, self.num_heads, head_dim))
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 1, 3))
        return hidden_state

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool) -> tf.Tensor:
        if self.with_cls_token:
            cls_token, hidden_state = tf.split(hidden_state, [1, height * width], 1)

        # rearrange "b (h w) c -> b c h w"
        batch_size, hidden_size, num_channels = shape_list(hidden_state)
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 1))
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, num_channels, height, width))

        key = self.convolution_projection_key(hidden_state)
        query = self.convolution_projection_query(hidden_state)
        value = self.convolution_projection_value(hidden_state)

        if self.with_cls_token:
            query = tf.concat((cls_token, query), axis=1)
            key = tf.concat((cls_token, key), axis=1)
            value = tf.concat((cls_token, value), axis=1)

        head_dim = self.embed_dim // self.num_heads
        query = self.rearrange_for_multi_head_attention(self.projection_query(query))
        key = self.rearrange_for_multi_head_attention(self.projection_key(key))
        value = self.rearrange_for_multi_head_attention(self.projection_value(value))

        attention_score = tf.matmul(query, key, transpose_b=True) * self.scale
        attention_probs = stable_softmax(logits=attention_score, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        context = tf.matmul(attention_probs, value)

        # rearrange "b h t d -> b t (h d)"
        _, _, hidden_size, _ = shape_list(context)
        context = tf.transpose(context, perm=(0, 2, 1, 3))
        context = tf.reshape(context, (batch_size, hidden_size, self.num_heads * head_dim))
        return context


class TFCvtSelfOutput(tf.keras.layers.Layer):
    """Output of the Attention layer."""

    def __init__(self, config: CvtConfig, embed_dim: int, drop_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = self.dense(inputs=hidden_state)
        hidden_state = self.dropout(inputs=hidden_state, training=training)
        return hidden_state


class TFCvtAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        attention_drop_rate: float,
        drop_rate: float,
        with_cls_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention = TFCvtSelfAttention(
            config,
            num_heads,
            embed_dim,
            kernel_size,
            stride_q,
            stride_kv,
            padding_q,
            padding_kv,
            qkv_projection_method,
            attention_drop_rate,
            with_cls_token,
            name="attention",
        )
        self.dense_output = TFCvtSelfOutput(config, embed_dim, drop_rate, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool):
        self_output = self.attention(hidden_state, height, width, training)
        attention_output = self.dense_output(self_output, training)
        return attention_output


class TFCvtIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: CvtConfig, embed_dim: int, mlp_ratio: int, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            units=int(embed_dim * mlp_ratio),
            kernel_initializer=get_initializer(config.initializer_range),
            activation="gelu",
            name="dense",
        )

    def call(self, hidden_state: tf.Tensor) -> tf.Tensor:
        hidden_state = self.dense(hidden_state)
        return hidden_state


class TFCvtOutput(tf.keras.layers.Layer):
    def __init__(self, config: CvtConfig, embed_dim: int, drop_rate: int, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, hidden_state: tf.Tensor, input_tensor: tf.Tensor) -> tf.Tensor:
        hidden_state = self.dense(inputs=hidden_state)
        hidden_state = self.dropout(inputs=hidden_state)
        hidden_state = hidden_state + input_tensor
        return hidden_state


class TFCvtLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        attention_drop_rate: float,
        drop_rate: float,
        mlp_ratio: float,
        drop_path_rate: float,
        with_cls_token: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.attention = TFCvtAttention(
            config,
            num_heads,
            embed_dim,
            kernel_size,
            stride_q,
            stride_kv,
            padding_q,
            padding_kv,
            qkv_projection_method,
            attention_drop_rate,
            drop_rate,
            with_cls_token,
            name="attention",
        )
        self.intermediate = TFCvtIntermediate(config, embed_dim, mlp_ratio, name="intermediate")
        self.dense_output = TFCvtOutput(config, embed_dim, drop_rate, name="output")
        # Using `layers.Activation` instead of `tf.identity` to better control `training` behaviour.
        self.drop_path = (
            TFCvtDropPath(drop_path_rate, name="drop_path")
            if drop_path_rate > 0.0
            else tf.keras.layers.Activation("linear", name="drop_path")
        )
        self.layernorm_before = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_before")
        self.layernorm_after = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layernorm_after")

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = False) -> tf.Tensor:
        self_attention_output = self.attention(
            # in Cvt, layernorm is applied before self-attention
            self.layernorm_before(hidden_state),
            width,
            height,
        )
        attention_output = self_attention_output
        attention_output = self.drop_path(attention_output)

        # first residual connection
        hidden_state = attention_output + hidden_state

        # in Cvt, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_state)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.dense_output(layer_output, hidden_state)
        layer_output = self.drop_path(layer_output, training=training)
        return layer_output


class TFCvtStage(tf.keras.layers.Layer):
    def __init__(self, config: CvtConfig, stage: int, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.stage = stage
        if self.config.cls_token[self.stage]:
            self.cls_token = self.add_weight(
                shape=(1, 1, self.config.embed_dim[-1]),
                initializer="zeros",
                trainable=True,
                name="cvt.encoder.stages.2.cls_token",
            )
        self.embedding = TFCvtEmbeddings(
            self.config,
            patch_size=config.patch_sizes[self.stage],
            stride=config.patch_stride[self.stage],
            embed_dim=config.embed_dim[self.stage],
            padding=config.patch_padding[self.stage],
            dropout_rate=config.drop_rate[self.stage],
            name="embedding",
        )

        drop_path_rates = tf.linspace(0.0, config.drop_path_rate[self.stage], config.depth[stage])
        drop_path_rates = [x.numpy().item() for x in drop_path_rates]
        self.layers = [
            TFCvtLayer(
                config,
                num_heads=config.num_heads[self.stage],
                embed_dim=config.embed_dim[self.stage],
                kernel_size=config.kernel_qkv[self.stage],
                stride_q=config.stride_q[self.stage],
                stride_kv=config.stride_kv[self.stage],
                padding_q=config.padding_q[self.stage],
                padding_kv=config.padding_kv[self.stage],
                qkv_projection_method=config.qkv_projection_method[self.stage],
                attention_drop_rate=config.attention_drop_rate[self.stage],
                drop_rate=config.drop_rate[self.stage],
                mlp_ratio=config.mlp_ratio[self.stage],
                drop_path_rate=drop_path_rates[self.stage],
                with_cls_token=config.cls_token[self.stage],
                name=f"layers.{j}",
            )
            for j in range(config.depth[self.stage])
        ]

    def call(self, hidden_state: tf.Tensor):
        cls_token = None
        hidden_state = self.embedding(hidden_state)

        batch_size, num_channels, height, width = shape_list(hidden_state)
        # rearrange b c h w -> b (h w) c"
        hidden_size = height * width
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, num_channels, hidden_size))
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 1))

        if self.config.cls_token[self.stage]:
            cls_token = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
            hidden_state = tf.concat((cls_token, hidden_state), axis=1)

        for layer in self.layers:
            layer_outputs = layer(hidden_state, height, width)
            hidden_state = layer_outputs

        if self.config.cls_token[self.stage]:
            cls_token, hidden_state = tf.split(hidden_state, [1, height * width], 1)

        # rearrange -> b (h w) c" -> b c h w
        hidden_state = tf.transpose(hidden_state, (0, 2, 1))
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, num_channels, height, width))
        return hidden_state, cls_token


class TFCvtEncoder(tf.keras.layers.Layer):
    config_class = CvtConfig

    def __init__(self, config: CvtConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.stages = [
            TFCvtStage(config, stage_idx, name=f"stages.{stage_idx}") for stage_idx in range(len(config.depth))
        ]

    def call(
        self,
        pixel_values: TFModelInputType,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        hidden_state = pixel_values

        cls_token = None
        for _, (stage_module) in enumerate(self.stages):
            hidden_state, cls_token = stage_module(hidden_state)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, cls_token, all_hidden_states] if v is not None)

        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=hidden_state,
            cls_token_value=cls_token,
            hidden_states=all_hidden_states,
        )


@keras_serializable
class TFCvtMainLayer(tf.keras.layers.Layer):
    config_class = CvtConfig

    def __init__(self, config: CvtConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.encoder = TFCvtEncoder(config, name="encoder")

    @unpack_inputs
    def call(
        self,
        pixel_values: Optional[TFModelInputType] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # pixel_values = tf.transpose(pixel_values, perm=(0, 3, 1, 2))
        # tried reshaping to to `NHWC` directly in main layer and using this format
        # throughout the model, but even though I get the same predictions as torch 
        # CVT model, our sequence_output have an absolute difference > 100
        
        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            # encoder_outputs -> [last_hidden_state, cls_token, all_hidden_states]
            return (sequence_output,) + encoder_outputs[1:]

        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=sequence_output,
            cls_token_value=encoder_outputs.cls_token_value,
            hidden_states=encoder_outputs.hidden_states,
        )


class TFCvtPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CvtConfig
    base_model_prefix = "cvt"
    main_input_name = "pixel_values"

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(3, self.config.num_channels, self.config.image_size, self.config.image_size), dtype=tf.float32
        )
        return {"pixel_values": tf.constant(VISION_DUMMY_INPUTS)}

    @tf.function(
        input_signature=[
            {
                "pixel_values": tf.TensorSpec((None, None, None, None), tf.float32, name="pixel_values"),
            }
        ]
    )
    def serving(self, inputs):
        """
        Method used for serving the model.

        Args:
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        output = self.call(inputs)
        return self.serving_output(output)


class TFCvtModel(TFCvtPreTrainedModel):
    def __init__(self, config: CvtConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.cvt = TFCvtMainLayer(config, name="cvt")

    @unpack_inputs
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        outputs = self.cvt(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        if not return_dict:
            # outputs -> [last_hidden_sate, cls_token_value, hidden_states]
            return (outputs[0],) + outputs[1:]

        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=outputs.last_hidden_state,
            cls_token_value=outputs.cls_token_value,
            hidden_states=outputs.hidden_states,
        )

    def serving_output(self, output: TFBaseModelOutputWithCLSToken) -> TFBaseModelOutputWithCLSToken:
        return TFBaseModelOutputWithCLSToken(
            last_hidden_state=output.last_hidden_state,
            cls_token_value=output.cls_token_value,
            hidden_states=output.hidden_states,
        )


class TFCvtForImageClassification(TFCvtPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: CvtConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.cvt = TFCvtMainLayer(config, name="cvt")
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

        # Classifier head
        self.classifier = tf.keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            use_bias=True,
            bias_initializer="zeros",
            name="classifier",
        )

    @unpack_inputs
    def call(
        self,
        pixel_values: Optional[tf.Tensor] = None,
        labels: Optional[tf.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFImageClassifierOutputWithNoAttention, Tuple[tf.Tensor]]:

        outputs = self.cvt(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs[0]
        cls_token = outputs[1]
        if self.config.cls_token[-1]:
            sequence_output = self.LayerNorm(cls_token)
        else:
            # rearrange "b c h w -> b (h w) c"
            batch_size, num_channels, height, width = shape_list(sequence_output)
            sequence_output = tf.reshape(sequence_output, shape=(batch_size, num_channels, height * width))
            sequence_output = tf.transpose(sequence_output, perm=(0, 2, 1))
            sequence_output = self.LayerNorm(sequence_output)

        sequence_output_mean = tf.reduce_mean(sequence_output, axis=1)
        logits = self.classifier(sequence_output_mean)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            # outputs -> [last_hidden_sate, cls_token_value, hidden_states]
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states
        )

    def serving_output(self, output: TFImageClassifierOutputWithNoAttention) -> TFImageClassifierOutputWithNoAttention:
        return TFImageClassifierOutputWithNoAttention(
            logits=output.logits,
            hidden_states=output.hidden_states
        )
