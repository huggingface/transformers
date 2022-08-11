import collections
from typing import Dict
from typing import Optional
from typing import Union

import tensorflow as tf

from ...modeling_tf_utils import (
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import shape_list
from .configuration_layoutlmv3 import LayoutLMv3Config

_CONFIG_FOR_DOC = "LayoutLMv3Config"

_DUMMY_INPUT_IDS = [
    [7, 6, 1],
    [1, 2, 0],
]

_DUMMY_BBOX = [
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
]


class TFLayoutLMv3PatchEmbeddings(tf.keras.layers.Layer):
    """LayoutLMv3 image (patch) embeddings."""

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        patch_sizes = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )
        self.proj = tf.keras.layers.Conv2D(
            filters=config.hidden_size,
            kernel_size=patch_sizes,
            strides=patch_sizes,
            padding="valid",
            data_format="channels_last",
            use_bias=True,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="proj",
        )
        self.hidden_size = config.hidden_size
        self.num_patches = (config.input_size**2) // (patch_sizes[0] * patch_sizes[1])

    def call(self, pixel_values):
        # When running on CPU, `tf.keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        pixel_values = tf.transpose(pixel_values, perm=[0, 2, 3, 1])

        embeddings = self.proj(pixel_values)
        embeddings = tf.reshape(embeddings, (-1, self.num_patches, self.hidden_size))
        return embeddings


class TFLayoutLMv3TextEmbeddings(tf.keras.layers.Layer):
    """
    LayoutLMv3 text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        self.word_embeddings = tf.keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="word_embeddings",
        )
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="token_type_embeddings",
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.padding_token_index = config.pad_token_id
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="position_embeddings",
        )
        self.x_position_embeddings = tf.keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.coordinate_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="x_position_embeddings",
        )
        self.y_position_embeddings = tf.keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.coordinate_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="y_position_embeddings",
        )
        self.h_position_embeddings = tf.keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.shape_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="h_position_embeddings",
        )
        self.w_position_embeddings = tf.keras.layers.Embedding(
            config.max_2d_position_embeddings,
            config.shape_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="w_position_embeddings",
        )
        self.max_2d_positions = config.max_2d_position_embeddings

    def calculate_spatial_position_embeddings(self, bbox):
        try:
            left_position_ids = bbox[:, :, 0]
            upper_position_ids = bbox[:, :, 1]
            right_position_ids = bbox[:, :, 2]
            lower_position_ids = bbox[:, :, 3]
        except IndexError as exception:
            raise IndexError("Bounding box is not of shape (B, N, 4).") from exception

        try:
            left_position_embeddings = self.x_position_embeddings(left_position_ids)
            upper_position_embeddings = self.y_position_embeddings(upper_position_ids)
            right_position_embeddings = self.x_position_embeddings(right_position_ids)
            lower_position_embeddings = self.y_position_embeddings(lower_position_ids)
        except IndexError as exception:
            raise IndexError(
                f"The `bbox` coordinate values should be within 0-{self.max_2d_positions} range."
            ) from exception

        max_position_id = self.max_2d_positions - 1
        h_position_embeddings = self.h_position_embeddings(
            tf.clip_by_value(bbox[:, :, 3] - bbox[:, :, 1], 0, max_position_id)
        )
        w_position_embeddings = self.w_position_embeddings(
            tf.clip_by_value(bbox[:, :, 2] - bbox[:, :, 0], 0, max_position_id)
        )

        # LayoutLMv1 sums the spatial embeddings, but LayoutLMv3 concatenates them.
        spatial_position_embeddings = tf.concat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            axis=-1,
        )
        return spatial_position_embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embds):
        """
        We are provided embeddings directly. We cannot infer which are padded, so just generate sequential position ids.
        """
        input_shape = tf.shape(inputs_embds)
        sequence_length = input_shape[1]
        start_index = self.padding_token_index + 1
        end_index = self.padding_token_index + sequence_length + 1
        position_ids = tf.range(start_index, end_index, dtype=tf.int32)
        batch_size = input_shape[0]
        position_ids = tf.reshape(position_ids, (1, sequence_length))
        position_ids = tf.tile(position_ids, (batch_size, 1))
        return position_ids

    def create_position_ids_from_input_ids(self, input_ids):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_token_index + 1.
        """
        mask = tf.cast(tf.not_equal(input_ids, self.padding_token_index), tf.int32)
        position_ids = tf.cumsum(mask, axis=1) * mask
        position_ids = position_ids + self.padding_token_index
        return position_ids

    def create_position_ids(self, input_ids, inputs_embeds):
        if input_ids is None:
            return self.create_position_ids_from_inputs_embeds(inputs_embeds)
        else:
            return self.create_position_ids_from_input_ids(input_ids)

    def call(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        training: bool = False,
    ):
        if position_ids is None:
            position_ids = self.create_position_ids(input_ids, inputs_embeds)

        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=tf.int32)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)

        embeddings += spatial_position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings


class TFLayoutLMv3PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutLMv3Config
    base_model_prefix = "layoutlmv3"

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        size = self.config.input_size
        image_shape = (2, self.config.num_channels, size, size)
        pixel_values = tf.random.uniform(shape=image_shape, minval=-1, maxval=1)
        return {
            "input_ids": tf.constant(_DUMMY_INPUT_IDS, dtype=tf.int64),
            "bbox": tf.constant(_DUMMY_BBOX, dtype=tf.int64),
            "pixel_values": pixel_values,
        }

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                "bbox": tf.TensorSpec((None, None, 4), tf.int32, name="bbox"),
                "pixel_values": tf.TensorSpec((None, None, None, None), tf.float32, name="pixel_values"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
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


class TFLayoutLMv3SelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attention_score_normaliser = sqrt(self.attention_head_size)

        self.query = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            bias_initializer="zeros",
            name="value",
        )

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

    def transpose_for_scores(self, x: tf.Tensor):
        new_shape = (
            *shape_list(x)[:-1],
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = tf.reshape(x, new_shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # B, H, N, D

    def cogview_attention(self, attention_scores: tf.Tensor, alpha: Union[float, int] = 32):
        """
        https://arxiv.org/abs/2105.13290 Section 2.4 Stabilization of training: Precision Bottleneck Relaxation
        (PB-Relax). A replacement of the original tf.keras.layers.Softmax(axis=-1)(attention_scores). Seems the
        new attention_probs will result in a slower speed and a little bias. Can use
        tf.debugging.assert_near(standard_attention_probs, cogview_attention_probs, atol=1e-08) for comparison.
        The smaller atol (e.g., 1e-08), the better.
        """
        scaled_attention_scores = attention_scores / alpha
        max_value = tf.expand_dims(tf.reduce_max(scaled_attention_scores, axis=-1), axis=-1)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return tf.math.softmax(new_attention_scores, axis=-1)

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor],
        head_mask: Optional[tf.Tensor],
        output_attentions: bool,
        rel_pos: Optional[tf.Tensor] = None,
        rel_2d_pos: Optional[tf.Tensor] = None,
        training: bool = False,
    ):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        normalised_query_layer = query_layer / self.attention_score_normaliser
        transposed_key_layer = tf.transpose(key_layer, perm=[0, 1, 3, 2])  # B, H, D, N
        attention_scores = tf.matmul(normalised_query_layer, transposed_key_layer)

        if self.has_relative_attention_bias and self.has_spatial_attention_bias:
            attention_scores += (rel_pos + rel_2d_pos) / self.attention_score_normaliser
        elif self.has_relative_attention_bias:
            attention_scores += rel_pos / self.attention_score_normaliser

        if attention_mask is not None:
            # Apply the attention mask (is precomputed for all layers in TFLayoutLMv3Model call() function)
            attention_scores += attention_mask

        # Normalize the attention scores to probabilities.
        # Use the trick of CogView paper to stabilize training.
        attention_probs = self.cogview_attention(attention_scores)

        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to.
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])  # B, N, H, D
        context_layer = tf.reshape(context_layer, (*shape_list(context_layer)[:2], self.all_head_size))  # B, N, H * D

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
