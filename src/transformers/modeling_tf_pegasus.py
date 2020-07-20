""" TF 2.0 PEGASUS model. """


import copy
import itertools
import logging
import math

import tensorflow as tf

from .configuration_pegasus import PegasusConfig
from .file_utils import DUMMY_INPUTS, DUMMY_MASK, add_start_docstrings, add_start_docstrings_to_callable
from .modeling_tf_utils import (
    TFPreTrainedModel,
    TFSharedEmbeddings,
    cast_bool_to_primitive,
    keras_serializable,
    shape_list,
)
from .tokenization_utils import BatchEncoding


logger = logging.getLogger(__name__)

TF_PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "pegasus-large",
    # See all PEGASUS models at https://huggingface.co/models?filter=pegasus
]

####################################################
# TF 2.0 Models are constructed using Keras imperative API by sub-classing
# - tf.keras.layers.Layer for the layers and
# - TFPreTrainedModel for the models (it-self a sub-class of tf.keras.Model)
####################################################

# TODO: layers including, attention, decoding, embeddings, timing, transformer_block

####################################################
# TFPegasusPreTrainedModel is a sub-class of tf.keras.Model
# which take care of loading and saving pre-trained weights
# and various common utilities.
# Here you just need to specify a few (self-explanatory)
# pointers for your model.
####################################################
# TODO: object -> TFPreTrainedModel
class TFPegasusPreTrainedModel(object):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pre-trained models.
    """

    config_class = PegasusConfig
    base_model_prefix = "transformer"

    def __init__(self, vocab_size, hidden_size, filter_size, num_heads,
                 num_encoder_layers, num_decoder_layers, label_smoothing,
                 dropout):
        self._dtype = tf.float32
        self._embedding_layer = embedding.Embedding(vocab_size, hidden_size,
                                                    "weights", self._dtype)
        block_fn = lambda: transformer_block.TransformerBlock(
            hidden_size, filter_size, num_heads, dropout)
        self._encoder_layers = [block_fn() for _ in range(num_encoder_layers)]
        self._decoder_layers = [block_fn() for _ in range(num_decoder_layers)]
        self._dropout_fn = lambda x, training: tf.compat.v2.nn.dropout(
            x, dropout, noise_shape=[x.shape[0], 1, x.shape[2]]) if training else x
        self._vocab_size = vocab_size
        self._num_heads = num_heads
        self._label_smoothing = label_smoothing
        self._decoder_scope_name = "decoder"
        self._layer_norm_encoder = tf.keras.layers.LayerNormalization(axis=2, epsilon=1e-12, name="LayerNorm")
        self._layer_norm_decoder = tf.keras.layers.LayerNormalization(axis=2, epsilon=1e-12, name="LayerNorm")

    def _encode(self, features, training):
        inputs_BxI = features["inputs"]
        inputs_bias_Bx1xI = attention.ids_to_bias(inputs_BxI, self._dtype)
        states_BxIxD = self._embedding_layer(inputs_BxI, True)
        states_BxIxD = self._dropout_fn(
            timing.add_time_signal(states_BxIxD), training)
        with tf.compat.v1.variable_scope("encoder", reuse=tf.compat.v1.AUTO_REUSE):
            states_BxIxD = transformer_block.stack(self._encoder_layers, training,
                                                   states_BxIxD, inputs_bias_Bx1xI,
                                                   None, None)
            states_BxIxD = self._layer_norm_encoder(states_BxIxD)
        return {"memory": states_BxIxD, "memory_bias": inputs_bias_Bx1xI}

    def __call__(self, features, training):
        """Create model.
        Args:
          features: dictionary of tensors including "inputs" [batch, input_len] and
            "targets" [batch, output_len]
          training: bool of whether the mode is training.
        Returns:
         Tuple of (loss, outputs): Loss is a scalar. Output is a dictionary of
           tensors, containing model's output logits.
        """
        if "inputs" not in features or "targets" not in features:
            raise ValueError("Require inputs and targets keys in features.")

        context = self._encode(features, training)
        self._context = context
        targets_BxT = features["targets"]
        bias_1xTxT = attention.upper_triangle_bias(
            tf.shape(input=targets_BxT)[1], self._dtype)
        states_BxTxD = self._embedding_layer(targets_BxT, True)
        states_BxTxD = tf.pad(tensor=states_BxTxD, paddings=[[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        states_BxTxD = timing.add_time_signal(states_BxTxD)
        states_BxTxD = self._dropout_fn(states_BxTxD, training)
        with tf.compat.v1.variable_scope(self._decoder_scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            states_BxTxD = transformer_block.stack(self._decoder_layers, training,
                                                   states_BxTxD, bias_1xTxT,
                                                   context["memory"],
                                                   context["memory_bias"])
            states_BxTxD = self._layer_norm_decoder(states_BxTxD)
        logits_BxTxV = self._embedding_layer(states_BxTxD, False)
        targets_mask_BxT = tf.cast(tf.greater(targets_BxT, 0), self._dtype)
        loss = tf.compat.v1.losses.softmax_cross_entropy(
            tf.one_hot(targets_BxT, self._vocab_size),
            logits_BxTxV,
            label_smoothing=self._label_smoothing,
            weights=targets_mask_BxT)
        return loss, {"logits": logits_BxTxV}

    def predict(self, features, max_decode_len, beam_size, **beam_kwargs):
        """Predict."""
        cache = self._encode(features, False)
        B, _, D = cache["memory"].shape
        T, V, H = max_decode_len, self._vocab_size, self._num_heads

        bias_1xTxT = attention.upper_triangle_bias(T, self._dtype)
        for i in range(len(self._decoder_layers)):
            cache[str(i)] = {
                "k": tf.zeros([B, H, T, D // H], self._dtype),
                "v": tf.zeros([B, H, T, D // H], self._dtype)
            }

        def symbols_to_logits_fn(dec_BxT, context, i):
            """Decode loop."""
            dec_Bx1 = tf.slice(dec_BxT, [0, tf.maximum(tf.cast(0, i.dtype), i - 1)],
                               [dec_BxT.shape[0], 1])
            bias_1x1xT = tf.slice(bias_1xTxT, [0, i, 0], [1, 1, T])
            dec_Bx1xD = self._embedding_layer(dec_Bx1, True)
            dec_Bx1xD *= tf.cast(tf.greater(i, 0), self._dtype)
            dec_Bx1xD = timing.add_time_signal(dec_Bx1xD, start_index=i)
            with tf.compat.v1.variable_scope(self._decoder_scope_name, reuse=tf.compat.v1.AUTO_REUSE):
                dec_Bx1xD = transformer_block.stack(self._decoder_layers, False,
                                                    dec_Bx1xD, bias_1x1xT,
                                                    context["memory"],
                                                    context["memory_bias"], context, i)
                dec_Bx1xD = self._layer_norm_decoder(dec_Bx1xD)
            logits_Bx1xV = self._embedding_layer(dec_Bx1xD, False)
            logits_BxV = tf.squeeze(logits_Bx1xV, axis=1)
            return logits_BxV

        decodes_BxT = decoding.left2right_decode(symbols_to_logits_fn, cache, B, T,
                                                 V, beam_size, **beam_kwargs)
        return {"outputs": decodes_BxT}


#TODO: add docstring
PEGASUS_START_DOCSTRING = r"""PEGASUS start"""

PEGASUS_INPUTS_DOCSTRING = r"""PEGASUS inputs"""


@add_start_docstrings(
    "The PEGASUS Model transformer",
    PEGASUS_START_DOCSTRING,
)
class TFPegasusModel(TFPegasusPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        print("init")

    @add_start_docstrings_to_callable(PEGASUS_INPUTS_DOCSTRING)
    def call(self, inputs, **kwargs):
        print("call")
        return tf.zeros((2, 2)), tf.zeros((2, 2)), tf.zeros((2, 2))
