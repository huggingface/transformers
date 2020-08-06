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


# import tensorflow_text as tf_text



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


_SHIFT_RESERVED_TOKENS = 103
_NEWLINE_SYMBOL = "<n>"


def encode(text: tf.Tensor, max_len: int, vocab_filename: str, encoder_type: str):
    """EncodeOp."""
    import tensorflow_text as tf_text
    if encoder_type not in ["sentencepiece", "sentencepiece_newline"]:
        raise ValueError("Unsupported encoder type: %s" % encoder_type)
    sp_model = tf.io.gfile.GFile(vocab_filename, "rb").read()
    tokenizer = tf_text.SentencepieceTokenizer(model=sp_model)
    batch_size = text.shape[0]
    if encoder_type == "sentencepiece_newline":
        text = tf.strings.regex_replace(text, "\n", _NEWLINE_SYMBOL)
    ids = tokenizer.tokenize(text)
    eos = tf.ragged.constant([[1]] * batch_size)
    ids = tf.concat([ids, eos], axis=1)
    ids = ids.to_tensor(default_value=0)
    ids = ids[:, :max_len]
    pad = max_len - tf.shape(input=ids)[1]
    ids = tf.pad(tensor=ids, paddings=[[0, 0], [0, pad]])
    ids.set_shape([ids.shape[0], max_len])
    ids = tf.compat.v1.where(ids > 1, ids + _SHIFT_RESERVED_TOKENS, ids)
    ids = tf.cast(ids, tf.int64)
    return ids


def decode(ids: tf.Tensor, vocab_filename: str, encoder_type: str):
    """DecodeOp."""
    import tensorflow_text as tf_text
    if encoder_type not in ["sentencepiece", "sentencepiece_newline"]:
        raise ValueError("Unsupported encoder type: %s" % encoder_type)
    sp_model = tf.io.gfile.GFile(vocab_filename, "rb").read()
    tokenizer = tf_text.SentencepieceTokenizer(model=sp_model)
    ids = tf.compat.v1.where(ids > 1 + _SHIFT_RESERVED_TOKENS, ids - _SHIFT_RESERVED_TOKENS, ids)
    ids = tf.cast(ids, tf.int32)
    text = tokenizer.detokenize(ids)
    text = tf.reshape(text, [-1])
    if encoder_type == "sentencepiece_newline":
        text = tf.strings.regex_replace(text, _NEWLINE_SYMBOL, "\n")
    return text


_MIN_TIMESCALE = 1.0
_MAX_TIMESCALE = 1.0e4


def add_time_signal(inputs_BxIxD, start_index=None):
    """Adds a transformer-style timing signal to inputs.

    Using periodic signals as in https://arxiv.org/abs/1706.03762.
    Generalized to allow each example in a batch to begin at a different index.

    Args:
      inputs_BxIxD: input representation.
      start_index: tensor of starting pos. [batch_size]

    Returns:
      output: representation with time signal added, same shape as input.
    """

    dtype = inputs_BxIxD.dtype
    B, I, D = inputs_BxIxD.shape
    if D % 2 != 0:
        raise ValueError("Input dimension must be even.")
    start_Bx1 = tf.zeros((B, 1), tf.int32) if start_index is None else start_index

    pos_1xI = tf.expand_dims(tf.range(I), 0)
    pos_BxI = tf.tile(pos_1xI, [B, 1]) + tf.cast(start_Bx1, tf.int32)
    pos_BxI = tf.cast(pos_BxI, dtype)
    N = D // 2
    log_time_incr = math.log(_MAX_TIMESCALE / _MIN_TIMESCALE) / tf.maximum(tf.cast(N, dtype) - 1, 1)
    inv_scale_N = _MIN_TIMESCALE * tf.exp(tf.cast(tf.range(N), dtype) * -log_time_incr)
    time_BxIxN = tf.expand_dims(pos_BxI, 2) * tf.reshape(inv_scale_N, [1, 1, -1])
    signal_BxIxD = tf.concat([tf.sin(time_BxIxN), tf.cos(time_BxIxN)], axis=2)
    signal_BxIxD = tf.reshape(signal_BxIxD, [B, I, D])
    return inputs_BxIxD + signal_BxIxD


def length_normalization(start, alpha, min_len, max_len, out_of_range_penalty):
    r"""Create length normalization function.

    Combines length penalty from https://arxiv.org/abs/1609.08144,
    and length constraint from https://www.aclweb.org/anthology/W18-2706.pdf.

    scores = \sum_j log(P_j) / ((start + lengths)/(1 + start))**alpha
            + out_of_range_penalty * (length > max_len or length < min_len)

    Args:
      start: int, length normalization start offset.
      alpha: float, [0, 1.0],  length normalization power.
      min_len: int, minimum decode length.
      max_len: int, maximum decode lengths.
      out_of_range_penalty: float, penalty for lengths outside min len and max
        len. Use a negative number that penalize out of range decodes, does hard
        constraint if set to -inf.

    Returns:
      fn(log_probs_BxM, length)->scores_BxM: a function to normalize sum log
      probabilities of sequence with current decoding lengths.
    """

    def length_norm_fn(log_probs_BxM, length_int):
        """Normalize sum log probabilities given a sequence length."""
        dtype = log_probs_BxM.dtype
        norm_flt = tf.pow(((start + tf.cast(length_int, dtype)) / (1.0 + start)), alpha)
        log_probs_BxM /= norm_flt
        too_short_bool = tf.less(length_int, min_len)
        too_long_bool = tf.logical_and(tf.greater(length_int, max_len), max_len > 0)
        out_of_range_bool = tf.logical_or(too_long_bool, too_short_bool)
        log_probs_BxM += out_of_range_penalty * tf.cast(out_of_range_bool, dtype)
        return log_probs_BxM

    return length_norm_fn


def beam_search(
    symbols_to_logits_fn, init_seq_BxT, initial_cache_BxU, vocab_size, beam_size, length_norm_fn, eos_id=1
):
    """Beam search.

    Args:
      symbols_to_logits_fn: fn(seq_BxT, cache_BxU, i) -> (logits_BxV, cache_BxU)
      init_seq_BxT: initial sequence ids.
      initial_cache_BxU: dictionary of tensors with shape BxU.
      vocab_size: vocabulary size.
      beam_size: beam size.
      length_norm_fn: length normalization function.
      eos_id: end of sequence.

    Returns:
      Tuple of (beams_BxMxT, scores_BxM). Beam searched sequences and scores.
  """
    B, T = init_seq_BxT.shape
    M, V = beam_size, vocab_size
    dtype = tf.float32
    int_dtype = init_seq_BxT.dtype

    def _loop_body(
        i, alive_seq_BxMxT, alive_log_probs_BxM, alive_cache_BxMxU, finished_seq_BxMxT, finished_scores_BxM
    ):
        """Beam search loop body."""
        # Decode one step with beam
        logits_BMxV, cache_BMxU = symbols_to_logits_fn(
            _flatten_beam_dim(alive_seq_BxMxT), tf.nest.map_structure(_flatten_beam_dim, alive_cache_BxMxU), i
        )
        logits_BxMxV = _unflatten_beam_dim(logits_BMxV, M)
        new_cache_BxMxU = tf.nest.map_structure(lambda t: _unflatten_beam_dim(t, M), cache_BMxU)

        # select top 2 * beam_size and fill alive and finished.
        log_probs_BxMxV = logits_BxMxV - tf.reduce_logsumexp(input_tensor=logits_BxMxV, axis=2, keepdims=True)
        log_probs_BxMxV += tf.expand_dims(alive_log_probs_BxM, axis=2)
        log_probs_BxMV = tf.reshape(log_probs_BxMxV, [B, -1])
        new_log_probs_Bx2M, topk_indices_Bx2M = tf.nn.top_k(log_probs_BxMV, k=2 * M)
        topk_beam_Bx2M = topk_indices_Bx2M // V
        topk_seq_Bx2MxT, new_cache_Bx2MxU = _gather_nested([alive_seq_BxMxT, new_cache_BxMxU], topk_beam_Bx2M)
        topk_ids_Bx2M = topk_indices_Bx2M % V
        new_seq_Bx2MxT = _update_i(topk_seq_Bx2MxT, topk_ids_Bx2M, i)
        new_finished_flags_Bx2M = tf.cast(tf.reduce_any(input_tensor=tf.equal(new_seq_Bx2MxT, eos_id), axis=-1), dtype)

        # get new alive
        _, topk_alive_indices_BxM = tf.nn.top_k(new_log_probs_Bx2M + new_finished_flags_Bx2M * dtype.min, k=M)
        (alive_seq_BxMxT, alive_log_probs_BxM, alive_cache_BxMxU) = _gather_nested(
            [new_seq_Bx2MxT, new_log_probs_Bx2M, new_cache_Bx2MxU], topk_alive_indices_BxM
        )

        # get new finished
        new_scores_Bx2M = length_norm_fn(new_log_probs_Bx2M, i + 1)
        new_scores_Bx2M += (1 - new_finished_flags_Bx2M) * dtype.min
        finished_seq_Bx3MxT = tf.concat([finished_seq_BxMxT, new_seq_Bx2MxT], axis=1)
        finished_scores_Bx3M = tf.concat([finished_scores_BxM, new_scores_Bx2M], axis=1)
        _, topk_finished_indices_BxM = tf.nn.top_k(finished_scores_Bx3M, k=M)
        (finished_seq_BxMxT, finished_scores_BxM) = _gather_nested(
            [finished_seq_Bx3MxT, finished_scores_Bx3M], topk_finished_indices_BxM
        )

        return [
            i + 1,
            alive_seq_BxMxT,
            alive_log_probs_BxM,
            alive_cache_BxMxU,
            finished_seq_BxMxT,
            finished_scores_BxM,
        ]

    # initialize.
    init_i = tf.constant(0, dtype=int_dtype)
    init_alive_seq_BxMxT = _expand_to_beam_size(init_seq_BxT, M)
    log_probs_1xM = tf.constant([[0.0] + [dtype.min] * (M - 1)], dtype=dtype)
    init_alive_log_probs_BxM = tf.tile(log_probs_1xM, [B, 1])
    init_alive_cache_BxMxU = tf.nest.map_structure(lambda t: _expand_to_beam_size(t, M), initial_cache_BxU)
    init_finished_seq_BxMxT = tf.zeros(tf.shape(input=init_alive_seq_BxMxT), int_dtype)
    init_finished_scores_BxM = tf.zeros([B, M], dtype=dtype) + dtype.min

    # run loop.
    (
        _,
        final_alive_seq_BxMxT,
        final_alive_scores_BxM,
        _,
        final_finished_seq_BxMxT,
        final_finished_scores_BxM,
    ) = tf.while_loop(
        cond=lambda *args: True,  # Always do T iterations
        body=_loop_body,
        loop_vars=[
            init_i,
            init_alive_seq_BxMxT,
            init_alive_log_probs_BxM,
            init_alive_cache_BxMxU,
            init_finished_seq_BxMxT,
            init_finished_scores_BxM,
        ],
        parallel_iterations=1,
        back_prop=False,
        maximum_iterations=T,
    )

    # process finished.
    final_finished_flag_BxMx1 = tf.reduce_any(
        input_tensor=tf.equal(final_finished_seq_BxMxT, eos_id), axis=-1, keepdims=True
    )
    final_seq_BxMxT = tf.compat.v1.where(
        tf.tile(final_finished_flag_BxMx1, [1, 1, T]), final_finished_seq_BxMxT, final_alive_seq_BxMxT
    )
    final_scores_BxM = tf.compat.v1.where(
        tf.squeeze(final_finished_flag_BxMx1, axis=-1), final_finished_scores_BxM, final_alive_scores_BxM
    )
    return final_seq_BxMxT, final_scores_BxM


def _update_i(tensor_BxNxT, updates_BxN, i):
    B, N, T = tensor_BxNxT.shape
    tensor_BNxT = tf.reshape(tensor_BxNxT, [-1, T])
    updates_BN = tf.reshape(updates_BxN, [-1])
    batch_BN = tf.range(B * N, dtype=tf.int64)
    i_BN = tf.fill([B * N], tf.cast(i, tf.int64))
    ind_BNx2 = tf.stack([batch_BN, i_BN], axis=-1)
    tensor_BNxT = tf.tensor_scatter_nd_update(tensor_BNxT, ind_BNx2, updates_BN)
    return tf.reshape(tensor_BNxT, [B, N, T])


def _expand_to_beam_size(tensor_BxU, beam_size):
    tensor_Bx1xU = tf.expand_dims(tensor_BxU, axis=1)
    tile_dims = [1] * tensor_Bx1xU.shape.ndims
    tile_dims[1] = beam_size
    tensor_BxMxU = tf.tile(tensor_Bx1xU, tile_dims)
    return tensor_BxMxU


def _flatten_beam_dim(tensor_BxMxU):
    shape = tensor_BxMxU.shape.as_list()
    tensor_BMxU = tf.reshape(tensor_BxMxU, [shape[0] * shape[1]] + shape[2:])
    return tensor_BMxU


def _unflatten_beam_dim(tensor_BMxU, M):
    shape = tensor_BMxU.shape.as_list()
    tensor_BxMxU = tf.reshape(tensor_BMxU, [shape[0] // M, M] + shape[1:])
    return tensor_BxMxU


def _gather_nested(nested_BxMxU, indices_BxN):
    def _gather_beam(tensor_BxMxU):
        tensor_BxNxU = tf.gather(tensor_BxMxU, indices_BxN, batch_dims=1, axis=1)
        return tensor_BxNxU

    return tf.nest.map_structure(_gather_beam, nested_BxMxU)


EOS_ID = 1


def process_logits(logits_BxN, top_k=0, top_p=0.0, temperature=0.0):
    """Process logits using gumbel noise and mask top_k or top_p.

    The downstream task can perform probability sampling using gumbel-max trick
    (taking the argmax of processed logits) (Statistical theory of extreme values
    and some practical applications: a series of lectures. 1954).
    Use cases:
      greedy: top_k=0, top_p=0.0, temperature=0.0
      random sampling: top_k=0, top_p=0.0, temperature=1.0
      topk sampling: top_k=k, top_p=0.0, temperature=1.0
      nucleus sampling: top_k=0, top_p=p, temperature=1.0
      random sampling biased toward greedy: top_k=0, top_p=0.0, temperature=0.5
    Notations:
      B: batch_size, N: number of logits, K: topk value.
    Args:
      logits_BxN: tensor of [batch_size vocab_size]
      top_k: k in top_k sampling.
      top_p: probability in necleus sampling.
      temperature: gumbel noise sampling temperature.

    Returns:
      logits: processed logits which is original logits add gumbel noise and
      values outside top_k and top_p set to -inf.
    """
    if top_k > 0 and top_p > 0:
        raise ValueError("Only one of the top_k and nucleus sampling should be specified.")

    if top_k > 0:
        top_values_BxK, _ = tf.math.top_k(logits_BxN, k=top_k, sorted=False)
        min_value_Bx1 = tf.reduce_min(input_tensor=top_values_BxK, axis=-1, keepdims=True)
        mask_BxN = tf.cast(tf.less(logits_BxN, min_value_Bx1), logits_BxN.dtype)
        logits_BxN -= mask_BxN * logits_BxN.dtype.max

    if top_p > 0:
        sort_indices_BxN = tf.argsort(logits_BxN, axis=-1, direction="DESCENDING")
        probs_BxN = tf.gather(tf.nn.softmax(logits_BxN), sort_indices_BxN, batch_dims=1)
        cumprobs_BxN = tf.cumsum(probs_BxN, axis=-1, exclusive=True)
        # The top 1 candidate always will not be masked.
        # This way ensures at least 1 indices will be selected.
        sort_mask_BxN = tf.cast(tf.greater(cumprobs_BxN, top_p), logits_BxN.dtype)
        batch_indices_BxN = tf.tile(tf.expand_dims(tf.range(logits_BxN.shape[0]), axis=-1), [1, logits_BxN.shape[1]])
        top_p_mask_BxN = tf.scatter_nd(
            tf.stack([batch_indices_BxN, sort_indices_BxN], axis=-1), sort_mask_BxN, logits_BxN.shape
        )
        logits_BxN -= top_p_mask_BxN * logits_BxN.dtype.max

    if temperature > 0:
        logits_shape = tf.shape(input=logits_BxN)
        uniform_noise_BxN = tf.random.uniform(logits_shape)
        logits_BxN += -tf.math.log(-tf.math.log(uniform_noise_BxN)) * temperature
    return logits_BxN


def inplace_update_i(tensor_BxL, updates_B, i):
    """Inplace update a tensor. B: batch_size, L: tensor length."""
    batch_size = tensor_BxL.shape[0]
    indices_Bx2 = tf.stack(
        [tf.range(batch_size, dtype=tf.int64), tf.fill([batch_size], tf.cast(i, tf.int64))], axis=-1
    )
    return tf.tensor_scatter_nd_update(tensor_BxL, indices_Bx2, updates_B)


def left2right_decode(
    symbols_to_logits_fn,
    context_BxU_dict,
    batch_size,
    max_decode_len,
    vocab_size,
    beam_size=1,
    beam_start=5,
    beam_alpha=0.6,
    beam_min=0,
    beam_max=-1,
    temperature=0.0,
    top_k=0,
    top_p=0.0,
    eos_id=EOS_ID,
):
    """left to right decode.

    Notations:
      B: batch_size, V: vocab_size, T: decode_len, U: undefined dimensions

    Args:
      symbols_to_logits_fn: logits = fn(decodes, context, i). Shoud take
        [batch_size, decoded_ids] and return [batch_size, vocab_size].
      context_BxU_dict: dict of Tensors.
      batch_size: int, decode batch size.
      max_decode_len: int, maximum number of steps to decode.
      vocab_size: int, output vocab size.
      beam_size: Number of beams to decode.
      beam_start: start length for scaling, default to 5.
      beam_alpha: Length penalty for decoding. Should be between 0 (shorter) and 1
        (longer), default to 0.6.
      beam_min: Minimum beam search lengths.
      beam_max: Maximum beam search lengths. Set -1 to use unlimited.
      temperature: Sampling temp for next token (0 for argmax), default to 0.0.
      top_k: Number of top symbols to consider at each time step, default to 0
        (consider all symbols).
      top_p: Nucleus sampling probability.
      eos_id: end of token id, default to 1.

    Returns:
      decodes: Tensor[batch, decode_len]
    """
    dtype = tf.int64
    # When beam_size=1, beam_search does not behave exactly like greedy.
    # This is due to using 2 * beam_size in grow_topk, and keep the top beam_size
    # ones that haven't reached EOS into alive.
    # In this case, alpha value for length penalty will take effect.
    if beam_size == 1:

        def decode_loop(i, decodes_BxT, cache_BxU_dict):
            logits_BxV = symbols_to_logits_fn(decodes_BxT, cache_BxU_dict, i)
            logits_BxV = process_logits(logits_BxV, top_k, top_p, temperature)
            decodes_BxT = inplace_update_i(decodes_BxT, tf.argmax(input=logits_BxV, axis=-1), i)
            return i + 1, decodes_BxT, cache_BxU_dict

        def loop_cond(i, decodes_BxT, unused_cache_BxU_dict):
            finished_B = tf.reduce_any(input_tensor=tf.equal(decodes_BxT, EOS_ID), axis=1)
            return tf.logical_and(i < max_decode_len, tf.logical_not(tf.reduce_all(input_tensor=finished_B)))

        init_dec_BxT = tf.zeros([batch_size, max_decode_len], dtype=dtype)
        _, decodes, _ = tf.while_loop(
            cond=loop_cond, body=decode_loop, loop_vars=[tf.constant(0, dtype=dtype), init_dec_BxT, context_BxU_dict]
        )
        return decodes

    else:

        def symbols_to_logits_fn_with_sampling(decodes_BxT, states_BxU_dict, i):
            logits_BxV = symbols_to_logits_fn(decodes_BxT, states_BxU_dict, i)
            logits_BxV = process_logits(logits_BxV, top_k, top_p, temperature)
            return logits_BxV, states_BxU_dict

        length_norm_fn = length_normalization(beam_start, beam_alpha, beam_min, beam_max, -1e3)
        beams, _ = beam_search(
            symbols_to_logits_fn_with_sampling,
            tf.zeros([batch_size, max_decode_len], dtype=tf.int32),
            context_BxU_dict,
            vocab_size,
            beam_size,
            length_norm_fn,
            eos_id,
        )
        return tf.cast(beams[:, 0, :], dtype)


class Embedding(object):
    """Embedding layer supporting shared input/output weights."""

    def __init__(self, vocab_size, hidden_size, name, dtype):
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._name = name
        self._dtype = dtype

    def __call__(self, tensor, is_input_layer):
        if is_input_layer:
            return self._ids_to_weights(tensor)
        else:
            return self._weights_to_logits(tensor)

    def _ids_to_weights(self, ids_BxI):
        """Maps IDs to embedding weights."""
        weights_BxIxD = tf.nn.embedding_lookup(params=self.weights_VxD, ids=ids_BxI)
        weights_BxIxD *= self._hidden_size ** 0.5
        return weights_BxIxD

    def _weights_to_logits(self, states_BxIxD):
        B, I, D = states_BxIxD.shape
        states_BIxD = tf.reshape(states_BxIxD, [-1, D])
        states_BIxV = tf.matmul(states_BIxD, self.weights_VxD, transpose_b=True)
        states_BxIxV = tf.reshape(states_BIxV, [B, I, self._vocab_size])
        return states_BxIxV

    @property
    def weights_VxD(self):
        """Gets embedding weights."""
        with tf.compat.v1.variable_scope("embeddings", reuse=tf.compat.v1.AUTO_REUSE):
            # Initialization is important here, and a normal distribution with stdev
            # equal to rsqrt hidden_size is significantly better than the default
            # initialization used for other layers (fan in / out avg).
            embeddings_VxD = tf.compat.v1.get_variable(
                self._name,
                [self._vocab_size, self._hidden_size],
                initializer=tf.compat.v1.random_normal_initializer(
                    stddev=self._hidden_size ** -0.5, dtype=self._dtype
                ),
                dtype=self._dtype,
            )
        return embeddings_VxD


def split_heads(tensor_BxIxD, num_heads):
    B, I, D = tensor_BxIxD.shape
    tensor_BxIxHxD = tf.reshape(tensor_BxIxD, [B, I, num_heads, D // num_heads])
    tensor_BxHxIxD = tf.transpose(a=tensor_BxIxHxD, perm=[0, 2, 1, 3])
    return tensor_BxHxIxD


class Attention:
    """Multihead scaled dot product attention."""

    def __init__(self, hidden_size, num_heads, attention_dropout):
        if hidden_size % num_heads != 0:
            raise ValueError("Number of attention heads must divide hidden size")

        self._q_layer = tf.compat.v1.layers.Dense(hidden_size, use_bias=False, name="q_proj")
        self._k_layer = tf.compat.v1.layers.Dense(hidden_size, use_bias=False, name="k_proj")
        self._v_layer = tf.compat.v1.layers.Dense(hidden_size, use_bias=False, name="v_proj")
        self._output_layer = tf.compat.v1.layers.Dense(hidden_size, use_bias=False, name="output_proj")
        self._num_heads = num_heads
        self._hidden_size = hidden_size
        self._attention_dropout = attention_dropout

    def __call__(self, input_BxIxDi, memory_BxMxDi, bias_BxIxM, training, cache=None, decode_i=None):
        B, I, _ = input_BxIxDi.shape
        M, H, D = memory_BxMxDi.shape[1], self._num_heads, self._hidden_size
        dtype = memory_BxMxDi.dtype

        q_BxHxIxDh = split_heads(self._q_layer(input_BxIxDi), H)
        q_BxHxIxDh *= (D // H) ** -0.5
        k_BxHxMxDh = split_heads(self._k_layer(memory_BxMxDi), H)
        v_BxHxMxDh = split_heads(self._v_layer(memory_BxMxDi), H)

        # cache saves previous activations before time decode_i during TPU decoding.
        if cache is not None and decode_i is not None:
            M = cache["k"].shape[2]
            indices_1x1xMx1 = tf.reshape(tf.one_hot(decode_i, M, dtype=dtype), [1, 1, M, 1])
            k_BxHxMxDh = cache["k"] + k_BxHxMxDh * indices_1x1xMx1
            v_BxHxMxDh = cache["v"] + v_BxHxMxDh * indices_1x1xMx1
            cache["k"] = k_BxHxMxDh
            cache["v"] = v_BxHxMxDh
        bias_BxHxIxM = tf.expand_dims(bias_BxIxM, axis=1)
        logits_BxHxIxM = tf.matmul(q_BxHxIxDh, k_BxHxMxDh, transpose_b=True) + bias_BxHxIxM
        alignment_BxHxIxM = tf.nn.softmax(logits_BxHxIxM)
        if training:
            alignment_BxHxIxM = tf.compat.v2.nn.dropout(
                alignment_BxHxIxM, self._attention_dropout, noise_shape=[1, 1, I, M]
            )
        outputs_BxHxIxDh = tf.matmul(alignment_BxHxIxM, v_BxHxMxDh)
        outputs_BxIxD = tf.reshape(tf.transpose(a=outputs_BxHxIxDh, perm=[0, 2, 1, 3]), [B, I, D])
        outputs_BxIxD = self._output_layer(outputs_BxIxD)
        return outputs_BxIxD


class SelfAttention(Attention):
    """Multihead scaled dot product self-attention."""

    def __call__(self, x, bias, training, cache=None, decode_i=None):
        return super().__call__(x, x, bias, training, cache=cache, decode_i=decode_i)


def _assert_equal(a,b):
    assert a ==b, f'{a} != {b}'

def ids_to_bias(ids_BxI, dtype=tf.float32, padding_id=0):
    """Convert ids to attention bias for attention."""
    pad_BxI = tf.cast(tf.equal(ids_BxI, padding_id), dtype)
    bias_Bx1xI = tf.expand_dims(pad_BxI * dtype.min, axis=1)
    return bias_Bx1xI


def upper_triangle_bias(D, dtype=tf.float32):
    """Create a upper triangle matrix for decoding bias."""
    upper_triangle_DxD = 1 - tf.linalg.band_part(tf.ones([D, D], dtype=dtype), -1, 0)
    #assert self._dtype == tf.float32, f'{self._dtype} != tf.float32'
    if isinstance(dtype, str):
        raise TypeError(dtype)
    min_val =dtype.min
    tensor_1xDxD = tf.expand_dims(upper_triangle_DxD * min_val, axis=0)
    return tensor_1xDxD


class TransformerBlock:
    """Transformer block.

    Attention block of self-attention, attention over external memory, and
    feedforward network.
    Initialize the block with
      block = TransformerBlock(hidden_size, filter_size, num_heads, dropout)
    To create an encoder self attention layer, use
      x = block(x, x_bias, None, None)
    To create a decoder attention layer, use
      y = block(y, upper_triangle_bias, x, x_bias)
    """

    def __init__(self, hidden_size, filter_size, num_heads, dropout):
        self._self_attn_layer = SelfAttention(hidden_size, num_heads, dropout)
        self._attn_layer = Attention(hidden_size, num_heads, dropout)
        self._relu_layer = tf.compat.v1.layers.Dense(filter_size, activation=tf.nn.relu)
        self._output_layer = tf.compat.v1.layers.Dense(hidden_size)
        self._dropout_fn = (
            lambda x, training: tf.compat.v2.nn.dropout(x, dropout, noise_shape=[x.shape[0], 1, x.shape[2]])
            if training
            else x
        )

        self._layer_norm_1 = tf.keras.layers.LayerNormalization(axis=2, epsilon=1e-12, name="LayerNorm")
        self._layer_norm_2 = tf.keras.layers.LayerNormalization(axis=2, epsilon=1e-12, name="LayerNorm")
        self._layer_norm_3 = tf.keras.layers.LayerNormalization(axis=2, epsilon=1e-12, name="LayerNorm")

    def __call__(self, training, inputs_BxIxD, bias_BxIxI, memory_BxMxD, bias_BxIxM, cache=None, decode_i=None):
        s_BxIxD = inputs_BxIxD
        with tf.compat.v1.variable_scope("self_attention"):
            y_BxIxD = self._layer_norm_1(s_BxIxD)
            y_BxIxD = self._self_attn_layer(y_BxIxD, bias_BxIxI, training, cache=cache, decode_i=decode_i)
            s_BxIxD += self._dropout_fn(y_BxIxD, training)
        if memory_BxMxD is not None:
            with tf.compat.v1.variable_scope("memory_attention"):
                y_BxIxD = self._layer_norm_2(s_BxIxD)
                y_BxIxD = self._attn_layer(y_BxIxD, memory_BxMxD, bias_BxIxM, training)
                s_BxIxD += self._dropout_fn(y_BxIxD, training)
        with tf.compat.v1.variable_scope("ffn"):
            y_BxIxD = self._layer_norm_3(s_BxIxD)
            y_BxIxD = self._dropout_fn(self._relu_layer(y_BxIxD), training)
            s_BxIxD += self._dropout_fn(self._output_layer(y_BxIxD), training)
        return s_BxIxD


def stack(layers, training, inputs_BxIxD, bias_BxIxI, memory_BxMxD, bias_BxIxM, cache=None, decode_i=None):
    """Stack AttentionBlock layers."""
    if (memory_BxMxD is None) != (bias_BxIxM is None):
        raise ValueError("memory and memory_bias need to be provided together.")
    s_BxIxD = inputs_BxIxD
    for i, layer in enumerate(layers):
        with tf.compat.v1.variable_scope("layer_%d" % i):
            s_BxIxD = layer(
                training,
                s_BxIxD,
                bias_BxIxI,
                memory_BxMxD,
                bias_BxIxM,
                cache=cache[str(i)] if cache is not None else None,
                decode_i=decode_i,
            )
    return s_BxIxD


class TFPegasusLegacyModel:
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pre-trained models.
    """

    config_class = PegasusConfig
    base_model_prefix = "transformer"

    def __init__(
        self,
        config: PegasusConfig,

    ):
        #super().__init__(config)
        # vocab_size,
        # hidden_size,
        # filter_size,
        # num_heads,
        # num_encoder_layers,
        # num_decoder_layers,
        # label_smoothing,
        # dropout,
        # FIXME(SS): should take config
        #import ipdb; ipdb.set_trace()
        self._dtype = tf.float32
        _assert_equal(type(self._dtype), type(tf.float32))
        self._embedding_layer = Embedding(config.vocab_size, config.d_model, "weights", self._dtype)
        #block_fn = lambda: TransformerBlock(c.d_model, c.encoder_ffn_dim, c.encoder_attention_heads, c.dropout)
        self._encoder_layers = [TransformerBlock(config.d_model, config.encoder_ffn_dim, config.encoder_attention_heads, config.dropout) for _ in range(config.encoder_layers)]
        self._decoder_layers = [TransformerBlock(config.d_model, config.decoder_ffn_dim, config.decoder_attention_heads, config.dropout) for _ in range(config.decoder_layers)]
        self._dropout_fn = (
            lambda x, training: tf.compat.v2.nn.dropout(x, config.dropout, noise_shape=[x.shape[0], 1, x.shape[2]])
            if training
            else x
        )
        self._vocab_size = config.vocab_size
        self._num_heads = config.encoder_attention_heads
        self._label_smoothing = 0.1
        self._decoder_scope_name = "decoder"
        self._layer_norm_encoder = tf.keras.layers.LayerNormalization(axis=2, epsilon=1e-12, name="LayerNorm")
        self._layer_norm_decoder = tf.keras.layers.LayerNormalization(axis=2, epsilon=1e-12, name="LayerNorm")
        _assert_equal(type(self._dtype), type(tf.float32))


    def _encode(self, features, training):
        inputs_BxI = features["inputs"]

        assert self._dtype == tf.float32, f'{self._dtype} != tf.float32'
        inputs_bias_Bx1xI = ids_to_bias(inputs_BxI)#, self._dtype)
        states_BxIxD = self._embedding_layer(inputs_BxI, True)
        states_BxIxD = self._dropout_fn(add_time_signal(states_BxIxD), training)
        with tf.compat.v1.variable_scope("encoder", reuse=tf.compat.v1.AUTO_REUSE):
            states_BxIxD = stack(self._encoder_layers, training, states_BxIxD, inputs_bias_Bx1xI, None, None)
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
        _assert_equal(type(self._dtype), type(tf.float32))
        context = self._encode(features, training)
        self._context = context
        targets_BxT = features["targets"]

        #import ipdb; ipdb.set_trace()
        bias_1xTxT = upper_triangle_bias(tf.shape(input=targets_BxT)[1], self._dtype)
        states_BxTxD = self._embedding_layer(targets_BxT, True)
        self._emb = states_BxTxD
        states_BxTxD = tf.pad(tensor=states_BxTxD, paddings=[[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        states_BxTxD = add_time_signal(states_BxTxD)
        #self._time_signal =
        states_BxTxD = self._dropout_fn(states_BxTxD, training)
        with tf.compat.v1.variable_scope(self._decoder_scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            states_BxTxD = stack(
                self._decoder_layers, training, states_BxTxD, bias_1xTxT, context["memory"], context["memory_bias"]
            )
            states_BxTxD = self._layer_norm_decoder(states_BxTxD)
        logits_BxTxV = self._embedding_layer(states_BxTxD, False)
        targets_mask_BxT = tf.cast(tf.greater(targets_BxT, 0), self._dtype)
        loss = tf.compat.v1.losses.softmax_cross_entropy(
            tf.one_hot(targets_BxT, self._vocab_size),
            logits_BxTxV,
            label_smoothing=self._label_smoothing,
            weights=targets_mask_BxT,
        )
        return loss, {"logits": logits_BxTxV}

    def predict(self, features, max_decode_len, beam_size, **beam_kwargs):
        """Predict."""
        cache = self._encode(features, False)
        B, _, D = cache["memory"].shape
        T, V, H = max_decode_len, self._vocab_size, self._num_heads

        bias_1xTxT = upper_triangle_bias(T, self._dtype)
        for i in range(len(self._decoder_layers)):
            cache[str(i)] = {
                "k": tf.zeros([B, H, T, D // H], self._dtype),
                "v": tf.zeros([B, H, T, D // H], self._dtype),
            }

        def symbols_to_logits_fn(dec_BxT, context, i):
            """Decode loop."""
            dec_Bx1 = tf.slice(dec_BxT, [0, tf.maximum(tf.cast(0, i.dtype), i - 1)], [dec_BxT.shape[0], 1])
            bias_1x1xT = tf.slice(bias_1xTxT, [0, i, 0], [1, 1, T])
            dec_Bx1xD = self._embedding_layer(dec_Bx1, True)
            dec_Bx1xD *= tf.cast(tf.greater(i, 0), self._dtype)
            dec_Bx1xD = add_time_signal(dec_Bx1xD, start_index=i)
            with tf.compat.v1.variable_scope(self._decoder_scope_name, reuse=tf.compat.v1.AUTO_REUSE):
                dec_Bx1xD = stack(
                    self._decoder_layers,
                    False,
                    dec_Bx1xD,
                    bias_1x1xT,
                    context["memory"],
                    context["memory_bias"],
                    context,
                    i,
                )
                dec_Bx1xD = self._layer_norm_decoder(dec_Bx1xD)
            logits_Bx1xV = self._embedding_layer(dec_Bx1xD, False)
            logits_BxV = tf.squeeze(logits_Bx1xV, axis=1)
            return logits_BxV

        decodes_BxT = left2right_decode(symbols_to_logits_fn, cache, B, T, V, beam_size, **beam_kwargs)
        return {"outputs": decodes_BxT}



class TFPegasusPretrainedModel(TFPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pre-trained models.
    """

    config_class = PegasusConfig
    base_model_prefix = "transformer"

    def __init__(
        self,
        config: PegasusConfig,

    ):
        #super().__init__(config)
        # vocab_size,
        # hidden_size,
        # filter_size,
        # num_heads,
        # num_encoder_layers,
        # num_decoder_layers,
        # label_smoothing,
        # dropout,
        # FIXME(SS): should take config
        #import ipdb; ipdb.set_trace()
        #self._dtype = tf.float32
        _assert_equal(type(self._dtype), type(tf.float32))
        self._embedding_layer = Embedding(config.vocab_size, config.d_model, "weights", self._dtype)
        #block_fn = lambda: TransformerBlock(c.d_model, c.encoder_ffn_dim, c.encoder_attention_heads, c.dropout)
        self._encoder_layers = [TransformerBlock(config.d_model, config.encoder_ffn_dim, config.encoder_attention_heads, config.dropout) for _ in range(config.encoder_layers)]
        self._decoder_layers = [TransformerBlock(config.d_model, config.decoder_ffn_dim, config.decoder_attention_heads, config.dropout) for _ in range(config.decoder_layers)]
        self._dropout_fn = (
            lambda x, training: tf.compat.v2.nn.dropout(x, config.dropout, noise_shape=[x.shape[0], 1, x.shape[2]])
            if training
            else x
        )
        self._vocab_size = config.vocab_size
        self._num_heads = config.encoder_attention_heads
        self._label_smoothing = 0.1
        self._decoder_scope_name = "decoder"
        self._layer_norm_encoder = tf.keras.layers.LayerNormalization(axis=2, epsilon=1e-12, name="LayerNorm")
        self._layer_norm_decoder = tf.keras.layers.LayerNormalization(axis=2, epsilon=1e-12, name="LayerNorm")
        _assert_equal(type(self._dtype), type(tf.float32))

    def _encode(self, features, training):
        inputs_BxI = features["inputs"]

        assert self._dtype == tf.float32, f'{self._dtype} != tf.float32'
        inputs_bias_Bx1xI = ids_to_bias(inputs_BxI)#, self._dtype)
        states_BxIxD = self._embedding_layer(inputs_BxI, True)
        states_BxIxD = self._dropout_fn(add_time_signal(states_BxIxD), training)
        with tf.compat.v1.variable_scope("encoder", reuse=tf.compat.v1.AUTO_REUSE):
            states_BxIxD = stack(self._encoder_layers, training, states_BxIxD, inputs_bias_Bx1xI, None, None)
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
        _assert_equal(type(self._dtype), type(tf.float32))
        context = self._encode(features, training)
        self._context = context
        targets_BxT = features["targets"]

        #import ipdb; ipdb.set_trace()
        bias_1xTxT = upper_triangle_bias(tf.shape(input=targets_BxT)[1], self._dtype)
        states_BxTxD = self._embedding_layer(targets_BxT, True)
        states_BxTxD = tf.pad(tensor=states_BxTxD, paddings=[[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        states_BxTxD = add_time_signal(states_BxTxD)
        states_BxTxD = self._dropout_fn(states_BxTxD, training)
        with tf.compat.v1.variable_scope(self._decoder_scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            states_BxTxD = stack(
                self._decoder_layers, training, states_BxTxD, bias_1xTxT, context["memory"], context["memory_bias"]
            )
            states_BxTxD = self._layer_norm_decoder(states_BxTxD)
        logits_BxTxV = self._embedding_layer(states_BxTxD, False)
        targets_mask_BxT = tf.cast(tf.greater(targets_BxT, 0), self._dtype)
        loss = tf.compat.v1.losses.softmax_cross_entropy(
            tf.one_hot(targets_BxT, self._vocab_size),
            logits_BxTxV,
            label_smoothing=self._label_smoothing,
            weights=targets_mask_BxT,
        )
        return loss, {"logits": logits_BxTxV}

    def predict(self, features, max_decode_len, beam_size, **beam_kwargs):
        """Predict."""
        cache = self._encode(features, False)
        B, _, D = cache["memory"].shape
        T, V, H = max_decode_len, self._vocab_size, self._num_heads

        bias_1xTxT = upper_triangle_bias(T, self._dtype)
        for i in range(len(self._decoder_layers)):
            cache[str(i)] = {
                "k": tf.zeros([B, H, T, D // H], self._dtype),
                "v": tf.zeros([B, H, T, D // H], self._dtype),
            }

        def symbols_to_logits_fn(dec_BxT, context, i):
            """Decode loop."""
            dec_Bx1 = tf.slice(dec_BxT, [0, tf.maximum(tf.cast(0, i.dtype), i - 1)], [dec_BxT.shape[0], 1])
            bias_1x1xT = tf.slice(bias_1xTxT, [0, i, 0], [1, 1, T])
            dec_Bx1xD = self._embedding_layer(dec_Bx1, True)
            dec_Bx1xD *= tf.cast(tf.greater(i, 0), self._dtype)
            dec_Bx1xD = add_time_signal(dec_Bx1xD, start_index=i)
            with tf.compat.v1.variable_scope(self._decoder_scope_name, reuse=tf.compat.v1.AUTO_REUSE):
                dec_Bx1xD = stack(
                    self._decoder_layers,
                    False,
                    dec_Bx1xD,
                    bias_1x1xT,
                    context["memory"],
                    context["memory_bias"],
                    context,
                    i,
                )
                dec_Bx1xD = self._layer_norm_decoder(dec_Bx1xD)
            logits_Bx1xV = self._embedding_layer(dec_Bx1xD, False)
            logits_BxV = tf.squeeze(logits_Bx1xV, axis=1)
            return logits_BxV

        decodes_BxT = left2right_decode(symbols_to_logits_fn, cache, B, T, V, beam_size, **beam_kwargs)
        return {"outputs": decodes_BxT}
