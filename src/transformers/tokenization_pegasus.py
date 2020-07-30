import tensorflow as tf
import tensorflow_text as tf_text

_SHIFT_RESERVED_TOKENS = 103
_NEWLINE_SYMBOL = "<n>"


def encode(text: tf.Tensor, max_len: int, vocab_filename: str, encoder_type: str):
    """EncodeOp."""
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


EOS_ID = 1
