from ..bart.modeling_bart import BartForConditionalGeneration
from .configuration_mbart import MBartConfig


_CONFIG_FOR_DOC = "MBartConfig"
_TOKENIZER_FOR_DOC = "MBartTokenizer"

MBART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mbart-large-cc25",
    "facebook/mbart-large-en-ro",
    # See all multilingual BART models at https://huggingface.co/models?filter=mbart
]


class MBartForConditionalGeneration(BartForConditionalGeneration):
    r"""
    This class overrides :class:`~transformers.BartForConditionalGeneration`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::
        >>> from transformers import MBartForConditionalGeneration, MBartTokenizer
        >>> model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
        >>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro")
        >>> article = "UN Chief Says There Is No Military Solution in Syria"
        >>> batch = tokenizer.prepare_seq2seq_batch(src_texts=[article], return_tensors="pt")
        >>> translated_tokens = model.generate(**batch)
        >>> translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        >>> assert translation == "Şeful ONU declară că nu există o soluţie militară în Siria"
    """
    model_type = "mbart"
    config_class = MBartConfig
    _keys_to_ignore_on_load_missing = [
        "model.encoder.embed_positions.weight",
        "model.decoder.embed_positions.weight",
    ]
    _keys_to_ignore_on_save = [
        "model.encoder.embed_positions.weight",
        "model.decoder.embed_positions.weight",
    ]
