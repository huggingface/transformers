from .configuration_mbart import MBartConfig
from .file_utils import add_start_docstrings
from .modeling_bart import BartForConditionalGeneration


_CONFIG_FOR_DOC = "MBartConfig"
_TOKENIZER_FOR_DOC = "MBartTokenizer"

MBART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mbart-large-cc25",
    "facebook/mbart-large-en-ro",
    # See all multilingual BART models at https://huggingface.co/models?filter=mbart
]

MBART_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.MBartConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for machine translation.", MBART_START_DOCSTRING
)
class MBartForConditionalGeneration(BartForConditionalGeneration):
    r"""
    This class overrides :class:`~transformers.BartForConditionalGeneration`. Please check the
    superclass for the appropriate documentation alongside usage examples.

    Examples::
        >>> from transformers import MBartForConditionalGeneration, MBartTokenizer
        >>> model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
        >>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro")
        >>> article = "UN Chief Says There Is No Military Solution in Syria"
        >>> batch = tokenizer.prepare_seq2seq_batch(src_texts=[article])
        >>> translated_tokens = model.generate(**batch)
        >>> translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        >>> assert translation == "Şeful ONU declară că nu există o soluţie militară în Siria"
    """

    config_class = MBartConfig
