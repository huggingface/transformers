from .configuration_xlm_prophetnet import XLMProphetNetConfig
from .modeling_prophetnet import ProphetNetForConditionalGeneration, ProphetNetModel
from .utils import logging


logger = logging.get_logger(__name__)

_TOKENIZER_FOR_DOC = "XLMProphetNetTokenizer"

XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/xprophetnet-large-wiki100-cased",
    # See all ProphetNet models at https://huggingface.co/models?filter=xprophetnet
]


class XLMProphetNetModel(ProphetNetModel):
    """
    This class overrides :class:`~transformers.ProphetNetModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMProphetNetConfig


class XLMProphetNetForConditionalGeneration(ProphetNetForConditionalGeneration):
    """
    This class overrides :class:`~transformers.ProphetNetForConditionalGeneration`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMProphetNetConfig
