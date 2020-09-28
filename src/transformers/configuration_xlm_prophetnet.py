""" XLM-ProphetNet model configuration """


import logging

from .configuration_prophetnet import ProphetNetConfig


logger = logging.getLogger(__name__)

XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/xprophetnet-large-wiki100-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/microsoft/xprophetnet-large-wiki100-cased/config.json",
}


class XLMProphetNetConfig(ProphetNetConfig):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    model_type = "xlm-prophetnet"
