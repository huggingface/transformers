# Follows OLMo's HF template

"""
OpenLM configuration
"""

from transformers import AutoConfig, PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class OpenLMConfig(PretrainedConfig):
    model_type = "openlm"

    def __init__(self, **kwargs):
        kwargs["architectures"] = ["OpenLMForCausalLM"]
        super().__init__(**kwargs)
