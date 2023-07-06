""" LaVIN model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

logger = logging.get_logger(__name__)

LAVIN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shauray/lavin": "https://huggingface.co/shauray/lavin/blob/main/config.json"
}

class lavin_config(PretrainedConfig):
  model_type = "lavin"
  keys_to_ignore_at_inference = ["past_key_values"]
  attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

  def __init__(self, vocab_size=2400, d_model=512, d_kv=64, d_ff=2048, ..., **kwargs):
    self.vocab_size = vocab_size
    super().__init__(pad_token_id=pad_token_id, **kwargs)


