# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_tf_available, is_torch_available
from .configuration_mt5 import MT5Config


if is_torch_available():
    from .modeling_mt5 import MT5ForConditionalGeneration, MT5Model

if is_tf_available():
    from .modeling_tf_mt5 import TFMT5ForConditionalGeneration, TFMT5Model
