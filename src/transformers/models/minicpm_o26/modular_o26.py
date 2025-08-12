
from typing import Callable, Optional

import torch
import torch.utils.checkpoint

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import LossKwargs, logging
from ..gemma.modeling_gemma import GemmaMLP
from ..llama.modeling_llama import (
    LlamaAttention,
)
from ..qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2ForQuestionAnswering,
    Qwen2ForSequenceClassification,
    Qwen2ForTokenClassification,
    Qwen2Model,
    Qwen2RMSNorm,
    Qwen2Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_minicpm_o26 import MiniCPM_o_2_6Config


# class MiniCPM_o_2_6BareModel(Qwen2Model):
#     pass


class MiniCPM_o_2_6ForModel(Qwen2ForCausalLM):
    pass


__all__ = [
    "MiniCPM_o_2_6Model",
]
