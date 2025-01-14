import torch.nn as nn

# Note that llama and cohere have different definitions for rotate_half
from transformers.models.cohere.modeling_cohere import rotate_half
from transformers.models.llama.modeling_llama import LlamaAttention


class SwitchFunctionAttention(LlamaAttention):
    pass
