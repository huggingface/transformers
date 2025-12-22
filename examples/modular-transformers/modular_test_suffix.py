import torch.nn as nn

from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class TestSuffixDecoderLayer(nn.module):
    pass


# Here, we want to add "Llama" as a suffix to the base `TestModel` name for all required dependencies
class TestSuffixLlamaDecoderLayer(LlamaDecoderLayer):
    pass
