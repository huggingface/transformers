import torch

from transformers.models.llama.modeling_llama import LlamaModel


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 4]
    x2 = x[..., x.shape[-1] // 4 :]
    return torch.cat((-x2, x1), dim=-1)


# example where we need some deps and some functions
class DummyModel(LlamaModel):
    pass
