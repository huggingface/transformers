import torch
from torch.nn import init


def uniform_(tensor: torch.Tensor, a: float = 0.0, b: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return init.uniform_(tensor, a=a, b=b, generator=generator)
    return tensor


def normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return init.normal_(tensor, mean=mean, std=std, generator=generator)
    return tensor


def constant_(tensor: torch.Tensor, val: float) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return init.constant_(tensor, val=val)
    return tensor


def ones_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return init.ones_(tensor)
    return tensor


def zeros_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return init.zeros_(tensor)
    return tensor


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return init.xavier_uniform_(tensor, gain=gain, generator=generator)
    return tensor


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return init.xavier_normal_(tensor, gain=gain, generator=generator)
    return tensor


def kaiming_uniform_(tensor: torch.Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu", generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity, generator=generator)
    return tensor


def kaiming_normal_(tensor: torch.Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu", generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity, generator=generator)
    return tensor


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b, generator=generator)
    return tensor


def orthogonal_(tensor: torch.Tensor, gain: float = 1, generator: torch.Generator | None = None,) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return init.orthogonal_(tensor, gain=gain, generator=generator)
    return tensor


def copy_(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        with torch.no_grad():
            return tensor.copy_(other)
    return tensor