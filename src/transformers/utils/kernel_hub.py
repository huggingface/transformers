from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict


if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class _Architecture:
    device_type: str

    def __eq__(self, other):
        return isinstance(other, _Architecture) and self.device_type == other.device_type

    def __hash__(self):
        return hash(self.device_type)


@dataclass
class LayerRepository:
    """
    Repository and name of a layer.
    """

    layer_name: str = field(metadata={"help": "The name of the layer in the kernel repository."})
    repo_id: str = field(metadata={"help": "The kernel hub repository with the layer."})
    revision: str = field(default="main", metadata={"help": "The revision of the layer."})


_KERNEL_MAPPING: Dict[str, Dict[_Architecture, LayerRepository]] = {}


def register_layer_mapping(kernel_name: str, *, device_type: str, layer_repository: LayerRepository):
    """
    Register a layer mapping.

    This function regiters a mapping from a layer identifier and device type
    to a layer in a kernel repository.
    """
    _KERNEL_MAPPING.setdefault(kernel_name, {})[_Architecture(device_type=device_type)] = layer_repository


def replace_hub_layer_forward(cls, layer_name: str, *, use_fallback: bool = True):
    """
    Replace the forward function of a layer using a layer from the kernel hub.

    This function monkeypatches a layer, replacing the `forward` method
    of the layer with that of a layer from the hub. The replacement is done
    when a layer matching `layer_name` and device type is registered through
    `register_layer_mapping`. The device type is inferred from the first
    argument to `forward`.
    """

    fallback_forward = cls.forward

    cached_forward = {}

    def forward(self, x, **args):
        kernel = _KERNEL_MAPPING.get(layer_name)
        if kernel is None:
            if not use_fallback:
                raise ValueError(f"No layer mapping for `{layer_name}`")
            return fallback_forward(self, x, **args)

        device = getattr(x, "device", None)
        if device is None:
            return fallback_forward(self, x, **args)

        # Short-circuit if we already loaded the layer.
        arch = _Architecture(device_type=device.type)
        layer_forward = cached_forward.get(arch, None)
        if layer_forward is not None:
            return layer_forward(self, x, **args)

        repo = kernel.get(arch)
        if repo is None:
            if not use_fallback:
                raise ValueError(f"No layer mapping for `{layer_name}` with device type `{device.type}`")
            return fallback_forward(self, x, **args)

        try:
            layer = _get_kernel_layer(
                repo_id=repo.repo_id,
                layer_name=repo.layer_name,
                revision=repo.revision,
            )
            layer_forward = layer.forward
            cached_forward[arch] = layer_forward
        except Exception as e:
            if not use_fallback:
                raise e
            layer_forward = fallback_forward

        return layer_forward(self, x, **args)

    cls.forward = forward


def use_hub_layer_forward(layer_name: str, *, use_fallback: bool = True):
    """
    Replace the forward function of a layer using a layer from the kernel hub.

    This decorator can be applied to a layer and replaces the forward method
    of the layer with that of a layer from the hub. The replacement is done
    when a layer matching `layer_name` and device type is registered through
    `register_layer_mapping`. The device type is inferred from the first
    argument to `forward`.
    """

    def decorator(cls):
        replace_hub_layer_forward(cls, layer_name, use_fallback=use_fallback)
        return cls

    return decorator


def _get_kernel_layer(*, repo_id: str, layer_name: str, revision: str) -> "nn.Module":
    """Get a layer from a kernel."""

    from kernels import get_kernel
    from torch import nn

    kernel = get_kernel(repo_id, revision=revision)

    if getattr(kernel, "layers", None) is None:
        raise ValueError(f"Kernel `{repo_id}` at revision `{revision}` does not define any layers.")

    layer = getattr(kernel.layers, layer_name, None)
    if layer is None:
        raise ValueError(f"Layer `{layer_name}` not found in kernel `{repo_id}`.")
    if not issubclass(layer, nn.Module):
        raise TypeError(f"Layer `{layer_name}` is not a Torch layer.")
    return layer
