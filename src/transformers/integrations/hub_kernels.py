from typing import Dict, Union

from kernels import (
    Device,
    LayerRepository,
    register_kernel_mapping,
    replace_kernel_forward_from_hub,
    use_kernel_forward_from_hub,
)


_KERNEL_MAPPING: Dict[str, Dict[Union[Device, str], LayerRepository]] = {
    "MultiScaleDeformableAttention": {
        "cuda": LayerRepository(
            repo_id="kernels-community/deformable-detr",
            layer_name="MultiScaleDeformableAttention",
        )
    }
}

register_kernel_mapping(_KERNEL_MAPPING)

__all__ = [
    "LayerRepository",
    "use_kernel_forward_from_hub",
    "register_kernel_mapping",
    "replace_kernel_forward_from_hub",
]
