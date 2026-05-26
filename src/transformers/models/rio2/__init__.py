# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team and the Rio2 contributors.
# Licensed under the Apache License, Version 2.0.

import sys

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

_import_structure = {
    "configuration_rio2": ["Rio2Config"],
    "processing_rio2": ["Rio2Processor"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_rio2"] = [
        "Rio2Model",
        "Rio2Output",
        "Rio2PreTrainedModel",
    ]
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
