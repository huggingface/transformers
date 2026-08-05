# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keeping GGUF weights in their blocks instead of unpacking them at load time.

Both halves of the decision live in `integrations.gguf.utils`: which weights *can* stay packed
(`get_gguf_plan`) and which really do, because a module exists to hold them
(`replace_with_gguf_modules`). What is left here are two loading hooks — swapping those modules in
before the weights arrive, and contributing this file's conversions afterwards.

Every quantized tensor is read as raw blocks; anything with no packed module to hold them gets a
`Dequantize` conversion, which is always correct, just larger.
"""

import torch

from ..integrations.gguf.kernels import get_gguf_kernel
from ..integrations.gguf.reader import GgufHeader
from ..integrations.gguf.utils import (
    add_gguf_dequantize_ops,
    get_gguf_conversion_mapping,
    get_gguf_plan,
    is_gguf_arch_supported,
    replace_with_gguf_modules,
)
from ..utils import is_gguf_available, is_torch_mps_available, logging
from ..utils.quantization_config import GgufConfig
from .base import HfQuantizer


logger = logging.get_logger(__name__)


class GgufHfQuantizer(HfQuantizer):
    """Loads a quantized GGUF checkpoint with its weights left in GGUF blocks."""

    quantization_config: "GgufConfig"
    header: GgufHeader  # set by `read_header`, before any of the loading hooks run
    requires_calibration = False
    requires_parameters_quantization = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.pre_quantized = True
        self.gguf_file = quantization_config.gguf_file
        self.keep_packed = set()
        self.quantized = {}
        self.mapping = []
        self.kernel = None
        self.header = None
        self.dtype = None
        # TODO: only for the legacy loader — drop this, and the two checks that read it, once every
        # architecture goes through this path and there is no fallback left
        self.supported = False

    def validate_environment(self, *args, **kwargs):
        if not is_gguf_available():
            raise ImportError("Loading a GGUF checkpoint requires the `gguf` package. Run `pip install gguf`.")
        if self.quantization_config.dequantize:
            return  # dense weights were asked for, and unpacking them at load needs no device kernel
        # Blocks are only worth keeping where something can compute on them, so there has to be a backend
        # to compute on. Where the weights then go is the caller's business: the torch unpacking path runs
        # on the host too, and `update_device_map` only picks the device when no `device_map` says.
        if not torch.cuda.is_available() and not is_torch_mps_available():
            raise RuntimeError(
                "Loading a GGUF checkpoint with its weights left quantized requires a CUDA or MPS backend. "
                "Pass `dequantize=True` in the quantization config to unpack the weights once at load "
                "instead, which runs anywhere and costs the memory of an unquantized model."
            )
        self.kernel = get_gguf_kernel()
        if self.kernel is None:
            # The weights still stay packed: torch can unpack a block, so only the speed goes. Every
            # forward unpacks its own weight instead of computing on the blocks — about a third of the
            # speed here, for a third of the memory.
            logger.warning(
                "No GGUF matmul kernel is available for this device. The weights stay quantized, so the "
                "memory saving holds, but every forward unpacks them, which costs speed. Pass "
                "`GgufConfig(dequantize=True)` to unpack once at load and trade the memory back."
            )

    def update_device_map(self, device_map):
        """Default to the backend the blocks are computed on, rather than the host.

        Only when the caller named none: an explicit `device_map` is left alone, host included.
        `validate_environment` has already established that one of these backends is there.
        """
        if device_map is None and not self.quantization_config.dequantize:
            device_map = {"": torch.device("cuda" if torch.cuda.is_available() else "mps")}
            logger.info(f"No `device_map` was passed; loading the GGUF weights on {device_map['']}.")
        return device_map

    def read_header(self, gguf_file: str):
        """Parse the file's metadata, once `from_pretrained` knows where the file is.

        Not in `__init__`: `gguf_file` can be a name inside a repo until the checkpoint is resolved to
        a local path. Called before the dtype is settled, because `update_dtype` reads the file's own
        float type off this header.
        """
        self.gguf_file = gguf_file
        self.header = GgufHeader.from_file(gguf_file)
        self.supported = is_gguf_arch_supported(self.header)

    def update_dtype(self, dtype):
        """Settle the dtype the model is loaded in, and keep it.

        `None` arrives from a `dtype="auto"` load: nothing outside the file can answer it, since the
        config is rebuilt from that same file and carries no dtype. A file written in one float type is
        loaded in that type — the checkpoint decides, as `auto` does everywhere else.

        Kept because `update_weight_conversions` needs it for the tensors it has to dequantize.
        """
        if dtype is None:
            file_dtype = self.header.dtype
            # default to bf16 in case the model is quantized
            dtype = file_dtype if file_dtype is not None else torch.bfloat16
        self.dtype = dtype
        return dtype

    def _process_model_before_weight_loading(self, model, **kwargs):
        """Swap in `GgufLinear` wherever the weight can stay packed."""
        if not self.supported:
            return
        # Built here, where the config is in scope, and kept for `update_weight_conversions`: it is the
        # same mapping both need, and the plan has to read a converter's operations before the unpacking
        # op is inserted into them.
        self.mapping = get_gguf_conversion_mapping(self.header.architecture, model.config)
        self.quantized, packable = get_gguf_plan(self.header, self.mapping)
        if self.quantization_config.dequantize:
            return
        self.keep_packed = set(replace_with_gguf_modules(model, packable, self.kernel))

    def update_weight_conversions(self, weight_conversions):
        """Prepend this file's conversions: the GGUF -> transformers mapping, and the unpacking.

        Quantized tensors are loaded as raw blocks whatever their destination. The ones whose module
        holds blocks keep them; the rest get a `Dequantize` at the head of their conversion chain, which
        runs once the pipeline has moved the bytes to the parameter's device.
        """
        if not self.supported:
            return weight_conversions
        to_unpack = {name: t for name, t in self.quantized.items() if name not in self.keep_packed}
        return add_gguf_dequantize_ops(self.mapping + weight_conversions, to_unpack, self.dtype)

    def param_needs_quantization(self, model, param_name: str, **kwargs) -> bool:
        return False

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True

    def is_serializable(self, safe_serialization=None) -> bool:
        return False
