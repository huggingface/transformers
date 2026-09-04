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
"""Keeping GGUF weights in their blocks instead of unpacking them at load time."""

from ..utils import is_torch_available, is_torch_mps_available, logging
from ..utils.quantization_config import GgufConfig
from .base import HfQuantizer


if is_torch_available():
    import torch

    from ..integrations.gguf.kernels import get_gguf_kernel
    from ..integrations.gguf.reader import GgufHeader, load_gguf_state_dict, read_gguf_metadata
    from ..integrations.gguf.utils import (
        add_gguf_load_ops,
        get_gguf_conversion_mapping,
        get_gguf_plan,
        is_gguf_arch_supported,
        replace_with_gguf_modules,
    )


logger = logging.get_logger(__name__)


class GgufHfQuantizer(HfQuantizer):
    """Loads a quantized GGUF checkpoint with its weights left in GGUF blocks."""

    quantization_config: "GgufConfig"
    header: "GgufHeader"  # set by `read_header`, before any of the loading hooks run
    requires_calibration = False
    requires_parameters_quantization = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.pre_quantized = True
        self.packed_modules = {}
        self.input_permutations = {}
        self.quantized = {}
        self.names = []
        self.mapping = []
        self.header = None
        self.kernel = None
        self.dtype = None
        # TODO: only for the legacy loader — drop this, and every hook that guards on it, once all
        # architectures go through this path and there is no fallback left
        self.supported = False

    def validate_environment(self, *args, **kwargs):
        if not self.supported:
            return
        if self.quantization_config.dequantize:
            return
        if not self.header.has_quantized_weights:
            # Nothing is in blocks, so there is nothing to keep packed and nothing to unpack: this is
            # the dequantized path already, on any device. Setting the flag is what leaves it an
            # ordinary dense model, and no fallback happened, so there is nothing to warn about.
            self.quantization_config.dequantize = True
            return
        self.kernel = get_gguf_kernel()
        if not self.kernel:
            self.quantization_config.dequantize = True
            logger.warning("No GGUF matmul kernel is available for this device. We will dequantize the entire model.")
            return

    def update_device_map(self, device_map):
        """Default to the backend the blocks are computed on, rather than the host."""
        # Rejected whatever the caller asked for, dequantized loads included, because this is not about
        # the blocks: a GGUF is one memory-mapped file, so there is no per-layer shard for the offload
        # machinery to leave on disk and page back in.
        if "disk" in {str(place) for place in getattr(device_map, "values", lambda: [device_map])()}:
            raise RuntimeError(
                "One or more modules is configured to be mapped to disk. Disk offload is not supported "
                "for models loaded from GGUF files."
            )
        if self.quantization_config.dequantize:
            return device_map
        if device_map is None and is_torch_mps_available():
            device_map = {"": torch.device("mps")}
            logger.info(f"No `device_map` was passed; loading the GGUF weights on {device_map['']}.")
        return device_map

    def read_header(self, gguf_file: str):
        """Parse the file's metadata, once `from_pretrained` knows where the file is."""
        metadata, _ = read_gguf_metadata(gguf_file)
        self.supported = is_gguf_arch_supported(metadata["general.architecture"])
        if self.supported:
            self.header = GgufHeader.from_file(gguf_file)
            self.validate_environment()

    def update_dtype(self, dtype):
        """Settle the dtype the model is loaded in, and keep it."""
        if dtype is None:
            if not self.supported:
                self.dtype = torch.get_default_dtype()
                return self.dtype
            dtype = self.header.dtype if self.header.dtype is not None else torch.float32
        # The kernels take f32 activations, so another dtype is cast in and out of every matmul: 9.3
        # tok/s against 40.7 on Qwen3.5-35B-A3B. The weights are quantized either way.
        if not self.quantization_config.dequantize and dtype != torch.float32:
            logger.warning(
                f"Loading a GGUF checkpoint with its weights packed is fastest in float32, and this "
                f"one was asked for in {dtype}. The kernels compute in float32, so every matmul will "
                f"cast its input and its output. Pass `dtype=torch.float32` to avoid it -- the weights "
                f"stay quantized regardless, so this costs no extra memory."
            )
        self.dtype = dtype
        return dtype

    def get_state_dict(self, checkpoint_file: str, model):
        """The file's tensors, quantized ones kept as raw blocks."""
        if self.supported:
            return load_gguf_state_dict(self.header)

        from ..modeling_gguf_pytorch_utils import load_gguf_checkpoint

        legacy = load_gguf_checkpoint(
            checkpoint_file, return_tensors=True, model_to_load=model, torch_dtype=self.dtype
        )
        return legacy["tensors"]

    def _process_model_before_weight_loading(self, model, **kwargs):
        """Swap in `GgufLinear` wherever the weight can stay packed."""
        if not self.supported:
            return
        self.mapping = get_gguf_conversion_mapping(self.header.architecture, model.config)
        self.quantized, packable, self.input_permutations, self.names = get_gguf_plan(self.header, self.mapping)
        if self.quantization_config.dequantize:
            return
        self.packed_modules = replace_with_gguf_modules(model, packable, self.kernel, self.dtype)

    def _process_model_after_weight_loading(self, model, **kwargs):
        """Fill in what needs the weights already in place: the input permutations, then the layer kernels."""
        if not self.supported:
            return model
        from ..integrations.gguf.kernels import kernelize_ggml_layers

        for param_name, module in self.packed_modules.items():
            permutation = self.input_permutations.get(param_name)
            if permutation is not None:
                module.input_permutation = permutation.to(module.weight.device)
        kernelize_ggml_layers(model)
        # `save_pretrained` writes a checkpoint back by reversing whatever loaded the model, and this
        # mapping does not reverse: `Dequantize` has no inverse worth defining, and nobody wants a GGUF
        # written back. Dropped once the weights are in place, so saving falls back on the model's own
        # mapping -- which is what it needs, a dequantized model being an ordinary dense one. A model
        # that kept its blocks never reaches that code: `is_serializable` refuses it first.
        model._weight_conversions = None
        return model

    def update_weight_conversions(self, weight_conversions):
        """Prepend this file's conversions: the GGUF -> transformers mapping, and the unpacking."""
        if not self.supported:
            return weight_conversions
        to_unpack = {name: t for name, t in self.quantized.items() if name not in self.packed_modules}
        return add_gguf_load_ops(self.mapping + weight_conversions, to_unpack, self.names, self.dtype)

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
