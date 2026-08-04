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
(`replace_with_gguf_modules`). What is left here are the loading hooks — running the swap before the
weights arrive, telling the reader which GGUF tensors to hand back as raw blocks, and re-tying
`lm_head` afterwards.

Anything with no packed module to hold it is dequantized at load, which is always correct, just
larger.
"""

from ..integrations.gguf.utils import (
    GgufEmbedding,
    get_gguf_plan,
    replace_with_gguf_modules,
    retie_gguf_lm_head,
)
from ..utils import is_gguf_available, logging
from ..utils.quantization_config import GgufConfig
from .base import HfQuantizer


logger = logging.get_logger(__name__)


class GgufHfQuantizer(HfQuantizer):
    """Loads a quantized GGUF checkpoint with its weights left in GGUF blocks."""

    quantization_config: "GgufConfig"
    requires_calibration = False
    requires_parameters_quantization = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.pre_quantized = True
        self.gguf_file = quantization_config.gguf_file
        self.keep_packed = set()
        self._embedding = None

    def validate_environment(self, *args, **kwargs):
        if not is_gguf_available():
            raise ImportError("Loading a GGUF checkpoint requires the `gguf` package. Run `pip install gguf`.")

    def _process_model_before_weight_loading(self, model, **kwargs):
        """Swap in `GgufLinear` wherever the weight can stay packed."""
        if self.quantization_config.dequantize:
            return  # asked for dense weights: leave every module alone and let the reader unpack
        plan = get_gguf_plan(self.gguf_file, model.config)
        replaced = replace_with_gguf_modules(model, plan, model.config.dtype)
        self.keep_packed = set(replaced)
        self._embedding = next((mod for mod in replaced.values() if isinstance(mod, GgufEmbedding)), None)

    def param_needs_quantization(self, model, param_name: str, **kwargs) -> bool:
        return False

    def _process_model_after_weight_loading(self, model, **kwargs):
        # Only relevant if the embedding itself is packed; the head is checked there.
        if self._embedding is not None:
            retie_gguf_lm_head(model, self._embedding)
        return model

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True

    def is_serializable(self, safe_serialization=None) -> bool:
        return False
