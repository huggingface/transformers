# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""The fine-grained quantizer: a model whose modules are in different block-scaled formats, each part loaded by its
own format's quantizer."""

from typing import TYPE_CHECKING

from ..utils.quantization_config import BlockFp8Config
from .block_scaled_utils import BlockScaledQuantizer
from .quantizers_utils import embedding_pattern, get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import FineGrainedConfig


class FineGrainedHfQuantizer(BlockScaledQuantizer):
    """Hosts one quantizer per entry of the config's `dtype_plan`, each the one its format config names in
    `AUTO_QUANTIZER_MAPPING`: what the formats share runs once over the whole plan, and each module's tensors take
    the ops of the quantizer its entry belongs to. DeepSeek-V4 is MXFP4 experts over block-FP8 linears; a modelopt
    export can mix NVFP4 and MXFP8."""

    quantization_config: "FineGrainedConfig"

    def __init__(self, quantization_config: "FineGrainedConfig", **kwargs):
        super().__init__(quantization_config, **kwargs)
        from .auto import AUTO_QUANTIZER_MAPPING

        # the embedding tables `modules_to_convert` names hold FP8 rows with one per-tensor scale
        self._embedding_config = BlockFp8Config(weight_block_size=None)
        configs = {id(config): config for config in self.plan().values()}.values()
        self._members = [AUTO_QUANTIZER_MAPPING[config.method.value](config, **kwargs) for config in configs]
        # the plan's embedding tables are the host's to name
        for member in self._members:
            member.embeddings = self.embeddings

    def plan(self):
        return {
            **dict.fromkeys(map(embedding_pattern, self.embeddings), self._embedding_config),
            **self.quantization_config.dtype_plan,
        }

    def members(self):
        return self._members

    def get_param_ops(self, model: "PreTrainedModel", param_name: str) -> list:
        member = self.member_of(get_module_from_name(model, param_name)[0])
        return member.get_param_ops(model, param_name) if member is not None else []
