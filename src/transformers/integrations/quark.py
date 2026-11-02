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
"Quark integration file"

from ..core_model_loading import ConversionOps
from ..utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn


class QuarkDequantize(ConversionOps):
    """
    Dense weights from the serialized `weight` and its `weight_scale` / `weight_zero_point`: quark's weight quantizer
    of a `QParamsLinear` built for the layer unpacks and dequantizes them, as in its forward.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, list[torch.Tensor]]:
        from quark.torch.export.main_export.quant_config_parser import get_layer_quant_config
        from quark.torch.export.nn.modules.qparamslinear import QParamsLinear

        from ..quantizers.quantizer_quark import CHECKPOINT_KEYS

        config = self.hf_quantizer.quantization_config
        module_name = full_layer_name.rpartition(".")[0]
        module = model.get_submodule(module_name)
        layer_config = get_layer_quant_config(config.quant_config, nn.Linear, module_name)
        output = {}
        for key, weights in input_dict.items():
            if not key.endswith("weight"):
                continue
            prefix = key.removesuffix("weight")
            # The state of the `QParamsLinear`, e.g. `weight_quantizer.scale`
            state = {
                CHECKPOINT_KEYS.get(name.removeprefix(prefix), name.removeprefix(prefix)): value
                for name, value in input_dict.items()
                if name.startswith(prefix) and name != key
            }
            weights = weights if isinstance(weights, list) else [weights]
            if not state:  # not quantized
                output[key] = weights
                continue
            output[key] = []
            for i, weight in enumerate(weights):
                # Without a linear (e.g. experts merged by a model converter), only unpacked weights have the dense shape
                in_features = getattr(module, "in_features", weight.shape[1])
                out_features = getattr(module, "out_features", weight.shape[0])
                with torch.device("meta"):
                    linear = nn.Linear(in_features, out_features, bias=False)
                with torch.device(weight.device):
                    qlinear = QParamsLinear(
                        linear, config.custom_mode, config.json_export_config.pack_method, layer_config
                    )
                params = {name: value[i] if isinstance(value, list) else value for name, value in state.items()}
                qlinear.load_state_dict({"weight": weight, **params}, strict=False, assign=True)
                output[key].append(qlinear._get_qweight(qlinear.weight))
        return output
