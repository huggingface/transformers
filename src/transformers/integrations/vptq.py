# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"VPTQ (Vector Post-Training Quantization) integration file"

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn

logger = logging.get_logger(__name__)


class VptqDequantize(ConversionOps):
    """
    Dequantized weights from the VPTQ `indices` into the `centroids`, with the optional outliers, residuals, permutation and
    norm, through the torch `vptq.ops.dequant` (what `vptq.ops.quant_gemm` runs without the CUDA kernels).
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, list[torch.Tensor]]:
        from vptq.ops import dequant

        if "indices" not in input_dict:
            # A model converter (e.g. merging experts) also collected the companions: unsupported, its sources load as before
            names = self.hf_quantizer.serialized_names["weight"]
            return {key: value for key, value in input_dict.items() if not key.endswith(tuple(names))}
        config = self.hf_quantizer.quantization_config
        module_name = full_layer_name.rpartition(".")[0]
        params = config.config_for_layers.get(module_name) or config.shared_layer_config.get(
            module_name.rsplit(".")[1]
        )
        out_features = model.get_submodule(module_name).out_features
        tensors = {name: value[0] if isinstance(value, list) else value for name, value in input_dict.items()}
        (outlier_vector_len, vector_len), (num_outlier_centroids, num_centroids) = (
            params["vector_lens"],
            params["num_centroids"],
        )
        num_res_centroids = params["num_res_centroids"][1]
        enable_outlier = outlier_vector_len > 1 and num_outlier_centroids > 0
        weight = dequant(
            indices=tensors["indices"],
            centroids=tensors["centroids.weight"].view(params["group_num"], num_centroids, vector_len),
            outlier_indices=tensors.get("outlier_indices"),
            outlier_centroids=tensors.get("outlier_centroids.weight"),
            res_indices=tensors.get("res_indices"),
            res_centroids=tensors.get("res_centroids.weight"),
            perm=tensors.get("perm"),
            weight_scale=tensors.get("weight_scale"),
            weight_bias=tensors.get("weight_bias"),
            is_indice_packed=True,
            enable_outlier=enable_outlier,
            enable_residual=num_res_centroids > 0,
            enable_perm=params["enable_perm"],
            enable_norm=params["enable_norm"],
            num_centroids=num_centroids,
            num_outlier_centroids=num_outlier_centroids,
            num_res_centroids=num_res_centroids,
            padding=-out_features % vector_len,
            outlier_padding=-out_features % outlier_vector_len if enable_outlier else 0,
            num_codebooks=params["group_num"],
            group_size=params["group_size"],
            outlier_size=params["outlier_size"],
            vector_len=vector_len,
            outlier_vector_len=outlier_vector_len,
        )
        return {"weight": [weight]}


def replace_with_vptq_linear(model, modules_to_not_convert: list[str] | None = None, quantization_config=None):
    """
    Public method that replaces the Linear layers of the given model with SPQR quantized layers.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        modules_to_not_convert (`list[str]`, *optional*, defaults to `None`):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        quantization_config (`VptqConfig`):
            The quantization config object that contains the quantization parameters.
    """
    from vptq import VQuantLinear

    has_been_replaced = False
    shared_layer_config = quantization_config.shared_layer_config
    config_for_layers = quantization_config.config_for_layers

    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        with torch.device("meta"):
            if isinstance(module, nn.Linear):
                layer_params = config_for_layers.get(module_name, None) or shared_layer_config.get(
                    module_name.rsplit(".")[1], None
                )
                new_module = VQuantLinear(
                    module.in_features,
                    module.out_features,
                    vector_lens=layer_params["vector_lens"],
                    num_centroids=layer_params["num_centroids"],
                    num_res_centroids=layer_params["num_res_centroids"],
                    group_num=layer_params["group_num"],
                    group_size=layer_params["group_size"],
                    outlier_size=layer_params["outlier_size"],
                    indices_as_float=layer_params["indices_as_float"],
                    enable_norm=layer_params["enable_norm"],
                    enable_perm=layer_params["enable_perm"],
                    is_indice_packed=True,
                    enable_proxy_error=False,
                    bias=module.bias is not None,
                )
                # Force requires grad to False to avoid unexpected errors
                model._modules[module_name].requires_grad_(False)
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using eetq but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
