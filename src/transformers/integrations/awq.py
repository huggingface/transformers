# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"AWQ (Activation aware Weight Quantization) integration file"

import importlib

from packaging import version

from ..activations import ACT2FN
from ..modeling_utils import PreTrainedModel
from ..utils import is_gptqmodel_available, is_torch_available, logging
from ..utils.quantization_config import (
    AwqBackendPackingMethod,
    AwqConfig,
    AWQLinearVersion,
    ExllamaVersion,
)


if is_torch_available():
    import torch
    import torch.nn as nn

logger = logging.get_logger(__name__)

AWQ_FUSED_MAPPINGS = {
    "mistral": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
        "use_alibi": False,
    },
    "mixtral": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["w1", "w3", "w2"],
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
        "use_alibi": False,
        "rope_theta": 1000000.0,
    },
    "llama": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
        "use_alibi": False,
    },
    "llava": {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
        "use_alibi": False,
    },
}

AWQ_SCALES_MAPPINGS = {
    "starcoder2": {"act": "act", "layer_before_act": "c_fc"},
    "RefinedWebModel": {"act": "act", "layer_before_act": "dense_h_to_4h"},
    "falcon": {"act": "act", "layer_before_act": "dense_h_to_4h"},
    "mpt": {"act": "act", "layer_before_act": "up_proj"},
    "gptj": {"act": "act", "layer_before_act": "fc_in"},
    "gpt_neox": {"act": "act", "layer_before_act": "dense_h_to_4h"},
    "gpt_bigcode": {"act": "act", "layer_before_act": "c_fc"},
    "bloom": {"act": "gelu_impl", "layer_before_act": "dense_h_to_4h"},
}


def replace_quantization_scales(model, model_type):
    from gptqmodel.quantization.awq.modules.act import ScaledActivation

    if model_type not in AWQ_SCALES_MAPPINGS:
        return model
    for name, module in model.named_children():
        act_name = AWQ_SCALES_MAPPINGS[model_type]["act"]
        layer_before_act_name = AWQ_SCALES_MAPPINGS[model_type]["layer_before_act"]
        if name == act_name and hasattr(model, layer_before_act_name):
            layer_before_act = getattr(model, AWQ_SCALES_MAPPINGS[model_type]["layer_before_act"])
            size = layer_before_act.out_features
            scale_like = torch.ones(size)
            model._modules[name] = ScaledActivation(module, scale_like)
        _ = replace_quantization_scales(module, model_type)
    return model


def replace_with_awq_linear(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
    current_key_name=None,
    has_been_replaced=False,
) -> bool:
    """
    Public method that recursively replaces the Linear layers of the given model with AWQ quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successful or not.

    During the module replacement, we also infer the backend to use through the `quantization_config` object.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`AwqConfig`):
            The quantization config object that contains the quantization parameters.
        modules_to_not_convert (`list`, *optional*):
            A list of modules to not convert. If a module name is in the list (e.g. `lm_head`), it will not be
            converted.
        current_key_name (`list`, *optional*):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    backend = quantization_config.backend

    if not is_gptqmodel_available():
        raise ValueError(
            "AWQ (either `llmawq`) is not available. Please install it with `pip install gptqmodel` or check out the installation guide in https://github.com/mit-han-lab/llm-awq"
        )

    if backend == AwqBackendPackingMethod.AUTOAWQ:
        if quantization_config.version == AWQLinearVersion.GEMM:
            from gptqmodel.nn_modules.qlinear.awq_gemm import AwqGEMMQuantLinear

            target_cls = AwqGEMMQuantLinear
        elif quantization_config.version == AWQLinearVersion.GEMV:
            from gptqmodel.nn_modules.qlinear.awq_gemv import AwqGEMVQuantLinear

            target_cls = AwqGEMVQuantLinear
        elif quantization_config.version == AWQLinearVersion.EXLLAMA:
            if quantization_config.exllama_config["version"] == ExllamaVersion.ONE:
                from gptqmodel.nn_modules.qlinear.awq_exllama import AwqExllamaQuantLinear

                target_cls = AwqExllamaQuantLinear
            elif quantization_config.exllama_config["version"] == ExllamaVersion.TWO:
                from gptqmodel.nn_modules.qlinear.awq_exllamav2 import AwqExllamaV2QuantLinear

                target_cls = AwqExllamaV2QuantLinear
            else:
                raise ValueError(f"Unrecognized Exllama version: {quantization_config.exllama_config['version']}")
        elif quantization_config.version == AWQLinearVersion.IPEX:
            from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqQuantLinear

            target_cls = TorchFusedAwqQuantLinear
        else:
            raise ValueError(f"Unrecognized AWQ version: {quantization_config.version}")
    else:
        from awq.quantize.qmodule import WQLinear

        target_cls = WQLinear

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                in_features = module.in_features
                out_features = module.out_features

                model._modules[name] = target_cls(
                    bits=quantization_config.bits,
                    sym=quantization_config.sym,
                    desc_act=quantization_config.desc_act,
                    group_size=quantization_config.group_size,
                    in_features=in_features,
                    out_features=out_features,
                    bias=module.bias is not None,
                    dev=module.weight.device,
                    register_buffers=True,
                )
                has_been_replaced = True

                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_awq_linear(
                module,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def post_init_awq_exllama_modules(model, exllama_config):
    """
    Runs post init for Exllama layers which performs:
        - Weights unpacking, reordering and repacking
        - Devices scratch space allocation
    """

    if exllama_config["version"] == ExllamaVersion.ONE:
        from gptqmodel.quantization.awq.modules.linear.exllama import exllama_post_init

        model = exllama_post_init(model)
    elif exllama_config["version"] == ExllamaVersion.TWO:
        from gptqmodel.quantization.awq.modules.linear.exllamav2 import exllamav2_post_init

        model = exllamav2_post_init(
            model,
            max_input_len=exllama_config["max_input_len"],
            max_batch_size=exllama_config["max_batch_size"],
        )
    else:
        raise ValueError(f"Unrecognized Exllama version: {exllama_config['version']}")

    return model


def post_init_awq_ipex_modules(model):
    """
    Runs post init for IPEX layers which performs:
        - Weights packing, reordering and repacking
    """

    from gptqmodel.quantization.awq.modules.linear.gemm_ipex import ipex_post_init

    model = ipex_post_init(model)

    return model
