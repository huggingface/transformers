from typing import TYPE_CHECKING, Any, Dict, List, Union
from .base import HfQuantizer

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_torch_available, logging
from ..integrations import replace_with_hqq_linear
from .quantizers_utils import get_module_from_name

if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


def is_hqq_available():
	available = True
	try:
		import hqq 
	except:
		available = False
	return available


try:
    from hqq.core.quantize import HQQLinear
except:
    HQQLinear = None


#
def autoname_modules(model):
    for name, module in model.named_modules():
        module.name = name

def find_parent(model, name):
    module_tree = name.split('.')[:-1]
    parent = model
    for m in module_tree:
        parent = parent._modules[m]
    return parent


class HQQHfQuantizer(HfQuantizer):
    """
	#TODO: 
    """

    use_keep_in_fp32_modules         = False  #False
    requires_parameters_quantization = True   #True
    requires_calibration             = False  #False
    required_packages                = ["hqq"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not (is_hqq_available()):
            raise ImportError("Using `HQQ` quantization requires `pip install hqq`")

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")

        device_map = kwargs.get("device_map", None)


    def check_quantized_param(
        self, model: "PreTrainedModel", param_value: "torch.Tensor", param_name: str, state_dict: Dict[str, Any]
    ) -> bool:
        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, torch.nn.Linear):
            return True
        else:
            return False

        return True

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: List[str],
    ):
        """
        TODO
        """

        module, tensor_name = get_module_from_name(model, param_name)
        layer_name          = param_name.replace('.weight', '').replace('.bias', '')

        if(type(module) is not torch.nn.Linear): 
            print(layer_name, 'not torch.nn.Linear')
            return 

        compute_dtype = torch.float16 #TODO coming from layer / torch_dtype


        #Create tmp linear layer
        tmp_linear_layer = torch.nn.Linear(in_features=module.in_features, out_features=module.out_features, bias=module.bias)
        tmp_layer_dict   = dict([(key.split('.')[-1], state_dict[key]) for key in state_dict if (layer_name in key)])
        tmp_linear_layer.load_state_dict(tmp_layer_dict)

        parent_module = find_parent(model, layer_name)
        node          = layer_name.split('.')[-1]

        if(hasattr(module, 'quant_config')):
            setattr(parent_module, node, HQQLinear(tmp_linear_layer, module.quant_config, compute_dtype=compute_dtype, device=target_device, del_orig=True))
        else:
            setattr(parent_module, node, tmp_linear_layer.to(target_device))

        del tmp_linear_layer

        #print('layer_name', layer_name, module.weight.device)

        


        import numpy as np
        # print('--------------------------------------------------------------------------------------------------------------------------')
        # print('model.model.embed_tokens', model.model.embed_tokens.weight.device.type)
        # print('model.lm_head', model.lm_head.weight.device.type)

        # print('model.model.layers[x].self_attn.q_proj', np.unique([layer.self_attn.q_proj.weight.device.type for layer in model.model.layers]))
        # print('model.model.layers[x].self_attn.k_proj', np.unique([layer.self_attn.k_proj.weight.device.type for layer in model.model.layers]))
        # print('model.model.layers[x].self_attn.v_proj', np.unique([layer.self_attn.v_proj.weight.device.type for layer in model.model.layers]))
        # print('model.model.layers[x].self_attn.o_proj', np.unique([layer.self_attn.o_proj.weight.device.type for layer in model.model.layers]))

        # print('model.model.layers[x].self_attn.rotary_emb.cos_cached', np.unique([layer.self_attn.rotary_emb.cos_cached.device.type for layer in model.model.layers]))
        # print('model.model.layers[x].self_attn.rotary_emb.sin_cached', np.unique([layer.self_attn.rotary_emb.sin_cached.device.type for layer in model.model.layers]))

        # print('model.model.layers[x].mlp.gate_proj', np.unique([layer.mlp.gate_proj.weight.device.type for layer in model.model.layers]))
        # print('model.model.layers[x].mlp.up_proj', np.unique([layer.mlp.up_proj.weight.device.type for layer in model.model.layers]))
        # print('model.model.layers[x].mlp.up_proj', np.unique([layer.mlp.up_proj.weight.device.type for layer in model.model.layers]))

        # print('model.model.layers[x].input_layernorm.weight', np.unique([layer.input_layernorm.weight.device.type for layer in model.model.layers]))
        # print('model.model.layers[x].post_attention_layernorm.weight', np.unique([layer.post_attention_layernorm.weight.device.type for layer in model.model.layers]))



    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.float16
        return torch_dtype

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer._process_model_before_weight_loading
    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        #from ..integrations import get_keys_to_not_convert
        #from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
        
        autoname_modules(model)

        #TOdo: how to get device from device_map
        device        = 'cuda'
        compute_dtype =  self.update_torch_dtype(None)
        model  = replace_with_hqq_linear(model, quantization_config=self.quantization_config, 
        										modules_to_not_convert=self.modules_to_not_convert, 
        										compute_dtype=compute_dtype, device=device)

        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        model.is_hqq_quantized = True
        model.is_hqq_serializable = self.is_serializable
        return model

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return True
