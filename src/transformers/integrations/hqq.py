
#Todo
#from ..utils import is_hqq_available

from ..utils import is_accelerate_available, logging

import torch
import torch.nn as nn

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import find_tied_parameters

logger = logging.get_logger(__name__)



def is_hqq_available():
	available = True
	try:
		import hqq 
	except:
		available = False
	return available

def autoname_modules(model):
	for name, module in model.named_modules():
		module.name = name

def _replace_with_hqq_linear(model, quantization_config, modules_to_not_convert, compute_dtype, device, has_been_replaced, current_key_name=None):
	if not is_hqq_available():
	    raise ValueError("HQQ is not available. Please install it with `pip install hqq`")

	modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert

	from hqq.core.quantize import HQQLinear

	#Autoname modules 
	#autoname_modules(model)

	for name, module in model.named_children():
		if current_key_name is None:
			current_key_name = []
		current_key_name.append(name)

		if (isinstance(module, nn.Linear)) and name not in modules_to_not_convert:
			# Check if the current key is not in the `modules_to_not_convert`
			if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):

				#TODO: FIGURE ACCELERATE THING
				print('Processing ', name)

				#with init_empty_weights():
				model._modules[name] = HQQLinear(module, quantization_config.to_dict(), compute_dtype=compute_dtype, device=device)

				has_been_replaced = True

				# Store the module class in case we need to transpose the weight later
				model._modules[name].source_cls = type(module)
				# Force requires grad to False to avoid unexpected errors
				model._modules[name].requires_grad_(False)

		if len(list(module.children())) > 0:
			_, has_been_replaced = _replace_with_hqq_linear(
				module,
				quantization_config,
				modules_to_not_convert,
				compute_dtype,
				device,
				has_been_replaced=has_been_replaced,
			)
		# Remove the last key for recursion
		current_key_name.pop(-1)

	return model, has_been_replaced


def replace_with_hqq_linear(model, quantization_config=None, modules_to_not_convert=None, compute_dtype=torch.float16, device='cuda', has_been_replaced=False):
    """
	TODO
    """

    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    model, has_been_replaced = _replace_with_hqq_linear(model, quantization_config=quantization_config, 
    														   modules_to_not_convert=modules_to_not_convert, 
    														   compute_dtype=compute_dtype, 
    														   device=device,
															   has_been_replaced=has_been_replaced, )

    if not has_been_replaced:
        logger.warning("No linear modules were found in your model for quantization.")

    return model




