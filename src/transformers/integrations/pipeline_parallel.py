import torch
import torch.nn as nn
from functools import partial

from ..distributed import DistributedConfig
from ..utils.generic import GeneralInterface
from transformers.utils import is_torch_greater_or_equal

_torch_distributed_available = torch.distributed.is_available()

if is_torch_greater_or_equal("2.5") and _torch_distributed_available:
    from torch.distributed.tensor import DTensor, Replicate

class PipelineParallelLayer:
    """
    General tensor parallel layer for transformers.
    """
    use_dtensor = True

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh): ...

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh): ...

    def partition_tensor(self, param, empty_param, param_type, param_casting_dtype, to_contiguous, rank, device_mesh):
        raise NotImplementedError

    def prepare_module_pp(self, module: nn.Module, device_mesh) -> nn.Module:
        if self.use_dtensor:
            distribute_module(
                module,
                device_mesh,
                partial(self._prepare_input_fn),
                partial(self._prepare_output_fn),
            )

class SendForwardLayer(PipelineParallelLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _prepare_input_fn(self, mod, inputs, device_mesh):
        return inputs
    
    def _prepare_output_fn(self, mod, outputs, device_mesh):
        if isinstance(outputs[0], DTensor):
            return (outputs[0].to_local(),) + outputs[1:]
        return outputs

class RecvForwardLayer(PipelineParallelLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _prepare_input_fn(self, mod, inputs, device_mesh):
        if not isinstance(inputs[0], DTensor):
            # Assumes the input is replicated on the TP mesh
            tp_device_mesh = mod.weight.device_mesh.parent
            replicated_placement = [Replicate() for _ in range(tp_device_mesh.ndim)]
            dtensor_input = DTensor.from_local(inputs[0], tp_device_mesh, replicated_placement, run_check=False)
            return (dtensor_input,) + inputs[1:]
        return inputs
    
    def _prepare_output_fn(self, mod, outputs, device_mesh):
        return outputs

class ParallelInterface(GeneralInterface):
    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given entry)
    _global_mapping = (
        {
            "send_forward": SendForwardLayer(),
            "recv_forward": RecvForwardLayer(),
        }
        if is_torch_greater_or_equal("2.5") and _torch_distributed_available
        else {}
    )


ALL_PARALLEL_STYLES: ParallelInterface = ParallelInterface()

def distribute_module(
    module: nn.Module,
    device_mesh=None,
    input_fn=None,
    output_fn=None,
) -> nn.Module:
    """
    Copy pasted from torch's function but we remove the communications (partitioning)
    as well as buffer registering that is similarly not efficient.
    """
    if len(module._forward_pre_hooks) == 0:
        if input_fn is not None:
            module.register_forward_pre_hook(lambda mod, inputs: input_fn(mod, inputs, device_mesh))
        if output_fn is not None:
            module.register_forward_hook(lambda mod, inputs, outputs: output_fn(mod, outputs, device_mesh))
    return module

def add_pipeline_parallel_hooks_to_module(
    model, module, pp_plan, layer_name, current_module_plan, device_mesh, parameter_name=None
):
    r"""
    This function is called in `PretrainedModel.post_init()`. It is responsible of adding hooks
    to the modules of the `model`, based on the `PretrainedModel._tp_plan`.

    This is the place where we add the `pre_forward` and `post_forwards` hooks. These are defined
    for each `TensorParallelLayer` as `_prepare_input_fn` and `_prepare_output_fn`.

    """
    if current_module_plan is not None:
        pp_layer = ALL_PARALLEL_STYLES[current_module_plan]
        try:
            pp_layer.prepare_module_pp(module, device_mesh)
        except NotImplementedError as e:
            print(
                f"Trying to prepare {layer_name}, but it's not supported. Corresponding module: {module} Fix it's PP plan: {e}"
            )

        module._hf_pp_plan = current_module_plan
        original_repr = module.__repr__
        module.__repr__ = lambda: f"{original_repr()}\nPP Plan: {current_module_plan}"

def distribute_model_on_pp_axis(model, distributed_config, device_mesh):
    model._pp_size = device_mesh.size()
    model._pp_device_mesh = device_mesh
    if distributed_config is not None:
        if isinstance(distributed_config, dict):
            distributed_config = DistributedConfig.from_dict(distributed_config)
        model.config.distributed_config = distributed_config
    
    #TODO(3outeille): for now dont care about pp_plan, just do the most naive
    # model_plan = model.pp_plan
    if is_torch_greater_or_equal("2.5") and _torch_distributed_available:
       num_layers = model.config.num_hidden_layers
       pp_rank, pp_world_size = device_mesh.get_local_rank(), device_mesh.size()
       layers_per_gpu = [num_layers // pp_world_size + (1 if i < num_layers % pp_world_size else 0) for i in range(pp_world_size)]
       start_layer = sum(layers_per_gpu[:pp_rank])
       
       #TODO(3outeille): use _no_split_modules for now. it uses the DecoderLayer
       assert len(model._no_split_modules) == 1, "_no_split_modules has more than 1 element"

       for name, module in model.named_modules():
            if name.startswith("model.layers."):
                layer_idx = int(name.split(".")[2])
                
                # First stage never recv as there is no stage to receive from
                if pp_rank > 0 and layer_idx == start_layer and type(module).__name__ == model._no_split_modules[0]:
                    add_pipeline_parallel_hooks_to_module(
                        model=model, 
                        module=module, 
                        pp_plan=None, 
                        layer_name=name, 
                        current_module_plan="recv_forward", 
                        device_mesh=device_mesh
                    )
                    module._is_hooked = True

                # Last layer in stage (needs to send to next stage)
                if pp_rank < pp_world_size - 1 and layer_idx == start_layer + layers_per_gpu[pp_rank] - 1 and type(module).__name__ == model._no_split_modules[0]:
                    add_pipeline_parallel_hooks_to_module(
                        model=model, 
                        module=module, 
                        pp_plan=None, 
                        layer_name=name, 
                        current_module_plan="send_forward", 
                        device_mesh=device_mesh
                    )
                    module._is_hooked = True

    # I want to print the model structure to know which layer has a pipeline parallel hook and which hook it has been used
    print(f"--- [Rank {pp_rank}] Final hook configuration ---")
    for name, module in model.named_modules():
        if hasattr(module, "_hf_pp_plan"):
            print(f"[Rank {pp_rank}] {name}: PP Plan = {module._hf_pp_plan}")

    return model
