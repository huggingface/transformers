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
"""
This file contains the utilities for distributed training that are needed
and not handled by the `accelerate` library outright.
"""

from .utils import logging

logger = logging.get_logger(__name__)

def apply_ipex_optimization(model, optimizer, training=False, dtype=torch.float32, is_in_train=False):
    """
    Applies IPEX optimizations onto the model and optimizer
    """
    if not is_ipex_available():
        raise ImportError(
            "Using IPEX but IPEX is not installed or IPEX's version does not match current PyTorch, please refer"
            " to https://github.com/intel/intel-extension-for-pytorch."
        )

    import intel_extension_for_pytorch as ipex

    if not training:
        model.eval()
        # conv_bn_folding is disabled as it fails in symbolic tracing, resulting in ipex warnings
        model = ipex.optimize(model, dtype=dtype, level="O1", conv_bn_folding=False, inplace=not self.is_in_train)
    else:
        if not model.training:
            model.train()
        model, self.optimizer = ipex.optimize(
            model, dtype=dtype, optimizer=self.optimizer, inplace=True, level="O1"
        )

    return model

def apply_fsdp_xla_optimization(model, fsdp_config, xla_fsdp_config, fsdp_v2_enabled=False):
    """
    Applies FSDP with XLA optimizations to `model`
    """
    try:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
        from torch_xla.distributed.fsdp import checkpoint_module
        from torch_xla.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
            transformer_auto_wrap_policy,
        )

        if fsdp_v2_enabled:
            from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
                SpmdFullyShardedDataParallel as FSDPv2,
            )
    except ImportError:
        raise ImportError("Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.")
    
    auto_wrap_policy = None
    auto_wrapper_callable = None
    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = fsdp_config.get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )
    
    if fsdp_config["min_num_params"] > 0:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=fsdp_config["min_num_params"]
        )
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(model, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # Transformer layer class to wrap
            transformer_layer_cls=transformer_cls_to_wrap,
        )
    if fsdp_config["xla_fsdp_grad_ckpt"]:
        if model.config.use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            model.config.use_cache = False

        # Apply gradient checkpointing to auto-wrapped sub-modules if specified
        def auto_wrapper_callable(m, *args, **kwargs):
            target_cls = FSDP if not fsdp_v2_enabled else FSDPv2
            return target_cls(checkpoint_module(m), *args, **kwargs)
    
    # Wrap the base model with an outer FSDP wrapper
    if not fsdp_v2_enabled:
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            auto_wrapper_callable=auto_wrapper_callable,
            **fsdp_kwargs,
        )
    else:
        def shard_output(output, mesh):
            from .modeling_outputs import CausalLMOutputWithPast

            real_output = None
            if isinstance(output, torch.Tensor):
                real_output = output
            elif isinstance(output, tuple):
                real_output = output[0]
            elif isinstance(output, CausalLMOutputWithPast):
                real_output = output.logits

            if real_output is None:
                raise ValueError("Something went wrong, the output of the model shouldn't be `None`")
            xs.mark_sharding(real_output, mesh, ("fsdp", None, None))

        model = FSDPv2(
            model,
            shard_output=shard_output,
            auto_wrap_policy=auto_wrap_policy,
            auto_wrapper_callable=auto_wrapper_callable,
        )

        return model