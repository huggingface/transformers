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

import functools
import os

import torch
from torch.utils.data import DataLoader

from ..utils import WEIGHTS_NAME, PushToHubMixin, is_torch_xla_available, logging


logger = logging.get_logger(__name__)


def tpu_spmd_dataloader(dataloader: DataLoader):
    if is_torch_xla_available():
        import torch_xla.distributed.parallel_loader as pl

        assert isinstance(dataloader, pl.MpDeviceLoader), (
            "The dataloader must be a `torch_xla.distributed.parallel_loader.MpDeviceLoader`."
        )

        # This is to support PyTorch/XLA FSDP via SPMD.
        # Here we shard the input data's 0th dim across the fsdp axis.
        import torch_xla.distributed.spmd as xs

        sharding_spec = xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None))
        dataloader._parallel_loader_kwargs["input_sharding"] = sharding_spec
        return dataloader
    else:
        return dataloader


def wrap_model_xla_fsdp(model, args, is_fsdp_xla_v2_enabled):
    """
    Wraps a model with XLA Fully Sharded Data Parallelism (FSDP).

    Handles both FSDP v1 (`XlaFullyShardedDataParallel`) and v2 (`SpmdFullyShardedDataParallel`),
    including auto-wrap policies, gradient checkpointing, and patching `xm.optimizer_step`.

    Args:
        model (`torch.nn.Module`): The model to wrap.
        args (`TrainingArguments`): The training arguments containing FSDP configuration.
        is_fsdp_xla_v2_enabled (`bool`): Whether FSDP v2 (SPMD) is enabled.

    Returns:
        `torch.nn.Module`: The FSDP-wrapped model.
    """
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs

    from ..trainer_pt_utils import get_module_class_from_name

    try:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
        from torch_xla.distributed.fsdp import checkpoint_module
        from torch_xla.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
            transformer_auto_wrap_policy,
        )

        if is_fsdp_xla_v2_enabled:
            from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
                SpmdFullyShardedDataParallel as FSDPv2,
            )
    except ImportError:
        raise ImportError("Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.")

    auto_wrap_policy = None
    auto_wrapper_callable = None
    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = args.fsdp_config.get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if args.fsdp_config["min_num_params"] > 0:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=args.fsdp_config["min_num_params"]
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

    fsdp_kwargs = args.xla_fsdp_config
    if args.fsdp_config["xla_fsdp_grad_ckpt"]:
        if model.config.use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            model.config.use_cache = False

        # Apply gradient checkpointing to auto-wrapped sub-modules if specified
        def auto_wrapper_callable(m, *args, **kwargs):
            target_cls = FSDP if not is_fsdp_xla_v2_enabled else FSDPv2
            return target_cls(checkpoint_module(m), *args, **kwargs)

    # Wrap the base model with an outer FSDP wrapper
    if is_fsdp_xla_v2_enabled:

        def shard_output(output, mesh):
            from ..modeling_outputs import CausalLMOutputWithPast

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
    else:
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            auto_wrapper_callable=auto_wrapper_callable,
            **fsdp_kwargs,
        )

    # Patch `xm.optimizer_step` should not reduce gradients in this case,
    # as FSDP does not need gradient reduction over sharded parameters.
    def patched_optimizer_step(optimizer, barrier=False, optimizer_args={}):
        loss = optimizer.step(**optimizer_args)
        if barrier:
            xm.mark_step()
        return loss

    xm.optimizer_step = patched_optimizer_step

    return model


def save_tpu_checkpoint(model, args, accelerator, processing_class, is_fsdp_xla_v1_enabled, output_dir=None):
    """
    Saves a model checkpoint on TPU/XLA devices.

    Handles FSDP v1 sharded checkpoints (with consolidation on master), as well as
    standard XLA model saving via `save_pretrained` or `xm.save`.

    Args:
        model (`torch.nn.Module`): The model to save.
        args (`TrainingArguments`): The training arguments.
        accelerator (`Accelerator`): The accelerator instance.
        processing_class: The processing class (tokenizer/processor) to save alongside the model.
        is_fsdp_xla_v1_enabled (`bool`): Whether FSDP XLA v1 is enabled.
        output_dir (`str`, *optional*): The directory to save to. Defaults to `args.output_dir`.
    """
    import torch_xla.core.xla_model as xm

    output_dir = output_dir if output_dir is not None else args.output_dir

    logger.info(f"Saving model checkpoint to {output_dir}")
    xm.mark_step()

    if xm.is_master_ordinal(local=False):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

    # Save a trained model and configuration using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    supported_classes = (PushToHubMixin,)
    xm.rendezvous("saving_checkpoint")
    if is_fsdp_xla_v1_enabled:
        ckpt = {
            "model": model.state_dict(),
            "shard_metadata": model.get_shard_metadata(),
        }
        ckpt_path = os.path.join(output_dir, f"rank{args.process_index}-of-{args.world_size}-{WEIGHTS_NAME}")
        # All ranks save sharded checkpoint
        xm.save(ckpt, ckpt_path, master_only=False)
        # Make sure all ranks have saved checkpoints
        xm.rendezvous("save_full_checkpoints")
        # Master save full checkpoint
        if args.should_save:
            from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints

            full_state_dict, _ = consolidate_sharded_model_checkpoints(
                ckpt_prefix=os.path.join(output_dir, ""),
                ckpt_suffix=f"rank*-of-*-{WEIGHTS_NAME}",
                save_model=False,
            )
            model = model.module.module
            unwrapped_model = accelerator.unwrap_model(model)
            if isinstance(unwrapped_model, supported_classes):
                unwrapped_model.save_pretrained(output_dir, state_dict=full_state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                xm.save(full_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    elif not isinstance(model, supported_classes):
        if isinstance(accelerator.unwrap_model(model), supported_classes):
            accelerator.unwrap_model(model).save_pretrained(
                output_dir,
                is_main_process=args.should_save,
                state_dict=xm._maybe_convert_to_cpu(model.state_dict()),
            )
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            state_dict = xm._maybe_convert_to_cpu(model.state_dict())
            xm.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    else:
        model.save_pretrained(
            output_dir,
            is_main_process=args.should_save,
            state_dict=xm._maybe_convert_to_cpu(model.state_dict()),
        )
    if processing_class is not None and args.should_save:
        processing_class.save_pretrained(output_dir)
