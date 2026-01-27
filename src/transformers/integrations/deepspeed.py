# Copyright 2020 The HuggingFace Team. All rights reserved.
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
Integration with Deepspeed
"""

import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod

from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn


logger = logging.get_logger(__name__)


def is_deepspeed_available():
    package_exists = importlib.util.find_spec("deepspeed") is not None

    # Check we're not importing a "deepspeed" directory somewhere but the actual library by trying to grab the version
    # AND checking it has an author field in the metadata that is HuggingFace.
    if package_exists:
        try:
            _ = importlib_metadata.metadata("deepspeed")
            return True
        except importlib_metadata.PackageNotFoundError:
            return False


if is_accelerate_available() and is_deepspeed_available():
    from accelerate.utils.deepspeed import HfDeepSpeedConfig as DeepSpeedConfig
else:
    # Inherits from a dummy `object` if accelerate is not available, so that python succeeds to import this file.
    # Deepspeed glue code will never inherit this dummy object as it checks if accelerate is available.
    from builtins import object as DeepSpeedConfig


class HfDeepSpeedConfig(DeepSpeedConfig):  # noqa UP004
    """
    This object contains a DeepSpeed configuration dictionary and can be quickly queried for things like zero stage.

    A `weakref` of this object is stored in the module's globals to be able to access the config from areas where
    things like the Trainer object is not available (e.g. `from_pretrained` and `_get_resized_embeddings`). Therefore
    it's important that this object remains alive while the program is still running.

    [`Trainer`] uses the `HfTrainerDeepSpeedConfig` subclass instead. That subclass has logic to sync the configuration
    with values of [`TrainingArguments`] by replacing special placeholder values: `"auto"`. Without this special logic
    the DeepSpeed configuration is not modified in any way.

    Args:
        config_file_or_dict (`Union[str, Dict]`): path to DeepSpeed config file or dict.

    """

    def __init__(self, config_file_or_dict):
        # set global weakref object
        set_hf_deepspeed_config(self)
        dep_version_check("accelerate")
        dep_version_check("deepspeed")
        super().__init__(config_file_or_dict)


class HfTrainerDeepSpeedConfig(HfDeepSpeedConfig):
    """
    The `HfTrainerDeepSpeedConfig` object is meant to be created during `TrainingArguments` object creation and has the
    same lifespan as the latter.
    """

    def __init__(self, config_file_or_dict):
        super().__init__(config_file_or_dict)
        self._dtype = None
        self.mismatches = []

    def dtype(self):
        if self._dtype is None:
            raise ValueError("trainer_config_process() wasn't called yet to tell dtype")
        return self._dtype

    def is_auto(self, ds_key_long):
        val = self.get_value(ds_key_long)
        if val is None:
            return False
        else:
            return val == "auto"

    def fill_match(self, ds_key_long, hf_val, hf_key=None, must_match=True):
        """
        A utility method that massages the config file and can optionally verify that the values match.

        1. Replace "auto" values with `TrainingArguments` value.

        2. If it wasn't "auto" and `must_match` is true, then check that DS config matches Trainer
        config values and if mismatched add the entry to `self.mismatched` - will assert during
        `trainer_config_finalize` for one or more mismatches.

        """
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return

        if config.get(ds_key) == "auto":
            config[ds_key] = hf_val
            return

        if not must_match:
            return

        ds_val = config.get(ds_key)
        if ds_val is not None and ds_val != hf_val:
            self.mismatches.append(f"- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}")

    fill_only = partialmethod(fill_match, must_match=False)

    def trainer_config_process(self, args, auto_find_batch_size=False):
        """
        Adjust the config with `TrainingArguments` values. This stage is run during `TrainingArguments` object
        creation.
        """
        # DeepSpeed does:
        # train_batch_size = world_size * train_micro_batch_size_per_gpu * gradient_accumulation_steps
        train_batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.fill_match(
            "train_micro_batch_size_per_gpu",
            args.per_device_train_batch_size,
            "per_device_train_batch_size",
            not auto_find_batch_size,
        )
        self.fill_match(
            "gradient_accumulation_steps",
            args.gradient_accumulation_steps,
            "gradient_accumulation_steps",
        )
        self.fill_match(
            "train_batch_size",
            train_batch_size,
            "train_batch_size (calculated)",
            not auto_find_batch_size,
        )
        self.fill_match("gradient_clipping", args.max_grad_norm, "max_grad_norm")

        self.fill_match("optimizer.params.lr", args.learning_rate, "learning_rate")
        self.fill_match(
            "optimizer.params.betas",
            [args.adam_beta1, args.adam_beta2],
            "adam_beta1+adam_beta2",
        )
        self.fill_match("optimizer.params.eps", args.adam_epsilon, "adam_epsilon")
        self.fill_match("optimizer.params.weight_decay", args.weight_decay, "weight_decay")

        self.fill_only("scheduler.params.warmup_min_lr", 0)  # not a trainer arg
        self.fill_match("scheduler.params.warmup_max_lr", args.learning_rate, "learning_rate")
        # total_num_steps - will get set in trainer_config_finalize

        if args.save_on_each_node:
            # deepspeed uses shared storage by default. Let's override this setting if save_on_each_node == True
            self.config["checkpoint"] = self.config.get("checkpoint", {})
            self.config["checkpoint"]["use_node_local_storage"] = args.save_on_each_node

        # amp: similar to the pytorch native amp - it has a bunch of optional params but we won't set
        # any here unless the user did the work
        self.fill_match("fp16.enabled", (args.fp16 or args.fp16_full_eval), "fp16|fp16_full_eval")
        self.fill_match("bf16.enabled", (args.bf16 or args.bf16_full_eval), "bf16|bf16_full_eval")

        # deepspeed's default mode is fp16 unless there is a config that says differently
        if self.is_true("bf16.enabled"):
            self._dtype = torch.bfloat16
        elif self.is_true("fp16.enabled"):
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

    def trainer_config_finalize(self, args, model, num_training_steps):
        """
        This stage is run after we have the model and know num_training_steps.

        Now we can complete the configuration process.
        """
        # zero

        # deal with config keys that use `auto` value and rely on model's hidden_size
        hidden_size_based_keys = [
            "zero_optimization.reduce_bucket_size",
            "zero_optimization.stage3_prefetch_bucket_size",
            "zero_optimization.stage3_param_persistence_threshold",
        ]
        hidden_size_auto_keys = [x for x in hidden_size_based_keys if self.is_auto(x)]

        if len(hidden_size_auto_keys) > 0:
            hidden_size = None
            if hasattr(model, "config"):
                if hasattr(model.config, "hidden_size"):
                    hidden_size = model.config.hidden_size
                elif hasattr(model.config, "hidden_sizes"):
                    # if there are many hidden sizes pick the largest one
                    hidden_size = max(model.config.hidden_sizes)
                elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
                    hidden_size = model.config.text_config.hidden_size
                elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_sizes"):
                    # if there are many hidden sizes pick the largest one
                    hidden_size = max(model.config.text_config.hidden_sizes)

            if hidden_size is None:
                raise ValueError(
                    "The model's config file has neither `hidden_size` nor `hidden_sizes` entry, "
                    "therefore it's not possible to automatically fill out the following `auto` entries "
                    f"in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing "
                    "`auto` values for these keys with an integer value of your choice."
                )

            self.fill_only("zero_optimization.reduce_bucket_size", hidden_size * hidden_size)
            if self.is_zero3():
                # automatically assign the optimal config values based on model config
                self.fill_only(
                    "zero_optimization.stage3_prefetch_bucket_size",
                    int(0.9 * hidden_size * hidden_size),
                )
                self.fill_only(
                    "zero_optimization.stage3_param_persistence_threshold",
                    10 * hidden_size,
                )

        # scheduler
        self.fill_match(
            "scheduler.params.total_num_steps",
            num_training_steps,
            "num_training_steps (calculated)",
        )
        self.fill_match(
            "scheduler.params.warmup_num_steps",
            args.get_warmup_steps(num_training_steps),
            "warmup_steps",
        )

        if len(self.mismatches) > 0:
            mismatches = "\n".join(self.mismatches)
            raise ValueError(
                "Please correct the following DeepSpeed config values that mismatch TrainingArguments"
                f" values:\n{mismatches}\nThe easiest method is to set these DeepSpeed config values to 'auto'."
            )


# keep the config object global to be able to access it anywhere during TrainingArguments life-cycle
_hf_deepspeed_config_weak_ref = None


def set_hf_deepspeed_config(hf_deepspeed_config_obj):
    # this is a special weakref global object to allow us to get to Deepspeed config from APIs
    # that don't have an easy way to get to the Deepspeed config outside of the Trainer domain.
    global _hf_deepspeed_config_weak_ref
    # will go away automatically when HfDeepSpeedConfig is destroyed (when TrainingArguments is destroyed)
    _hf_deepspeed_config_weak_ref = weakref.ref(hf_deepspeed_config_obj)


def unset_hf_deepspeed_config():
    # useful for unit tests to ensure the global state doesn't leak - call from `tearDown` method
    global _hf_deepspeed_config_weak_ref
    _hf_deepspeed_config_weak_ref = None


def is_deepspeed_zero3_enabled():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        try:
            return _hf_deepspeed_config_weak_ref().is_zero3()
        except AttributeError:
            return False
    else:
        return False


def deepspeed_config():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().config
    else:
        return None


def _apply_weight_conversions_to_state_dict(model, state_dict, weight_mapping):
    """
    Apply weight conversions (renaming and merging/splitting operations) to a state dict.
    This is a simplified version that handles the conversion without loading into the model.
    """
    # Check for Tensor Parallelism - weight conversions are not tested with TP
    # TP uses ReplaceWithTensorSlicing which may conflict with our weight conversions
    ds_config = deepspeed_config()
    if ds_config is not None:
        # Check training config (tensor_parallel.autotp_size)
        tp_size = ds_config.get("tensor_parallel", {}).get("autotp_size", 1)
        # Check inference config (inference.tensor_parallel.tp_size)
        inference_config = ds_config.get("inference", {})
        if isinstance(inference_config, dict):
            tp_size = max(tp_size, inference_config.get("tensor_parallel", {}).get("tp_size", 1))
        if tp_size > 1:
            raise NotImplementedError(
                "Weight conversions (e.g., MoE expert fusion) with DeepSpeed Tensor Parallelism "
                "are not yet implemented but support is coming soon. Please disable tensor_parallel "
                "in your DeepSpeed config or convert your checkpoint to the expected format first."
            )

    from ..core_model_loading import WeightConverter, WeightRenaming, dot_natural_key, rename_source_key

    # Preserve metadata from the original state dict
    metadata = getattr(state_dict, "_metadata", None)

    prefix = model.base_model_prefix

    # Build a meta state dict for matching - only keys/shapes, no actual tensor data
    # This minimizes memory since we don't duplicate the model's parameters
    model_state_dict = {}
    for key, param in model.state_dict().items():
        model_state_dict[key] = torch.empty(param.shape, dtype=param.dtype, device="meta")

    renamings = [entry for entry in weight_mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in weight_mapping if isinstance(entry, WeightConverter)]

    # Fast path: if we only have simple renamings and no converters, we can skip the expensive collection logic
    if len(converters) == 0:
        new_state_dict = {}
        for original_key, tensor in state_dict.items():
            renamed_key, _ = rename_source_key(original_key, renamings, [], prefix, model_state_dict)
            if renamed_key in model_state_dict:
                new_state_dict[renamed_key] = tensor
        # Attach metadata to the new state dict
        if metadata is not None:
            new_state_dict._metadata = metadata
        return new_state_dict

    # Full path: we have WeightConverter operations that require tensor fusion/splitting
    pattern_to_converter = {k: converter for converter in converters for k in converter.source_patterns}

    # Build a mapping of what needs to be converted
    # Sort keys to ensure consistent ordering (important for MoE conversions)
    # Iterate over sorted keys and pop from state_dict to free memory immediately
    conversion_mapping = {}
    key_rename_cache = {}  # Cache rename results to avoid redundant processing
    sorted_keys = sorted(state_dict.keys(), key=lambda k: dot_natural_key(k))
    for original_key in sorted_keys:
        tensor = state_dict.pop(original_key)  # Pop to free memory immediately
        # Rename the key according to all renaming pattern and optional weight converter patterns
        renamed_key, source_pattern = rename_source_key(original_key, renamings, converters, prefix, model_state_dict)

        # Cache the rename result for use in the cleanup loop
        key_rename_cache[original_key] = renamed_key

        # Only process if the renamed key is in the model's state dict
        if renamed_key in model_state_dict:
            if source_pattern is not None:
                new_converter = copy.deepcopy(pattern_to_converter[source_pattern])
                mapping = conversion_mapping.setdefault(renamed_key, new_converter)
            else:
                mapping = conversion_mapping.setdefault(renamed_key, WeightRenaming(original_key, renamed_key))
                source_pattern = original_key

            # Add the tensor directly (not a Future, since it's already materialized)
            mapping.add_tensor(renamed_key, original_key, source_pattern, tensor)

    # Apply the conversions and build the new state dict
    new_state_dict = {}
    # Track which renamed_keys came from WeightConverter (need to skip their originals)
    converted_renamed_keys = set()
    for renamed_key, mapping in conversion_mapping.items():
        try:
            # Only WeightConverter needs convert(); WeightRenaming is just a simple rename
            if not isinstance(mapping, WeightConverter):
                continue
            realized_value, _ = mapping.convert(
                renamed_key,
                model=model,
                config=model.config,
            )
            for target_name, param in realized_value.items():
                param = param[0] if isinstance(param, list) else param
                new_state_dict[target_name] = param
            # Track that this key was converted
            converted_renamed_keys.add(renamed_key)
            # Free memory by clearing source tensors
            if hasattr(mapping, "source_tensors"):
                mapping.source_tensors = {}
        except Exception as e:
            raise RuntimeError(
                f"Failed to apply weight conversion for '{renamed_key}'. "
                f"This likely means the checkpoint format is incompatible with the current model version. "
                f"Error: {e}"
            ) from e

    # Add any keys that didn't need conversion (use cached rename results)
    # At this point, state_dict only contains unconverted keys (others were popped)
    for key in list(state_dict.keys()):
        renamed_key = key_rename_cache.get(key)
        if renamed_key is None:
            # Key wasn't in our cache, compute rename
            renamed_key, _ = rename_source_key(key, renamings, [], prefix, model_state_dict)
        if renamed_key not in new_state_dict and renamed_key in model_state_dict:
            new_state_dict[renamed_key] = state_dict.pop(key)

    # Attach metadata to the new state dict
    if metadata is not None:
        new_state_dict._metadata = metadata

    return new_state_dict


def _load_state_dict_into_zero3_model(model_to_load, state_dict, load_config=None):
    """
    Loads state dict into a model specifically for Zero3, since DeepSpeed does not support the `transformers`
    tensor parallelism API.

    Nearly identical code to PyTorch's `_load_from_state_dict`

    Args:
        model_to_load: The model to load weights into
        state_dict: The state dict containing the weights
        load_config: Optional LoadStateDictConfig containing weight_mapping and other loading options
    """
    # copy state_dict so `_load_state_dict_into_zero3_model` can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # Extract weight_mapping from load_config if provided
    weight_mapping = None
    if load_config is not None:
        weight_mapping = getattr(load_config, "weight_mapping", None)

    # Apply weight conversions if provided
    if weight_mapping is not None and len(weight_mapping) > 0:
        state_dict = _apply_weight_conversions_to_state_dict(model_to_load, state_dict, weight_mapping)
        # Keep the current weight conversion mapping for later saving (in case it was coming directly from the user)
        model_to_load._weight_conversions = weight_mapping

    error_msgs = []
    meta_model_state_dict = model_to_load.state_dict()
    missing_keys = set(meta_model_state_dict.keys())

    prefix_model = getattr(model_to_load, "base_model_prefix", None)
    # take care of the case where in the checkpoint we don't have the prefix
    state_dict = {
        (f"{prefix_model}.{k}" if meta_model_state_dict.get(f"{prefix_model}.{k}") is not None else k): v
        for k, v in state_dict.items()
    }

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix="", assign_to_params_buffers=False):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        local_metadata["assign_to_params_buffers"] = assign_to_params_buffers

        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if is_deepspeed_zero3_enabled():
            import deepspeed

            # In sharded models, each shard has only part of the full state_dict, so only gather
            # parameters that are in the current state_dict.
            named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
            params_to_gather = []
            for k in named_parameters:
                if k in state_dict:
                    param = named_parameters[k]
                    # crutial to not init the weight again
                    param._is_hf_initialized = True
                    params_to_gather.append(param)
                    missing_keys.discard(k)

            if len(params_to_gather) > 0:
                # because zero3 puts placeholders in model params, this context
                # manager gathers (unpartitions) the params of the current layer, then loads from
                # the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".", assign_to_params_buffers)

    load(model_to_load, state_dict, assign_to_params_buffers=False)

    return error_msgs, missing_keys


def deepspeed_optim_sched(trainer, hf_deepspeed_config, args, num_training_steps, model_parameters):
    """
    A convenience wrapper that deals with optimizer and lr scheduler configuration.
    """
    from accelerate.utils import DummyOptim, DummyScheduler

    config = hf_deepspeed_config.config

    # Mixing and matching DS schedulers and optimizers is supported unless Offload is enabled in which case it's:
    # 1. DS scheduler + DS optimizer: Yes
    # 2. HF scheduler + HF optimizer: Mostly*
    # 3. DS scheduler + HF optimizer: Mostly*
    # 4. HF scheduler + DS optimizer: Yes
    #
    # Mostly*: All non-native DeepSpeed optimizers that have both CPU and GPU implementation should work (except LAMB)

    optimizer = None
    if "optimizer" in config:
        optimizer = DummyOptim(params=model_parameters)
    else:
        if hf_deepspeed_config.is_offload():
            logger.info(
                "Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the"
                " custom optimizer has both CPU and GPU implementation (except LAMB)"
            )

        # ds supports Adam, OneBitAdam, and Lamb optimizers and can import other optimizers from torch.
        # But trainer uses AdamW by default.
        optimizer = trainer.create_optimizer()
        # To use other optimizers requires voiding warranty with: `zero_allow_untested_optimizer`
        config["zero_allow_untested_optimizer"] = True

    lr_scheduler = None
    if "scheduler" in config:
        lr_scheduler = DummyScheduler(optimizer)
    else:
        if isinstance(optimizer, DummyOptim):

            def _lr_scheduler_callable(optimizer):
                # create a shallow copy first, so later modifications do not affect original trainer
                trainer_copy = copy.copy(trainer)
                # at the time _lr_scheduler_callable is called, trainer.lr_scheduler has been set
                # update it to None so that we can re-create a new scheduler
                trainer_copy.lr_scheduler = None
                lr_scheduler = trainer_copy.create_scheduler(
                    num_training_steps=num_training_steps, optimizer=optimizer
                )
                return lr_scheduler

            lr_scheduler = DummyScheduler(optimizer, lr_scheduler_callable=_lr_scheduler_callable)
        else:
            lr_scheduler = trainer.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    return optimizer, lr_scheduler


def deepspeed_init(trainer, num_training_steps, inference=False):
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

    If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
        inference: launch in inference mode (no optimizer and no lr scheduler)
        auto_find_batch_size: whether to ignore the `train_micro_batch_size_per_gpu` argument as it's being
            set automatically by the auto batch size finder

    Returns: optimizer, lr_scheduler

    We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
    https://github.com/deepspeedai/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
    can't resume from a checkpoint after it did some stepping https://github.com/deepspeedai/DeepSpeed/issues/1612

    """
    from deepspeed.utils import logger as ds_logger

    model = trainer.model
    args = trainer.args

    hf_deepspeed_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config

    # resume config update - some bits like `model` and `num_training_steps` only become available during train
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

    # set the Deepspeed log level consistent with the Trainer
    ds_logger.setLevel(args.get_process_log_level())

    if inference:
        # only Z3 makes sense for the inference
        if not hf_deepspeed_config.is_zero3():
            raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")

        # in case the training config is re-used for inference
        hf_deepspeed_config.del_config_sub_tree("optimizer")
        hf_deepspeed_config.del_config_sub_tree("lr_scheduler")
        optimizer, lr_scheduler = None, None
        model_parameters = None
    else:
        trainer.optimizer = None  # important for when deepspeed_init is used as re-init
        deepspeed_tp_size = hf_deepspeed_config.config.get("tensor_parallel", {}).get("autotp_size", 1)
        if deepspeed_tp_size > 1:
            import deepspeed

            model = deepspeed.tp_model_init(
                model=model,
                tp_size=deepspeed_tp_size,
                dtype=hf_deepspeed_config.dtype(),
                config=hf_deepspeed_config.config,
            )
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer, lr_scheduler = deepspeed_optim_sched(
            trainer, hf_deepspeed_config, args, num_training_steps, model_parameters
        )

    # keep for quick debug:
    # from pprint import pprint; pprint(config)

    return optimizer, lr_scheduler


def deepspeed_load_checkpoint(deepspeed_engine, checkpoint_path, load_module_strict=True):
    # it's possible that the user is trying to resume from model_path, which doesn't necessarily
    # contain a deepspeed checkpoint. e.g. examples just check if the dir exists and assume it's
    # a resume from a checkpoint and not just a local pretrained weight. So we check here if the
    # path contains what looks like a deepspeed checkpoint
    import glob

    deepspeed_checkpoint_dirs = sorted(glob.glob(f"{checkpoint_path}/global_step*"))

    if len(deepspeed_checkpoint_dirs) > 0:
        logger.info(f"Attempting to resume from {checkpoint_path}")
        # this magically updates self.optimizer and self.lr_scheduler
        load_path, _ = deepspeed_engine.load_checkpoint(
            checkpoint_path,
            load_module_strict=load_module_strict,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise ValueError(f"[deepspeed] failed to resume from checkpoint {checkpoint_path}")
    else:
        raise ValueError(f"Can't find a valid checkpoint at {checkpoint_path}")
