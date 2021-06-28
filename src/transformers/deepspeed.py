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

import importlib.util
import io
import json
import weakref
from copy import deepcopy
from functools import partialmethod

from .dependency_versions_check import dep_version_check
from .file_utils import is_torch_available
from .utils import logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


def is_deepspeed_available():
    return importlib.util.find_spec("deepspeed") is not None


class HfDeepSpeedConfig:
    """
    This object contains a DeepSpeed configuration dictionary and can be quickly queried for things like zero stage.

    A ``weakref`` of this object is stored in the module's globals to be able to access the config from areas where
    things like the Trainer object is not available (e.g. ``from_pretrained`` and ``_get_resized_embeddings``).
    Therefore it's important that this object remains alive while the program is still running.

    :class:`~transformers.Trainer` uses the ``HfTrainerDeepSpeedConfig`` subclass instead. That subclass has logic to
    sync the configuration with values of :class:`~transformers.TrainingArguments` by replacing special placeholder
    values: ``"auto"``. Without this special logic the DeepSpeed configuration is not modified in any way.

    Args:
        config_file_or_dict (:obj:`Union[str, Dict]`) - path to DeepSpeed config file or dict.

    """

    def __init__(self, config_file_or_dict):
        # set global weakref object
        set_hf_deepspeed_config(self)

        dep_version_check("deepspeed")

        if isinstance(config_file_or_dict, dict):
            # Don't modify user's data should they want to reuse it (e.g. in tests), because once we
            # modified it, it will not be accepted here again, since `auto` values would have been overriden
            config = deepcopy(config_file_or_dict)
        elif isinstance(config_file_or_dict, str):
            with io.open(config_file_or_dict, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise ValueError("expecting either a path to a DeepSpeed config file or a pre-populated dict")
        self.config = config

        # zero stage - this is done as early as possible, before model is created, to allow
        # ``is_deepspeed_zero3_enabled`` query and getting to the early deepspeed config object
        # during ``zero.Init()`` which needs whether fp16 is enabled, dtype, etc.
        self._stage = self.get_value("zero_optimization.stage", -1)

        # offload
        self._offload = False
        if self.is_zero2() or self.is_zero3():
            offload_devices_valid = set(["cpu", "nvme"])
            offload_devices = set(
                [
                    self.get_value("zero_optimization.offload_optimizer.device"),
                    self.get_value("zero_optimization.offload_param.device"),
                ]
            )
            if len(offload_devices & offload_devices_valid) > 0:
                self._offload = True

    def find_config_node(self, ds_key_long):
        config = self.config

        # find the config node of interest if it exists
        nodes = ds_key_long.split(".")
        ds_key = nodes.pop()
        for node in nodes:
            config = config.get(node)
            if config is None:
                return None, ds_key

        return config, ds_key

    def get_value(self, ds_key_long, default=None):
        """
        Returns the set value or ``default`` if no value is set
        """
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return default
        return config.get(ds_key, default)

    def is_true(self, ds_key_long):
        """
        Returns :obj:`True`/:obj:`False` only if the value is set, always :obj:`False` otherwise. So use this method to
        ask the very specific question of whether the value is set to :obj:`True` (and it's not set to :obj:`False` or
        isn't set).

        """
        value = self.get_value(ds_key_long)
        return False if value is None else bool(value)

    def is_false(self, ds_key_long):
        """
        Returns :obj:`True`/:obj:`False` only if the value is set, always :obj:`False` otherwise. So use this method to
        ask the very specific question of whether the value is set to :obj:`False` (and it's not set to :obj:`True` or
        isn't set).
        """
        value = self.get_value(ds_key_long)
        return False if value is None else not bool(value)

    def is_zero2(self):
        return self._stage == 2

    def is_zero3(self):
        return self._stage == 3

    def is_offload(self):
        return self._offload


class HfTrainerDeepSpeedConfig(HfDeepSpeedConfig):
    """
    The ``HfTrainerDeepSpeedConfig`` object is meant to be created during ``TrainingArguments`` object creation and has
    the same lifespan as the latter.
    """

    def __init__(self, config_file_or_dict):
        super().__init__(config_file_or_dict)
        self._dtype = torch.float16
        self.mismatches = []

    def dtype(self):
        return self._dtype

    def fill_match(self, ds_key_long, hf_val, hf_key=None, must_match=True):
        """
        A utility method that massages the config file and can optionally verify that the values match.

        1. Replace "auto" values with ``TrainingArguments`` value.

        2. If it wasn't "auto" and ``must_match`` is true, then check that DS config matches Trainer
        config values and if mismatched add the entry to ``self.mismatched`` - will assert during
        ``trainer_config_finalize`` for one or more mismatches.

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

    def trainer_config_process(self, args):
        """
        Adjust the config with ``TrainingArguments`` values. This stage is run during ``TrainingArguments`` object
        creation.
        """
        # DeepSpeed does:
        # train_batch_size = world_size * train_micro_batch_size_per_gpu * gradient_accumulation_steps
        train_batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.fill_match(
            "train_micro_batch_size_per_gpu", args.per_device_train_batch_size, "per_device_train_batch_size"
        )
        self.fill_match("gradient_accumulation_steps", args.gradient_accumulation_steps, "gradient_accumulation_steps")
        self.fill_match("train_batch_size", train_batch_size, "train_batch_size (calculated)")
        self.fill_match("gradient_clipping", args.max_grad_norm, "max_grad_norm")

        self.fill_match("optimizer.params.lr", args.learning_rate, "learning_rate")
        self.fill_match("optimizer.params.betas", [args.adam_beta1, args.adam_beta2], "adam_beta1+adam_beta2")
        self.fill_match("optimizer.params.eps", args.adam_epsilon, "adam_epsilon")
        self.fill_match("optimizer.params.weight_decay", args.weight_decay, "weight_decay")

        self.fill_only("scheduler.params.warmup_min_lr", 0)  # not a trainer arg
        self.fill_match("scheduler.params.warmup_max_lr", args.learning_rate, "learning_rate")
        self.fill_match("scheduler.params.warmup_num_steps", args.warmup_steps, "warmup_steps")
        # total_num_steps - will get set in trainer_config_finalize

        # fp16
        if args.fp16:
            fp16_backend = "apex" if args.fp16_backend == "apex" else "amp"
        else:
            fp16_backend = None

        # amp: similar to the pytorch native amp - it has a bunch of optional params but we won't set
        # any here unless the user did the work
        self.fill_match("fp16.enabled", fp16_backend == "amp", "fp16+fp16_backend(amp)")

        # apex: delegates amp work to apex (which needs to be available), but it cannot be used with any
        # ZeRO features
        self.fill_match("amp.enabled", fp16_backend == "apex", "fp16+fp16_backend(apex)")
        self.fill_match("amp.opt_level", args.fp16_opt_level, "fp16_opt_level")

        # only if we have an explicit fp16.enabled = False then it's fp32, if it's True or this
        # whole config section is missing then the fallback is fp16
        if self.is_false("fp16.enabled"):
            self._dtype = torch.float32
        # later there will be other dtypes besides just fp16 and fp32
        # also not quite sure what dtype should be under apex, defaulting to fp16 for now

    def trainer_config_finalize(self, args, model, num_training_steps):
        """
        This stage is run after we have the model and know num_training_steps.

        Now we we can complete the configuration process.
        """
        # zero
        if self.is_zero3():
            # automatically assign the optimal config values based on model config
            hidden_size = model.config.hidden_size
            self.fill_only("zero_optimization.reduce_bucket_size", hidden_size * hidden_size)
            self.fill_only("zero_optimization.stage3_prefetch_bucket_size", 0.9 * hidden_size * hidden_size)
            self.fill_only("zero_optimization.stage3_param_persistence_threshold", 10 * hidden_size)

        # scheduler
        self.fill_match("scheduler.params.total_num_steps", num_training_steps, "num_training_steps (calculated)")

        if len(self.mismatches) > 0:
            mismatches = "\n".join(self.mismatches)
            raise ValueError(
                f"Please correct the following DeepSpeed config values that mismatch TrainingArguments values:\n{mismatches}\n"
                "The easiest method is to set these DeepSpeed config values to 'auto'."
            )


# keep the config object global to be able to access it anywhere during TrainingArguments life-cycle
_hf_deepspeed_config_weak_ref = None


def set_hf_deepspeed_config(hf_deepspeed_config_obj):
    # this is a special weakref global object to allow us to get to Deepspeed config from APIs
    # that don't have an easy way to get to the Deepspeed config outside of the Trainer domain.
    global _hf_deepspeed_config_weak_ref
    # will go away automatically when HfDeepSpeedConfig is destroyed (when TrainingArguments is destroyed)
    _hf_deepspeed_config_weak_ref = weakref.ref(hf_deepspeed_config_obj)


def is_deepspeed_zero3_enabled():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().is_zero3()
    else:
        return False


def deepspeed_config():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().config
    else:
        return None


def deepspeed_init(trainer, num_training_steps, resume_from_checkpoint=None):
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

    If ``resume_from_checkpoint`` was passed then an attempt to resume from a previously saved checkpoint will be made.

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load

    Returns: model, optimizer, lr_scheduler

    """
    import deepspeed
    from deepspeed.utils import logger as ds_logger

    model = trainer.model
    args = trainer.args

    hf_deepspeed_config = args.hf_deepspeed_config
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

    # resume config update - some bits like `model` and `num_training_steps` only become available during train
    config = hf_deepspeed_config.config

    # Optimizer + Scheduler
    # Currently supported combos:
    # 1. DS scheduler + DS optimizer: Yes
    # 2. HF scheduler + HF optimizer: Yes
    # 3. DS scheduler + HF optimizer: Yes
    # 4. HF scheduler + DS optimizer: No
    #
    # Unless Offload is enabled in which case it's:
    # 1. DS scheduler + DS optimizer: Yes
    # 2. HF scheduler + HF optimizer: No
    # 3. DS scheduler + HF optimizer: No
    # 4. HF scheduler + DS optimizer: No

    optimizer = None
    if "optimizer" in config:
        if args.adafactor:
            raise ValueError(
                "--adafactor was passed, but also found `optimizer` configured in the DeepSpeed config. "
                "Only one optimizer can be configured."
            )
    else:
        if hf_deepspeed_config.is_offload():
            raise ValueError("ZeRO Offload can only work with DeepSpeed optimizers")

        # ds supports Adam, OneBitAdam, and Lamb optimizers and can import other optimizers from torch.
        # But trainer uses AdamW by default.
        trainer.create_optimizer()
        optimizer = trainer.optimizer
        # To use other optimizers requires voiding warranty with: `zero_allow_untested_optimizer`
        config["zero_allow_untested_optimizer"] = True

    # DS schedulers (deepspeed/runtime/lr_schedules.py):
    #
    # DS name      | --lr_scheduler_type  | HF func                           | Notes
    # -------------| ---------------------|-----------------------------------|--------------------
    # LRRangeTest  | na                   | na                                | LRRT
    # OneCycle     | na                   | na                                | 1CLR
    # WarmupLR     | constant_with_warmup | get_constant_schedule_with_warmup | w/ warmup_min_lr=0
    # WarmupDecayLR| linear               | get_linear_schedule_with_warmup   |
    lr_scheduler = None
    if "scheduler" not in config:
        if "optimizer" in config:
            # to make this option work, we need to init DS optimizer first, then init HS scheduler,
            # then pass the HS scheduler to DS init, which is not possible at the moment
            raise ValueError("At the moment HF scheduler + DeepSpeed optimizer combination is not possible")
        else:
            trainer.create_scheduler(num_training_steps=num_training_steps)
            lr_scheduler = trainer.lr_scheduler

    # keep for quick debug:
    # from pprint import pprint; pprint(config)

    # set the Deepspeed log level consistent with the trainer
    ds_logger.setLevel(args.get_process_log_level())

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config_params=config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    if resume_from_checkpoint is not None:

        # it's possible that the user is trying to resume from model_path, which doesn't necessarily
        # contain a deepspeed checkpoint. e.g. examples just check if the dir exists and assume it's
        # a resume from a checkpoint and not just a local pretrained weight. So we check here if the
        # path contains what looks like a deepspeed checkpoint
        import glob

        deepspeed_checkpoint_dirs = sorted(glob.glob(f"{resume_from_checkpoint}/global_step*"))

        if len(deepspeed_checkpoint_dirs) > 0:
            logger.info(f"Attempting to resume from {resume_from_checkpoint}")
            # this magically updates self.optimizer and self.lr_scheduler
            load_path, _ = model.load_checkpoint(
                resume_from_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
            )
            if load_path is None:
                raise ValueError(f"[deepspeed] failed to resume from checkpoint {resume_from_checkpoint}")
        else:
            logger.info(f"{resume_from_checkpoint} doesn't have deepspeed checkpoints, doing nothing")

    return model, optimizer, lr_scheduler
