# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


# Integrations must be imported before ML frameworks:
# isort: off
from .integrations import (
    get_reporting_integration_callbacks,
    propagate_args_to_deepspeed,
    hp_params,
)

# isort: on
import importlib

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .feature_extraction_sequence_utils import SequenceFeatureExtractor
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import (
    PreTrainedModel,
    is_peft_model,
    load_peft_adapters,
    load_sagemaker_model,
    load_sharded_checkpoint,
)
from .models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import (
    is_torch_greater_or_equal_than_1_13,
    is_torch_greater_or_equal_than_2_3,
)
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    InputTokenTrackingCallback,
    NefttuneCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    apply_fsdp_xla_optimization,
    apply_ipex_optimization,
    clip_grad_norm_,
    create_accelerator,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_decay_parameter_names,
    get_model_param_count,
    get_rng_state,
    model_sanity_checks,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    optimizer_sanity_checks,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
    set_rng_state,
    wait_for_everyone,
)
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from .training_args import OptimizerGroups, OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_galore_torch_available,
    is_in_notebook,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_cuda_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    logging,
    strtobool,
)
from .utils.quantization_config import QuantizationMethod


# isort: off

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel

if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_gather
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets

logger = logging.get_logger(__name__)


def _get_fsdp_ckpt_kwargs():
    return {"adapter_only": True} if is_accelerate_available("0.27.0") else {}


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.

            <Tip>

            [`Trainer`] is optimized to work with the [`PreTrainedModel`] provided by the library. You can still use
            your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers
            models.

            </Tip>

        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise.
        train_dataset (Union[`torch.utils.data.Dataset`, `torch.utils.data.IterableDataset`, `datasets.Dataset`], *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed.

            Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (Union[`torch.utils.data.Dataset`, Dict[str, `torch.utils.data.Dataset`, `datasets.Dataset`]), *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.
        tokenizer ([`PreTrainedTokenizerBase`], *optional*):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs to the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (`Callable[[], PreTrainedModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
            from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
            be able to choose different architectures according to hyper parameters (such as layer count, sizes of
            inner layers, dropout probabilities etc).
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
            `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
            after the last eval batch to signal that the function needs to calculate and return the global summary
            statistics rather than accumulating the batch-level statistics.
        callbacks (List of [`TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](callback).

            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to `False` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return `False` .
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)

    """

    # Those are used as methods of the Trainer in examples.
    from .trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        if args.batch_eval_metrics and compute_metrics is not None:
            if "compute_result" not in inspect.signature(compute_metrics).parameters.keys():
                raise ValueError(
                    "When using `batch_eval_metrics`, your `compute_metrics` function must take a `compute_result`"
                    " boolean argument which will be triggered after the last batch of the eval set to signal that the"
                    " summary statistics should be returned by the function."
                )
        self.args = args
        # Seed must be set before instantiating the model
        enable_full_determinism(args.seed) if args.full_determinism else set_seed(args.seed)

        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        self.create_accelerator_and_postprocess()

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices

        # Init and verify data related items
        if tokenizer is not None and isinstance(tokenizer, (PreTrainedTokenizerBase, SequenceFeatureExtractor)):
            self.data_collator = DataCollatorWithPadding(tokenizer)
        else:
            self.data_collator = data_collator if data_collator is not None else default_data_collator
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if train_dataset is not None and not has_length(train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        if (
            train_dataset is not None
            and isinstance(train_dataset, torch.utils.data.IterableDataset)
            and args.group_by_length
        ):
            raise ValueError("the `--group_by_length` option is only available for `Dataset`, not `IterableDataset")
        self._signature_columns = None

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.optimizer, self.lr_scheduler = optimizers

        if model is None:
            if model_init is None:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
            if self.optimizer is None or self.lr_scheduler is None:
                raise RuntimeError(
                    "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                    "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
                )
            self.model_init = model_init
            model = self.call_model_init()
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method. This will become a fatal error in the next"
                    " release.",
                    FutureWarning,
                )
            self.model_init = model_init

        self.setup_model_parallelism(model)

        # one place to sort out whether to place the model on device or not
        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        # 3. full bf16 or fp16 eval - since the model needs to be cast to the right dtype first
        # 4. FSDP - same as MP
        self.place_model_on_device = args.place_model_on_device
        if (
            self.is_model_parallel
            or ((args.fp16_full_eval or args.bf16_full_eval) and not args.do_train)
            or self.using_model_orchestration
        ):
            self.place_model_on_device = False
        # Bnb quantized models don't support `to` operation
        if (
            self.place_model_on_device
            and not getattr(model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
        ):
            self.model = self._move_model_to_device(model, args.device)

        model_sanity_checks(model)
        optimizer_sanity_checks(model, self.optimizer, self.lr_scheduler, self.using_model_orchestration)

        # We keep two references so that later we can check if `self.model is self.model_wrapped`
        self.model_wrapped = self.model = model

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # Mixed precision setup
        self.use_apex = False
        self.use_cpu_amp = False

        # Mixed precision: SageMaker Model Parallel
        if is_sagemaker_mp_enabled():
            # BF16 + model parallelism in SageMaker: currently not supported, raise an error
            if args.bf16:
                raise ValueError("SageMaker Model Parallelism does not support BF16 yet. Please use FP16 instead ")

            if IS_SAGEMAKER_MP_POST_1_10:
                # When there's mismatch between SMP config and trainer argument, use SMP config as truth
                if args.fp16 != smp.state.cfg.fp16:
                    logger.warning(
                        f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
                        f"but FP16 provided in trainer argument is {args.fp16}, "
                        f"setting to {smp.state.cfg.fp16}"
                    )
                    args.fp16 = smp.state.cfg.fp16
            else:
                # smp < 1.10 does not support fp16 in trainer.
                if hasattr(smp.state.cfg, "fp16"):
                    logger.warning(
                        f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
                        "but SageMaker Model Parallelism < 1.10 does not support FP16 in trainer."
                    )

        # Mixed precision (when not using `Accelerator`)
        if args.half_precision_backend == "auto":
            if args.device == torch.device("cpu"):
                if args.fp16 and not is_torch_greater_or_equal_than_2_3:
                    raise ValueError("Tried to use `fp16` but it is not supported on cpu")
                elif args.bf16:
                    args.half_precision_backend = "cpu_amp"
            logger.info(f"Using {args.half_precision_backend} half precision backend")

        if (args.fp16 or args.bf16) and not (self.is_deepspeed_enabled or is_sagemaker_mp_enabled()):
            # deepspeed and SageMaker Model Parallel manage their own half precision
            if args.half_precision_backend == "cpu_amp":
                self.use_cpu_amp = True
                self.amp_dtype = torch.bfloat16
            elif args.half_precision_backend == "apex":
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to"
                        " https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True

        # Label smoothing
        self.label_smoother = None
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)

        # Callbacks
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler, self.accelerator
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        if self.args.include_num_input_tokens_seen:
            self.token_tracking_cb = self.add_callback(InputTokenTrackingCallback)
        if self.args.neftune_noise_alpha is not None:
            self.add_callback(NefttuneCallback(self.args.neftune_noise_alpha))

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        self.control = TrainerControl()

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0

        self.hp_search_backend = None
        self.label_names = (
            self.args.label_names if self.args.label_names is not None else find_labels(self.model.__class__)
        )
        self.can_return_loss = can_return_loss(self.model.__class__)

        # Internal variables to help with automatic batch size reduction
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False

        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # very last
        self._memory_tracker.stop_and_update_metrics()

        # torch.compile
        if args.torch_compile and not is_torch_compile_available():
            raise RuntimeError("Using torch.compile requires PyTorch 2.0 or higher.")

        self.is_fsdp_xla_v2_enabled = args.fsdp_config.get("xla_fsdp_v2", False)

    @property
    def using_model_orchestration(self):
        "Returns whether the underlying model is using training orchestration such as FSDP or DeepSpeed"
        return self.is_fsdp_enabled or self.is_fsdp_xla_enabled or self.is_deepspeed_enabled

    @property
    def is_fsdp_xla_enabled(self):
        return self.args.fsdp_config.get("xla", False)

    def call_model_init(self, trial=None):
        "Instantiates a model using the model init function."
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("`model_init` should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("`model_init` should not return None.")

        return model

    def perform_distributed_sanity_checks(self):
        # Model orchestration verifications
        if self.using_model_orchestration:
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            # `save_only_model` can't be used with DeepSpeed/FSDP along with `load_best_model_at_end`
            if self.args.save_only_model and self.args.load_best_model_at_end:
                raise ValueError(
                    f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`."
                )
            # `auto_find_batch_size` isn't yet supported with DeepSpeed/FSDP
            if self.args.auto_find_batch_size:
                raise NotImplementedError(f"`{wrapper}` doesn't support `auto_find_batch_size`.")
            if len(self.args.fsdp) > 0:
                if self.is_deepspeed_enabled:
                    raise ValueError(
                        "Using --fsdp xxx together with --deepspeed is not possible, deactivate one of those flags."
                    )
                if not self.is_fsdp_xla_enabled and self.args.parallel_mode != ParallelMode.DISTRIBUTED:
                    raise ValueError("Using fsdp only works in distributed training.")

    def setup_model_parallelism(self, model):
        # First check based on model attributes
        self.is_model_parallel = getattr(model, "is_parallelizable", False) and getattr(model, "model_parallel", False)
        # Then base it off the `device_map`
        if getattr(model, "hf_device_map", None) is not None:
            devices = [device for device in set(model.hf_device_map.values()) if device not in ["cpu", "disk"]]
            self.is_model_parallel = len(devices) > 1 or (
                len(devices) == 1 and self.args.device != torch.device(devices[0])
            )
            # warn users
            if self.is_model_parallel:
                logger.warn(
                    "You have loaded a model on multiple GPUs. `is_model_parallel` attribute will be force-set"
                    " to `True` to avoid any unexpected behavior such as device placement mismatching."
                )

    def _wrap_model(self, model, training=True, dataloader=None):
        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if self.accelerator.unwrap_model(model) is not model:
            return model

        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model, self.optimizer = apply_ipex_optimization(
                model, self.optimizer, training, dtype=dtype, is_in_train=self.args.is_in_train
            )

        if is_sagemaker_mp_enabled():
            # Wrapping the base model twice in a DistributedModel will raise an error.
            if isinstance(self.model_wrapped, smp.model.DistributedModel):
                return self.model_wrapped
            return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization) / 8bit models does not support DDP
        if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
            model = nn.DataParallel(model)

        if self.args.jit_mode_eval:
            start_time = time.time()
            model = self.torch_jit_model_eval(model, dataloader, training)
            self.jit_compilation_time = round(time.time() - start_time, 4)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Distributed training (should be after apex fp16 initialization)
        # Distributed training using PyTorch FSDP on XLA (not supported in accelerate yet)
        if self.is_fsdp_xla_enabled:
            model = apply_fsdp_xla_optimization(
                model, self.optimizer, self.args.fsdp_config, self.is_fsdp_xla_v2_enabled
            )

            # Patch `xm.optimizer_step` should not reduce gradients in this case,
            # as FSDP does not need gradient reduction over sharded parameters.
            def patched_optimizer_step(optimizer, barrier=False, optimizer_args={}):
                loss = optimizer.step(**optimizer_args)
                if barrier:
                    xm.mark_step()
                return loss

            xm.optimizer_step = patched_optimizer_step
        # Sagemaker DP
        elif is_sagemaker_dp_enabled():
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))]
            )
        # Otherwise make sure that `Accelerate` has the proper args
        elif self.args.parallel_mode == ParallelMode.DISTRIBUTED and not is_torch_neuroncore_available():
            kwargs = {"find_unused_parameters": True}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing

            kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb

            if self.args.ddp_broadcast_buffers is not None:
                kwargs["broadcast_buffers"] = self.args.ddp_broadcast_buffers

            self.accelerator.ddp_handler = DistributedDataParallelKwargs(**kwargs)
        return model

    def torch_jit_model_eval(self, model, dataloader, training=False):
        if not training:
            if dataloader is None:
                logger.warning("failed to use PyTorch jit mode due to current dataloader is none.")
                return model
            example_batch = next(iter(dataloader))
            example_batch = self._prepare_inputs(example_batch)
            try:
                jit_model = copy.copy(model)
                jit_model.eval()
                original_forward = jit_model.__dict__.pop("_original_forward", None)
                # remove mixed precision hooks from the model
                if original_forward:
                    jit_model.forward = original_forward
                with self.accelerator.autocast(cache_enabled=False), torch.no_grad():
                    if version.parse(version.parse(torch.__version__).base_version) >= version.parse("2.0.0"):
                        if isinstance(example_batch, dict):
                            jit_model = torch.jit.trace(jit_model, example_kwarg_inputs=example_batch, strict=False)
                        else:
                            jit_model = torch.jit.trace(
                                jit_model,
                                example_kwarg_inputs={key: example_batch[key] for key in example_batch},
                                strict=False,
                            )
                    else:
                        jit_inputs = []
                        for key in example_batch:
                            example_tensor = torch.ones_like(example_batch[key])
                            jit_inputs.append(example_tensor)
                        jit_inputs = tuple(jit_inputs)
                        jit_model = torch.jit.trace(jit_model, jit_inputs, strict=False)
                jit_model = torch.jit.freeze(jit_model)
                with torch.no_grad():
                    jit_model(**example_batch)
                    jit_model(**example_batch)
                model = jit_model
                self.use_cpu_amp = False
            except (RuntimeError, TypeError, ValueError, NameError, IndexError) as e:
                logger.warning(f"failed to use PyTorch jit mode due to: {e}.")

        return model

    def _move_model_to_device(self, model, device):
        model = model.to(device)
        # Moving a model to an XLA device disconnects the tied weights, so we have to retie them.
        if self.args.parallel_mode == ParallelMode.TPU and hasattr(model, "tie_weights"):
            model.tie_weights()

    def add_callback(self, callback):
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        self.callback_handler.remove_callback(callback)

    def create_accelerator_and_postprocess(self):
        # create accelerator object
        self.accelerator = create_accelerator(self.args)
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        if "use_gather_object" in inspect.signature(self.gather_function).parameters.keys():
            self.gather_function = functools.partial(
                self.gather_function, use_gather_object=self.args.eval_use_gather_object
            )

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        # NOTE: This should be simplified to just build a FSDP plugin manually w/ overrides
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
                "limit_all_gathers", fsdp_plugin.limit_all_gathers
            )
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
                    "activation_checkpointing", fsdp_plugin.activation_checkpointing
                )
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError(
                        "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                        "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                        "when using FSDP."
                    )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            propagate_args_to_deepspeed(self.args)

        self.perform_distributed_sanity_checks()

    def _update_qlora_fsdp_plugin(self):
        if self.is_fsdp_enabled and is_peft_model(self.model):
            from peft import LoraConfig
            from peft.utils.other import fsdp_auto_wrap_policy

            if isinstance(self.model.active_peft_config, LoraConfig):
                fsdp_plugin = self.accelerator.state.fsdp_plugin
                fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(self.mode)

                if (
                    getattr(self.model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
                    and self.model.hf_quantizer.quantization_config.bnb_4bit_quant_storage.is_floating_point
                    and is_accelerate_available("0.28.0")
                ):
                    fsdp_plugin.set_mixed_precision(
                        self.model.hf_quantizer.quantization_config.bnb_4bit_quant_storage, override=True
                    )

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is not None:
            return

        # Determine which model to inspect
        model_to_inspect = self.model
        if is_peft_model(self.model):
            if hasattr(self.model, "get_base_model"):
                model_to_inspect = self.model.get_base_model()
            else:
                # PeftMixedModel does not provide a `get_base_model` method
                model_to_inspect = self.model.base_model.model

        signature = inspect.signature(model_to_inspect.forward)
        self._signature_columns = list(signature.parameters.keys())

        # Labels may be named label or label_ids, the default data collator handles that.
        label_columns = set(["label", "label_ids"] + self.label_names)
        self._signature_columns.extend(label_columns)

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.warning(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if len(columns) == 0:
            raise ValueError(
                "No columns in the dataset match the model's forward method signature. "
                f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "
                "Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def _get_sampler(self, dataset=None, evaluation: bool = False) -> Optional[torch.utils.data.Sampler]:
        if evaluation and self.args.world_size < 1 or dataset is None or not has_length(dataset):
            return None

        if evaluation:
            return SequentialSampler(dataset)

        # Build the sampler.
        if self.args.group_by_length:
            lengths = None
            if (
                is_datasets_available()
                and isinstance(dataset, datasets.Dataset)
                and self.args.length_column_name in dataset.column_names
            ):
                lengths = dataset[self.args.length_column_name]
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return RandomSampler(dataset)

    def _prepare_dataset_for_dataloader(self, dataset, data_collator, dataset_type: str = "train"):
        """
        Prepare a dataset for a dataloader.

        Returns:
            A tuple of (dataset, dataloader_params)
        """
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=dataset_type)
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description=dataset_type)

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "batch_size": self._train_batch_size if dataset_type == "train" else self.args.eval_batch_size,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_sampler(dataset, evaluation=dataset_type != "train")
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if dataset_type == "train":
                dataloader_params["worker_init_fn"] = seed_worker
        return dataset, dataloader_params

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset, dataloader_params = self._prepare_dataset_for_dataloader(self.train_dataset, self.data_collator)
        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        if self.is_fsdp_xla_v2_enabled:
            dataloader = tpu_spmd_dataloader(dataloader)
        return dataloader

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        # Grab the right `eval_dataset` if needed
        if isinstance(eval_dataset, str):
            eval_dataset = self.eval_dataset[eval_dataset]
        elif eval_dataset is None:
            eval_dataset = self.eval_dataset

        eval_dataset, dataloader_params = self._prepare_dataset_for_dataloader(
            eval_dataset, self.data_collator, dataset_type="evaluation"
        )
        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        eval_dataloader = self.accelerator.prepare(eval_dataloader)

        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        return eval_dataloader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        test_dataset, dataloader_params = self._prepare_dataset_for_dataloader(
            test_dataset, self.data_collator, dataset_type="test"
        )
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def num_tokens(self, train_dl: DataLoader, max_steps: Optional[int] = None) -> int:
        """
        Helper to get number of tokens in a [`~torch.utils.data.DataLoader`] by enumerating dataloader.
        """
        train_tokens = 0
        try:
            for step, batch in enumerate(train_dl):
                tokens = batch["input_ids"].numel()
                if max_steps is not None:
                    return tokens * max_steps
                train_tokens += tokens
            return train_tokens
        except KeyError:
            logger.warning("Cannot get num_tokens from dataloader")
            return train_tokens

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is not None:
            if is_sagemaker_mp_enabled():
                self.optimizer = smp.DistributedOptimizer(self.optimizer)
            return self.optimizer
        decay_parameters = get_decay_parameter_names(opt_model)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

        # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
        # e.g. for GaLore optimizer.
        if "params" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("params")

        # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
        # e.g. for LOMO optimizer.
        if "model" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("model")

        # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
        # to avoid arguments conflicts.
        if "optimizer_dict" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if optimizer_cls.__name__ == "Adam8bit":
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    logger.info(f"skipped {module}: {skipped/2**20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        optimizer = self.optimizer
        # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
        if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
            optimizer = optimizer.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        # Special case for SageMaker ModelParallel since there process_index is dp_process_index, not the global
        # process index.
        if is_sagemaker_mp_enabled():
            return smp.rank() == 0
        else:
            return self.args.process_index == 0

    def get_num_trainable_parameters(self):
        """
        Get the number of trainable parameters.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_learning_rates(self):
        """
        Returns the learning rate of each parameter from self.optimizer.
        """
        if self.optimizer is None:
            raise ValueError("Trainer optimizer is None, please make sure you have setup the optimizer before.")
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_optimizer_group(self, param: Optional[Union[str, torch.nn.parameter.Parameter]] = None):
        """
        Returns optimizer group for a parameter if given, else returns all optimizer groups for params.
        Args:
            param (`str` or `torch.nn.parameter.Parameter`, *optional*):
                The parameter for which optimizer group needs to be returned.
        """
        if self.optimizer is None:
            raise ValueError("Trainer optimizer is None, please make sure you have setup the optimizer before.")
        if param is not None:
            for group in self.optimizer.param_groups:
                if param in group["params"]:
                    return group
        return [group["params"] for group in self.optimizer.param_groups]

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_cls = None
        optimizer_kwargs = {"lr": args.learning_rate}

        class_name = None
        optimizer_groups = OptimizerGroups.find_groups(args.optim)

        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif "ADAMW" in optimizer_groups:
            optimizer_kwargs.update(
                {
                    "betas": (args.adam_beta1, args.adam_beta2),
                    "eps": args.adam_epsilon,
                }
            )
            class_name = "AdamW"
            import_to_try = None

            # Based on the optimizer, import/grab the right `__init__`
            if args.optim == OptimizerNames.ADAMW_HF:
                from .optimization import AdamW
            elif "ADAMW_TORCH" in optimizer_groups:
                from torch.optim import AdamW

                if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                    optimizer_kwargs.update({"fused": True})
            elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
                import_to_try = "torch_xla.amp.syncfree"
            elif args.optim == OptimizerNames.ADAMW_TORCH_NPU_FUSED:
                import_to_try = "torch_npu.optim"
                class_name = "NpuFusedAdamW"
            elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
                import_to_try = "apex.optimizers"
                class_name = "FusedAdam"
            elif args.optim == OptimizerNames.ADAMW_ANYPRECISION:
                import_to_try = "torchdistx.optimizers"
                class_name = "AnyPrecisionAdamW"
                # TODO Change dtypes back to M=FP32, Var = BF16, Kahan = False once they can be cast together in torchdistx.
                optimizer_kwargs.update(
                    {
                        "use_kahan_summation": strtobool(optim_args.get("use_kahan_summation", "False")),
                        "momentum_dtype": getattr(torch, optim_args.get("momentum_dtype", "float32")),
                        "variance_dtype": getattr(torch, optim_args.get("variance_dtype", "float32")),
                        "compensation_buffer_dtype": getattr(
                            torch, optim_args.get("compensation_buffer_dtype", "bfloat16")
                        ),
                    }
                )

            elif "BNB_COMPATIBLE" in optimizer_groups:
                import_to_try = "bitsandbytes.optim"

            # If we're not needing to import anything, use the earlier imported AdamW
            if import_to_try is None:
                optimizer_cls = AdamW
            else:
                try:
                    module = importlib.import_module(import_to_try)
                    optimizer_cls = getattr(module, class_name)
                except (ModuleNotFoundError, AttributeError):
                    raise ImportError(
                        f"Trainer failed to import {class_name} from {import_to_try}. Please ensure the library you are using is installed."
                    )

        elif "RMSPROP" in optimizer_groups:
            try:
                from bitsandbytes.optim import RMSprop
            except ImportError:
                raise ImportError("Trainer tried to instantiate bnb optimizer but bnb is not installed!")
            optimizer_cls = RMSprop
            # Above we pass all `adam_kwargs` to the optimizer, here
            # we only pass `optim_args` which can be passed by the user.
            optimizer_kwargs.update(optim_args)
        elif "LION" in optimizer_groups:
            try:
                from bitsandbytes.optim import Lion
            except ImportError:
                raise ImportError("Trainer tried to instantiate bnb optimizer but bnb is not installed!")
            optimizer_cls = Lion
            optimizer_kwargs.update({"betas": (args.adam_beta1, args.adam_beta2)})

        # Now add in the bnb specifics
        if "BNB_COMPATIBLE" in optimizer_groups:
            optimizer_kwargs.update(
                {
                    "optim_bits": 32 if "8bit" not in args.optim else 8,
                    "is_paged": "paged" in args.optim and "rmsprop" not in args.optim,
                }
            )

        # And potentially return if ready
        if optimizer_cls is not None:
            return optimizer_cls, optimizer_kwargs

        if "TORCH_NATIVE" in optimizer_groups:
            optimizer_cls = getattr(torch.optim, args.optim)
            return optimizer_cls, optimizer_kwargs

        if "GALORE" in optimizer_groups:
            if not is_galore_torch_available():
                raise ImportError(
                    "You need to install `galore_torch` in order to use GaLore optimizers"
                    " install it with `pip install git+https://github.com/jiaweizzhao/GaLore`"
                )
            from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit

            is_layerwise = args.optim.lower().endswith("layerwise")
            if is_layerwise and args.parallel_mode == ParallelMode.DISTRIBUTED:
                raise NotImplementedError("Layer-wise GaLore does not support DDP at this time")

            if args.optim_target_modules is None:
                raise ValueError(
                    "You need to define a `optim_target_modules` in order to properly use GaLore optimizers"
                )

            if model is None:
                raise ValueError("You need to pass a model in order to correctly initialize a GaLore optimizer.")

            logger.warning(
                "Activated GaLoRE fine-tuning, depending on your model size and hardware, the training might take a while before starting. Please be patient!"
            )

            if args.optim in [OptimizerNames.GALORE_ADAMW, OptimizerNames.GALORE_ADAMW_LAYERWISE]:
                optimizer_cls = GaLoreAdamW
            elif args.optim in [OptimizerNames.GALORE_ADAMW_8BIT, OptimizerNames.GALORE_ADAMW_8BIT_LAYERWISE]:
                optimizer_cls = GaLoreAdamW8bit
            elif args.optim in [OptimizerNames.GALORE_ADAFACTOR, OptimizerNames.GALORE_ADAFACTOR_LAYERWISE]:
                optimizer_cls = GaLoreAdafactor

            all_linear = (
                isinstance(args.optim_target_modules, str)
                and args.optim_target_modules.replace("_", "-") == "all-linear"
            )

            galore_params = []
            galore_params_names = []
            for module_name, module in model.named_modules():
                target_module_exists, is_regex = check_target_module_exists(
                    args.optim_target_modules, module_name, return_is_regex=True
                )

                if not isinstance(module, nn.Linear):
                    # Warn in case we match but it's not a linear layer
                    if target_module_exists and not is_regex:
                        logger.warning(
                            f"{module_name} has been matched but ignored as GaLore only supports linear layers. Please double check your `optim_target_modules`!"
                        )

                    continue

                if not target_module_exists and not all_linear:
                    continue

                galore_params.append(module.weight)
                galore_params_names.append(module_name + ".weight")

            if len(galore_params) == 0:
                raise ValueError(
                    f"None of the target modules were found! ({args.optim_target_modules}). Please make sure to pass a valid `target_modules`."
                )

            non_galore_params = [p for n, p in model.named_parameters() if n not in galore_params_names]

            galore_optim_kwargs = {
                "rank": int(optim_args.pop("rank", 128)),
                "update_proj_gap": int(optim_args.pop("update_proj_gap", 200)),
                "scale": float(optim_args.pop("scale", 0.25)),
                "proj_type": optim_args.pop("proj_type", "std"),
            }

            # The default args are from the official repository: https://github.com/jiaweizzhao/GaLore
            param_groups = [
                {"params": non_galore_params},
                {"params": galore_params, **galore_optim_kwargs},
            ]

            if is_layerwise:
                # For layer-wise optimizers, the optimization step is done through post accumulation
                # gradient hooks. The trick is to first attach these hooks to the model parameters then
                # create a dummy optimizer that will perform no-ops in the Trainer.
                # See the original implementation or the nice implementation from @hiyouga
                # here: https://github.com/hiyouga/LLaMA-Factory/commit/8664262cde3919e10eaecbd66e8c5d356856362e#diff-ebe08ab14496dfb9e06075f0fdd36799ef6d1535cc4dd4715b74c4e3e06fe3ba
                if args.gradient_accumulation_steps != 1:
                    raise ValueError("Layerwise GaLoRE optimizer do not support gradient accumulation !")

                optimizer_dict = {}
                for param in non_galore_params:
                    param_groups = [{"params": [param]}]
                    optimizer_dict[param] = optimizer_cls(param_groups, **optimizer_kwargs)
                for param in galore_params:
                    param_groups = [{"params": [param], **galore_optim_kwargs}]
                    optimizer_dict[param] = optimizer_cls(param_groups, **optimizer_kwargs)

                def optimizer_hook(param):
                    if param.grad is not None:
                        optimizer_dict[param].step()
                        optimizer_dict[param].zero_grad()

                for param in model.parameters():
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(optimizer_hook)

                optimizer_cls = LayerWiseDummyOptimizer
                optimizer_kwargs.update({"optimizer_dict": optimizer_dict})

            optimizer_kwargs.update({"params": param_groups})

            if args.optim == OptimizerNames.GALORE_ADAFACTOR:
                optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
            return optimizer_cls, optimizer_kwargs

        if "LOMO" in optimizer_groups:
            if not is_lomo_available():
                raise ImportError(
                    "You need to install `lomo_optim` in order to use LOMO optimizers"
                    " install it with `pip install lomo-optim`"
                )
            if not is_accelerate_available("0.30.0"):
                raise ImportError("You need to have `accelerate>=0.30.0` to be able to use LOMO optimizers")

            if model is None:
                raise ValueError("You need to pass a `model` in order to correctly initialize a LOMO optimizer.")

            from lomo_optim import AdaLomo, Lomo

            if "ada" in args.optim:
                optimizer_cls = AdaLomo
            else:
                optimizer_cls = Lomo

            optimizer_kwargs.update({"model": model})
            return optimizer_cls, optimizer_kwargs

        # Finally raise an error if we're here
        raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")

    def _inner_training_loop(
        self,
        batch_size=None,
        args: TrainingArguments = None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")

        train_dataloader = self.get_train_dataloader()
        if not has_length(train_dataloader) and args.max_steps < 1:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None

        # Since the actual `max_steps` depends on if we have a dataloader length, number of epochs, or max_steps,
        # we keep it defined here first as the "base case" if we don't have a dataloader length later
        max_steps = args.max_steps
        # If max_steps is negative, we use the number of epochs to determine the number of total steps later
        epoch_based = max_steps < 0

        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_train_epochs = math.ceil(args.num_train_epochs) if epoch_based else sys.maxsize
        num_update_steps_per_epoch = max_steps
        num_examples = total_train_batch_size * max_steps
        # May be slightly incorrect if the last batch in the training dataloader has a smaller size
        # but it's the best we can do
        num_train_samples = max_steps * total_train_batch_size

        # Case 2: We have a dataloader length and can extrapolate
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = max(len_dataloader // args.gradient_accumulation_steps, 1)
            num_examples = self.num_examples(train_dataloader)
            if epoch_based:
                # Since we have a dataloader length and are utilizing epochs, we can extrapolate the number of steps
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            else:
                # Otherwise, we calculate the number of epochs from max_steps & update steps
                num_train_epochs = max_steps // num_update_steps_per_epoch + int(
                    max_steps % num_update_steps_per_epoch > 0
                )

        if args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            if epoch_based:
                num_train_tokens *= args.gradient_accumulation_steps

        # Can this be a callback?
        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_enabled or self.is_fsdp_xla_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
            is_hyper_param_search=trial is not None,
            train_batch_size=self._train_batch_size,
        )
        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.populate_initial_step_values(args, max_steps)

        # Deal with model setup
        # 1. Gradient checkpointing
        if args.gradient_checkpointing:
            gradient_checkpointing_kwargs = {}
            if args.gradient_checkpointing_kwargs is not None:
                gradient_checkpointing_kwargs.update(args.gradient_checkpointing_kwargs)

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        # 2. Wrap the model
        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # unhandled cases such as FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._update_qlora_fsdp_plugin()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # 3. call with prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases like "DummyScheduler" for DeepSpeed via the config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif "LOMO" in OptimizerGroups.find_groups(self.args.optim):
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # Load from checkpointing
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)
        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None

        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        grad_norm = clip_grad_norm_(args, model, self.optimizer, self.accelerator)
                        if self.is_deepspeed_enabled:
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()

                    self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run and not isinstance(self.lr_scheduler, ReduceLROnPlateau):
                        # We delay optimizer scheduling until the metrics are generated
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

            if args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of training
                delattr(self, "_past")

            logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
            if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
                # Wait for everyone to get here so we are sure the model has been saved by process 0.
                wait_for_everyone()
                self._load_best_model()

            # add remaining tr_loss
            self._total_loss_scalar += tr_loss.item()
            effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
            train_loss = self._total_loss_scalar / effective_global_step

            metrics = speed_metrics(
                "train",
                start_time,
                num_samples=num_train_samples,
                num_steps=self.state.max_steps,
                num_tokens=num_train_tokens,
            )
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
            metrics["train_loss"] = train_loss

            self.is_in_train = False

            self._memory_tracker.stop_and_update_metrics(metrics)

            self.log(metrics)

            run_dir = self._get_output_dir(trial)
            checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

            # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
            if (
                self.args.should_save
                and self.state.best_model_checkpoint is not None
                and self.args.save_total_limit == 1
            ):
                for checkpoint in checkpoints_sorted:
                    if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                        shutil.rmtree(checkpoint, ignore_errors=True)
            self.control = self.callback_handler.on_train_end(args, self.state, self.control)

            # Wait for the checkpoint to be uploaded.
            self._finish_current_push()

            return TrainOutput(self.state.global_step, train_loss, metrics)

    def compute_loss_context_manager(self, cache_enabled=False):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        Only really enabled when `use_cpu_amp=True`, else delegates it to `Accelerate`
        """
        if self.use_cpu_amp:
            ctx_manager = torch.cpu.amp.autocast(cache_enabled=cache_enabled, dtype=self.amp_dtype)
        else:
            ctx_manager = contextlib.nullcontext()

        return ctx_manager

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learning rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.
        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in `transformers` 5.0.0. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        if args.push_to_hub:
            try:
                # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop

        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    kwargs = {"predictions": logits, "label_ids": labels}
                    if args.include_inputs_for_metrics:
                        kwargs["inputs"] = inputs
                    metrics = self.compute_metrics(
                        EvalPrediction(**kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            kwargs = {"predictions": all_preds, "label_ids": all_labels}
            if args.include_inputs_for_metrics:
                kwargs["inputs"] = all_inputs
            metrics = self.compute_metrics(EvalPrediction(**kwargs))
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.token_tracking_cb.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _hp_search_setup(self, trial: Union["optuna.Trial", Dict[str, Any]]):
        """HP search setup code"""
        self._trial = trial

        if self.hp_search_backend is None or trial is None:
            return
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            params = self.hp_space(trial)
        elif self.hp_search_backend == HPSearchBackend.RAY:
            params = trial
            params.pop("wandb", None)
        elif self.hp_search_backend == HPSearchBackend.SIGOPT:
            params = {k: int(v) if isinstance(v, str) else v for k, v in trial.assignments.items()}
        elif self.hp_search_backend == HPSearchBackend.WANDB:
            params = trial

        for key, value in params.items():
            if not hasattr(self.args, key):
                logger.warning(
                    f"Trying to set {key} in the hyperparameter search but there is no corresponding field in"
                    " `TrainingArguments`."
                )
                continue
            old_attr = getattr(self.args, key, None)
            # Casting value to the proper type
            if old_attr is not None:
                value = type(old_attr)(value)

            setattr(self.args, key, value)
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            logger.info(f"Trial: {trial.params}")
        if self.hp_search_backend == HPSearchBackend.SIGOPT:
            logger.info(f"SigOpt Assignments: {trial.assignments}")
        if self.hp_search_backend == HPSearchBackend.WANDB:
            logger.info(f"W&B Sweep parameters: {trial}")
        if self.is_deepspeed_enabled:
            if self.args.deepspeed is None:
                raise ValueError("For sweeps with deepspeed, `args.deepspeed` must be set")
            # Rebuild the deepspeed config to reflect the updated training parameters
            from accelerate.utils import DeepSpeedPlugin

            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

            self.args.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.args.deepspeed)
            self.args.hf_deepspeed_config.trainer_config_process(self.args)
            self.args.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.args.hf_deepspeed_config)

        self.create_accelerator_and_postprocess()

    def _report_to_hp_search(self, trial: Union["optuna.Trial", Dict[str, Any]], step: int, metrics: Dict[str, float]):
        if self.hp_search_backend is None or trial is None:
            return
        metrics = metrics.copy()
        self.objective = self.compute_objective(metrics)
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            import optuna

            if not trial.study._is_multi_objective():
                trial.report(self.objective, step)
                if trial.should_prune():
                    self.callback_handler.on_train_end(self.args, self.state, self.control)
                    raise optuna.TrialPruned()
        elif self.hp_search_backend == HPSearchBackend.RAY:
            import ray.train

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                if self.control.should_save:
                    self._tune_save_checkpoint(checkpoint_dir=temp_checkpoint_dir)
                    checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
                metrics["objective"] = self.objective
                ray.train.report(metrics, checkpoint=checkpoint)

    def hyperparameter_search(
        self,
        hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
        compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
        n_trials: int = 20,
        direction: Union[str, List[str]] = "minimize",
        backend: Optional[Union["str", HPSearchBackend]] = None,
        hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
        **kwargs,
    ) -> Union[BestRun, List[BestRun]]:
        """
        Launch an hyperparameter search using `optuna` or `Ray Tune` or `SigOpt`. The optimized quantity is determined
        by `compute_objective`, which defaults to a function returning the evaluation loss when no metric is provided,
        the sum of all metrics otherwise.

        <Tip warning={true}>

        To use this method, you need to have provided a `model_init` when initializing your [`Trainer`]: we need to
        reinitialize the model at each new run. This is incompatible with the `optimizers` argument, so you need to
        subclass [`Trainer`] and override the method [`~Trainer.create_optimizer_and_scheduler`] for custom
        optimizer/scheduler.

        </Tip>

        Args:
            hp_space (`Callable[["optuna.Trial"], Dict[str, float]]`, *optional*):
                A function that defines the hyperparameter search space. Will default to
                [`~trainer_utils.default_hp_space_optuna`] or [`~trainer_utils.default_hp_space_ray`] or
                [`~trainer_utils.default_hp_space_sigopt`] depending on your backend.
            compute_objective (`Callable[[Dict[str, float]], float]`, *optional*):
                A function computing the objective to minimize or maximize from the metrics returned by the `evaluate`
                method. Will default to [`~trainer_utils.default_compute_objective`].
            n_trials (`int`, *optional*, defaults to 100):
                The number of trial runs to test.
            direction (`str` or `List[str]`, *optional*, defaults to `"minimize"`):
                If it's single objective optimization, direction is `str`, can be `"minimize"` or `"maximize"`, you
                should pick `"minimize"` when optimizing the validation loss, `"maximize"` when optimizing one or
                several metrics. If it's multi objectives optimization, direction is `List[str]`, can be List of
                `"minimize"` and `"maximize"`, you should pick `"minimize"` when optimizing the validation loss,
                `"maximize"` when optimizing one or several metrics.
            backend (`str` or [`~training_utils.HPSearchBackend`], *optional*):
                The backend to use for hyperparameter search. Will default to optuna or Ray Tune or SigOpt, depending
                on which one is installed. If all are installed, will default to optuna.
            hp_name (`Callable[["optuna.Trial"], str]]`, *optional*):
                A function that defines the trial/run name. Will default to None.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to `optuna.create_study` or `ray.tune.run`. For more
                information see:

                - the documentation of
                  [optuna.create_study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html)
                - the documentation of [tune.run](https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run)
                - the documentation of [sigopt](https://app.sigopt.com/docs/endpoints/experiments/create)

        Returns:
            [`trainer_utils.BestRun` or `List[trainer_utils.BestRun]`]: All the information about the best run or best
            runs for multi-objective optimization. Experiment summary can be found in `run_summary` attribute for Ray
            backend.
        """
        if backend is None:
            backend = default_hp_search_backend()
        backend = HPSearchBackend(backend)
        backend_obj = ALL_HYPERPARAMETER_SEARCH_BACKENDS[backend]()
        backend_obj.ensure_available()
        self.hp_search_backend = backend
        if self.model_init is None:
            raise RuntimeError(
                "To use hyperparameter search, you need to pass your model through a model_init function."
            )

        self.hp_space = backend_obj.default_hp_space if hp_space is None else hp_space
        self.hp_name = hp_name
        self.compute_objective = default_compute_objective if compute_objective is None else compute_objective

        best_run = backend_obj.run(self, n_trials, direction, **kwargs)

        self.hp_search_backend = None
        return best_run

    def _tune_save_checkpoint(self, checkpoint_dir: str):
        output_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        self.save_model(output_dir, _internal_call=True)
        if self.args.should_save:
            # Update the `TrainerControl` state to where we are currently
            self.state.stateful_callbacks["TrainerControl"] = self.control.state()
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    def _get_output_dir(self, trial):
        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.RAY:
                import ray.train

                run_id = ray.train.get_context().get_trial_id()
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            elif self.hp_search_backend == HPSearchBackend.WANDB:
                import wandb

                run_id = wandb.run.id
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
        return run_dir

    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if (
            self.state.best_model_checkpoint is not None
            and str(Path(self.state.best_model_checkpoint)) in checkpoints_sorted
        ):
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(
                self.model_wrapped,
                self.state.best_model_checkpoint,
                load_module_strict=not is_peft_model(self.model),
            )
        elif self.is_fsdp_enabled:
            load_result = load_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                model,
                self.state.best_model_checkpoint,
                **_get_fsdp_ckpt_kwargs(),
            )
        elif (
            os.path.exists(best_model_path)
            or os.path.exists(best_safe_model_path)
            or os.path.exists(best_adapter_model_path)
            or os.path.exists(best_safe_adapter_model_path)
        ):
            has_been_loaded = True
            weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
            if is_sagemaker_mp_enabled():
                load_result = load_sagemaker_model(self.state.best_model_checkpoint, model, best_safe_model_path)
            else:
                if is_peft_model(model):
                    has_been_loaded = load_peft_adapters(self.state.best_model_checkpoint, model, [])
                    # Load_adapter has no return value present, modify it when appropriate.
                    from torch.nn.modules.module import _IncompatibleKeys

                    load_result = _IncompatibleKeys([], [])
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                        state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                    else:
                        state_dict = torch.load(
                            best_model_path,
                            map_location="cpu",
                            **weights_only_kwarg,
                        )

                    # If the model is on the GPU, it still works!
                    # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                    # which takes *args instead of **kwargs
                    load_result = model.load_state_dict(state_dict, False)
                if not is_sagemaker_mp_enabled() and has_been_loaded:
                    self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_INDEX_NAME)) or os.path.exists(
            os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)
        ):
            load_result = load_sharded_checkpoint(
                model, self.state.best_model_checkpoint, strict=is_sagemaker_mp_enabled()
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])

        is_parallel = self.args.parallel_mode == ParallelMode.DISTRIBUTED
        if is_torch_cuda_available():
            set_rng_state(torch.cuda, checkpoint_rng_state["cuda"], is_parallel)
        if is_torch_xla_available():
            set_rng_state(xm, checkpoint_rng_state["xla"])
        if is_torch_npu_available():
            set_rng_state(torch.npu, checkpoint_rng_state["npu"], is_parallel)
        if is_torch_mlu_available():
            set_rng_state(torch.mlu, checkpoint_rng_state["mlu"], is_parallel)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)

        is_fsdp_checkpoint = os.path.isdir(resume_from_checkpoint) and (
            # Check for `SHARDED_STATE_DICT`
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            # Check for `FULL_STATE_DICT`
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        if is_fsdp_checkpoint and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        # If multiple adapters exist, they get saved in subdirectories
        adapter_subdirs = []
        if os.path.isdir(resume_from_checkpoint):
            adapter_subdirs = [
                folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                and (
                    os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_WEIGHTS_NAME))
                    or os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_SAFE_WEIGHTS_NAME))
                )
            ]

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_checkpoint
            or adapter_subdirs
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if any([os.path.isfile(weights_file), os.path.isfile(safe_weights_file), is_fsdp_checkpoint]):
            weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
            if is_sagemaker_mp_enabled():
                load_result = load_sagemaker_model(resume_from_checkpoint, model, weights_file, self.args.fp16)
            elif self.is_fsdp_enabled:
                load_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    self.state.best_model_at_checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                else:
                    state_dict = torch.load(
                        weights_file,
                        map_location="cpu",
                        **weights_only_kwarg,
                    )

                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, False)
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        elif is_peft_model(model):
            load_peft_adapters(resume_from_checkpoint, model, adapter_subdirs)

        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
                reissue_pt_warnings(caught_warnings)
            return

        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, OPTIMIZER_NAME) + "_*")
            if is_sagemaker_mp_enabled()
            else (
                os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME))
                or os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME_BIN))
                or (
                    os.path.isdir(checkpoint)
                    and any(
                        OPTIMIZER_NAME_BIN.split(".")[0] in folder_name
                        for folder_name in os.listdir(checkpoint)
                        if os.path.isdir(os.path.join(checkpoint, folder_name))
                    )
                )
            )
        )

        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
            # Load in optimizer and scheduler states
            if is_torch_xla_available():
                # On TPU we have to take some extra precautions to properly load the states on the right device.
                optimizer_state = torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location="cpu")
                with warnings.catch_warnings(record=True) as caught_warnings:
                    lr_scheduler_state = torch.load(os.path.join(checkpoint, SCHEDULER_NAME), map_location="cpu")
                reissue_pt_warnings(caught_warnings)

                xm.send_cpu_data_to_device(optimizer_state, self.args.device)
                xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)

                self.optimizer.load_state_dict(optimizer_state)
                self.lr_scheduler.load_state_dict(lr_scheduler_state)
            else:
                if is_sagemaker_mp_enabled():
                    if os.path.isfile(os.path.join(checkpoint, "user_content.pt")):
                        # Optimizer checkpoint was saved with smp >= 1.10
                        def opt_load_hook(mod, opt):
                            opt.load_state_dict(smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True))

                    else:
                        # Optimizer checkpoint was saved with smp < 1.10
                        def opt_load_hook(mod, opt):
                            if IS_SAGEMAKER_MP_POST_1_10:
                                opt.load_state_dict(
                                    smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True, back_compat=True)
                                )
                            else:
                                opt.load_state_dict(smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True))

                    self.model_wrapped.register_post_step_hook(opt_load_hook)
                else:
                    # We use the CPU when training on one GPU to avoid OOM for GPU RAM when training big models.
                    # In distributed training however, we load directly on each GPU and risk the GPU OOM as it's more
                    # likely to get OOM on CPU (since we load num_gpu times the optimizer state
                    map_location = self.args.device if self.args.world_size > 1 else "cpu"
                    if self.is_fsdp_enabled:
                        load_fsdp_optimizer(
                            self.accelerator.state.fsdp_plugin,
                            self.accelerator,
                            self.optimizer,
                            self.model,
                            checkpoint,
                            **_get_fsdp_ckpt_kwargs(),
                        )
                    else:
                        self.optimizer.load_state_dict(
                            torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location)
                        )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
                reissue_pt_warnings(caught_warnings)

    def _load_callback_state(self):
        """If callback states exist and were passed in, restore their states if enabled"""
        if not self.args.restore_callback_states_from_checkpoint:
            return
        # Callback states are stored in stateful_callbacks
        not_found = []
        new_callbacks = []
        original_callbacks = self.callback_handler.callbacks + [self.control]
        for stored_callback, data in self.state.stateful_callbacks.items():
            if not isinstance(data, list):
                data = [data]
            if any(callback.__class__.__name__ == stored_callback for callback in original_callbacks):
                # We can load/restore from multiple callbacks of the same type.
                duplicates = [
                    callback for callback in original_callbacks if callback.__class__.__name__ == stored_callback
                ]
                for callback, callback_data in zip(duplicates, data):
                    args = callback_data.get("args", {})
                    attributes = callback_data.get("attributes", {})
                    new_callback = type(callback)(**args)
                    for attribute, value in attributes.items():
                        setattr(new_callback, attribute, value)
                    if isinstance(callback, TrainerControl):
                        # Specifically for restoring the `control` state
                        self.control = new_callback
                    else:
                        new_callbacks.append(new_callback)
                    # We remove the existing callback and add it to the list of new callbacks
                    self.callback_handler.remove_callback(type(new_callback))
                logger.info("Continuing training from checkpoint, restoring any callbacks that were passed in")
            else:
                not_found.append(stored_callback)
        if len(not_found) > 0:
            logger.warning(
                f"Checkpoint included callbacks not included in current configuration. Ignoring. ({', '.join(not_found)})"
            )
        for callback in new_callbacks:
            self.callback_handler.add_callback(callback)

    def _issue_warnings_after_load(self, load_result):
        if len(load_result.missing_keys) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                self.model._keys_to_ignore_on_save
            ):
                self.model.tie_weights()
            else:
                logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _nested_gather(self, tensors, name=None):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        if is_torch_xla_available():
            if name is None:
                name = "nested_gather"
            tensors = nested_xla_mesh_reduce(tensors, name)
        elif is_sagemaker_mp_enabled():
            tensors = smp_gather(tensors)
        elif (self.args.distributed_state is not None and self.args.distributed_state.distributed_type != "NO") or (
            self.args.distributed_state is None and self.args.local_rank != -1
        ):
            tensors = distributed_concat(tensors)
        return tensors

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.

        Args:
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    def store_flos(self):
        # Storing the number of floating-point operations that went into the model
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            self.state.total_flos += (
                distributed_broadcast_scalars([self.current_flos], device=self.args.device).sum().item()
            )
            self.current_flos = 0
        else:
            self.state.total_flos += self.current_flos
            self.current_flos = 0

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.
        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        state_dict = None
        if is_torch_xla_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif self.is_fsdp_enabled:
            if ("FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)) and (
                version.parse(accelerate_version) > version.parse("0.24.1")
            ):
                state_dict = self.accelerator.get_state_dict(self.model)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
        elif self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model_wrapped.save_checkpoint(output_dir)
        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model = self.model
        if not isinstance(model, supported_classes):
            if state_dict is None:
                state_dict = model.state_dict()

            model = self.accelerator.unwrap_model(model)
            if not isinstance(model, supported_classes):
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        logger.info(f"Saving model checkpoint to {output_dir}")
        model = self.model
        xm.mark_step()

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        supported_classes = (PushToHubMixin,)
        xm.rendezvous("saving_checkpoint")
        if not isinstance(model, supported_classes):
            if isinstance(self.accelerator.unwrap_model(model), supported_classes):
                self.accelerator.unwrap_model(model).save_pretrained(
                    output_dir,
                    is_main_process=self.args.should_save,
                    state_dict=xm._maybe_convert_to_cpu(model.state_dict()),
                    save_function=xm.save,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                state_dict = xm._maybe_convert_to_cpu(model.state_dict())
                xm.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(
                output_dir,
                is_main_process=self.args.should_save,
                save_function=xm.save,
                safe_serialization=self.args.save_safetensors,
                state_dict=xm._maybe_convert_to_cpu(model.state_dict()),
            )
        if self.tokenizer is not None and self.args.should_save:
            self.tokenizer.save_pretrained(output_dir)

    def _save_rng_state(self, output_dir):
        fname = os.path.join(
            output_dir, f"rng_state_{self.args.process_index}.pth" if self.args.world_size > 1 else "rng_state.pth"
        )

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }

        is_parallel = self.args.parallel_mode == ParallelMode.DISTRIBUTED
        if torch.cuda.is_available():
            rng_states["cuda"] = get_rng_state(torch.cuda, is_parallel)

        if is_torch_xla_available():
            rng_states["xla"] = get_rng_state(xm)

        if is_torch_npu_available():
            rng_states["npu"] = get_rng_state(torch.npu, is_parallel)

        if is_torch_mlu_available():
            rng_states["mlu"] = get_rng_state(torch.mlu, is_parallel)

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        torch.save(rng_states, fname)

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                metric_value = metrics[metric_to_check]
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            # Update the `TrainerControl` state to where we are currently
            self.state.stateful_callbacks["TrainerControl"] = self.control.state()
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def _save_optimizer_and_scheduler(self, output_dir):
        if is_torch_xla_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
        elif self.is_deepspeed_enabled:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            accept_exclude_frozen_parameters = "exclude_frozen_parameters" in set(
                inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys()
            )
            if accept_exclude_frozen_parameters and is_peft_model(self.model):
                self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
            else:
                self.model_wrapped.save_checkpoint(output_dir)
        elif self.is_fsdp_enabled:
            # save fsdp specific ckpt for resuming from ckpt
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir, **_get_fsdp_ckpt_kwargs()
            )
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
            )
        elif self.args.should_save:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

        # Save SCHEDULER & SCALER
        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if (
            self.args.should_save
            and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
            and not is_torch_xla_available()
        ):
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)

    def _push_from_checkpoint(self, checkpoint_folder):
        # Only push from one node.
        if not self.is_world_process_zero() or self.args.hub_strategy == HubStrategy.END:
            return
        # If we haven't finished the last push, we don't do this one unless args.hub_always_push=True.
        if not self.args.hub_always_push and self.push_in_progress is not None and not self.push_in_progress.is_done():
            return

        output_dir = self.args.output_dir
        # To avoid a new synchronization of all model weights, we just copy the file from the checkpoint folder
        modeling_files = [CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
        #  Add sharded checkpoints if we have an index
        for index_file in [WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME]:
            index_path = os.path.join(checkpoint_folder, index_file)
            if os.path.isfile(index_path):
                modeling_files.append(index_file)
                with open(index_path) as f:
                    index = json.loads(f.read())
                shard_files = list(set(index["weight_map"].values()))
                modeling_files.extend(shard_files)
        if is_peft_available():
            modeling_files.extend([ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME])
        for modeling_file in modeling_files:
            if os.path.isfile(os.path.join(checkpoint_folder, modeling_file)):
                shutil.copy(os.path.join(checkpoint_folder, modeling_file), os.path.join(output_dir, modeling_file))
        # Saving the tokenizer is fast and we don't know how many files it may have spawned, so we resave it to be sure.
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # Same for the training arguments
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.args.save_strategy == IntervalStrategy.STEPS:
            commit_message = f"Training in progress, step {self.state.global_step}"
        else:
            commit_message = f"Training in progress, epoch {int(self.state.epoch)}"

        model_push_job = upload_folder(
            repo_id=self.hub_model_id,
            folder_path=output_dir,
            commit_message=commit_message,
            token=self.args.hub_token,
            run_as_future=True,
            ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
        )

        push_jobs = [model_push_job]

        if self.args.hub_strategy in [HubStrategy.CHECKPOINT, HubStrategy.ALL_CHECKPOINTS]:
            path_in_repo = (
                "last-checkpoint" if self.args.hub_strategy == HubStrategy.CHECKPOINT else Path(checkpoint_folder).name
            )
            checkpoint_push = upload_folder(
                repo_id=self.hub_model_id,
                folder_path=checkpoint_folder,
                path_in_repo=path_in_repo,
                commit_message=commit_message + ", checkpoint",
                token=self.args.hub_token,
                run_as_future=True,
            )
            push_jobs.append(checkpoint_push)

        if self.push_in_progress is None or self.push_in_progress.is_done():
            self.push_in_progress = PushInProgress(push_jobs)
        else:
            self.push_in_progress.jobs.extend(push_jobs)

    def _finish_current_push(self):
        if not hasattr(self, "push_in_progress"):
            return
        if self.push_in_progress is not None and not self.push_in_progress.is_done():
            logger.info("Waiting for the current checkpoint push to be finished, this might take a couple of minutes.")
            self.push_in_progress.wait_until_done()

    def init_hf_repo(self, token: Optional[str] = None):
        """
        Initializes a git repo in `self.args.hub_model_id`.
        """
        # Only on process zero
        if not self.is_world_process_zero():
            return

        if self.args.hub_model_id is None:
            repo_name = Path(self.args.output_dir).absolute().name
        else:
            repo_name = self.args.hub_model_id

        token = token if token is not None else self.args.hub_token
        repo_url = create_repo(repo_name, token=token, private=self.args.hub_private_repo, exist_ok=True)
        self.hub_model_id = repo_url.repo_id
        self.push_in_progress = None

    def create_model_card(
        self,
        language: Optional[str] = None,
        license: Optional[str] = None,
        tags: Union[str, List[str], None] = None,
        model_name: Optional[str] = None,
        finetuned_from: Optional[str] = None,
        tasks: Union[str, List[str], None] = None,
        dataset_tags: Union[str, List[str], None] = None,
        dataset: Union[str, List[str], None] = None,
        dataset_args: Union[str, List[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            language (`str`, *optional*):
                The language of the model (if applicable)
            license (`str`, *optional*):
                The license of the model. Will default to the license of the pretrained model used, if the original
                model given to the `Trainer` comes from a repo on the Hub.
            tags (`str` or `List[str]`, *optional*):
                Some tags to be included in the metadata of the model card.
            model_name (`str`, *optional*):
                The name of the model.
            finetuned_from (`str`, *optional*):
                The name of the model used to fine-tune this one (if applicable). Will default to the name of the repo
                of the original model given to the `Trainer` (if it comes from the Hub).
            tasks (`str` or `List[str]`, *optional*):
                One or several task identifiers, to be included in the metadata of the model card.
            dataset_tags (`str` or `List[str]`, *optional*):
                One or several dataset tags, to be included in the metadata of the model card.
            dataset (`str` or `List[str]`, *optional*):
                One or several dataset identifiers, to be included in the metadata of the model card.
            dataset_args (`str` or `List[str]`, *optional*):
               One or several dataset arguments, to be included in the metadata of the model card.
        """
        if not self.is_world_process_zero():
            return

        model_card_filepath = os.path.join(self.args.output_dir, "README.md")
        is_peft_library = False
        if os.path.exists(model_card_filepath):
            library_name = ModelCard.load(model_card_filepath).data.get("library_name")
            is_peft_library = library_name == "peft"

            # Append existing tags in `tags`
            existing_tags = ModelCard.load(model_card_filepath).data.tags
            if tags is not None and existing_tags is not None:
                if isinstance(tags, str):
                    tags = [tags]
                for tag in existing_tags:
                    if tag not in tags:
                        tags.append(tag)

        training_summary = TrainingSummary.from_trainer(
            self,
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
        )
        model_card = training_summary.to_model_card()
        with open(model_card_filepath, "w") as f:
            f.write(model_card)

        if is_peft_library:
            self.accelerator.unwrap_model(self.model).create_or_update_model_card(self.args.output_dir)

    def push_to_hub(
        self,
        commit_message: Optional[str] = "End of training",
        blocking: bool = True,
        token: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Upload `self.model` and `self.tokenizer` to the 🤗 model hub on the repo `self.args.hub_model_id`.
        Parameters:
            commit_message (`str`, *optional*, defaults to `"End of training"`):
                Message to commit while pushing.
            blocking (`bool`, *optional*, defaults to `True`):
                Whether the function should return only when the `git push` has finished.
            token (`str`, *optional*, defaults to `None`):
                Token with write permission to overwrite Trainer's original args.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to [`~Trainer.create_model_card`].
        Returns:
            The URL of the repository where the model was pushed if `blocking=False`, or a `Future` object tracking the
            progress of the commit if `blocking=True`.
        """
        model_name = kwargs.pop("model_name", None)
        if model_name is None and self.args.should_save:
            if self.args.hub_model_id is None:
                model_name = Path(self.args.output_dir).name
            else:
                model_name = self.args.hub_model_id.split("/")[-1]
        token = token if token is not None else self.args.hub_token

        # In case the user calls this method with args.push_to_hub = False
        if self.hub_model_id is None:
            self.init_hf_repo(token=token)

        # Needs to be executed on all processes for TPU training, but will only save on the processed determined by
        # self.args.should_save.
        self.save_model(_internal_call=True)

        # Only push from one node.
        if not self.is_world_process_zero():
            return

        # Add additional tags in the case the model has already some tags and users pass
        # "tags" argument to `push_to_hub` so that trainer automatically handles internal tags
        # from all models since Trainer does not call `model.push_to_hub`.
        if getattr(self.model, "model_tags", None) is not None:
            if "tags" not in kwargs:
                kwargs["tags"] = []

            # If it is a string, convert it to a list
            if isinstance(kwargs["tags"], str):
                kwargs["tags"] = [kwargs["tags"]]

            for model_tag in self.model.model_tags:
                if model_tag not in kwargs["tags"]:
                    kwargs["tags"].append(model_tag)

        self.create_model_card(model_name=model_name, **kwargs)

        # Wait for the current upload to be finished.
        self._finish_current_push()
        return upload_folder(
            repo_id=self.hub_model_id,
            folder_path=self.args.output_dir,
            commit_message=commit_message,
            token=token,
            run_as_future=not blocking,
            ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
        )

    #
    # Deprecated code
    #
    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        if not has_length(dataloader):
            raise ValueError("dataloader must implement a working __len__")

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info(f"\n***** Running {description} *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")

        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        inputs_host: Union[torch.Tensor, List[torch.Tensor]] = None
        metrics: Optional[dict] = None

        world_size = max(1, args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, SequentialDistributedSampler):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            inputs_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)

        model.eval()

        if args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and preds_host is not None and labels_host is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    if args.include_inputs_for_metrics:
                        metrics = self.compute_metrics(
                            EvalPrediction(predictions=preds_host, label_ids=labels_host, inputs=inputs_host),
                            compute_result=is_last_step,
                        )
                    else:
                        metrics = self.compute_metrics(
                            EvalPrediction(predictions=preds_host, label_ids=labels_host),
                            compute_result=is_last_step,
                        )

            if self.args.batch_eval_metrics or (
                args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0
            ):
                # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
                    inputs_gatherer.add_arrays(self._gather_and_numpify(inputs_host, "eval_inputs_ids"))

                # Set back to None to begin a new accumulation
                del losses_host, preds_host, labels_host, inputs_host
                torch.cuda.empty_cache()
                losses_host, preds_host, labels_host, inputs_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
            inputs_gatherer.add_arrays(self._gather_and_numpify(inputs_host, "eval_inputs_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None
        inputs_ids = inputs_gatherer.finalize() if not prediction_loss_only else None

        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
            and not self.args.batch_eval_metrics
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=preds, label_ids=label_ids, inputs=inputs_ids)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=preds, label_ids=label_ids, metrics=metrics, num_samples=num_examples)

    def _gather_and_numpify(self, tensors, name):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        if is_torch_xla_available():
            tensors = nested_xla_mesh_reduce(tensors, name)
        elif is_sagemaker_mp_enabled():
            tensors = smp_gather(tensors)
        elif self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            tensors = distributed_concat(tensors)

        return nested_numpify(tensors)
