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

import contextlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .debug_utils import DebugOption
from .file_utils import (
    ExplicitEnum,
    cached_property,
    get_full_repo_name,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_bf16_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    torch_required,
)
from .trainer_utils import EvaluationStrategy, HubStrategy, IntervalStrategy, SchedulerType, ShardedDDPOption
from .utils import logging


if is_torch_available():
    import torch

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as sm_dist

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    smp.init()


logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAMW_HF = "adamw_hf"
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        output_dir (`str`):
            The output directory where the model predictions and checkpoints will be written.
        overwrite_output_dir (`bool`, *optional*, defaults to `False`):
            If `True`, overwrite the content of the output directory. Use this to continue training if `output_dir`
            points to a checkpoint directory.
        do_train (`bool`, *optional*, defaults to `False`):
            Whether to run training or not. This argument is not directly used by [`Trainer`], it's intended to be used
            by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/master/examples) for more details.
        do_eval (`bool`, *optional*):
            Whether to run evaluation on the validation set or not. Will be set to `True` if `evaluation_strategy` is
            different from `"no"`. This argument is not directly used by [`Trainer`], it's intended to be used by your
            training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/master/examples) for more details.
        do_predict (`bool`, *optional*, defaults to `False`):
            Whether to run predictions on the test set or not. This argument is not directly used by [`Trainer`], it's
            intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/master/examples) for more details.
        evaluation_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
            The evaluation strategy to adopt during training. Possible values are:

                - `"no"`: No evaluation is done during training.
                - `"steps"`: Evaluation is done (and logged) every `eval_steps`.
                - `"epoch"`: Evaluation is done at the end of each epoch.

        prediction_loss_only (`bool`, *optional*, defaults to `False`):
            When performing evaluation and generating predictions, only returns the loss.
        per_device_train_batch_size (`int`, *optional*, defaults to 8):
            The batch size per GPU/TPU core/CPU for training.
        per_device_eval_batch_size (`int`, *optional*, defaults to 8):
            The batch size per GPU/TPU core/CPU for evaluation.
        gradient_accumulation_steps (`int`, *optional*, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

            <Tip warning={true}>

            When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging,
            evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training examples.

            </Tip>

        eval_accumulation_steps (`int`, *optional*):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but
            requires more memory).
        learning_rate (`float`, *optional*, defaults to 5e-5):
            The initial learning rate for [`AdamW`] optimizer.
        weight_decay (`float`, *optional*, defaults to 0):
            The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`]
            optimizer.
        adam_beta1 (`float`, *optional*, defaults to 0.9):
            The beta1 hyperparameter for the [`AdamW`] optimizer.
        adam_beta2 (`float`, *optional*, defaults to 0.999):
            The beta2 hyperparameter for the [`AdamW`] optimizer.
        adam_epsilon (`float`, *optional*, defaults to 1e-8):
            The epsilon hyperparameter for the [`AdamW`] optimizer.
        max_grad_norm (`float`, *optional*, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        num_train_epochs(`float`, *optional*, defaults to 3.0):
            Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
            the last epoch before stopping training).
        max_steps (`int`, *optional*, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
            In case of using a finite iterable dataset the training may stop before reaching the set number of steps
            when all data is exhausted
        lr_scheduler_type (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
            The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values.
        warmup_ratio (`float`, *optional*, defaults to 0.0):
            Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
        warmup_steps (`int`, *optional*, defaults to 0):
            Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`.
        log_level (`str`, *optional*, defaults to `passive`):
            Logger log level to use on the main process. Possible choices are the log levels as strings: 'debug',
            'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets the
            application set the level.
        log_level_replica (`str`, *optional*, defaults to `passive`):
            Logger log level to use on replicas. Same choices as `log_level`"
        log_on_each_node (`bool`, *optional*, defaults to `True`):
            In multinode distributed training, whether to log using `log_level` once per node, or only on the main
            node.
        logging_dir (`str`, *optional*):
            [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
            *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
        logging_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The logging strategy to adopt during training. Possible values are:

                - `"no"`: No logging is done during training.
                - `"epoch"`: Logging is done at the end of each epoch.
                - `"steps"`: Logging is done every `logging_steps`.

        logging_first_step (`bool`, *optional*, defaults to `False`):
            Whether to log and evaluate the first `global_step` or not.
        logging_steps (`int`, *optional*, defaults to 500):
            Number of update steps between two logs if `logging_strategy="steps"`.
        logging_nan_inf_filter (`bool`, *optional*, defaults to `True`):
            Whether to filter `nan` and `inf` losses for logging. If set to `True` the loss of every step that is `nan`
            or `inf` is filtered and the average loss of the current logging window is taken instead.

            <Tip>

            `logging_nan_inf_filter` only influences the logging of loss values, it does not change the behavior the
            gradient is computed or applied to the model.

            </Tip>

        save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                - `"no"`: No save is done during training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`.
        save_steps (`int`, *optional*, defaults to 500):
            Number of updates steps before two checkpoint saves if `save_strategy="steps"`.
        save_total_limit (`int`, *optional*):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            `output_dir`.
        save_on_each_node (`bool`, *optional*, defaults to `False`):
            When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on
            the main one.

            This should not be activated when the different nodes use the same storage as the files will be saved with
            the same names for each node.
        no_cuda (`bool`, *optional*, defaults to `False`):
            Whether to not use CUDA even when it is available or not.
        seed (`int`, *optional*, defaults to 42):
            Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
            [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized parameters.
        bf16 (`bool`, *optional*, defaults to `False`):
            Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher
            NVIDIA architecture. This is an experimental API and it may change.
        fp16 (`bool`, *optional*, defaults to `False`):
            Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
        fp16_opt_level (`str`, *optional*, defaults to 'O1'):
            For `fp16` training, Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details on
            the [Apex documentation](https://nvidia.github.io/apex/amp).
        fp16_backend (`str`, *optional*, defaults to `"auto"`):
            This argument is deprecated. Use `half_precision_backend` instead.
        half_precision_backend (`str`, *optional*, defaults to `"auto"`):
            The backend to use for mixed precision training. Must be one of `"auto"`, `"amp"` or `"apex"`. `"auto"`
            will use AMP or APEX depending on the PyTorch version detected, while the other choices will force the
            requested backend.
        bf16_full_eval (`bool`, *optional*, defaults to `False`):
            Whether to use full bfloat16 evaluation instead of 32-bit. This will be faster and save memory but can harm
            metric values. This is an experimental API and it may change.
        fp16_full_eval (`bool`, *optional*, defaults to `False`):
            Whether to use full float16 evaluation instead of 32-bit. This will be faster and save memory but can harm
            metric values.
        tf32 (`bool`, *optional*):
            Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental API
            and it may change.
        local_rank (`int`, *optional*, defaults to -1):
            Rank of the process during distributed training.
        xpu_backend (`str`, *optional*):
            The backend to use for xpu distributed training. Must be one of `"mpi"` or `"ccl"`.
        tpu_num_cores (`int`, *optional*):
            When training on TPU, the number of TPU cores (automatically passed by launcher script).
        dataloader_drop_last (`bool`, *optional*, defaults to `False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        eval_steps (`int`, *optional*):
            Number of update steps between two evaluations if `evaluation_strategy="steps"`. Will default to the same
            value as `logging_steps` if not set.
        dataloader_num_workers (`int`, *optional*, defaults to 0):
            Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
            main process.
        past_index (`int`, *optional*, defaults to -1):
            Some models like [TransformerXL](../model_doc/transformerxl) or [XLNet](../model_doc/xlnet) can make use of
            the past hidden states for their predictions. If this argument is set to a positive int, the `Trainer` will
            use the corresponding output (usually index 2) as the past state and feed it to the model at the next
            training step under the keyword argument `mems`.
        run_name (`str`, *optional*):
            A descriptor for the run. Typically used for [wandb](https://www.wandb.com/) and
            [mlflow](https://www.mlflow.org/) logging.
        disable_tqdm (`bool`, *optional*):
            Whether or not to disable the tqdm progress bars and table of metrics produced by
            [`~notebook.NotebookTrainingTracker`] in Jupyter Notebooks. Will default to `True` if the logging level is
            set to warn or lower (default), `False` otherwise.
        remove_unused_columns (`bool`, *optional*, defaults to `True`):
            If using `datasets.Dataset` datasets, whether or not to automatically remove the columns unused by the
            model forward method.

            (Note that this behavior is not implemented for [`TFTrainer`] yet.)
        label_names (`List[str]`, *optional*):
            The list of keys in your dictionary of inputs that correspond to the labels.

            Will eventually default to `["labels"]` except if the model used is one of the `XxxForQuestionAnswering` in
            which case it will default to `["start_positions", "end_positions"]`.
        load_best_model_at_end (`bool`, *optional*, defaults to `False`):
            Whether or not to load the best model found during training at the end of training.

            <Tip>

            When set to `True`, the parameters `save_strategy` needs to be the same as `eval_strategy`, and in the case
            it is "steps", `save_steps` must be a round multiple of `eval_steps`.

            </Tip>

        metric_for_best_model (`str`, *optional*):
            Use in conjunction with `load_best_model_at_end` to specify the metric to use to compare two different
            models. Must be the name of a metric returned by the evaluation with or without the prefix `"eval_"`. Will
            default to `"loss"` if unspecified and `load_best_model_at_end=True` (to use the evaluation loss).

            If you set this value, `greater_is_better` will default to `True`. Don't forget to set it to `False` if
            your metric is better when lower.
        greater_is_better (`bool`, *optional*):
            Use in conjunction with `load_best_model_at_end` and `metric_for_best_model` to specify if better models
            should have a greater metric or not. Will default to:

            - `True` if `metric_for_best_model` is set to a value that isn't `"loss"` or `"eval_loss"`.
            - `False` if `metric_for_best_model` is not set, or set to `"loss"` or `"eval_loss"`.
        ignore_data_skip (`bool`, *optional*, defaults to `False`):
            When resuming training, whether or not to skip the epochs and batches to get the data loading at the same
            stage as in the previous training. If set to `True`, the training will begin faster (as that skipping step
            can take a long time) but will not yield the same results as the interrupted training would have.
        sharded_ddp (`bool`, `str` or list of [`~trainer_utils.ShardedDDPOption`], *optional*, defaults to `False`):
            Use Sharded DDP training from [FairScale](https://github.com/facebookresearch/fairscale) (in distributed
            training only). This is an experimental feature.

            A list of options along the following:

            - `"simple"`: to use first instance of sharded DDP released by fairscale (`ShardedDDP`) similar to ZeRO-2.
            - `"zero_dp_2"`: to use the second instance of sharded DPP released by fairscale (`FullyShardedDDP`) in
              Zero-2 mode (with `reshard_after_forward=False`).
            - `"zero_dp_3"`: to use the second instance of sharded DPP released by fairscale (`FullyShardedDDP`) in
              Zero-3 mode (with `reshard_after_forward=True`).
            - `"offload"`: to add ZeRO-offload (only compatible with `"zero_dp_2"` and `"zero_dp_3"`).

            If a string is passed, it will be split on space. If a bool is passed, it will be converted to an empty
            list for `False` and `["simple"]` for `True`.
        deepspeed (`str` or `dict`, *optional*):
            Use [Deepspeed](https://github.com/microsoft/deepspeed). This is an experimental feature and its API may
            evolve in the future. The value is either the location of DeepSpeed json config file (e.g.,
            `ds_config.json`) or an already loaded json file as a `dict`"
        label_smoothing_factor (`float`, *optional*, defaults to 0.0):
            The label smoothing factor to use. Zero means no label smoothing, otherwise the underlying onehot-encoded
            labels are changed from 0s and 1s to `label_smoothing_factor/num_labels` and `1 - label_smoothing_factor +
            label_smoothing_factor/num_labels` respectively.
        debug (`str` or list of [`~debug_utils.DebugOption`], *optional*, defaults to `""`):
            Enable one or more debug features. This is an experimental feature.

            Possible options are:

            - `"underflow_overflow"`: detects overflow in model's input/outputs and reports the last frames that led to
              the event
            - `"tpu_metrics_debug"`: print debug metrics on TPU

            The options should be separated by whitespaces.
        optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_hf"`):
            The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor.
        adafactor (`bool`, *optional*, defaults to `False`):
            This argument is deprecated. Use `--optim adafactor` instead.
        group_by_length (`bool`, *optional*, defaults to `False`):
            Whether or not to group together samples of roughly the same length in the training dataset (to minimize
            padding applied and be more efficient). Only useful if applying dynamic padding.
        length_column_name (`str`, *optional*, defaults to `"length"`):
            Column name for precomputed lengths. If the column exists, grouping by length will use these values rather
            than computing them on train startup. Ignored unless `group_by_length` is `True` and the dataset is an
            instance of `Dataset`.
        report_to (`str` or `List[str]`, *optional*, defaults to `"all"`):
            The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
            `"comet_ml"`, `"mlflow"`, `"tensorboard"` and `"wandb"`. Use `"all"` to report to all integrations
            installed, `"none"` for no integrations.
        ddp_find_unused_parameters (`bool`, *optional*):
            When using distributed training, the value of the flag `find_unused_parameters` passed to
            `DistributedDataParallel`. Will default to `False` if gradient checkpointing is used, `True` otherwise.
        ddp_bucket_cap_mb (`int`, *optional*):
            When using distributed training, the value of the flag `bucket_cap_mb` passed to `DistributedDataParallel`.
        dataloader_pin_memory (`bool`, *optional*, defaults to `True`):
            Whether you want to pin memory in data loaders or not. Will default to `True`.
        skip_memory_metrics (`bool`, *optional*, defaults to `True`):
            Whether to skip adding of memory profiler reports to metrics. This is skipped by default because it slows
            down the training and evaluation speed.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether or not to push the model to the Hub every time the model is saved. If this is activated,
            `output_dir` will begin a git directory synced with the the repo (determined by `hub_model_id`) and the
            content will be pushed each time a save is triggered (depending on your `save_strategy`). Calling
            [`~Trainer.save_model`] will also trigger a push.

            <Tip warning={true}>

            If `output_dir` exists, it needs to be a local clone of the repository to which the [`Trainer`] will be
            pushed.

            </Tip>

        resume_from_checkpoint (`str`, *optional*):
            The path to a folder with a valid checkpoint for your model. This argument is not directly used by
            [`Trainer`], it's intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/master/examples) for more details.
        hub_model_id (`str`, *optional*):
            The name of the repository to keep in sync with the local *output_dir*. It can be a simple model ID in
            which case the model will be pushed in your namespace. Otherwise it should be the whole repository name,
            for instance `"user_name/model"`, which allows you to push to an organization you are a member of with
            `"organization_name/model"`. Will default to `user_name/output_dir_name` with *output_dir_name* being the
            name of `output_dir`.

            Will default to to the name of `output_dir`.
        hub_strategy (`str` or [`~trainer_utils.HubStrategy`], *optional*, defaults to `"every_save"`):
            Defines the scope of what is pushed to the Hub and when. Possible values are:

            - `"end"`: push the model, its configuration, the tokenizer (if passed along to the [`Trainer`]) and a
              draft of a model card when the [`~Trainer.save_model`] method is called.
            - `"every_save"`: push the model, its configuration, the tokenizer (if passed along to the [`Trainer`]) and
              a draft of a model card each time there is a model save. The pushes are asynchronous to not block
              training, and in case the save are very frequent, a new push is only attempted if the previous one is
              finished. A last push is made with the final model at the end of training.
            - `"checkpoint"`: like `"every_save"` but the latest checkpoint is also pushed in a subfolder named
              last-checkpoint, allowing you to resume training easily with
              `trainer.train(resume_from_checkpoint="last-checkpoint")`.
            - `"all_checkpoints"`: like `"checkpoint"` but all checkpoints are pushed like they appear in the output
              folder (so you will get one checkpoint folder per folder in your final repository)

        hub_token (`str`, *optional*):
            The token to use to push the model to the Hub. Will default to the token in the cache folder obtained with
            `huggingface-cli login`.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    """

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluation_strategy: IntervalStrategy = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
            "Batch size per GPU/TPU core/CPU for training."
        },
    )
    per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_eval_batch_size` is preferred. "
            "Batch size per GPU/TPU core/CPU for evaluation."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    log_level: Optional[str] = field(
        default="passive",
        metadata={
            "help": "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets the application set the level. Defaults to 'passive'.",
            "choices": trainer_log_levels.keys(),
        },
    )
    log_level_replica: Optional[str] = field(
        default="passive",
        metadata={
            "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
            "choices": trainer_log_levels.keys(),
        },
    )
    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help": "When doing a multinode distributed training, whether to log once per node or just once on the main node."
        },
    )
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    logging_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    logging_nan_inf_filter: str = field(default=True, metadata={"help": "Filter nan and inf losses for logging."})
    save_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": "When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on the main one"
        },
    )
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    bf16: bool = field(
        default=False,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture. This is an experimental API and it may change."
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    half_precision_backend: str = field(
        default="auto",
        metadata={"help": "The backend to be used for half precision.", "choices": ["auto", "amp", "apex"]},
    )
    bf16_full_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may change."
        },
    )
    fp16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use full float16 evaluation instead of 32-bit"},
    )
    tf32: bool = field(
        default=None,
        metadata={
            "help": "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental API and it may change."
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    xpu_backend: str = field(
        default=None,
        metadata={"help": "The backend to be used for distributed training on Intel XPU.", "choices": ["mpi", "ccl"]},
    )
    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    )
    tpu_metrics_debug: bool = field(
        default=False,
        metadata={
            "help": "Deprecated, the use of `--debug tpu_metrics_debug` is preferred. TPU: Whether to print debug metrics"
        },
    )
    debug: str = field(
        default="",
        metadata={
            "help": "Whether or not to enable debug mode. Current options: "
            "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
            "`tpu_metrics_debug` (print debug metrics on TPU)."
        },
    )

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process."
        },
    )

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": "When resuming training, whether or not to skip the first epochs and batches to get to the same training data."
        },
    )
    sharded_ddp: str = field(
        default="",
        metadata={
            "help": "Whether or not to use sharded DDP training (in distributed training only). The base option "
            "should be `simple`, `zero_dp_2` or `zero_dp_3` and you can add CPU-offload to `zero_dp_2` or `zero_dp_3` "
            "like this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or "
            "with the same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`.",
        },
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already loaded json file as a dict"
        },
    )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )
    optim: OptimizerNames = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Whether or not to group samples of roughly the same length together when batching."},
    )
    length_column_name: Optional[str] = field(
        default="length",
        metadata={"help": "Column name with precomputed lengths to use when grouping by length."},
    )
    report_to: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=None,
        metadata={
            "help": "When using distributed training, the value of the flag `find_unused_parameters` passed to "
            "`DistributedDataParallel`."
        },
    )
    ddp_bucket_cap_mb: Optional[int] = field(
        default=None,
        metadata={
            "help": "When using distributed training, the value of the flag `bucket_cap_mb` passed to "
            "`DistributedDataParallel`."
        },
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
    use_legacy_prediction_loop: bool = field(
        default=False, metadata={"help": "Whether or not to use the legacy prediction_loop in the Trainer."}
    )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_strategy: HubStrategy = field(
        default="every_save",
        metadata={"help": "The hub strategy to use when `--push_to_hub` is activated."},
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    # Deprecated arguments
    fp16_backend: str = field(
        default="auto",
        metadata={"help": "Deprecated. Use half_precision_backend instead", "choices": ["auto", "amp", "apex"]},
    )
    push_to_hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to which push the `Trainer`."}
    )
    push_to_hub_organization: str = field(
        default=None, metadata={"help": "The name of the organization in with to which push the `Trainer`."}
    )
    push_to_hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    _n_gpu: int = field(init=False, repr=False, default=-1)
    mp_parameters: str = field(
        default="",
        metadata={"help": "Used by the SageMaker launcher to send mp-specific args. Ignored in Trainer"},
    )

    def __post_init__(self):
        # Handle --use_env option in torch.distributed.launch (local_rank not passed as an arg then).
        # This needs to happen before any call to self.device or self.n_gpu.
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != self.local_rank:
            self.local_rank = env_local_rank

        # convert to int
        self.log_level = trainer_log_levels[self.log_level]
        self.log_level_replica = trainer_log_levels[self.log_level_replica]

        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        #  see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        if isinstance(self.evaluation_strategy, EvaluationStrategy):
            warnings.warn(
                "using `EvaluationStrategy` for `evaluation_strategy` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `IntervalStrategy` instead",
                FutureWarning,
            )
            # Go back to the underlying string or we won't be able to instantiate `IntervalStrategy` on it.
            self.evaluation_strategy = self.evaluation_strategy.value

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)
        self.hub_strategy = HubStrategy(self.hub_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir

        if self.fp16_backend and self.fp16_backend != "auto":
            warnings.warn(
                "`fp16_backend` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `half_precision_backend` instead",
                FutureWarning,
            )
            self.half_precision_backend = self.fp16_backend

        if (self.bf16 or self.bf16_full_eval) and not is_torch_bf16_available():
            raise ValueError("Your setup doesn't support bf16. You need Ampere GPU, torch>=1.10, cuda>=11.0")

        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")
        if self.bf16:
            if self.half_precision_backend == "apex":
                raise ValueError(
                    " `--half_precision_backend apex`: bf16 is not supported by apex. Use `--half_precision_backend amp` instead"
                )
            if not (self.sharded_ddp == "" or not self.sharded_ddp):
                raise ValueError("sharded_ddp is not supported with bf16")

        self.optim = OptimizerNames(self.optim)
        if self.adafactor:
            warnings.warn(
                "`--adafactor` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--optim adafactor` instead",
                FutureWarning,
            )
            self.optim = OptimizerNames.ADAFACTOR

        if (
            is_torch_available()
            and (self.device.type != "cuda")
            and not (self.device.type == "xla" and "GPU_NUM_DEVICES" in os.environ)
            and (self.fp16 or self.fp16_full_eval or self.bf16 or self.bf16_full_eval)
        ):
            raise ValueError(
                "Mixed precision training with AMP or APEX (`--fp16` or `--bf16`) and half precision evaluation (`--fp16_full_eval` or `--bf16_full_eval`) can only be used on CUDA devices."
            )

        if is_torch_available() and self.tf32 is not None:
            if self.tf32:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                else:
                    raise ValueError("--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7")
            else:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                # no need to assert on else

        if self.report_to is None:
            logger.info(
                "The default value for the training argument `--report_to` will change in v5 (from all installed "
                "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
                "now. You should start updating your code and make this info disappear :-)."
            )
            self.report_to = "all"
        if self.report_to == "all" or self.report_to == ["all"]:
            # Import at runtime to avoid a circular import.
            from .integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()
        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio during training"
            )

        if isinstance(self.sharded_ddp, bool):
            self.sharded_ddp = "simple" if self.sharded_ddp else ""
        if isinstance(self.sharded_ddp, str):
            self.sharded_ddp = [ShardedDDPOption(s) for s in self.sharded_ddp.split()]
        if self.sharded_ddp == [ShardedDDPOption.OFFLOAD]:
            raise ValueError(
                "`--sharded_ddp offload` can't work on its own. It needs to be added to `--sharded_ddp zero_dp_2` or "
                '`--sharded_ddp zero_dp_3`. For example, `--sharded_ddp "zero_dp_2 offload"`.'
            )
        elif len(self.sharded_ddp) > 1 and ShardedDDPOption.SIMPLE in self.sharded_ddp:
            raise ValueError("`--sharded_ddp simple` is not compatible with any other option.")
        elif ShardedDDPOption.ZERO_DP_2 in self.sharded_ddp and ShardedDDPOption.ZERO_DP_3 in self.sharded_ddp:
            raise ValueError("`--sharded_ddp zero_dp_2` is not compatible with `--sharded_ddp zero_dp_3`.")

        if self.tpu_metrics_debug:
            warnings.warn(
                "using `--tpu_metrics_debug` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--debug tpu_metrics_debug` instead",
                FutureWarning,
            )
            self.debug += " tpu_metrics_debug"
            self.tpu_metrics_debug = False
        if isinstance(self.debug, str):
            self.debug = [DebugOption(s) for s in self.debug.split()]

        if self.deepspeed:
            # - must be run very last in arg parsing, since it will use a lot of these settings.
            # - must be run before the model is created.
            from transformers.deepspeed import HfTrainerDeepSpeedConfig

            # will be used later by the Trainer
            # note: leave self.deepspeed unmodified in case a user relies on it not to be modified)
            self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)

        if self.push_to_hub_token is not None:
            warnings.warn(
                "`--push_to_hub_token` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                "`--hub_token` instead.",
                FutureWarning,
            )
            self.hub_token = self.push_to_hub_token

        if self.push_to_hub_model_id is not None:
            self.hub_model_id = get_full_repo_name(
                self.push_to_hub_model_id, organization=self.push_to_hub_organization, token=self.hub_token
            )
            if self.push_to_hub_organization is not None:
                warnings.warn(
                    "`--push_to_hub_model_id` and `--push_to_hub_organization` are deprecated and will be removed in "
                    "version 5 of 🤗 Transformers. Use `--hub_model_id` instead and pass the full repo name to this "
                    f"argument (in this case {self.hub_model_id}).",
                    FutureWarning,
                )
            else:
                warnings.warn(
                    "`--push_to_hub_model_id` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                    "`--hub_model_id` instead and pass the full repo name to this argument (in this case "
                    f"{self.hub_model_id}).",
                    FutureWarning,
                )
        elif self.push_to_hub_organization is not None:
            self.hub_model_id = f"{self.push_to_hub_organization}/{Path(self.output_dir).name}"
            warnings.warn(
                "`--push_to_hub_organization` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                "`--hub_model_id` instead and pass the full repo name to this argument (in this case "
                f"{self.hub_model_id}).",
                FutureWarning,
            )

    def __str__(self):
        self_as_dict = asdict(self)

        # Remove deprecated arguments. That code should be removed once
        # those deprecated arguments are removed from TrainingArguments. (TODO: v5)
        del self_as_dict["per_gpu_train_batch_size"]
        del self_as_dict["per_gpu_eval_batch_size"]

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from `per_gpu_train_batch_size` in distributed training).
        """
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from `per_gpu_eval_batch_size` in distributed training).
        """
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        eval_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return eval_batch_size

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
            if self.local_rank != -1:
                # Initializes distributed backend for cpu
                if self.xpu_backend not in ("mpi", "ccl"):
                    raise ValueError(
                        "CPU distributed training backend is not properly set. "
                        "Please set '--xpu_backend' to either 'mpi' or 'ccl'."
                    )
                torch.distributed.init_process_group(backend=self.xpu_backend)
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif is_sagemaker_mp_enabled():
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            self._n_gpu = 1
        elif is_sagemaker_dp_enabled():
            sm_dist.init_process_group()
            self.local_rank = sm_dist.get_local_rank()
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.deepspeed:
            # deepspeed inits torch.distributed internally
            from .deepspeed import is_deepspeed_available

            if not is_deepspeed_available():
                raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
            import deepspeed

            deepspeed.init_distributed()

            # workaround for setups like notebooks where the launcher can't be used,
            # but deepspeed requires a dist env.
            # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
            self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    @torch_required
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu

    @property
    @torch_required
    def parallel_mode(self):
        """
        The current mode used for parallelism if multiple GPUs/TPU cores are available. One of:

        - `ParallelMode.NOT_PARALLEL`: no parallelism (CPU or one GPU).
        - `ParallelMode.NOT_DISTRIBUTED`: several GPUs in one single process (uses `torch.nn.DataParallel`).
        - `ParallelMode.DISTRIBUTED`: several GPUs, each having its own process (uses
          `torch.nn.DistributedDataParallel`).
        - `ParallelMode.TPU`: several TPU cores.
        """
        if is_torch_tpu_available():
            return ParallelMode.TPU
        elif is_sagemaker_mp_enabled():
            return ParallelMode.SAGEMAKER_MODEL_PARALLEL
        elif is_sagemaker_dp_enabled():
            return ParallelMode.SAGEMAKER_DATA_PARALLEL
        elif self.local_rank != -1:
            return ParallelMode.DISTRIBUTED
        elif self.n_gpu > 1:
            return ParallelMode.NOT_DISTRIBUTED
        else:
            return ParallelMode.NOT_PARALLEL

    @property
    @torch_required
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        if is_torch_tpu_available():
            return xm.xrt_world_size()
        elif is_sagemaker_mp_enabled():
            return smp.dp_size() if not smp.state.cfg.prescaled_batch else smp.rdp_size()
        elif is_sagemaker_dp_enabled():
            return sm_dist.get_world_size()
        elif self.local_rank != -1:
            return torch.distributed.get_world_size()
        return 1

    @property
    @torch_required
    def process_index(self):
        """
        The index of the current process used.
        """
        if is_torch_tpu_available():
            return xm.get_ordinal()
        elif is_sagemaker_mp_enabled():
            return smp.dp_rank() if not smp.state.cfg.prescaled_batch else smp.rdp_rank()
        elif is_sagemaker_dp_enabled():
            return sm_dist.get_rank()
        elif self.local_rank != -1:
            return torch.distributed.get_rank()
        return 0

    @property
    @torch_required
    def local_process_index(self):
        """
        The index of the local process used.
        """
        if is_torch_tpu_available():
            return xm.get_local_ordinal()
        elif is_sagemaker_mp_enabled():
            return smp.local_rank()
        elif is_sagemaker_dp_enabled():
            return sm_dist.get_rank()
        elif self.local_rank != -1:
            return self.local_rank
        return 0

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        if self.log_on_each_node:
            return self.local_process_index == 0
        else:
            if is_sagemaker_mp_enabled():
                return smp.rank() == 0
            else:
                return self.process_index == 0

    @property
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
        if self.save_on_each_node:
            return self.local_process_index == 0
        else:
            if is_sagemaker_mp_enabled():
                return smp.rank() == 0
            else:
                return self.process_index == 0

    def get_process_log_level(self):
        """
        Returns the log level to be used depending on whether this process is the main process of node 0, main process
        of node non-0, or a non-main process.

        For the main process the log level defaults to `logging.INFO` unless overridden by `log_level` argument.

        For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica`
        argument.

        The choice between the main and replica process settings is made according to the return value of `should_log`.
        """

        log_level_main_node = logging.INFO if self.log_level == -1 else self.log_level
        log_level_replica_node = logging.WARNING if self.log_level_replica == -1 else self.log_level_replica
        return log_level_main_node if self.should_log else log_level_replica_node

    @property
    def place_model_on_device(self):
        """
        Can be subclassed and overridden for some specific integrations.
        """
        return not is_sagemaker_mp_enabled()

    @property
    def _no_sync_in_gradient_accumulation(self):
        """
        Whether or not to use no_sync for the gradients when doing gradient accumulation.
        """
        return not (self.deepspeed or is_sagemaker_dp_enabled() or is_sagemaker_mp_enabled())

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        """
        A context manager for torch distributed environment where on needs to do something on the main process, while
        blocking replicas, and when it's finished releasing the replicas.

        One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
        which upon completion saves a cached version of results and which then automatically gets loaded by the
        replicas.

        Args:
            local (`bool`, *optional*, defaults to `True`):
                if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
                rank 0 In multi-node environment with a shared filesystem you most likely will want to use
                `local=False` so that only the main process of the first node will do the processing. If however, the
                filesystem is not shared, then the main process of each node will need to do the processing, which is
                the default behavior.
            desc (`str`, *optional*, defaults to `"work"`):
                a work description to be used in debug logs

        """
        if is_torch_available() and self.world_size > 1:
            main_process_desc = "main process"
            if local:
                is_main_process = self.local_process_index == 0
                main_process_desc = "main local process"
            elif is_sagemaker_mp_enabled():
                is_main_process = smp.rank() == 0
            else:
                is_main_process = self.process_index == 0

            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")
                    if is_torch_tpu_available():
                        xm.rendezvous(desc)
                    elif is_sagemaker_dp_enabled():
                        sm_dist.barrier()
                    else:
                        torch.distributed.barrier()
                yield
            finally:
                if is_main_process:
                    # the wait is over
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    if is_torch_tpu_available():
                        xm.rendezvous(desc)
                    elif is_sagemaker_dp_enabled():
                        sm_dist.barrier()
                    else:
                        torch.distributed.barrier()
        else:
            yield

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}


class ParallelMode(Enum):
    NOT_PARALLEL = "not_parallel"
    NOT_DISTRIBUTED = "not_distributed"
    DISTRIBUTED = "distributed"
    SAGEMAKER_MODEL_PARALLEL = "sagemaker_model_parallel"
    SAGEMAKER_DATA_PARALLEL = "sagemaker_data_parallel"
    TPU = "tpu"
