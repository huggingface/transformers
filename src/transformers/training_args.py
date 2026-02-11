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
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from functools import cached_property
from typing import Any

from .debug_utils import DebugOption
from .trainer_utils import (
    FSDPOption,
    HubStrategy,
    IntervalStrategy,
    SaveStrategy,
    SchedulerType,
)
from .utils import (
    ACCELERATE_MIN_VERSION,
    ExplicitEnum,
    is_accelerate_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tf32_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    logging,
    requires_backends,
)
from .utils.generic import strtobool
from .utils.import_utils import enable_tf32, is_optimum_neuron_available


logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)

if is_torch_available():
    import torch
    import torch.distributed as dist

if is_accelerate_available():
    from accelerate.state import AcceleratorState, PartialState
    from accelerate.utils import DistributedType

    from .trainer_pt_utils import AcceleratorConfig

if is_accelerate_available("1.10.1"):
    from accelerate.parallelism_config import ParallelismConfig
else:
    ParallelismConfig = Any

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_torch_neuroncore_available(check_device=False):
    # torchrun support
    # https://github.com/pytorch/xla/pull/3609
    if os.environ.get("TORCHELASTIC_RUN_ID"):
        if is_optimum_neuron_available():
            logger.info(
                "Make sure that you are performing the training with the NeuronTrainer from optimum[neuron], this "
                "will fail otherwise."
            )
        else:
            logger.warning(
                "Please use the NeuronTrainer from optimum[neuron] instead of the Transformers library to perform "
                "training on AWS Trainium instances. More information here: "
                "https://github.com/huggingface/optimum-neuron"
            )
            import torch_xla.distributed.xla_backend as xbn

            if not isinstance(dist.group.WORLD, xbn.ProcessGroupXla):
                dist.init_process_group(backend="xla")
                if not isinstance(dist.group.WORLD, xbn.ProcessGroupXla):
                    raise AssertionError("Failed to initialize torch.distributed process group using XLA backend.")


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    smp.init()


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_TORCH_NPU_FUSED = "adamw_torch_npu_fused"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    ADAMW_TORCH_4BIT = "adamw_torch_4bit"
    ADAMW_TORCH_8BIT = "adamw_torch_8bit"
    ADEMAMIX = "ademamix"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAMW_BNB = "adamw_bnb_8bit"
    ADAMW_8BIT = "adamw_8bit"  # just an alias for adamw_bnb_8bit
    ADEMAMIX_8BIT = "ademamix_8bit"
    LION_8BIT = "lion_8bit"
    LION = "lion_32bit"
    PAGED_ADAMW = "paged_adamw_32bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_ADEMAMIX = "paged_ademamix_32bit"
    PAGED_ADEMAMIX_8BIT = "paged_ademamix_8bit"
    PAGED_LION = "paged_lion_32bit"
    PAGED_LION_8BIT = "paged_lion_8bit"
    RMSPROP = "rmsprop"
    RMSPROP_BNB = "rmsprop_bnb"
    RMSPROP_8BIT = "rmsprop_bnb_8bit"
    RMSPROP_32BIT = "rmsprop_bnb_32bit"
    GALORE_ADAMW = "galore_adamw"
    GALORE_ADAMW_8BIT = "galore_adamw_8bit"
    GALORE_ADAFACTOR = "galore_adafactor"
    GALORE_ADAMW_LAYERWISE = "galore_adamw_layerwise"
    GALORE_ADAMW_8BIT_LAYERWISE = "galore_adamw_8bit_layerwise"
    GALORE_ADAFACTOR_LAYERWISE = "galore_adafactor_layerwise"
    LOMO = "lomo"
    ADALOMO = "adalomo"
    GROKADAMW = "grokadamw"
    SCHEDULE_FREE_RADAM = "schedule_free_radam"
    SCHEDULE_FREE_ADAMW = "schedule_free_adamw"
    SCHEDULE_FREE_SGD = "schedule_free_sgd"
    APOLLO_ADAMW = "apollo_adamw"
    APOLLO_ADAMW_LAYERWISE = "apollo_adamw_layerwise"
    STABLE_ADAMW = "stable_adamw"


def _convert_str_dict(passed_value: dict):
    "Safely checks that a passed value is a dictionary and converts any string values to their appropriate types."
    for key, value in passed_value.items():
        if isinstance(value, dict):
            passed_value[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            # First check for bool and convert
            if value.lower() in ("true", "false"):
                passed_value[key] = value.lower() == "true"
            # Check for digit
            elif value.isdigit():
                passed_value[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                passed_value[key] = float(value)

    return passed_value


@dataclass
class TrainingArguments:
    """
    Configuration class for controlling all aspects of model training with the Trainer.
    TrainingArguments centralizes all hyperparameters, optimization settings, logging preferences, and infrastructure choices needed for training.

    [`HfArgumentParser`] can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        output_dir (`str`, *optional*, defaults to `"trainer_output"`):
            The output directory where the model predictions and checkpoints will be written.

        > Training Duration and Batch Size

        per_device_train_batch_size (`int`, *optional*, defaults to 8):
            The batch size *per device*. The **global batch size** is computed as:
            `per_device_train_batch_size * number_of_devices` in multi-GPU or distributed setups.
        num_train_epochs(`float`, *optional*, defaults to 3.0):
            Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
            the last epoch before stopping training).
        max_steps (`int`, *optional*, defaults to -1):
            Overrides `num_train_epochs`. If set to a positive number, the total number of training steps to perform.
            For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
            `max_steps` is reached.

        > Learning Rate & Scheduler

        learning_rate (`float`, *optional*, defaults to 5e-5):
            The initial learning rate for the optimizer. This is typically the peak learning rate when using a scheduler with warmup.
        lr_scheduler_type (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
            The learning rate scheduler type to use. See [`SchedulerType`] for all possible values. Common choices:
                - "linear" = [`get_linear_schedule_with_warmup`]
                - "cosine" = [`get_cosine_schedule_with_warmup`]
                - "constant" =  [`get_constant_schedule`]
                - "constant_with_warmup" = [`get_constant_schedule_with_warmup`]
        lr_scheduler_kwargs (`dict` or `str`, *optional*, defaults to `None`):
            The extra arguments for the lr_scheduler. See the documentation of each scheduler for possible values.
        warmup_steps (`int` or `float`, *optional*, defaults to 0):
            Number of steps for a linear warmup from 0 to `learning_rate`. Warmup helps stabilize training in the initial phase. Can be:
                - An integer: exact number of warmup steps
                - A float in range [0, 1): interpreted as ratio of total training steps

        > Optimizer

        optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_torch"` (for torch>=2.8 `"adamw_torch_fused"`)):
            The optimizer to use. Common options:
                - `"adamw_torch"`: PyTorch's AdamW (recommended default)
                - `"adamw_torch_fused"`: Fused AdamW kernel
                - `"adamw_hf"`: HuggingFace's AdamW implementation
                - `"sgd"`: Stochastic Gradient Descent with momentum
                - `"adafactor"`: Memory-efficient optimizer for large models
                - `"adamw_8bit"`: 8-bit AdamW (requires bitsandbytes)
                See [`OptimizerNames`] for the complete list.
        optim_args (`str`, *optional*):
            Optional arguments that are supplied to optimizers such as AnyPrecisionAdamW, AdEMAMix, and GaLore.
        weight_decay (`float`, *optional*, defaults to 0):
            Weight decay coefficient applied by the optimizer (not the loss function). Adds L2
            regularization to prevent overfitting by penalizing large weights. Automatically
            excluded from bias and LayerNorm parameters. Typical values: 0.01 (standard), 0.1
            (stronger regularization), 0.0 (no regularization).
        adam_beta1 (`float`, *optional*, defaults to 0.9):
            The exponential decay rate for the first moment estimates (momentum) in Adam-based
            optimizers. Controls how much history of gradients to retain.
        adam_beta2 (`float`, *optional*, defaults to 0.999):
            The exponential decay rate for the second moment estimates (variance) in Adam-based
            optimizers. Controls adaptive learning rate scaling.
        adam_epsilon (`float`, *optional*, defaults to 1e-8):
            Epsilon value for numerical stability in Adam-based optimizers. Prevents division by
            zero in the denominator of the update rule.
        optim_target_modules (`Union[str, list[str]]`, *optional*):
            The target modules to optimize, i.e. the module names that you would like to train.
            Currently used for the [GaLore algorithm](https://huggingface.co/papers/2403.03507) and [APOLLO algorithm](https://huggingface.co/papers/2412.05270).
            See [GaLore implementation](https://github.com/jiaweizzhao/GaLore) and [APOLLO implementation](https://github.com/zhuhanqing/APOLLO) for more details.
            You need to make sure to pass a valid GaLore or APOLLO optimizer, e.g., one of: "apollo_adamw", "galore_adamw", "galore_adamw_8bit", "galore_adafactor" and make sure that the target modules are `nn.Linear` modules only.

        > Regularization & Training Stability

        gradient_accumulation_steps (`int`, *optional*, defaults to 1):
            Number of update steps to accumulate gradients before performing a backward/update pass.
            Simulates larger batch sizes without additional memory. Effective batch size =
            `per_device_train_batch_size × num_devices × gradient_accumulation_steps`.
            > [!TIP]
            > When using gradient accumulation, one "step" is counted as one step with a backward pass. Therefore, logging, evaluation, and saving will occur every `gradient_accumulation_steps × xxx_step` training examples.
        average_tokens_across_devices (`bool`, *optional*, defaults to `True`):
            Whether or not to average tokens across devices. If enabled, will use all_reduce to synchronize
            num_tokens_in_batch for precise loss calculation. Reference:
            https://github.com/huggingface/transformers/issues/34242
        max_grad_norm (`float`, *optional*, defaults to 1.0):
            Maximum gradient norm for gradient clipping. Applied after backward pass, before
            optimizer step. Prevents gradient explosion by scaling down gradients when their global
            norm exceeds this threshold. Set to 0 to disable clipping. Typical values:
            1.0 (standard), 0.5 (more conservative), 5.0 (less aggressive).
        label_smoothing_factor (`float`, *optional*, defaults to 0.0):
            Label smoothing factor to prevent overconfidence. Replaces hard 0/1 targets with soft
            targets: 0 becomes `ε/num_labels` and 1 becomes `1 - ε + ε/num_labels`, where
            ε = `label_smoothing_factor`. Zero means no smoothing. Typical range: 0.0 to 0.1.

        > Mixed Precision Training

        bf16 (`bool`, *optional*, defaults to `False`):
            Enable bfloat16 (BF16) mixed precision training
            Generally preferred over FP16 due to better numerical stability and no loss scaling required.
        fp16 (`bool`, *optional*, defaults to `False`):
            Enable float16 (FP16) mixed precision training.
            Consider using BF16 instead if your hardware supports it.
        bf16_full_eval (`bool`, *optional*, defaults to `False`):
            Use full BF16 precision for evaluation (not just mixed precision). Faster and saves
            memory but may affect metric values slightly. Only applies during evaluation.
        fp16_full_eval (`bool`, *optional*, defaults to `False`):
            Use full FP16 precision for evaluation (not just mixed precision). Faster and saves
            memory but may affect metric values slightly. Only applies during evaluation.
        tf32 (`bool`, *optional*):
            Enable TensorFloat-32 (TF32) mode on Ampere and newer GPUs. TF32 uses 19-bit precision
            for matrix multiplications (instead of FP32's 23-bit), providing up to 8x speedup with
            negligible accuracy loss. Default depends on PyTorch version. See
            [TF32 docs](https://huggingface.co/docs/transformers/perf_train_gpu_one#tf32).

        > Gradient Checkpointing

        gradient_checkpointing (`bool`, *optional*, defaults to `False`):
            Enable gradient checkpointing to trade compute for memory. Reduces memory usage by
            clearing activations during forward pass and recomputing them during backward pass.
            Enables training larger models or batch sizes at the cost of ~20% slower training.
        gradient_checkpointing_kwargs (`dict`, *optional*, defaults to `None`):
            Keyword arguments passed to `gradient_checkpointing_enable()`.

        > Compilation

        torch_compile (`bool`, *optional*, defaults to `False`):
            Compile the model using PyTorch 2.0's `torch.compile()` for faster training. Can provide
            20-50% speedup with no code changes. Uses default compilation settings unless
            `torch_compile_backend` or `torch_compile_mode` are specified.
        torch_compile_backend (`str`, *optional*):
            Backend for `torch.compile()`. If set, automatically enables `torch_compile`. Options
            include `"inductor"` (default), `"aot_eager"`, `"cudagraphs"`. Backends vary by PyTorch
            version - see PyTorch docs for available options.
        torch_compile_mode (`str`, *optional*):
            Compilation mode for `torch.compile()`. If set, automatically enables `torch_compile`.
            Options: `"default"`, `"reduce-overhead"` (minimize Python overhead), `"max-autotune"`
            (aggressive optimization, slower compile time).

        > Kernels

        use_liger_kernel (`bool`, *optional*, defaults to `False`):
            Enable [Liger Kernel](https://github.com/linkedin/Liger-Kernel) optimizations. Increases
            multi-GPU throughput by ~20% and reduces memory usage by ~60%. Works with Flash Attention,
            FSDP, and DeepSpeed. Currently supports Llama, Mistral, Mixtral, and Gemma models.
        liger_kernel_config (`Optional[dict]`, *optional*):
            Configuration for Liger Kernel. Passed as kwargs to `_apply_liger_kernel_to_instance()`.
            Options typically include: `"rope"`, `"swiglu"`, `"cross_entropy"`,
            `"fused_linear_cross_entropy"`, `"rms_norm"`. If `None`, uses default configuration.

        > Additional Optimizations

        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not to enable cache for the model. For training, this is usually not needed apart from some PEFT methods that uses `past_key_values`.
        neftune_noise_alpha (`Optional[float]`):
            If not `None`, this will activate NEFTune noise embeddings. This can drastically improve model performance
            for instruction fine-tuning. Check out the [original paper](https://huggingface.co/papers/2310.05914) and the
            [original code](https://github.com/neelsjain/NEFTune). Support transformers `PreTrainedModel` and also
            `PeftModel` from peft. The original paper used values in the range [5.0, 15.0].
        torch_empty_cache_steps (`int`, *optional*):
            Number of steps to wait before calling `torch.<device>.empty_cache()`. If left unset or set to None, cache will not be emptied.
            This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage at a cost of about [10% slower performance](https://github.com/huggingface/transformers/issues/31372).
        auto_find_batch_size (`bool`, *optional*, defaults to `False`)
            Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding
            CUDA Out-of-Memory errors.

        > Logging & Monitoring Training

        logging_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The logging strategy to adopt during training. Possible values are:
                - `"no"`: No logging is done during training.
                - `"epoch"`: Logging is done at the end of each epoch.
                - `"steps"`: Logging is done every `logging_steps`.
        logging_steps (`int` or `float`, *optional*, defaults to 500):
            Number of update steps between two logs if `logging_strategy="steps"`. Should be an integer or a float in
            range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
        logging_first_step (`bool`, *optional*, defaults to `False`):
            Whether to log the first `global_step` or not.
        log_on_each_node (`bool`, *optional*, defaults to `True`):
            In multinode distributed training, whether to log using `log_level` once per node, or only on the main
            node.
        logging_nan_inf_filter (`bool`, *optional*, defaults to `True`):
             Filter out NaN and Inf losses when logging. If `True`, replaces NaN/Inf losses with the
            average of recent valid losses. Does not affect gradient computation, only logging.
        include_num_input_tokens_seen (`Optional[Union[str, bool]]`, *optional*, defaults to "no"):
            Whether to track the number of input tokens seen. Must be one of ["all", "non_padding", "no"] or a boolean value which map to "all" or "no".
            May be slower in distributed training as gather operations must be called.

        > Logging

        log_level (`str`, *optional*, defaults to `passive`):
            Logging level for the main process. Options: `"debug"`, `"info"`, `"warning"`, `"error"`,
            `"critical"`, or `"passive"` (doesn't change the current Transformers logging level,
            which defaults to `"warning"`)
        log_level_replica (`str`, *optional*, defaults to `"warning"`):
            Logging level for replica processes in distributed training. Same options as `log_level`.
        disable_tqdm (`bool`, *optional*):
            Disable tqdm progress bars. Defaults to `True` if `log_level` is warning or lower, `False` otherwise.

        > Experiment Tracking Integration

        report_to (`str` or `list[str]`, *optional*, defaults to `"none"`):
            The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
            `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`, `"swanlab"`,
            `"tensorboard"`, `"trackio"` and `"wandb"`. Use `"all"` to report to all integrations installed, `"none"`
            for no integrations.
        run_name (`str`, *optional*):
            A descriptor for the run. Typically used for [trackio](https://github.com/gradio-app/trackio),
            [wandb](https://www.wandb.com/), [mlflow](https://www.mlflow.org/), [comet](https://www.comet.com/site) and
            [swanlab](https://swanlab.cn) logging.
        project (`str`, *optional*, defaults to `"huggingface"`):
            The name of the project to use for logging. Currently, only used by Trackio.
        trackio_space_id (`str` or `None`, *optional*, defaults to `"trackio"`):
            The Hugging Face Space ID to deploy to when using Trackio. Should be a complete Space name like
            `'username/reponame'` or `'orgname/reponame'`, or just `'reponame'` in which case the Space will be
            created in the currently-logged-in Hugging Face user's namespace. If `None`, will log to a local directory.
            Note that this Space will be public unless you set `hub_private_repo=True` or your organization's default
            is to create private Spaces."

        > Evaluation

        eval_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
            When to run evaluation. Options:
            - `"no"`: No evaluation during training
            - `"steps"`: Evaluate every `eval_steps`
            - `"epoch"`: Evaluate at the end of each epoch
        eval_steps (`int` or `float`, *optional*):
            Number of update steps between two evaluations if `eval_strategy="steps"`. Will default to the same
            value as `logging_steps` if not set. Should be an integer or a float in range `[0,1)`. If smaller than 1,
            will be interpreted as ratio of total training steps.
        eval_delay (`float`, *optional*):
            Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
            eval_strategy.
        per_device_eval_batch_size (`int`, *optional*, defaults to 8):
            The batch size per device accelerator core/CPU for evaluation.
        prediction_loss_only (`bool`, *optional*, defaults to `False`):
            When performing evaluation and generating predictions, only returns the loss.
        eval_on_start (`bool`, *optional*, defaults to `False`):
            Whether to perform a evaluation step (sanity check) before the training to ensure the validation steps works correctly.
        eval_do_concat_batches (`bool`, *optional*, defaults to `True`):
            Whether to recursively concat inputs/losses/labels/predictions across batches. If `False`,
            will instead store them as lists, with each batch kept separate.
        eval_use_gather_object (`bool`, *optional*, defaults to `False`):
            Whether to run recursively gather object in a nested list/tuple/dictionary of objects from all devices. This should only be enabled if users are not just returning tensors, and this is actively discouraged by PyTorch.
            This is useful when the labels structure is non standard, like in computer vision tasks.
        eval_accumulation_steps (`int`, *optional*):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on the device accelerator before being moved to the CPU (faster but
            requires more memory).

        > Metrics Computation

        include_for_metrics (`list[str]`, *optional*, defaults to `[]`):
            Include additional data in the `compute_metrics` function if needed for metrics computation.
            Possible options to add to `include_for_metrics` list:
            - `"inputs"`: Input data passed to the model, intended for calculating input dependent metrics.
            - `"loss"`: Loss values computed during evaluation, intended for calculating loss dependent metrics.
        batch_eval_metrics (`bool`, *optional*, defaults to `False`):
            If set to `True`, evaluation will call compute_metrics at the end of each batch to accumulate statistics
            rather than saving all eval logits in memory. When set to `True`, you must pass a compute_metrics function
            that takes a boolean argument `compute_result`, which when passed `True`, will trigger the final global
            summary statistics from the batch-level summary statistics you've accumulated over the evaluation set.

        > Checkpointing & Saving

        save_only_model (`bool`, *optional*, defaults to `False`):
            Save only model weights, not optimizer/scheduler/RNG state. Significantly reduces
            checkpoint size but prevents resuming training from the checkpoint. Use when you only
            need the trained model for inference, not continued training.
            You can only load the model using `from_pretrained` with this option set to `True`.
        save_strategy (`str` or [`~trainer_utils.SaveStrategy`], *optional*, defaults to `"steps"`):
            The checkpoint save strategy to adopt during training. Possible values are:
                - `"no"`: No save is done during training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`.
                - `"best"`: Save is done whenever a new `best_metric` is achieved.
        save_steps (`int` or `float`, *optional*, defaults to 500):
            Number of updates steps before two checkpoint saves if `save_strategy="steps"`. Should be an integer or a
            float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
        save_on_each_node (`bool`, *optional*, defaults to `False`):
            When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on
            the main one.
            This should not be activated when the different nodes use the same storage as the files will be saved with
            the same names for each node.
        save_total_limit (`int`, *optional*):
            Maximum number of checkpoints to keep. Deletes older checkpoints in `output_dir`. When
            `load_best_model_at_end=True`, the best checkpoint is always retained plus the most
            recent ones. For example, `save_total_limit=5` keeps the 4 most recent plus the best
        enable_jit_checkpoint (`bool`, *optional*, defaults to `False`):
            Enable Just-In-Time checkpointing on SIGTERM signal for graceful termination on
            preemptible workloads. **Important**: Configure your orchestrator's graceful shutdown
            period to allow sufficient time. For Kubernetes, set `terminationGracePeriodSeconds`
            (default 30s is usually insufficient). For Slurm, use `--signal=USR1@<seconds>`.
            Required grace period ≥ longest iteration time + checkpoint save time.

        > Hugging Face Hub Integration

        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether or not to push the model to the Hub every time the model is saved. If this is activated,
            `output_dir` will begin a git directory synced with the repo (determined by `hub_model_id`) and the content
            will be pushed each time a save is triggered (depending on your `save_strategy`). Calling
            [`~Trainer.save_model`] will also trigger a push.
        hub_token (`str`, *optional*):
            The token to use to push the model to the Hub. Will default to the token in the cache folder obtained with
            `hf auth login`.
        hub_private_repo (`bool`, *optional*):
            Whether to make the repo private. If `None` (default), the repo will be public unless the organization's
            default is private. This value is ignored if the repo already exists. If reporting to Trackio with
            deployment to Hugging Face Spaces enabled, the same logic determines whether the Space is private.
        hub_model_id (`str`, *optional*):
            The name of the repository to keep in sync with the local *output_dir*. It can be a simple model ID in
            which case the model will be pushed in your namespace. Otherwise it should be the whole repository name,
            for instance `"user_name/model"`, which allows you to push to an organization you are a member of with
            `"organization_name/model"`. Will default to `user_name/output_dir_name` with *output_dir_name* being the
            name of `output_dir`.
        hub_strategy (`str` or [`~trainer_utils.HubStrategy`], *optional*, defaults to `"every_save"`):
            Defines what and when to push to Hub. Options:
                - `"end"`: Push only at the end of training
                - `"every_save"`: Push on each save (async to not block training)
                - `"checkpoint"`: Like `"every_save"` plus push latest checkpoint to `"last-checkpoint"` subfolder for easy resuming
                - `"all_checkpoints"`: Push all checkpoints as they appear
        hub_always_push (`bool`, *optional*, defaults to `False`):
            Unless this is `True`, the `Trainer` will skip pushing a checkpoint when the previous push is not finished.
        hub_revision (`str`, *optional*):
            The revision to use when pushing to the Hub. Can be a branch name, a tag, or a commit hash.

        > Best Model Tracking

        load_best_model_at_end (`bool`, *optional*, defaults to `False`):
            Load the best checkpoint at the end of training. Requires `eval_strategy` to be set.
            When enabled, the best checkpoint is always saved (see `save_total_limit`).
            <Tip>
            When `True`, `save_strategy` must match `eval_strategy`, and if using `"steps"`,
            `save_steps` must be a multiple of `eval_steps`.
            </Tip>
        metric_for_best_model (`str`, *optional*):
            Metric to use for comparing models when `load_best_model_at_end=True`. Must be a metric
            name returned by evaluation, with or without the `"eval_"` prefix. Defaults to `"loss"`.
            If you set this, `greater_is_better` will default to `True` unless the name ends with
            `"loss"`. Examples: `"accuracy"`, `"f1"`, `"eval_bleu"`.
        greater_is_better (`bool`, *optional*):
            Whether higher metric values are better. Defaults based on `metric_for_best_model`:
            `True` if the metric name doesn't end in `"loss"`, `False` otherwise.

        > Resuming Training

        ignore_data_skip (`bool`, *optional*, defaults to `False`):
            When resuming training, skip fast-forwarding through the dataset to reach the previous
            state. If `True`, training starts from the beginning of the dataset (faster resume but
            results won't match interrupted training). If `False`, skips seen data (slower resume
            but exact continuation).
        restore_callback_states_from_checkpoint (`bool`, *optional*, defaults to `False`):
            Restore callback states from checkpoint when resuming. If `True`, will override callbacks
            passed to Trainer if they exist in the checkpoint.

        > Reproducibility

        full_determinism (`bool`, *optional*, defaults to `False`)
            If `True`, [`enable_full_determinism`] is called instead of [`set_seed`] to ensure reproducible results in
            distributed training. Important: this will negatively impact the performance, so only use it for debugging.
        seed (`int`, *optional*, defaults to 42):
            Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
            [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized parameters.
        data_seed (`int`, *optional*):
            Random seed to be used with data samplers. If not set, random generators for data sampling will use the
            same seed as `seed`. This can be used to ensure reproducibility of data sampling, independent of the model
            seed.

        > Hardware Configuration

        use_cpu (`bool`, *optional*, defaults to `False`):
            Whether or not to use cpu. If set to False, we will use the available torch device/backend.

        > Accelerate Configuration

        accelerator_config (`str`, `dict`, or `AcceleratorConfig`, *optional*):
            Configuration for the internal Accelerate integration. Can be:
                - Path to JSON config file: `"accelerator_config.json"`
                - Dictionary with config options
                - `AcceleratorConfig` instance
            Key options:
                - `split_batches` (`bool`, defaults to `False`): Whether to split batches across devices.
                    If `True`, actual batch size is the same on all devices (total must be divisible by
                    num_processes). If `False`, each device gets the specified batch size.
                - `dispatch_batches` (`bool`): If `True`, only main process iterates through dataloader
                    and dispatches batches to devices. Defaults to `True` for `IterableDataset`, `False`
                    otherwise.
                - `even_batches` (`bool`, defaults to `True`): Duplicate samples from dataset start to
                    ensure all workers get equal batch sizes.
                - `use_seedable_sampler` (`bool`, defaults to `True`): Use fully seedable random sampler
                    for reproducibility.
                - `use_configured_state` (`bool`, defaults to `False`): Use pre-initialized
                    `AcceleratorState`/`PartialState` instead of creating new one. May cause issues with
                    hyperparameter tuning.

        parallelism_config (`ParallelismConfig`, *optional*):
            Parallelism configuration for the training run. Requires Accelerate `1.10.1`

        > Dataloader

        dataloader_drop_last (`bool`, *optional*, defaults to `False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        dataloader_num_workers (`int`, *optional*, defaults to 0):
            Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
            main process.
        dataloader_pin_memory (`bool`, *optional*, defaults to `True`):
            Whether you want to pin memory in data loaders or not. Will default to `True`.
        dataloader_persistent_workers (`bool`, *optional*, defaults to `False`):
            If True, the data loader will not shut down the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will
            increase RAM usage. Will default to `False`.
        dataloader_prefetch_factor (`int`, *optional*):
            Number of batches loaded in advance by each worker.
            2 means there will be a total of 2 * num_workers batches prefetched across all workers.
        remove_unused_columns (`bool`, *optional*, defaults to `True`):
            Whether or not to automatically remove the columns unused by the model forward method.
        label_names (`list[str]`, *optional*):
            The list of keys in your dictionary of inputs that correspond to the labels.
            Will eventually default to the list of argument names accepted by the model that contain the word "label",
            except if the model used is one of the `XxxForQuestionAnswering` in which case it will also include the
            `["start_positions", "end_positions"]` keys.
            You should only specify `label_names` if you're using custom label names or if your model's `forward` consumes multiple label tensors (e.g., extractive QA).
        train_sampling_strategy (`str`, *optional*, defaults to `"random"`):
            The sampler to use for the training dataloader. Possible values are:

                - `"random"`: Uses `RandomSampler` (default).
                - `"sequential"`: Uses `SequentialSampler`.
                - `"group_by_length"`: Uses `LengthGroupedSampler` to group samples of roughly the same length
                  together (to minimize padding and be more efficient).

            Note: When using an `IterableDataset`, this argument is ignored.
        length_column_name (`str`, *optional*, defaults to `"length"`):
            Column name for precomputed lengths. If the column exists, grouping by length will use these values rather
            than computing them on train startup. Ignored unless `train_sampling_strategy` is `"group_by_length"` and the dataset
            is an instance of `Dataset`.

        > DDP (DistributedDataParallel)

        ddp_find_unused_parameters (`bool`, *optional*):
            When using distributed training, the value of the flag `find_unused_parameters` passed to
            `DistributedDataParallel`. Will default to `False` if gradient checkpointing is used, `True` otherwise.
        ddp_bucket_cap_mb (`int`, *optional*):
            When using distributed training, the value of the flag `bucket_cap_mb` passed to `DistributedDataParallel`.
        ddp_broadcast_buffers (`bool`, *optional*):
            When using distributed training, the value of the flag `broadcast_buffers` passed to
            `DistributedDataParallel`. Will default to `False` if gradient checkpointing is used, `True` otherwise.
        ddp_backend (`str`, *optional*):
            The backend to use for distributed training. Must be one of `"nccl"`, `"mpi"`, `"xccl"`, `"gloo"`, `"hccl"`.
        ddp_timeout (`int`, *optional*, defaults to 1800):
            The timeout for `torch.distributed.init_process_group` calls, used to avoid GPU socket timeouts when
            performing slow operations in distributed runnings. Please refer to the [PyTorch documentation](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) for more
            information.

        > FSDP (Fully Sharded Data Parallel)

        fsdp (`bool`, `str` or list of [`~trainer_utils.FSDPOption`], *optional*, defaults to `None`):
            Enable PyTorch Fully Sharded Data Parallel (FSDP) for distributed training. Options:
                - `"full_shard"`: Shard parameters, gradients, and optimizer states (most memory efficient)
                - `"shard_grad_op"`: Shard only optimizer states and gradients (ZeRO-2)
                - `"hybrid_shard"`: Full shard within nodes, replicate across nodes
                - `"hybrid_shard_zero2"`: Shard gradients/optimizer within nodes, replicate across nodes
                - `"offload"`: Offload parameters and gradients to CPU (only with `"full_shard"` or
                    `"shard_grad_op"`)
                - `"auto_wrap"`: Automatically wrap layers using `default_auto_wrap_policy`
        fsdp_config (`str` or `dict`, *optional*):
            Config to be used with fsdp (Pytorch Distributed Parallel Training). The value is either a location of
            fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`.

            A List of config and its options:
                - fsdp_version (`int`, *optional*, defaults to `1`):
                    The version of FSDP to use. Defaults to 1.
                - min_num_params (`int`, *optional*, defaults to `0`):
                    FSDP's minimum number of parameters for Default Auto Wrapping. (useful only when `fsdp` field is
                    passed).
                - transformer_layer_cls_to_wrap (`list[str]`, *optional*):
                    List of transformer layer class names (case-sensitive) to wrap, e.g, `BertLayer`, `GPTJBlock`,
                    `T5Block` .... (useful only when `fsdp` flag is passed).
                - backward_prefetch (`str`, *optional*)
                    FSDP's backward prefetch mode. Controls when to prefetch next set of parameters (useful only when
                    `fsdp` field is passed).

                    A list of options along the following:

                    - `"backward_pre"` : Prefetches the next set of parameters before the current set of parameter's
                      gradient computation.
                    - `"backward_post"` : This prefetches the next set of parameters after the current set of
                      parameter's gradient computation.
                - forward_prefetch (`bool`, *optional*, defaults to `False`)
                    FSDP's forward prefetch mode (useful only when `fsdp` field is passed).
                     If `"True"`, then FSDP explicitly prefetches the next upcoming all-gather while executing in the
                     forward pass.
                - limit_all_gathers (`bool`, *optional*, defaults to `False`)
                    FSDP's limit_all_gathers (useful only when `fsdp` field is passed).
                     If `"True"`, FSDP explicitly synchronizes the CPU thread to prevent too many in-flight
                     all-gathers.
                - use_orig_params (`bool`, *optional*, defaults to `True`)
                    If `"True"`, allows non-uniform `requires_grad` during init, which means support for interspersed
                    frozen and trainable parameters. Useful in cases such as parameter-efficient fine-tuning. Please
                    refer this
                    [blog](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019
                - sync_module_states (`bool`, *optional*, defaults to `True`)
                    If `"True"`, each individually wrapped FSDP unit will broadcast module parameters from rank 0 to
                    ensure they are the same across all ranks after initialization
                - cpu_ram_efficient_loading (`bool`, *optional*, defaults to `False`)
                    If `"True"`, only the first process loads the pretrained model checkpoint while all other processes
                    have empty weights.  When this setting as `"True"`, `sync_module_states` also must to be `"True"`,
                    otherwise all the processes except the main process would have random weights leading to unexpected
                    behaviour during training.
                - activation_checkpointing (`bool`, *optional*, defaults to `False`):
                    If `"True"`, activation checkpointing is a technique to reduce memory usage by clearing activations of
                    certain layers and recomputing them during a backward pass. Effectively, this trades extra
                    computation time for reduced memory usage.
                - xla (`bool`, *optional*, defaults to `False`):
                    Whether to use PyTorch/XLA Fully Sharded Data Parallel Training. This is an experimental feature
                    and its API may evolve in the future.
                - xla_fsdp_settings (`dict`, *optional*)
                    The value is a dictionary which stores the XLA FSDP wrapping parameters.

                    For a complete list of options, please see [here](
                    https://github.com/pytorch/xla/blob/master/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py).
                - xla_fsdp_grad_ckpt (`bool`, *optional*, defaults to `False`):
                    Will use gradient checkpointing over each nested XLA FSDP wrapped layer. This setting can only be
                    used when the xla flag is set to true, and an auto wrapping policy is specified through
                    fsdp_min_num_params or fsdp_transformer_layer_cls_to_wrap.

        > DeepSpeed

        deepspeed (`str` or `dict`, *optional*):
             Enable [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) integration. Value is either:
                - Path to DeepSpeed JSON config file: `"ds_config.json"`
                - Loaded config as dictionary
          > [!TIP]
          > If using ZeRO initialization, instantiate your model *after* initializing
          `TrainingArguments`, otherwise ZeRO won't be applied.

        > Debugging & Profiling (Experimental)

        debug (`str` or list of [`~debug_utils.DebugOption`], *optional*, defaults to `""`):
            Enable one or more debug features. This is an experimental feature.
            Possible options are:
            - "underflow_overflow": detects overflow in model's input/outputs and reports the last frames that led to
              the event
            - "tpu_metrics_debug": print debug metrics on TPU
        skip_memory_metrics (`bool`, *optional*, defaults to `True`):
            Whether to skip adding of memory profiler reports to metrics. This is skipped by default because it slows
            down the training and evaluation speed.

        > External Script Flags (not used by Trainer)

        do_train (`bool`, *optional*, defaults to `False`):
            Whether to run training or not. This argument is not directly used by [`Trainer`], it's intended to be used
            by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        do_eval (`bool`, *optional*):
            Whether to run evaluation on the validation set or not. Will be set to `True` if `eval_strategy` is
            different from `"no"`. This argument is not directly used by [`Trainer`], it's intended to be used by your
            training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        do_predict (`bool`, *optional*, defaults to `False`):
            Whether to run predictions on the test set or not. This argument is not directly used by [`Trainer`], it's
            intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        resume_from_checkpoint (`str`, *optional*):
            The path to a folder with a valid checkpoint for your model. This argument is not directly used by
            [`Trainer`], it's intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
    """

    # Fields that accept dict values via CLI as JSON strings (e.g., '{"key": "value"}').
    # Any new dict-typed arg must be added here and typed as `dict | str | None`.
    _VALID_DICT_FIELDS = [
        "accelerator_config",
        "fsdp_config",
        "deepspeed",
        "gradient_checkpointing_kwargs",
        "lr_scheduler_kwargs",
    ]

    # --- Output ---
    output_dir: str | None = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    # --- Training Duration and Batch Size ---
    per_device_train_batch_size: int = field(default=8, metadata={"help": "The batch size per device for training."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={
            "help": "Overrides `num_train_epochs`. If set to a positive number, the total number of training steps to perform."
        },
    )

    # --- Learning Rate & Scheduler ---
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for the optimizer."})
    lr_scheduler_type: SchedulerType | str = field(
        default="linear",
        metadata={"help": "The learning rate scheduler type to use. See `SchedulerType` for all possible values."},
    )
    lr_scheduler_kwargs: dict | str | None = field(
        default=None,
        metadata={
            "help": "The extra arguments for the lr_scheduler. See the documentation of each scheduler for possible values."
        },
    )
    warmup_steps: float = field(
        default=0,
        metadata={
            "help": "Number of steps for a linear warmup from 0 to `learning_rate`. Can be an integer (exact steps) or a float in [0, 1) (ratio of total steps)."
        },
    )

    # --- Optimizer ---
    default_optim = "adamw_torch"
    if is_torch_available():
        from .pytorch_utils import is_torch_greater_or_equal_than_2_8

        if is_torch_greater_or_equal_than_2_8:
            default_optim = "adamw_torch_fused"
    optim: OptimizerNames | str = field(
        default=default_optim,
        metadata={"help": "The optimizer to use. See `OptimizerNames` for the complete list."},
    )
    optim_args: str | None = field(
        default=None,
        metadata={
            "help": "Optional arguments supplied to optimizers such as AnyPrecisionAdamW, AdEMAMix, and GaLore."
        },
    )
    weight_decay: float = field(
        default=0.0,
        metadata={
            "help": "Weight decay coefficient applied by the optimizer. Automatically excluded from bias and LayerNorm parameters."
        },
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={
            "help": "The exponential decay rate for the first moment estimates (momentum) in Adam-based optimizers."
        },
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={
            "help": "The exponential decay rate for the second moment estimates (variance) in Adam-based optimizers."
        },
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon value for numerical stability in Adam-based optimizers."}
    )
    optim_target_modules: None | str | list[str] = field(
        default=None,
        metadata={"help": "The target modules to optimize. Currently used for the GaLore and APOLLO algorithms."},
    )

    # --- Regularization & Training Stability ---
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Number of update steps to accumulate gradients before performing a backward/update pass."
                " Effective batch size = per_device_train_batch_size * num_devices * gradient_accumulation_steps."
            )
        },
    )
    average_tokens_across_devices: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to average tokens across devices. If enabled, will use all_reduce to "
            "synchronize num_tokens_in_batch for precise loss calculation. Reference: "
            "https://github.com/huggingface/transformers/issues/34242"
        },
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Maximum gradient norm for gradient clipping. Set to 0 to disable."}
    )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "Label smoothing factor to prevent overconfidence. Zero means no smoothing."}
    )

    # --- Mixed Precision ---
    bf16: bool = field(
        default=False,
        metadata={
            "help": "Enable bfloat16 (BF16) mixed precision training. Generally preferred over FP16 due to better numerical stability."
        },
    )
    fp16: bool = field(
        default=False,
        metadata={
            "help": "Enable float16 (FP16) mixed precision training. Consider using BF16 instead if your hardware supports it."
        },
    )
    bf16_full_eval: bool = field(
        default=False,
        metadata={
            "help": "Use full BF16 precision for evaluation (not just mixed precision). Faster and saves memory."
        },
    )
    fp16_full_eval: bool = field(
        default=False,
        metadata={
            "help": "Use full FP16 precision for evaluation (not just mixed precision). Faster and saves memory."
        },
    )
    tf32: bool | None = field(
        default=None,
        metadata={
            "help": "Enable TF32 mode on Ampere and newer GPUs. Provides up to 8x speedup with negligible accuracy loss."
        },
    )

    # --- Gradient Checkpointing ---
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Enable gradient checkpointing to trade compute for memory. Reduces memory at the cost of ~20% slower training."
        },
    )
    gradient_checkpointing_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={"help": "Keyword arguments passed to `gradient_checkpointing_enable()`."},
    )

    # --- Compilation ---
    torch_compile: bool = field(
        default=False, metadata={"help": "Compile the model using `torch.compile()` for faster training."}
    )
    torch_compile_backend: str | None = field(
        default=None,
        metadata={
            "help": "Backend for `torch.compile()`. If set, automatically enables `torch_compile`.",
        },
    )
    torch_compile_mode: str | None = field(
        default=None,
        metadata={
            "help": "Compilation mode for `torch.compile()`. If set, automatically enables `torch_compile`.",
        },
    )

    # --- Kernels ---
    use_liger_kernel: bool = field(
        default=False,
        metadata={
            "help": "Enable Liger Kernel optimizations. Increases throughput by ~20% and reduces memory by ~60%."
        },
    )
    liger_kernel_config: dict[str, bool] | None = field(
        default=None,
        metadata={
            "help": "Configuration for Liger Kernel. Passed as kwargs to `_apply_liger_kernel_to_instance()`. If None, uses default configuration."
        },
    )

    # --- Additional Optimizations ---
    use_cache: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use cache for the model For training, this is usually not needed apart from some PEFT methods that uses `past_key_values`."
        },
    )
    neftune_noise_alpha: float | None = field(
        default=None,
        metadata={
            "help": "If not None, activates NEFTune noise embeddings. Can drastically improve performance for instruction fine-tuning. Typical range: [5.0, 15.0]."
        },
    )
    torch_empty_cache_steps: int | None = field(
        default=None,
        metadata={
            "help": "Number of steps to wait before calling `torch.<device>.empty_cache()`. Helps avoid CUDA OOM at a cost of ~10% slower performance. If None, cache will not be emptied."
        },
    )
    auto_find_batch_size: bool = field(
        default=False,
        metadata={
            "help": "Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding CUDA Out-of-Memory errors."
        },
    )

    # --- Logging & Monitoring ---
    logging_strategy: IntervalStrategy | str = field(
        default="steps",
        metadata={"help": "The logging strategy to adopt during training. Options: 'no', 'epoch', 'steps'."},
    )
    logging_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    logging_first_step: bool = field(
        default=False, metadata={"help": "Whether to log the first `global_step` or not."}
    )
    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
        },
    )
    logging_nan_inf_filter: bool = field(
        default=True,
        metadata={
            "help": "Filter out NaN and Inf losses when logging. Does not affect gradient computation, only logging."
        },
    )
    include_num_input_tokens_seen: str | bool = field(
        default="no",
        metadata={
            "help": (
                "Whether to track the number of input tokens seen. "
                "Must be one of [`all`, `non_padding`, `no`] or a boolean value which map to `all` or `no`"
            )
        },
    )

    # --- Log Levels ---
    log_level: str = field(
        default="passive",
        metadata={
            "help": "Logging level for the main process. Options: 'debug', 'info', 'warning', 'error', 'critical', 'passive'.",
            "choices": trainer_log_levels.keys(),
        },
    )
    log_level_replica: str = field(
        default="warning",
        metadata={
            "help": "Logging level for replica processes in distributed training. Same options as `log_level`.",
            "choices": trainer_log_levels.keys(),
        },
    )
    disable_tqdm: bool | None = field(
        default=None,
        metadata={"help": "Disable tqdm progress bars. Defaults to True if log_level is warning or lower."},
    )

    # --- Experiment Tracking ---
    report_to: None | str | list[str] = field(
        default="none",
        metadata={
            "help": "The list of integrations to report the results and logs to. Use 'all' for all installed integrations, 'none' for no integrations."
        },
    )
    run_name: str | None = field(
        default=None,
        metadata={
            "help": (
                "An optional descriptor for the run. Notably used for trackio, wandb, mlflow comet and swanlab "
                "logging."
            )
        },
    )
    project: str = field(
        default="huggingface",
        metadata={"help": "The name of the project to use for logging. Currently, only used by Trackio."},
    )
    trackio_space_id: str | None = field(
        default="trackio",
        metadata={
            "help": "The Hugging Face Space ID to deploy to when using Trackio. Should be a complete Space name like "
            "'username/reponame' or 'orgname/reponame', or just 'reponame' in which case the Space will be created in "
            "the currently-logged-in Hugging Face user's namespace. If `None`, will log to a local directory. Note "
            "that this Space will be public unless you set `hub_private_repo=True` or your organization's "
            "default is to create private Spaces."
        },
    )

    # --- Evaluation ---
    eval_strategy: IntervalStrategy | str = field(
        default="no",
        metadata={"help": "When to run evaluation. Options: 'no', 'steps', 'epoch'."},
    )
    eval_steps: float | None = field(
        default=None,
        metadata={
            "help": (
                "Number of update steps between evaluations if `eval_strategy='steps'`. Defaults to `logging_steps` if not set."
                " Should be an integer or a float in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    eval_delay: float = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " eval_strategy."
            )
        },
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "The batch size per device (GPU/TPU core/CPU) for evaluation."}
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and generating predictions, only returns the loss."},
    )
    eval_on_start: bool = field(
        default=False,
        metadata={
            "help": "Whether to run through the entire `evaluation` step at the very beginning of training as a sanity check."
        },
    )
    eval_do_concat_batches: bool = field(
        default=True,
        metadata={
            "help": "Whether to recursively concat inputs/losses/labels/predictions across batches. If `False`, will instead store them as lists, with each batch kept separate."
        },
    )
    eval_use_gather_object: bool = field(
        default=False,
        metadata={
            "help": "Whether to run recursively gather object in a nested list/tuple/dictionary of objects from all devices."
        },
    )
    eval_accumulation_steps: int | None = field(
        default=None,
        metadata={
            "help": "Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If unset, predictions are accumulated on the accelerator before being moved to the CPU."
        },
    )

    # --- Metrics ---
    include_for_metrics: list[str] = field(
        default_factory=list,
        metadata={"help": "Include additional data in the `compute_metrics` function. Options: 'inputs', 'loss'."},
    )
    batch_eval_metrics: bool = field(
        default=False,
        metadata={"help": "Break eval metrics calculation into batches to save memory."},
    )

    # --- Checkpointing & Saving ---
    save_only_model: bool = field(
        default=False,
        metadata={
            "help": "Save only model weights, not optimizer/scheduler/RNG state. Prevents resuming training from checkpoint."
        },
    )
    save_strategy: SaveStrategy | str = field(
        default="steps",
        metadata={
            "help": "The checkpoint save strategy to adopt during training. Options: 'no', 'epoch', 'steps', 'best'."
        },
    )
    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one"
            )
        },
    )
    save_total_limit: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of checkpoints to keep. Deletes older checkpoints in `output_dir`. The best checkpoint is always retained when `load_best_model_at_end=True`."
        },
    )
    enable_jit_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Enable JIT checkpointing on SIGTERM signal for graceful termination on preemptible workloads. Configure your orchestrator's graceful shutdown period accordingly."
        },
    )

    # --- Hub Integration ---
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to push the model to the Hub every time the model is saved."}
    )
    hub_token: str | None = field(
        default=None,
        metadata={
            "help": "The token to use to push the model to the Hub. Defaults to the token from `hf auth login`."
        },
    )
    hub_private_repo: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to make the repo private. If `None` (default), the repo will be public unless the "
            "organization's default is private. This value is ignored if the repo already exists. If reporting to "
            "Trackio with deployment to Hugging Face Spaces enabled, the same logic determines whether the Space is "
            "private."
        },
    )
    hub_model_id: str | None = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_strategy: HubStrategy | str = field(
        default="every_save",
        metadata={
            "help": "Defines what and when to push to Hub. Options: 'end', 'every_save', 'checkpoint', 'all_checkpoints'."
        },
    )
    hub_always_push: bool = field(
        default=False,
        metadata={"help": "Unless `True`, the Trainer will skip pushes if the previous one wasn't finished yet."},
    )
    hub_revision: str | None = field(
        default=None,
        metadata={
            "help": "The revision to use when pushing to the Hub. Can be a branch name, a tag, or a commit hash."
        },
    )

    # --- Best Model Tracking ---
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Load the best checkpoint at the end of training. Requires `eval_strategy` to be set."},
    )
    metric_for_best_model: str | None = field(
        default=None,
        metadata={
            "help": "Metric to use for comparing models when `load_best_model_at_end=True`. Defaults to 'loss'."
        },
    )
    greater_is_better: bool | None = field(
        default=None,
        metadata={"help": "Whether higher metric values are better. Defaults based on `metric_for_best_model`."},
    )

    # --- Resuming Training ---
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": "When resuming training, skip fast-forwarding through the dataset to reach the previous state. If True, training starts from the beginning of the dataset."
        },
    )
    restore_callback_states_from_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Whether to restore the callback states from the checkpoint. If `True`, will override callbacks passed to the `Trainer` if they exist in the checkpoint."
        },
    )

    # --- Reproducibility ---
    full_determinism: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed"
                " training. Important: this will negatively impact the performance, so only use it for debugging."
            )
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    data_seed: int | None = field(
        default=None,
        metadata={"help": "Random seed to be used with data samplers. If not set, uses the same seed as `seed`."},
    )

    # --- Hardware ---
    use_cpu: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use cpu. If set to False, we will use the available torch device/backend."
        },
    )

    # --- Accelerate ---
    accelerator_config: dict | str | None = field(
        default=None,
        metadata={
            "help": "Configuration for the internal Accelerate integration. Can be a path to a JSON config file or a dict."
        },
    )
    parallelism_config: ParallelismConfig | None = field(
        default=None,
        metadata={"help": "Parallelism configuration for the training run. Requires Accelerate `1.10.1`."},
    )

    # --- Dataloader ---
    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": "If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will increase RAM usage."
        },
    )
    dataloader_prefetch_factor: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of batches loaded in advance by each worker. "
                "2 means there will be a total of 2 * num_workers batches prefetched across all workers. "
            )
        },
    )
    remove_unused_columns: bool = field(
        default=True,
        metadata={"help": "Whether or not to automatically remove the columns unused by the model forward method."},
    )
    label_names: list[str] | None = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    train_sampling_strategy: str = field(
        default="random",
        metadata={
            "help": "Sampler for training: 'random' (default), 'sequential', or 'group_by_length'.",
            "choices": ["random", "sequential", "group_by_length"],
        },
    )
    length_column_name: str = field(
        default="length",
        metadata={
            "help": "Column name for precomputed lengths. Ignored unless `train_sampling_strategy` is 'group_by_length'."
        },
    )

    # --- DDP ---
    ddp_find_unused_parameters: bool | None = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_bucket_cap_mb: int | None = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `bucket_cap_mb` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_broadcast_buffers: bool | None = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `broadcast_buffers` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_backend: str | None = field(
        default=None,
        metadata={
            "help": "The backend to use for distributed training. Must be one of 'nccl', 'mpi', 'xccl', 'gloo', 'hccl'.",
            "choices": ["nccl", "gloo", "mpi", "xccl", "hccl", "cncl", "mccl"],
        },
    )
    ddp_timeout: int = field(
        default=1800,
        metadata={"help": "The timeout for `torch.distributed.init_process_group` calls (in seconds)."},
    )

    # --- FSDP ---
    fsdp: list[FSDPOption] | str | None = field(
        default=None,
        metadata={
            "help": "Enable PyTorch FSDP for distributed training. Options: 'full_shard', 'shard_grad_op', 'hybrid_shard', 'hybrid_shard_zero2', 'offload', 'auto_wrap'.",
        },
    )
    fsdp_config: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": (
                "Config to be used with FSDP (Pytorch Fully Sharded Data Parallel). The value is either a "
                "fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`."
            )
        },
    )

    # --- DeepSpeed ---
    deepspeed: dict | str | None = field(
        default=None,
        metadata={"help": "Enable DeepSpeed integration. Value is a path to a JSON config file or a dict."},
    )

    # --- Debugging ---
    debug: str | list[DebugOption] = field(
        default="",
        metadata={
            "help": "Enable one or more debug features. Options: 'underflow_overflow' (detect overflow in model I/O), 'tpu_metrics_debug' (print TPU metrics)."
        },
    )
    skip_memory_metrics: bool = field(
        default=True,
        metadata={
            "help": "Whether to skip adding memory profiler reports to metrics. Skipped by default because it slows down training."
        },
    )

    # --- External Script Flags ---
    do_train: bool = field(
        default=False,
        metadata={
            "help": "Whether to run training. Not directly used by Trainer; intended for training/evaluation scripts."
        },
    )
    do_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to run evaluation. Not directly used by Trainer; intended for training/evaluation scripts."
        },
    )
    do_predict: bool = field(
        default=False,
        metadata={
            "help": "Whether to run predictions on the test set. Not directly used by Trainer; intended for training/evaluation scripts."
        },
    )
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={
            "help": "Path to a folder with a valid checkpoint for your model. Not directly used by Trainer; intended for training/evaluation scripts."
        },
    )

    # --- Deprecated / Internal ---
    warmup_ratio: float | None = field(
        default=None,
        metadata={
            "help": "This argument is deprecated and will be removed in v5.2. Use `warmup_steps` instead as it also works with float values."
        },
    )
    logging_dir: str | None = field(
        default=None,
        metadata={
            "help": "Deprecated and will be removed in v5.2. Set env var `TENSORBOARD_LOGGING_DIR` instead. TensorBoard log directory."
        },
    )
    local_rank: int = field(
        default=-1,
        metadata={
            "help": "When using torch.distributed.launch (Deprecated), it will pass `local_rank` in the script, so we need this for the parser. To get the local rank, prefer using the property `local_process_index`"
        },
    )

    def __post_init__(self):
        # ── 1. Defaults & Normalization ──
        if self.output_dir is None:
            self.output_dir = "trainer_output"
            logger.info(
                "No output directory specified, defaulting to 'trainer_output'. "
                "To change this behavior, specify --output_dir when creating TrainingArguments."
            )

        # Parse JSON string dict args from CLI (e.g., '{"key": "value"}').
        # Only parses strings starting with '{'; other strings are treated as file paths.
        for valid_field in self._VALID_DICT_FIELDS:
            passed_value = getattr(self, valid_field)
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, valid_field, loaded_dict)

        # Expand ~ in paths so os.makedirs works correctly (#10628)
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        if self.warmup_ratio is not None:
            logger.warning("warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.")
            self.warmup_steps = self.warmup_ratio

        if self.logging_dir is not None:
            logger.warning(
                "`logging_dir` is deprecated and will be removed in v5.2. Please set `TENSORBOARD_LOGGING_DIR` instead."
            )

        if isinstance(self.include_num_input_tokens_seen, bool):
            self.include_num_input_tokens_seen = "all" if self.include_num_input_tokens_seen else "no"

        # ── 2. Enum / Type Conversions ──
        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = SaveStrategy(self.save_strategy)
        self.hub_strategy = HubStrategy(self.hub_strategy)
        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        self.optim = OptimizerNames(self.optim)

        if isinstance(self.debug, str):
            self.debug = [DebugOption(s) for s in self.debug.split()]
        elif self.debug is None:
            self.debug = []

        # ── 3. Auto-derived Values ──
        if self.do_eval is False and self.eval_strategy != IntervalStrategy.NO:
            self.do_eval = True

        # Fall back to logging_steps if eval_steps is unset
        if self.eval_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.eval_strategy} requires either non-zero --eval_steps or"
                    " --logging_steps"
                )

        if (
            self.load_best_model_at_end or self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU
        ) and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = not self.metric_for_best_model.endswith("loss")

        if self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        # ── 4. Validation ──
        self._validate_args()

        # ── 5. Mixed Precision ──
        # Read from env first; DeepSpeed may override this later
        self.mixed_precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
        if self.fp16:
            self.mixed_precision = "fp16"
        elif self.bf16:
            self.mixed_precision = "bf16"

        # ── 6. Torch Compile ──
        if (self.torch_compile_mode is not None or self.torch_compile_backend is not None) and not self.torch_compile:
            self.torch_compile = True
        if self.torch_compile and self.torch_compile_backend is None:
            if not self.use_cpu and is_torch_hpu_available():
                self.torch_compile_backend = "hpu_backend"
            else:
                self.torch_compile_backend = "inductor"

        if self.torch_compile:
            # TODO: remove env var fallback once minimum accelerate >= 1.2.0
            if not is_accelerate_available("1.2.0"):
                os.environ["ACCELERATE_DYNAMO_BACKEND"] = self.torch_compile_backend
                if self.torch_compile_mode is not None:
                    os.environ["ACCELERATE_DYNAMO_MODE"] = self.torch_compile_mode

        # ── 7. Accelerator Config (must come before self.device) ──
        if is_accelerate_available():
            if not isinstance(self.accelerator_config, AcceleratorConfig):
                if self.accelerator_config is None:
                    self.accelerator_config = AcceleratorConfig()
                elif isinstance(self.accelerator_config, dict):
                    self.accelerator_config = AcceleratorConfig(**self.accelerator_config)
                # Reject uninstantiated class (e.g. AcceleratorConfig instead of AcceleratorConfig())
                elif isinstance(self.accelerator_config, type):
                    raise NotImplementedError(
                        "Tried passing in a callable to `accelerator_config`, but this is not supported. "
                        "Please pass in a fully constructed `AcceleratorConfig` object instead."
                    )
                else:
                    self.accelerator_config = AcceleratorConfig.from_json_file(self.accelerator_config)
            if self.accelerator_config.split_batches:
                logger.info(
                    "Using `split_batches=True` in `accelerator_config` will override the `per_device_train_batch_size` "
                    "Batches will be split across all processes equally when using `split_batches=True`."
                )

        # ── 8. Device Init ──
        if is_torch_available():
            self.device

        # ── 9. TF32 ──
        if is_torch_available() and self.torch_compile:
            if is_torch_tf32_available():
                if self.tf32 is None and not self.fp16 or self.bf16:
                    device_str = "MUSA" if is_torch_musa_available() else "CUDA"
                    logger.info(
                        f"Setting TF32 in {device_str} backends to speedup torch compile, you won't see any improvement"
                        " otherwise."
                    )
                    enable_tf32(True)
            else:
                logger.warning(
                    "The speedups for torchdynamo mostly come with GPU Ampere or higher and which is not detected here."
                )
        if is_torch_available() and self.tf32 is not None:
            if self.tf32:
                if is_torch_tf32_available():
                    enable_tf32(True)
                else:
                    raise ValueError("--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7")
            else:
                if is_torch_tf32_available():
                    enable_tf32(False)
                # TF32 not available, nothing to disable

        # ── 10. Hardware Overrides ──
        if self.use_cpu:
            self.dataloader_pin_memory = False

        # ── 11. FSDP ──
        # Store args only (not the plugin itself) to avoid pickle issues
        self.fsdp_plugin_args = self._process_fsdp_args()

        # ── 12. DeepSpeed (must be last) ──
        self.deepspeed_plugin = None
        if self.deepspeed:
            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

            # Leave self.deepspeed unmodified; users may rely on the original value
            self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)

            from accelerate.utils import DeepSpeedPlugin

            self.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.hf_deepspeed_config)
        elif strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")):
            from accelerate.utils import DeepSpeedPlugin

            self.deepspeed_plugin = DeepSpeedPlugin()
            self.deepspeed_plugin.set_mixed_precision(self.mixed_precision)
            self.deepspeed_plugin.set_deepspeed_weakref()

    def _validate_args(self):
        """Validate argument combinations and value constraints."""
        if self.torch_empty_cache_steps is not None:
            if not (isinstance(self.torch_empty_cache_steps, int) and self.torch_empty_cache_steps > 0):
                raise ValueError(
                    f"`torch_empty_cache_steps` must be an integer bigger than 0, got {self.torch_empty_cache_steps}."
                )

        # logging_steps must be non-zero when logging_strategy="steps"
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps > 1:
            if self.logging_steps != int(self.logging_steps):
                raise ValueError(f"--logging_steps must be an integer if bigger than 1: {self.logging_steps}")
            self.logging_steps = int(self.logging_steps)
        if self.eval_strategy == IntervalStrategy.STEPS and self.eval_steps > 1:
            if self.eval_steps != int(self.eval_steps):
                raise ValueError(f"--eval_steps must be an integer if bigger than 1: {self.eval_steps}")
            self.eval_steps = int(self.eval_steps)
        if self.save_strategy == SaveStrategy.STEPS and self.save_steps > 1:
            if self.save_steps != int(self.save_steps):
                raise ValueError(f"--save_steps must be an integer if bigger than 1: {self.save_steps}")
            self.save_steps = int(self.save_steps)

        # load_best_model_at_end requires compatible save and eval strategies
        if self.load_best_model_at_end and self.save_strategy != SaveStrategy.BEST:
            if self.eval_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.eval_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.eval_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                if self.eval_steps < 1 or self.save_steps < 1:
                    if not (self.eval_steps < 1 and self.save_steps < 1):
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            "steps, which cannot get guaranteed when mixing ratio and absolute steps for save_steps "
                            f"{self.save_steps} and eval_steps {self.eval_steps}."
                        )
                    # Use integer arithmetic to avoid floating point precision issues
                    LARGE_MULTIPLIER = 1_000_000
                    if (self.save_steps * LARGE_MULTIPLIER) % (self.eval_steps * LARGE_MULTIPLIER) != 0:
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            f"steps, but found {self.save_steps}, which is not a multiple of {self.eval_steps}."
                        )
                else:
                    raise ValueError(
                        "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                        f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                    )

        if is_torch_available():
            if self.bf16 or self.bf16_full_eval:
                if not self.use_cpu and not is_torch_bf16_gpu_available() and not is_torch_xla_available():
                    error_message = "Your setup doesn't support bf16/gpu. You need to assign use_cpu if you want to train the model on CPU."
                    if is_torch_cuda_available():
                        error_message += " You need Ampere+ GPU with cuda>=11.0."
                    raise ValueError(error_message)

        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")

        if self.fp16_full_eval and self.bf16_full_eval:
            raise ValueError("At most one of fp16 and bf16 can be True for full eval, but not both")

        if self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            if self.eval_strategy == IntervalStrategy.NO:
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires an eval strategy")
            if not is_torch_available():
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires torch>=0.2.0")

        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be an integer or a float")

        if self.dataloader_num_workers == 0 and self.dataloader_prefetch_factor is not None:
            raise ValueError(
                "--dataloader_prefetch_factor can only be set when data is loaded in a different process, i.e."
                " when --dataloader_num_workers > 0."
            )

    def __str__(self):
        self_as_dict = asdict(self)

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training.
        """
        train_batch_size = self.per_device_train_batch_size * max(1, self.n_gpu)
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation.
        """
        eval_batch_size = self.per_device_eval_batch_size * max(1, self.n_gpu)
        return eval_batch_size

    @property
    def ddp_timeout_delta(self) -> timedelta:
        """
        The actual timeout for torch.distributed.init_process_group since it expects a timedelta variable.
        """
        return timedelta(seconds=self.ddp_timeout)

    @cached_property
    def _setup_devices(self) -> "torch.device":
        requires_backends(self, ["torch"])
        logger.info("PyTorch: setting up devices")
        if not is_sagemaker_mp_enabled():
            if not is_accelerate_available():
                raise ImportError(
                    f"Using the `Trainer` with `PyTorch` requires `accelerate>={ACCELERATE_MIN_VERSION}`: "
                    f"Please run `pip install transformers[torch]` or `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
                )
        # Build kwargs for PartialState; actual init happens below
        accelerator_state_kwargs: dict[str, Any] = {"enabled": True, "use_configured_state": False}
        if isinstance(self.accelerator_config, AcceleratorConfig):
            accelerator_state_kwargs["use_configured_state"] = self.accelerator_config.pop(
                "use_configured_state", False
            )
        if accelerator_state_kwargs["use_configured_state"]:
            if PartialState._shared_state == {}:
                raise ValueError(
                    "Passing `'use_configured_state':True` to the AcceleratorConfig requires a pre-configured "
                    "`AcceleratorState` or `PartialState` to be defined before calling `TrainingArguments`. "
                )
            self.distributed_state = PartialState(cpu=self.use_cpu)
            if self.deepspeed and self.distributed_state.distributed_type != DistributedType.DEEPSPEED:
                raise RuntimeError(
                    "Tried to use an already configured `Accelerator` or `PartialState` that was not initialized for DeepSpeed, "
                    "but also passed in a `deepspeed` configuration to the `TrainingArguments`. Please set "
                    "`use_configured_state:False` instead or setup your `Accelerator` or `PartialState` properly."
                )
        else:
            AcceleratorState._reset_state(reset_partial_state=True)
            self.distributed_state = None

        self._n_gpu = 1
        if self.use_cpu or strtobool(os.environ.get("ACCELERATE_USE_CPU", "False")):
            accelerator_state_kwargs["cpu"] = True
            accelerator_state_kwargs["backend"] = self.ddp_backend
            self._n_gpu = 0
        elif is_sagemaker_mp_enabled():
            accelerator_state_kwargs["enabled"] = False
            device = torch.device("cuda", smp.local_rank())
            torch.cuda.set_device(device)
        elif is_sagemaker_dp_enabled():
            accelerator_state_kwargs["_use_sagemaker_dp"] = True
        elif self.deepspeed:
            accelerator_state_kwargs["use_deepspeed"] = True
            accelerator_state_kwargs["timeout"] = timedelta(seconds=self.ddp_timeout)
        else:
            accelerator_state_kwargs["backend"] = self.ddp_backend
            accelerator_state_kwargs["timeout"] = timedelta(seconds=self.ddp_timeout)

        # Initialize PartialState with the accumulated kwargs
        if accelerator_state_kwargs.pop("enabled", False) and not accelerator_state_kwargs.pop(
            "use_configured_state", False
        ):
            # Temporarily set env var so Accelerate detects DeepSpeed
            use_deepspeed = accelerator_state_kwargs.pop("use_deepspeed", False)
            if use_deepspeed:
                os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.distributed_state = PartialState(**accelerator_state_kwargs)
            if use_deepspeed:
                del os.environ["ACCELERATE_USE_DEEPSPEED"]
        if not is_sagemaker_mp_enabled():
            device = self.distributed_state.device
        if dist.is_available() and dist.is_initialized() and self.parallel_mode != ParallelMode.DISTRIBUTED:
            logger.warning(
                "torch.distributed process group is initialized, but parallel_mode != ParallelMode.DISTRIBUTED. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
            )
        if is_torch_xla_available():
            device = self.distributed_state.device
            self._n_gpu = 0
        elif is_sagemaker_dp_enabled() or is_sagemaker_mp_enabled():
            pass  # _n_gpu already set above
        elif self.distributed_state.distributed_type == DistributedType.NO:
            if self.use_cpu:
                device = torch.device("cpu")
            elif is_torch_mps_available():
                device = torch.device("mps")
            elif is_torch_xpu_available():
                device = torch.device("xpu:0")
                torch.xpu.set_device(device)
            elif is_torch_mlu_available():
                device = torch.device("mlu:0")
                torch.mlu.set_device(device)
            elif is_torch_musa_available():
                device = torch.device("musa:0")
                torch.musa.set_device(device)
            elif is_torch_npu_available():
                device = torch.device("npu:0")
                torch.npu.set_device(device)
            elif is_torch_hpu_available():
                device = torch.device("hpu:0")
                torch.hpu.set_device(device)
            else:
                # Default to cuda:0 (respects CUDA_VISIBLE_DEVICES); nn.DataParallel handles n_gpu > 1
                device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else os.environ.get("ACCELERATE_TORCH_DEVICE", "cpu")
                )
                # _n_gpu may not have been set yet if _setup_devices is called early
                self._n_gpu = torch.cuda.device_count()
                if device.type == "cuda":
                    torch.cuda.set_device(device)
        return device

    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        requires_backends(self, ["torch"])
        return self._setup_devices

    @property
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        requires_backends(self, ["torch"])
        # Ensure _setup_devices has been called
        if not hasattr(self, "_n_gpu"):
            _ = self._setup_devices
        return self._n_gpu

    @property
    def parallel_mode(self):
        """
        The current mode used for parallelism if multiple GPUs/TPU cores are available. One of:

        - `ParallelMode.NOT_PARALLEL`: no parallelism (CPU or one GPU).
        - `ParallelMode.NOT_DISTRIBUTED`: several GPUs in one single process (uses `torch.nn.DataParallel`).
        - `ParallelMode.DISTRIBUTED`: several GPUs, each having its own process (uses
          `torch.nn.DistributedDataParallel`).
        - `ParallelMode.TPU`: several TPU cores.
        """
        requires_backends(self, ["torch"])
        if is_torch_xla_available():
            return ParallelMode.TPU
        elif is_sagemaker_mp_enabled():
            return ParallelMode.SAGEMAKER_MODEL_PARALLEL
        elif is_sagemaker_dp_enabled():
            return ParallelMode.SAGEMAKER_DATA_PARALLEL
        elif self.distributed_state is not None and self.distributed_state.distributed_type != DistributedType.NO:
            return ParallelMode.DISTRIBUTED
        elif self.n_gpu > 1:
            return ParallelMode.NOT_DISTRIBUTED
        else:
            return ParallelMode.NOT_PARALLEL

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        requires_backends(self, ["torch"])
        if self.distributed_state is not None:
            return self.distributed_state.num_processes
        elif is_sagemaker_mp_enabled():
            return smp.dp_size() if not smp.state.cfg.prescaled_batch else smp.rdp_size()
        return 1

    @property
    def process_index(self):
        """
        The index of the current process used.
        """
        requires_backends(self, ["torch"])
        if self.distributed_state is not None:
            return self.distributed_state.process_index
        elif is_sagemaker_mp_enabled():
            return smp.dp_rank() if not smp.state.cfg.prescaled_batch else smp.rdp_rank()
        return 0

    @property
    def local_process_index(self):
        """
        The index of the local process used.
        """
        requires_backends(self, ["torch"])

        if self.distributed_state is not None:
            return self.distributed_state.local_process_index
        elif is_sagemaker_mp_enabled():
            return smp.local_rank()
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

        For the main process the log level defaults to the logging level set (`logging.WARNING` if you didn't do
        anything) unless overridden by `log_level` argument.

        For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica`
        argument.

        The choice between the main and replica process settings is made according to the return value of `should_log`.
        """

        # convert to int
        log_level = trainer_log_levels[self.log_level]
        log_level_replica = trainer_log_levels[self.log_level_replica]

        log_level_main_node = logging.get_verbosity() if log_level == -1 else log_level
        log_level_replica_node = logging.get_verbosity() if log_level_replica == -1 else log_level_replica
        return log_level_main_node if self.should_log else log_level_replica_node

    @property
    def place_model_on_device(self) -> bool | None:
        """
        Can be subclassed and overridden for some specific integrations.
        """
        return None

    @property
    def _no_sync_in_gradient_accumulation(self):
        """
        Whether or not to use no_sync for the gradients when doing gradient accumulation.
        """
        return not (
            self.deepspeed or is_sagemaker_dp_enabled() or is_sagemaker_mp_enabled() or is_torch_neuroncore_available()
        )

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
            main_process_desc = "main local process" if local else "main process"
            if self.distributed_state is not None:
                is_main_process = (
                    self.distributed_state.is_local_main_process if local else self.distributed_state.is_main_process
                )
            elif is_sagemaker_mp_enabled():
                is_main_process = smp.rank() == 0

            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")

                    if is_torch_xla_available():
                        xm.rendezvous(desc)
                    else:
                        dist.barrier()
                yield
            finally:
                if is_main_process:
                    # the wait is over
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    if is_torch_xla_available():
                        xm.rendezvous(desc)
                    else:
                        dist.barrier()
        else:
            yield

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            int(self.warmup_steps) if self.warmup_steps >= 1 else math.ceil(num_training_steps * self.warmup_steps)
        )
        return warmup_steps

    def _dict_dtype_to_str(self, d: dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("dtype") is not None and not isinstance(d["dtype"], str):
            d["dtype"] = str(d["dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self._dict_dtype_to_str(value)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # Exclude non-init fields (they aren't user-facing config)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
            # Serialize AcceleratorConfig to dict
            if is_accelerate_available() and isinstance(v, AcceleratorConfig):
                d[k] = v.to_dict()
            # Serialize quantization_config if nested inside model_init_kwargs
            if k == "model_init_kwargs" and isinstance(v, dict) and "quantization_config" in v:
                quantization_config = v.get("quantization_config")
                if quantization_config and not isinstance(quantization_config, dict):
                    d[k]["quantization_config"] = quantization_config.to_dict()
            if k == "parallelism_config" and v is not None:
                d[k] = v.to_json()

        self._dict_dtype_to_str(d)

        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard's hparams
        """
        d = self.to_dict()
        d = {**d, "train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}

        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}

    # Convenience setters for grouped configuration
    def set_training(
        self,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        weight_decay: float = 0,
        num_epochs: float = 3,
        max_steps: int = -1,
        gradient_accumulation_steps: int = 1,
        seed: int = 42,
        gradient_checkpointing: bool = False,
    ):
        """
        A method that regroups all basic arguments linked to the training.

        <Tip>

        Calling this method will automatically set `self.do_train` to `True`.

        </Tip>

        Args:
            learning_rate (`float`, *optional*, defaults to 5e-5):
                The initial learning rate for the optimizer.
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for training.
            weight_decay (`float`, *optional*, defaults to 0):
                The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in the
                optimizer.
            num_train_epochs(`float`, *optional*, defaults to 3.0):
                Total number of training epochs to perform (if not an integer, will perform the decimal part percents
                of the last epoch before stopping training).
            max_steps (`int`, *optional*, defaults to -1):
                If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
                For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
                `max_steps` is reached.
            gradient_accumulation_steps (`int`, *optional*, defaults to 1):
                Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

                <Tip warning={true}>

                When using gradient accumulation, one step is counted as one step with backward pass. Therefore,
                logging, evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training
                examples.

                </Tip>

            seed (`int`, *optional*, defaults to 42):
                Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use
                the [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized
                parameters.
            gradient_checkpointing (`bool`, *optional*, defaults to `False`):
                If True, use gradient checkpointing to save memory at the expense of slower backward pass.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_training(learning_rate=1e-4, batch_size=32)
        >>> args.learning_rate
        1e-4
        ```
        """
        self.do_train = True
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = batch_size
        self.weight_decay = weight_decay
        self.num_train_epochs = num_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        self.gradient_checkpointing = gradient_checkpointing
        return self

    def set_evaluate(
        self,
        strategy: str | IntervalStrategy = "no",
        steps: int = 500,
        batch_size: int = 8,
        accumulation_steps: int | None = None,
        delay: float | None = None,
        loss_only: bool = False,
    ):
        """
        A method that regroups all arguments linked to evaluation.

        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
                The evaluation strategy to adopt during training. Possible values are:

                    - `"no"`: No evaluation is done during training.
                    - `"steps"`: Evaluation is done (and logged) every `steps`.
                    - `"epoch"`: Evaluation is done at the end of each epoch.

                Setting a `strategy` different from `"no"` will set `self.do_eval` to `True`.
            steps (`int`, *optional*, defaults to 500):
                Number of update steps between two evaluations if `strategy="steps"`.
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for evaluation.
            accumulation_steps (`int`, *optional*):
                Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU.
                If left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster
                but requires more memory).
            delay (`float`, *optional*):
                Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
                eval_strategy.
            loss_only (`bool`, *optional*, defaults to `False`):
                Ignores all outputs except the loss.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_evaluate(strategy="steps", steps=100)
        >>> args.eval_steps
        100
        ```
        """
        self.eval_strategy = IntervalStrategy(strategy)
        if self.eval_strategy == IntervalStrategy.STEPS and steps == 0:
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        self.do_eval = self.eval_strategy != IntervalStrategy.NO
        self.eval_steps = steps
        self.per_device_eval_batch_size = batch_size
        self.eval_accumulation_steps = accumulation_steps
        self.eval_delay = delay
        self.prediction_loss_only = loss_only
        return self

    def set_testing(
        self,
        batch_size: int = 8,
        loss_only: bool = False,
    ):
        """
        A method that regroups all basic arguments linked to testing on a held-out dataset.

        <Tip>

        Calling this method will automatically set `self.do_predict` to `True`.

        </Tip>

        Args:
            batch_size (`int` *optional*, defaults to 8):
                The batch size per device (GPU/TPU core/CPU...) used for testing.
            loss_only (`bool`, *optional*, defaults to `False`):
                Ignores all outputs except the loss.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_testing(batch_size=32)
        >>> args.per_device_eval_batch_size
        32
        ```
        """
        self.do_predict = True
        self.per_device_eval_batch_size = batch_size
        self.prediction_loss_only = loss_only
        return self

    def set_save(
        self,
        strategy: str | IntervalStrategy = "steps",
        steps: int = 500,
        total_limit: int | None = None,
        on_each_node: bool = False,
    ):
        """
        A method that regroups all arguments linked to checkpoint saving.

        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
                The checkpoint save strategy to adopt during training. Possible values are:

                    - `"no"`: No save is done during training.
                    - `"epoch"`: Save is done at the end of each epoch.
                    - `"steps"`: Save is done every `save_steps`.

            steps (`int`, *optional*, defaults to 500):
                Number of updates steps before two checkpoint saves if `strategy="steps"`.
            total_limit (`int`, *optional*):
                If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
                `output_dir`.
            on_each_node (`bool`, *optional*, defaults to `False`):
                When doing multi-node distributed training, whether to save models and checkpoints on each node, or
                only on the main one.

                This should not be activated when the different nodes use the same storage as the files will be saved
                with the same names for each node.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_save(strategy="steps", steps=100)
        >>> args.save_steps
        100
        ```
        """
        self.save_strategy = SaveStrategy(strategy)
        if self.save_strategy == SaveStrategy.STEPS and steps == 0:
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        self.save_steps = steps
        self.save_total_limit = total_limit
        self.save_on_each_node = on_each_node
        return self

    def set_logging(
        self,
        strategy: str | IntervalStrategy = "steps",
        steps: int = 500,
        report_to: str | list[str] = "none",
        level: str = "passive",
        first_step: bool = False,
        nan_inf_filter: bool = False,
        on_each_node: bool = False,
        replica_level: str = "passive",
    ):
        """
        A method that regroups all arguments linked to logging.

        Args:
            strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
                The logging strategy to adopt during training. Possible values are:

                    - `"no"`: No logging is done during training.
                    - `"epoch"`: Logging is done at the end of each epoch.
                    - `"steps"`: Logging is done every `logging_steps`.

            steps (`int`, *optional*, defaults to 500):
                Number of update steps between two logs if `strategy="steps"`.
            level (`str`, *optional*, defaults to `"passive"`):
                Logger log level to use on the main process. Possible choices are the log levels as strings: `"debug"`,
                `"info"`, `"warning"`, `"error"` and `"critical"`, plus a `"passive"` level which doesn't set anything
                and lets the application set the level.
            report_to (`str` or `list[str]`, *optional*, defaults to `"none"`):
                The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
                `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`,
                `"swanlab"`, `"tensorboard"`, `"trackio"` and `"wandb"`. Use `"all"` to report to all integrations
                installed, `"none"` for no integrations.
            first_step (`bool`, *optional*, defaults to `False`):
                Whether to log and evaluate the first `global_step` or not.
            nan_inf_filter (`bool`, *optional*, defaults to `True`):
                Whether to filter `nan` and `inf` losses for logging. If set to `True` the loss of every step that is
                `nan` or `inf` is filtered and the average loss of the current logging window is taken instead.

                <Tip>

                `nan_inf_filter` only influences the logging of loss values, it does not change the behavior the
                gradient is computed or applied to the model.

                </Tip>

            on_each_node (`bool`, *optional*, defaults to `True`):
                In multinode distributed training, whether to log using `log_level` once per node, or only on the main
                node.
            replica_level (`str`, *optional*, defaults to `"passive"`):
                Logger log level to use on replicas. Same choices as `log_level`

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_logging(strategy="steps", steps=100)
        >>> args.logging_steps
        100
        ```
        """
        self.logging_strategy = IntervalStrategy(strategy)
        if self.logging_strategy == IntervalStrategy.STEPS and steps == 0:
            raise ValueError("Setting `strategy` as 'steps' requires a positive value for `steps`.")
        self.logging_steps = steps
        self.report_to = report_to
        self.log_level = level
        self.logging_first_step = first_step
        self.logging_nan_inf_filter = nan_inf_filter
        self.log_on_each_node = on_each_node
        self.log_level_replica = replica_level
        return self

    def set_push_to_hub(
        self,
        model_id: str,
        strategy: str | HubStrategy = "every_save",
        token: str | None = None,
        private_repo: bool | None = None,
        always_push: bool = False,
        revision: str | None = None,
    ):
        """
        A method that regroups all arguments linked to synchronizing checkpoints with the Hub.

        <Tip>

        Calling this method will set `self.push_to_hub` to `True`, which means the `output_dir` will begin a git
        directory synced with the repo (determined by `model_id`) and the content will be pushed each time a save is
        triggered (depending on your `self.save_strategy`). Calling [`~Trainer.save_model`] will also trigger a push.

        </Tip>

        Args:
            model_id (`str`):
                The name of the repository to keep in sync with the local *output_dir*. It can be a simple model ID in
                which case the model will be pushed in your namespace. Otherwise it should be the whole repository
                name, for instance `"user_name/model"`, which allows you to push to an organization you are a member of
                with `"organization_name/model"`.
            strategy (`str` or [`~trainer_utils.HubStrategy`], *optional*, defaults to `"every_save"`):
                Defines the scope of what is pushed to the Hub and when. Possible values are:

                - `"end"`: push the model, its configuration, the processing_class e.g. tokenizer (if passed along to the [`Trainer`]) and a
                draft of a model card when the [`~Trainer.save_model`] method is called.
                - `"every_save"`: push the model, its configuration, the processing_class e.g. tokenizer (if passed along to the [`Trainer`])
                  and
                a draft of a model card each time there is a model save. The pushes are asynchronous to not block
                training, and in case the save are very frequent, a new push is only attempted if the previous one is
                finished. A last push is made with the final model at the end of training.
                - `"checkpoint"`: like `"every_save"` but the latest checkpoint is also pushed in a subfolder named
                last-checkpoint, allowing you to resume training easily with
                `trainer.train(resume_from_checkpoint="last-checkpoint")`.
                - `"all_checkpoints"`: like `"checkpoint"` but all checkpoints are pushed like they appear in the
                  output
                folder (so you will get one checkpoint folder per folder in your final repository)

            token (`str`, *optional*):
                The token to use to push the model to the Hub. Will default to the token in the cache folder obtained
                with `hf auth login`.
            private_repo (`bool`, *optional*, defaults to `False`):
                Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            always_push (`bool`, *optional*, defaults to `False`):
                Unless this is `True`, the `Trainer` will skip pushing a checkpoint when the previous push is not
                finished.
            revision (`str`, *optional*):
                The revision to use when pushing to the Hub. Can be a branch name, a tag, or a commit hash.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_push_to_hub("me/awesome-model")
        >>> args.hub_model_id
        'me/awesome-model'
        ```
        """
        self.push_to_hub = True
        self.hub_model_id = model_id
        self.hub_strategy = HubStrategy(strategy)
        self.hub_token = token
        self.hub_private_repo = private_repo
        self.hub_always_push = always_push
        self.hub_revision = revision
        return self

    def set_optimizer(
        self,
        name: str | OptimizerNames = "adamw_torch",
        learning_rate: float = 5e-5,
        weight_decay: float = 0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        args: str | None = None,
    ):
        """
        A method that regroups all arguments linked to the optimizer and its hyperparameters.

        Args:
            name (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_torch"`):
                The optimizer to use: `"adamw_torch"`, `"adamw_torch_fused"`, `"adamw_apex_fused"`,
                `"adamw_anyprecision"` or `"adafactor"`.
            learning_rate (`float`, *optional*, defaults to 5e-5):
                The initial learning rate.
            weight_decay (`float`, *optional*, defaults to 0):
                The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights.
            beta1 (`float`, *optional*, defaults to 0.9):
                The beta1 hyperparameter for the adam optimizer or its variants.
            beta2 (`float`, *optional*, defaults to 0.999):
                The beta2 hyperparameter for the adam optimizer or its variants.
            epsilon (`float`, *optional*, defaults to 1e-8):
                The epsilon hyperparameter for the adam optimizer or its variants.
            args (`str`, *optional*):
                Optional arguments that are supplied to AnyPrecisionAdamW (only useful when
                `optim="adamw_anyprecision"`).

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_optimizer(name="adamw_torch", beta1=0.8)
        >>> args.optim
        'adamw_torch'
        ```
        """
        self.optim = OptimizerNames(name)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_beta1 = beta1
        self.adam_beta2 = beta2
        self.adam_epsilon = epsilon
        self.optim_args = args
        return self

    def set_lr_scheduler(
        self,
        name: str | SchedulerType = "linear",
        num_epochs: float = 3.0,
        max_steps: int = -1,
        warmup_steps: float = 0,
        warmup_ratio: float | None = None,
    ):
        """
        A method that regroups all arguments linked to the learning rate scheduler and its hyperparameters.

        Args:
            name (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
                The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values.
            num_epochs(`float`, *optional*, defaults to 3.0):
                Total number of training epochs to perform (if not an integer, will perform the decimal part percents
                of the last epoch before stopping training).
            max_steps (`int`, *optional*, defaults to -1):
                If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
                For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
                `max_steps` is reached.
            warmup_steps (`float`, *optional*, defaults to 0):
                Number of steps used for a linear warmup from 0 to `learning_rate`.  Should be an integer or a float in range `[0,1)`.
                If smaller than 1, will be interpreted as ratio of steps used for a linear warmup from 0 to `learning_rate`.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_lr_scheduler(name="cosine", warmup_steps=0.05)
        >>> args.warmup_steps
        0.05
        ```
        """
        if warmup_ratio is not None:
            logger.warning("warmup_ratio is deprecated and will be removed in v5.2 . Use `warmup_steps` instead.")
            warmup_steps = warmup_ratio

        self.lr_scheduler_type = SchedulerType(name)
        self.num_train_epochs = num_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        return self

    def set_dataloader(
        self,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        auto_find_batch_size: bool = False,
        ignore_data_skip: bool = False,
        sampler_seed: int | None = None,
    ):
        """
        A method that regroups all arguments linked to the dataloaders creation.

        Args:
            drop_last (`bool`, *optional*, defaults to `False`):
                Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch
                size) or not.
            num_workers (`int`, *optional*, defaults to 0):
                Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in
                the main process.
            pin_memory (`bool`, *optional*, defaults to `True`):
                Whether you want to pin memory in data loaders or not. Will default to `True`.
            persistent_workers (`bool`, *optional*, defaults to `False`):
                If True, the data loader will not shut down the worker processes after a dataset has been consumed
                once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training,
                but will increase RAM usage. Will default to `False`.
            prefetch_factor (`int`, *optional*):
                Number of batches loaded in advance by each worker.
                2 means there will be a total of 2 * num_workers batches prefetched across all workers.
            auto_find_batch_size (`bool`, *optional*, defaults to `False`)
                Whether to find a batch size that will fit into memory automatically through exponential decay,
                avoiding CUDA Out-of-Memory errors. Requires accelerate to be installed (`pip install accelerate`)
            ignore_data_skip (`bool`, *optional*, defaults to `False`):
                When resuming training, whether or not to skip the epochs and batches to get the data loading at the
                same stage as in the previous training. If set to `True`, the training will begin faster (as that
                skipping step can take a long time) but will not yield the same results as the interrupted training
                would have.
            sampler_seed (`int`, *optional*):
                Random seed to be used with data samplers. If not set, random generators for data sampling will use the
                same seed as `self.seed`. This can be used to ensure reproducibility of data sampling, independent of
                the model seed.

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_dataloader(train_batch_size=16, eval_batch_size=64)
        >>> args.per_device_train_batch_size
        16
        ```
        """
        self.per_device_train_batch_size = train_batch_size
        self.per_device_eval_batch_size = eval_batch_size
        self.dataloader_drop_last = drop_last
        self.dataloader_num_workers = num_workers
        self.dataloader_pin_memory = pin_memory
        self.dataloader_persistent_workers = persistent_workers
        self.dataloader_prefetch_factor = prefetch_factor
        self.auto_find_batch_size = auto_find_batch_size
        self.ignore_data_skip = ignore_data_skip
        self.data_seed = sampler_seed
        return self

    def _process_fsdp_args(self):
        if not self.fsdp:
            self.fsdp = []
        elif self.fsdp is True:
            self.fsdp = [FSDPOption.FULL_SHARD]
        elif isinstance(self.fsdp, str):
            self.fsdp = [FSDPOption(s) for s in self.fsdp.split()]

        if self.fsdp == [FSDPOption.OFFLOAD]:
            raise ValueError(
                "`--fsdp offload` can't work on its own. It needs to be added to `--fsdp full_shard` or "
                '`--fsdp shard_grad_op`. For example, `--fsdp "full_shard offload"`.'
            )
        elif FSDPOption.FULL_SHARD in self.fsdp and FSDPOption.SHARD_GRAD_OP in self.fsdp:
            raise ValueError("`--fsdp full_shard` is not compatible with `--fsdp shard_grad_op`.")

        if self.gradient_checkpointing and (
            FSDPOption.FULL_SHARD in self.fsdp or FSDPOption.HYBRID_SHARD in self.fsdp
        ):
            logger.warning(
                "When using FSDP full shard, instead of using `gradient_checkpointing` in TrainingArguments, please"
                " use `activation_checkpointing` in `fsdp_config`. The former introduces a redundant AllGather"
                " operation in backward pass. Reference: https://github.com/huggingface/transformers/issues/30404"
            )

        if self.fsdp_config is None:
            self.fsdp_config = {}

        if isinstance(self.fsdp_config, str):
            if len(self.fsdp) == 0:
                warnings.warn("`--fsdp_config` is useful only when `--fsdp` is specified.")
            with open(self.fsdp_config, encoding="utf-8") as f:
                self.fsdp_config = json.load(f)

        if self.fsdp_config is not None and isinstance(self.fsdp_config, dict):
            for k in list(self.fsdp_config.keys()):
                if k.startswith("fsdp_"):
                    v = self.fsdp_config.pop(k)
                    self.fsdp_config[k[5:]] = v

        self.fsdp_config["min_num_params"] = self.fsdp_config.get("min_num_params", 0)

        # Normalize transformer_layer_cls_to_wrap from string to list
        if isinstance(self.fsdp_config.get("transformer_layer_cls_to_wrap", None), str):
            self.fsdp_config["transformer_layer_cls_to_wrap"] = [self.fsdp_config["transformer_layer_cls_to_wrap"]]

        if len(self.fsdp) == 0 and self.fsdp_config["min_num_params"] > 0:
            warnings.warn("`min_num_params` is useful only when `--fsdp` is specified.")

        if len(self.fsdp) == 0 and self.fsdp_config.get("transformer_layer_cls_to_wrap", None) is not None:
            warnings.warn("`transformer_layer_cls_to_wrap` is useful only when `--fsdp` is specified.")

        if (
            len(self.fsdp) > 0
            and self.fsdp_config["min_num_params"] > 0
            and self.fsdp_config.get("transformer_layer_cls_to_wrap", None) is not None
        ):
            raise ValueError("`min_num_params` and `transformer_layer_cls_to_wrap` are mutually exclusive.")
        self.fsdp_config["xla"] = self.fsdp_config.get("xla", False)
        self.fsdp_config["xla_fsdp_v2"] = self.fsdp_config.get("xla_fsdp_v2", False)
        self.fsdp_config["xla_fsdp_grad_ckpt"] = self.fsdp_config.get("xla_fsdp_grad_ckpt", False)
        if self.fsdp_config["xla"]:
            if len(self.fsdp) > 0:
                # Copy to avoid mutating the original (needed for JSON serialization)
                self.xla_fsdp_config = self.fsdp_config.get("xla_fsdp_settings", {}).copy()
                # Convert string dtype names to torch.dtype
                if "compute_dtype" in self.xla_fsdp_config:
                    self.xla_fsdp_config["compute_dtype"] = getattr(torch, self.xla_fsdp_config["compute_dtype"])
                if "buffer_dtype" in self.xla_fsdp_config:
                    self.xla_fsdp_config["buffer_dtype"] = getattr(torch, self.xla_fsdp_config["buffer_dtype"])
            else:
                warnings.warn("XLA FSDP can be used only when `--fsdp` is specified.")
        else:
            if self.fsdp_config["xla_fsdp_grad_ckpt"]:
                warnings.warn("`--xla_fsdp_grad_ckpt` is useful only when `--xla` is set to true.")

        # Build kwargs for Accelerate's FSDPPlugin
        fsdp_plugin_args = None
        if len(self.fsdp) > 0 and not self.fsdp_config["xla"]:
            from accelerate.utils.constants import (
                FSDP_AUTO_WRAP_POLICY,
                FSDP_SHARDING_STRATEGY,
            )

            fsdp_plugin_args = {}
            for fsdp_option in self.fsdp:
                if fsdp_option.upper() in FSDP_SHARDING_STRATEGY:
                    fsdp_plugin_args["sharding_strategy"] = fsdp_option
                elif fsdp_option == FSDPOption.OFFLOAD:
                    fsdp_plugin_args["cpu_offload"] = True
                elif fsdp_option == FSDPOption.AUTO_WRAP:
                    fsdp_plugin_args["auto_wrap_policy"] = FSDP_AUTO_WRAP_POLICY[0]
                    if self.fsdp_config["min_num_params"] > 0:
                        fsdp_plugin_args["min_num_params"] = self.fsdp_config["min_num_params"]
                        fsdp_plugin_args["auto_wrap_policy"] = FSDP_AUTO_WRAP_POLICY[1]
                    elif self.fsdp_config.get("transformer_layer_cls_to_wrap", None) is not None:
                        fsdp_plugin_args["transformer_cls_names_to_wrap"] = ",".join(
                            self.fsdp_config["transformer_layer_cls_to_wrap"]
                        )
            fsdp_version = int(self.fsdp_config.get("version", 1))
            fsdp_plugin_args["fsdp_version"] = fsdp_version
            prefetch_policy = self.fsdp_config.get("backward_prefetch", "NO_PREFETCH")
            if fsdp_version == 2:
                fsdp_plugin_args["reshard_after_forward"] = str_to_bool(
                    str(self.fsdp_config.get("reshard_after_forward", "false")).lower()
                )
            else:
                fsdp_plugin_args["forward_prefetch"] = str_to_bool(
                    str(self.fsdp_config.get("forward_prefetch", "false")).lower()
                )
                fsdp_plugin_args["backward_prefetch"] = prefetch_policy.upper()
                fsdp_plugin_args["reshard_after_forward"] = str(
                    self.fsdp_config.get("reshard_after_forward", "FULL_SHARD")
                ).lower()
                fsdp_plugin_args["use_orig_params"] = str_to_bool(
                    str(self.fsdp_config.get("use_orig_params", "true")).lower()
                )

            sync_module_states = str(self.fsdp_config.get("sync_module_states", "true")).lower()
            cpu_ram_efficient_loading = str(self.fsdp_config.get("cpu_ram_efficient_loading", "false")).lower()
            if sync_module_states == "false" and cpu_ram_efficient_loading == "true":
                # Without sync, non-main processes would have random weights
                raise ValueError('`sync_module_states` must be `"True"` if `cpu_ram_efficient_loading` is `"True"`')

            # Set env var to suppress Accelerate warning and for transformers to read
            fsdp_plugin_args["cpu_ram_efficient_loading"] = str_to_bool(cpu_ram_efficient_loading)
            os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = cpu_ram_efficient_loading

            fsdp_plugin_args["sync_module_states"] = str_to_bool(sync_module_states)

        return fsdp_plugin_args


class ParallelMode(Enum):
    NOT_PARALLEL = "not_parallel"
    NOT_DISTRIBUTED = "not_distributed"
    DISTRIBUTED = "distributed"
    SAGEMAKER_MODEL_PARALLEL = "sagemaker_model_parallel"
    SAGEMAKER_DATA_PARALLEL = "sagemaker_data_parallel"
    TPU = "tpu"


def str_to_bool(value, to_bool: bool = True) -> int | bool:
    """
    Converts a string representation of truth to `True` (1) or `False` (0).

    True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1 if not to_bool else True
    elif value in ("n", "no", "f", "false", "off", "0"):
        return 0 if not to_bool else False
    else:
        raise ValueError(f"invalid truth value {value}")
