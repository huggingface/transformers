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
Integrations with other Python libraries.
"""
import importlib.util
import io
import json
import numbers
import os
import tempfile
import weakref
from copy import deepcopy
from pathlib import Path

from .dependency_versions_check import dep_version_check
from .utils import logging


logger = logging.get_logger(__name__)


# comet_ml requires to be imported before any ML frameworks
_has_comet = importlib.util.find_spec("comet_ml") is not None and os.getenv("COMET_MODE", "").upper() != "DISABLED"
if _has_comet:
    try:
        import comet_ml  # noqa: F401

        if hasattr(comet_ml, "config") and comet_ml.config.get_config("comet.api_key"):
            _has_comet = True
        else:
            if os.getenv("COMET_MODE", "").upper() != "DISABLED":
                logger.warning("comet_ml is installed but `COMET_API_KEY` is not set.")
            _has_comet = False
    except (ImportError, ValueError):
        _has_comet = False

from .file_utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402
from .trainer_callback import TrainerCallback  # noqa: E402
from .trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402


# Integration functions:
def is_wandb_available():
    # any value of WANDB_DISABLED disables wandb
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        logger.warning(
            "Using the `WAND_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        return False
    return importlib.util.find_spec("wandb") is not None


def is_comet_available():
    return _has_comet


def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


def is_optuna_available():
    return importlib.util.find_spec("optuna") is not None


def is_ray_available():
    return importlib.util.find_spec("ray") is not None


def is_ray_tune_available():
    if not is_ray_available():
        return False
    return importlib.util.find_spec("ray.tune") is not None


def is_azureml_available():
    if importlib.util.find_spec("azureml") is None:
        return False
    if importlib.util.find_spec("azureml.core") is None:
        return False
    return importlib.util.find_spec("azureml.core.run") is not None


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None


def is_fairscale_available():
    return importlib.util.find_spec("fairscale") is not None


def is_deepspeed_available():
    return importlib.util.find_spec("deepspeed") is not None


def hp_params(trial):
    if is_optuna_available():
        import optuna

        if isinstance(trial, optuna.Trial):
            return trial.params
    if is_ray_tune_available():
        if isinstance(trial, dict):
            return trial

    raise RuntimeError(f"Unknown type for trial {trial.__class__}")


def default_hp_search_backend():
    if is_optuna_available():
        return "optuna"
    elif is_ray_tune_available():
        return "ray"


def run_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import optuna

    def _objective(trial, checkpoint_dir=None):
        checkpoint = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    checkpoint = os.path.join(checkpoint_dir, subdir)
        trainer.objective = None
        trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
        return trainer.objective

    timeout = kwargs.pop("timeout", None)
    n_jobs = kwargs.pop("n_jobs", 1)
    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
    best_trial = study.best_trial
    return BestRun(str(best_trial.number), best_trial.value, best_trial.params)


def run_hp_search_ray(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import ray

    def _objective(trial, local_trainer, checkpoint_dir=None):
        checkpoint = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    checkpoint = os.path.join(checkpoint_dir, subdir)
        local_trainer.objective = None
        local_trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(local_trainer, "objective", None) is None:
            metrics = local_trainer.evaluate()
            local_trainer.objective = local_trainer.compute_objective(metrics)
            local_trainer._tune_save_checkpoint()
            ray.tune.report(objective=local_trainer.objective, **metrics, done=True)

    # The model and TensorBoard writer do not pickle so we have to remove them (if they exists)
    # while doing the ray hp search.

    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    trainer.model = None
    # Setup default `resources_per_trial`.
    if "resources_per_trial" not in kwargs:
        # Default to 1 CPU and 1 GPU (if applicable) per trial.
        kwargs["resources_per_trial"] = {"cpu": 1}
        if trainer.args.n_gpu > 0:
            kwargs["resources_per_trial"]["gpu"] = 1
        resource_msg = "1 CPU" + (" and 1 GPU" if trainer.args.n_gpu > 0 else "")
        logger.info(
            "No `resources_per_trial` arg was passed into "
            "`hyperparameter_search`. Setting it to a default value "
            f"of {resource_msg} for each trial."
        )
    # Make sure each trainer only uses GPUs that were allocated per trial.
    gpus_per_trial = kwargs["resources_per_trial"].get("gpu", 0)
    trainer.args._n_gpu = gpus_per_trial

    # Setup default `progress_reporter`.
    if "progress_reporter" not in kwargs:
        from ray.tune import CLIReporter

        kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])
    if "keep_checkpoints_num" in kwargs and kwargs["keep_checkpoints_num"] > 0:
        # `keep_checkpoints_num=0` would disabled checkpointing
        trainer.use_tune_checkpoints = True
        if kwargs["keep_checkpoints_num"] > 1:
            logger.warning(
                f"Currently keeping {kwargs['keep_checkpoint_num']} checkpoints for each trial. "
                "Checkpoints are usually huge, "
                "consider setting `keep_checkpoints_num=1`."
            )
    if "scheduler" in kwargs:
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule, PopulationBasedTraining

        # Check if checkpointing is enabled for PopulationBasedTraining
        if isinstance(kwargs["scheduler"], PopulationBasedTraining):
            if not trainer.use_tune_checkpoints:
                logger.warning(
                    "You are using PopulationBasedTraining but you haven't enabled checkpointing. "
                    "This means your trials will train from scratch everytime they are exploiting "
                    "new configurations. Consider enabling checkpointing by passing "
                    "`keep_checkpoints_num=1` as an additional argument to `Trainer.hyperparameter_search`."
                )

        # Check for `do_eval` and `eval_during_training` for schedulers that require intermediate reporting.
        if isinstance(
            kwargs["scheduler"], (ASHAScheduler, MedianStoppingRule, HyperBandForBOHB, PopulationBasedTraining)
        ) and (not trainer.args.do_eval or trainer.args.evaluation_strategy == IntervalStrategy.NO):
            raise RuntimeError(
                "You are using {cls} as a scheduler but you haven't enabled evaluation during training. "
                "This means your trials will not report intermediate results to Ray Tune, and "
                "can thus not be stopped early or used to exploit other trials parameters. "
                "If this is what you want, do not use {cls}. If you would like to use {cls}, "
                "make sure you pass `do_eval=True` and `evaluation_strategy='steps'` in the "
                "Trainer `args`.".format(cls=type(kwargs["scheduler"]).__name__)
            )

    analysis = ray.tune.run(
        ray.tune.with_parameters(_objective, local_trainer=trainer),
        config=trainer.hp_space(None),
        num_samples=n_trials,
        **kwargs,
    )
    best_trial = analysis.get_best_trial(metric="objective", mode=direction[:3])
    best_run = BestRun(best_trial.trial_id, best_trial.last_result["objective"], best_trial.config)
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)
    return best_run


def get_available_reporting_integrations():
    integrations = []
    if is_azureml_available():
        integrations.append("azure_ml")
    if is_comet_available():
        integrations.append("comet_ml")
    if is_mlflow_available():
        integrations.append("mlflow")
    if is_tensorboard_available():
        integrations.append("tensorboard")
    if is_wandb_available():
        integrations.append("wandb")
    return integrations


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


def _is_true(config, key):
    if config is None:
        return False
    return bool(config.get(key))


def _set_if_auto(config, key, val):
    if config is None:
        return
    if config.get(key) == "auto":
        config[key] = val


class DeepSpeedConfigHF:
    """
    This object contains Deepspeed configuration and can be quickly queried for things like zero stage.

    We store a ``weakref`` of this object in the module's global to be able to access the config from areas where the
    Trainer is not available (e.g. `from_pretrained` and `_get_resized_embeddings`).

    The ``DeepSpeedConfigHF`` object is meant to be created during ``TrainingArguments`` object creation and has the
    same lifespan as the latter.
    """

    def __init__(self, args):
        self.config = None
        self.stage = 0
        self.offload = False

        dep_version_check("deepspeed")

        self.config_process(args)

        # set global weakref object
        deepspeed_config_hf_set(self)

    def is_zero2(self):
        return self.stage == 2

    def is_zero3(self):
        return self.stage == 3

    def is_offload(self):
        return self.offload

    def config_process(self, args):
        """
        1. load json if the ``args.deepspeed`` is a path
        2. replace any ``auto`` values in the config with the correct or recommended value

        This is done as early as possible, before model is created, to allow ``is_deepspeed_zero3_enabled`` query and
        getting to the early deepspeed config object during ``zero.Init()`` which needs whether fp16 is enabled, dtype,
        etc.

        """
        config_file_or_dict = args.deepspeed
        if isinstance(config_file_or_dict, dict):
            # Don't modify user's data should they want to reuse it (e.g. in tests), because once we
            # modified it, it will not be accepted here again, since `auto` values would have been overriden
            config = deepcopy(config_file_or_dict)
        elif isinstance(config_file_or_dict, str):
            with io.open(config_file_or_dict, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise ValueError("expecting either a path to a config file or a pre-populated dict")

        self.config = config

        # DeepSpeed does:
        # train_batch_size = world_size * train_micro_batch_size_per_gpu * gradient_accumulation_steps
        train_batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        _set_if_auto(config, "train_micro_batch_size_per_gpu", args.per_device_train_batch_size)
        _set_if_auto(config, "gradient_accumulation_steps", args.gradient_accumulation_steps)
        _set_if_auto(config, "train_batch_size", train_batch_size)
        _set_if_auto(config, "gradient_clipping", args.max_grad_norm)

        # zero
        config_zero = config.get("zero_optimization", {})
        self.stage = config_zero.get("stage", 0)

        config_optim = config.get("optimizer", {})
        if config_optim != {}:
            config_optim_params = config_optim.get("params")
            _set_if_auto(config_optim_params, "lr", args.learning_rate)
            _set_if_auto(config_optim_params, "betas", [args.adam_beta1, args.adam_beta2])
            _set_if_auto(config_optim_params, "eps", args.adam_epsilon)
            _set_if_auto(config_optim_params, "weight_decay", args.weight_decay)

        config_sched = config.get("scheduler", {})
        if config_sched != {}:
            config_sched_params = config_sched.get("params")
            _set_if_auto(config_sched_params, "warmup_min_lr", 0)
            _set_if_auto(config_sched_params, "warmup_max_lr", args.learning_rate)
            _set_if_auto(config_sched_params, "warmup_num_steps", args.warmup_steps)
            # total_num_steps - will get set in deepspeed_init

        # fp16
        if args.fp16:
            fp16_backend = "apex" if args.fp16_backend == "apex" else "amp"
        else:
            fp16_backend = None

        # amp: similar to the pytorch native amp - it has a bunch of optional params but we won't set
        # any here unless the user did the work
        config_fp16 = config.get("fp16")
        _set_if_auto(config_fp16, "enabled", fp16_backend == "amp")

        # apex: delegates amp work to apex (which needs to be available), but it cannot be used with any
        # ZeRO features, so probably best to be avoided.
        config_amp = config.get("amp")
        _set_if_auto(config_amp, "enabled", fp16_backend == "apex")
        _set_if_auto(config_amp, "opt_level", args.fp16_opt_level)

        config_zero = config.get("zero_optimization", {})
        if self.is_zero2():
            self.offload = _is_true(config_zero, "cpu_offload")
        elif self.is_zero3():
            offload_devices = ["cpu", "nvme"]
            if config_zero.get("offload_optimizer", {}).get("device") in offload_devices:
                self.offload = True
            if config_zero.get("offload_param", {}).get("device") in offload_devices:
                self.offload = True

    def config_finalize(self, args, model, num_training_steps):
        """
        This stage is run after we have the model and know num_training_steps.

        Now we we can complete the configuration process.

        """
        config = self.config

        # zero
        config_zero = config.get("zero_optimization", {})
        if self.is_zero3():
            # automatically assign the optimal config values based on model config
            hidden_size = model.config.hidden_size
            _set_if_auto(config_zero, "reduce_bucket_size", hidden_size * hidden_size)
            _set_if_auto(config_zero, "stage3_prefetch_bucket_size", 0.9 * hidden_size * hidden_size)
            _set_if_auto(config_zero, "stage3_param_persistence_threshold", 10 * hidden_size)

        # scheduler
        config_sched = config.get("scheduler", {})
        config_sched_params = config_sched.get("params", {})
        _set_if_auto(config_sched_params, "total_num_steps", num_training_steps)


# keep the config object global to be able to access it anywhere during TrainingArguments life-cycle
_deepspeed_config_hf_weak_ref = None


def deepspeed_config_hf_set(deepspeed_config_hf_obj):
    # this is a special weakref global object to allow us to get to Deepspeed config from APIs
    # that don't have an easy way to get to the Deepspeed config outside of the Trainer domain.
    global _deepspeed_config_hf_weak_ref
    # will go away automatically when DeepSpeedConfigHF is destroyed (when TrainingArguments is destroyed)
    _deepspeed_config_hf_weak_ref = weakref.ref(deepspeed_config_hf_obj)


def is_deepspeed_zero3_enabled():
    if _deepspeed_config_hf_weak_ref is not None and _deepspeed_config_hf_weak_ref() is not None:
        return _deepspeed_config_hf_weak_ref().is_zero3()
    else:
        return False


def deepspeed_config():
    if _deepspeed_config_hf_weak_ref is not None and _deepspeed_config_hf_weak_ref() is not None:
        return _deepspeed_config_hf_weak_ref().config
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

    model = trainer.model

    deepspeed_config_hf = trainer.args.deepspeed_config_hf
    deepspeed_config_hf.config_finalize(trainer.args, model, num_training_steps)

    # resume config update - some bits like `model` and `num_training_steps` only become available during train
    config = deepspeed_config_hf.config

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
    if "optimizer" not in config:
        if deepspeed_config_hf.is_offload():
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


class TensorBoardCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `TensorBoard
    <https://www.tensorflow.org/tensorboard>`__.

    Args:
        tb_writer (:obj:`SummaryWriter`, `optional`):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writer=None):
        has_tensorboard = is_tensorboard_available()
        assert (
            has_tensorboard
        ), "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or install tensorboardX."
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401

                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter

                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writer = tb_writer

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writer = self._SummaryWriter(log_dir=log_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        self._init_summary_writer(args, log_dir)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(self.tb_writer, "add_hparams"):
                self.tb_writer.add_hparams(args.to_sanitized_dict(), metric_dict={})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()


class WandbCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `Weight and Biases <https://www.wandb.com/>`__.
    """

    def __init__(self):
        has_wandb = is_wandb_available()
        assert has_wandb, "WandbCallback requires wandb to be installed. Run `pip install wandb`."
        if has_wandb:
            import wandb

            self._wandb = wandb
        self._initialized = False
        # log outputs
        self._log_model = os.getenv("WANDB_LOG_MODEL", "FALSE").upper() in ENV_VARS_TRUE_VALUES.union({"TRUE"})

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information `here
        <https://docs.wandb.ai/integrations/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_LOG_MODEL (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to log model as artifact at the end of training. Use along with
                `TrainingArguments.load_best_model_at_end` to upload best model.
            WANDB_WATCH (:obj:`str`, `optional` defaults to :obj:`"gradients"`):
                Can be :obj:`"gradients"`, :obj:`"all"` or :obj:`"false"`. Set to :obj:`"false"` to disable gradient
                logging or :obj:`"all"` to log gradients and parameters.
            WANDB_PROJECT (:obj:`str`, `optional`, defaults to :obj:`"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name

            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    name=run_name,
                    **init_args,
                )
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
                )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._wandb is None:
            return
        hp_search = state.is_hyper_param_search
        if hp_search:
            self._wandb.finish()
        if not self._initialized:
            self.setup(args, state, model, **kwargs)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model and self._initialized and state.is_world_process_zero:
            from .trainer import Trainer

            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                    }
                )
                artifact = self._wandb.Artifact(name=f"model-{self._wandb.run.id}", type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})


class CometCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `Comet ML <https://www.comet.ml/site/>`__.
    """

    def __init__(self):
        assert _has_comet, "CometCallback requires comet-ml to be installed. Run `pip install comet-ml`."
        self._initialized = False

    def setup(self, args, state, model):
        """
        Setup the optional Comet.ml integration.

        Environment:
            COMET_MODE (:obj:`str`, `optional`):
                "OFFLINE", "ONLINE", or "DISABLED"
            COMET_PROJECT_NAME (:obj:`str`, `optional`):
                Comet.ml project name for experiments
            COMET_OFFLINE_DIRECTORY (:obj:`str`, `optional`):
                Folder to use for saving offline experiments when :obj:`COMET_MODE` is "OFFLINE"

        For a number of configurable items in the environment, see `here
        <https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables>`__.
        """
        self._initialized = True
        if state.is_world_process_zero:
            comet_mode = os.getenv("COMET_MODE", "ONLINE").upper()
            args = {"project_name": os.getenv("COMET_PROJECT_NAME", "huggingface")}
            experiment = None
            if comet_mode == "ONLINE":
                experiment = comet_ml.Experiment(**args)
                logger.info("Automatic Comet.ml online logging enabled")
            elif comet_mode == "OFFLINE":
                args["offline_directory"] = os.getenv("COMET_OFFLINE_DIRECTORY", "./")
                experiment = comet_ml.OfflineExperiment(**args)
                logger.info("Automatic Comet.ml offline logging enabled; use `comet upload` when finished")
            if experiment is not None:
                experiment._set_model_graph(model, framework="transformers")
                experiment._log_parameters(args, prefix="args/", framework="transformers")
                if hasattr(model, "config"):
                    experiment._log_parameters(model.config, prefix="config/", framework="transformers")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            experiment = comet_ml.config.get_global_experiment()
            if experiment is not None:
                experiment._log_metrics(logs, step=state.global_step, epoch=state.epoch, framework="transformers")


class AzureMLCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `AzureML
    <https://pypi.org/project/azureml-sdk/>`__.
    """

    def __init__(self, azureml_run=None):
        assert (
            is_azureml_available()
        ), "AzureMLCallback requires azureml to be installed. Run `pip install azureml-sdk`."
        self.azureml_run = azureml_run

    def on_init_end(self, args, state, control, **kwargs):
        from azureml.core.run import Run

        if self.azureml_run is None and state.is_world_process_zero:
            self.azureml_run = Run.get_context()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.azureml_run:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.azureml_run.log(k, v, description=k)


class MLflowCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `MLflow <https://www.mlflow.org/>`__.
    """

    def __init__(self):
        assert is_mlflow_available(), "MLflowCallback requires mlflow to be installed. Run `pip install mlflow`."
        import mlflow

        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH

        self._initialized = False
        self._log_artifacts = False
        self._ml_flow = mlflow

    def setup(self, args, state, model):
        """
        Setup the optional MLflow integration.

        Environment:
            HF_MLFLOW_LOG_ARTIFACTS (:obj:`str`, `optional`):
                Whether to use MLflow .log_artifact() facility to log artifacts.

                This only makes sense if logging to a remote server, e.g. s3 or GCS. If set to `True` or `1`, will copy
                whatever is in :class:`~transformers.TrainingArguments`'s ``output_dir`` to the local or remote
                artifact storage. Using it without a remote storage will just copy the files to your artifact location.
        """
        log_artifacts = os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper()
        if log_artifacts in {"TRUE", "1"}:
            self._log_artifacts = True
        if state.is_world_process_zero:
            self._ml_flow.start_run()
            combined_dict = args.to_dict()
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f"Trainer is attempting to log a value of "
                        f'"{value}" for key "{name}" as a parameter. '
                        f"MLflow's log_param() only accepts values no longer than "
                        f"250 characters so we dropped this attribute."
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                self._ml_flow.log_params(dict(combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH]))
        self._initialized = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self._ml_flow.log_metric(k, v, step=state.global_step)
                else:
                    logger.warning(
                        f"Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a metric. '
                        f"MLflow's log_metric() only accepts float and "
                        f"int types so we dropped this attribute."
                    )

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            if self._log_artifacts:
                logger.info("Logging artifacts. This may take time.")
                self._ml_flow.log_artifacts(args.output_dir)

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if self._ml_flow.active_run is not None:
            self._ml_flow.end_run()


INTEGRATION_TO_CALLBACK = {
    "azure_ml": AzureMLCallback,
    "comet_ml": CometCallback,
    "mlflow": MLflowCallback,
    "tensorboard": TensorBoardCallback,
    "wandb": WandbCallback,
}


def get_reporting_integration_callbacks(report_to):
    for integration in report_to:
        if integration not in INTEGRATION_TO_CALLBACK:
            raise ValueError(
                f"{integration} is not supported, only {', '.join(INTEGRATION_TO_CALLBACK.keys())} are supported."
            )
    return [INTEGRATION_TO_CALLBACK[integration] for integration in report_to]
