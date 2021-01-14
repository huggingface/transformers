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
import math
import numbers
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .trainer_utils import SchedulerType
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
from .trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, EvaluationStrategy  # noqa: E402


# Integration functions:
def is_wandb_available():
    if os.getenv("WANDB_DISABLED"):
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
        model_path = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    model_path = os.path.join(checkpoint_dir, subdir)
        trainer.objective = None
        trainer.train(model_path=model_path, trial=trial)
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

    def _objective(trial, checkpoint_dir=None):
        model_path = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    model_path = os.path.join(checkpoint_dir, subdir)
        trainer.objective = None
        trainer.train(model_path=model_path, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
            trainer._tune_save_checkpoint()
            ray.tune.report(objective=trainer.objective, **metrics, done=True)

    # The model and TensorBoard writer do not pickle so we have to remove them (if they exists)
    # while doing the ray hp search.

    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    trainer.model = None
    # Setup default `resources_per_trial` and `reporter`.
    if "resources_per_trial" not in kwargs and trainer.args.n_gpu > 0:
        # `args.n_gpu` is considered the total number of GPUs that will be split
        # among the `n_jobs`
        n_jobs = int(kwargs.pop("n_jobs", 1))
        num_gpus_per_trial = trainer.args.n_gpu
        if num_gpus_per_trial / n_jobs >= 1:
            num_gpus_per_trial = int(math.ceil(num_gpus_per_trial / n_jobs))
        kwargs["resources_per_trial"] = {"gpu": num_gpus_per_trial}

    if "progress_reporter" not in kwargs:
        from ray.tune import CLIReporter

        kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])
    if "keep_checkpoints_num" in kwargs and kwargs["keep_checkpoints_num"] > 0:
        # `keep_checkpoints_num=0` would disabled checkpointing
        trainer.use_tune_checkpoints = True
        if kwargs["keep_checkpoints_num"] > 1:
            logger.warning(
                "Currently keeping {} checkpoints for each trial. Checkpoints are usually huge, "
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
        ) and (not trainer.args.do_eval or trainer.args.evaluation_strategy == EvaluationStrategy.NO):
            raise RuntimeError(
                "You are using {cls} as a scheduler but you haven't enabled evaluation during training. "
                "This means your trials will not report intermediate results to Ray Tune, and "
                "can thus not be stopped early or used to exploit other trials parameters. "
                "If this is what you want, do not use {cls}. If you would like to use {cls}, "
                "make sure you pass `do_eval=True` and `evaluation_strategy='steps'` in the "
                "Trainer `args`.".format(cls=type(kwargs["scheduler"]).__name__)
            )

    analysis = ray.tune.run(_objective, config=trainer.hp_space(None), num_samples=n_trials, **kwargs)
    best_trial = analysis.get_best_trial(metric="objective", mode=direction[:3])
    best_run = BestRun(best_trial.trial_id, best_trial.last_result["objective"], best_trial.config)
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)
    return best_run


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


def init_deepspeed(trainer, num_training_steps):
    """
    Init DeepSpeed, after converting any relevant Trainer's args into DeepSpeed configuration

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu

    Returns: model, optimizer, lr_scheduler
    """
    import deepspeed

    args = trainer.args
    ds_config_file = args.deepspeed
    model = trainer.model

    with io.open(ds_config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # The following code translates relevant trainer's cl args into the DS config

    # First to ensure that there is no mismatch between cl args values and presets in the config
    # file, ask to not set in ds config file:
    # - "train_batch_size",
    # - "train_micro_batch_size_per_gpu",
    # - "gradient_accumulation_steps"
    bs_keys = ["train_batch_size", "train_micro_batch_size_per_gpu"]
    if len([x for x in bs_keys if x in config.keys()]):
        raise ValueError(
            f"Do not include {bs_keys} entries in the ds config file, as they will be set via --per_device_train_batch_size or its default"
        )
    if "gradient_accumulation_steps" in config.keys():
        raise ValueError(
            "Do not include gradient_accumulation_steps entries in the ds config file, as they will be set via --gradient_accumulation_steps or its default"
        )

    # DeepSpeed does:
    #   train_batch_size = n_gpus * train_micro_batch_size_per_gpu * gradient_accumulation_steps
    # therefore we just need to set:
    config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    config["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    if "gradient_clipping" in config:
        logger.info(
            f"Keeping the `gradient_clipping` config from {ds_config_file} intact, ignoring any gradient clipping-specific cl args"
        )
    else:  # override only if the ds config doesn't already have this section
        config["gradient_clipping"] = args.max_grad_norm

    if "optimizer" in config:
        logger.info(
            f"Keeping the `optimizer` config from {ds_config_file} intact, ignoring any optimizer-specific cl args"
        )
    else:  # override only if the ds config doesn't already have this section
        # ds supports Adam, OneBitAdam, and Lamb optimizers and can import other optimizers from torch.
        # But trainer uses AdamW by default.
        # To use other optimizers so using a different scheduler requires voiding warranty with: `zero_allow_untested_optimizer`

        optimizer_configs = {
            "AdamW": {
                "lr": args.learning_rate,
                "betas": [args.adam_beta1, args.adam_beta2],
                "eps": args.adam_epsilon,
                "weight_decay": args.weight_decay,
            }
        }
        optimizer = "AdamW"

        config["zero_allow_untested_optimizer"] = True
        config["optimizer"] = {
            "type": optimizer,
            "params": optimizer_configs[optimizer],
        }

    # DS schedulers (deepspeed/runtime/lr_schedules.py):
    #
    # DS name      | --lr_scheduler_type  | HF func                           | Notes
    # -------------| ---------------------|-----------------------------------|--------------------
    # LRRangeTest  | na                   | na                                | LRRT
    # OneCycle     | na                   | na                                | 1CLR
    # WarmupLR     | constant_with_warmup | get_constant_schedule_with_warmup | w/ warmup_min_lr=0
    # WarmupDecayLR| linear               | get_linear_schedule_with_warmup   |
    if "scheduler" in config:
        logger.info(
            f"Keeping the `scheduler` config from {ds_config_file} intact, ignoring any scheduler-specific cl args"
        )
    else:  # override only if the ds config doesn't already have this section
        if args.lr_scheduler_type == SchedulerType.LINEAR:
            scheduler = "WarmupDecayLR"
            params = {
                "last_batch_iteration": -1,
                "total_num_steps": num_training_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps,
            }
        elif args.lr_scheduler_type == SchedulerType.CONSTANT_WITH_WARMUP:
            scheduler = "WarmupLR"
            params = {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps,
            }
        else:
            raise ValueError(f"{args.lr_scheduler_type} scheduler type is not supported by DeepSpeed")

        config["scheduler"] = {
            "type": scheduler,
            "params": params,
        }

    # fp16
    if trainer.fp16_backend is not None:
        # Deepspeed has 2 possible fp16 config entries:
        # - `fp16`: for the native amp - it has a bunch of optional params but we won't set any here unless the user did the work
        # - `amp`: which delegates amp work to apex (which needs to be available), but it cannot be used with any ZeRO features, so probably best to be avoided.
        if trainer.fp16_backend == "apex":
            if "amp" in config:
                logger.info(
                    f"Keeping the `amp` config from {ds_config_file} intact, ignoring any amp-specific cl args"
                )
            else:
                config["amp"] = {
                    "enabled": True,
                    "opt_level": args.fp16_opt_level,
                }
        elif trainer.fp16_backend == "amp":
            if "fp16" in config:
                logger.info(
                    f"Keeping the `fp16` config from {ds_config_file} intact, ignoring any fp16-specific cl args"
                )
            else:
                config["fp16"] = {
                    "enabled": True,
                }

    # for clarity extract the specific cl args that are being passed to deepspeed
    ds_args = dict(local_rank=args.local_rank)

    # init that takes part of the config via `args`, and the bulk of it via `config_params`
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=SimpleNamespace(**ds_args),  # expects an obj
        model=model,
        model_parameters=model_parameters,
        config_params=config,
    )

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
        self.tb_writer = tb_writer
        self._SummaryWriter = SummaryWriter

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
        if state.is_world_process_zero:
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
                        '"%s" of type %s for key "%s" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute.",
                        v,
                        type(v),
                        k,
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

            wandb.ensure_configured()
            if wandb.api.api_key is None:
                has_wandb = False
                logger.warning(
                    "W&B installed but not logged in. Run `wandb login` or set the WANDB_API_KEY env variable."
                )
                self._wandb = None
            else:
                self._wandb = wandb
        self._initialized = False

    def setup(self, args, state, model, reinit, **kwargs):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information `here
        <https://docs.wandb.com/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_LOG_MODEL (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to log model as artifact at the end of training.
            WANDB_WATCH (:obj:`str`, `optional` defaults to :obj:`"gradients"`):
                Can be :obj:`"gradients"`, :obj:`"all"` or :obj:`"false"`. Set to :obj:`"false"` to disable gradient
                logging or :obj:`"all"` to log gradients and parameters.
            WANDB_PROJECT (:obj:`str`, `optional`, defaults to :obj:`"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable wandb entirely.
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

            self._wandb.init(
                project=os.getenv("WANDB_PROJECT", "huggingface"),
                config=combined_dict,
                name=run_name,
                reinit=reinit,
                **init_args,
            )

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
                )

            # log outputs
            self._log_model = os.getenv("WANDB_LOG_MODEL", "FALSE").upper() in ENV_VARS_TRUE_VALUES.union({"TRUE"})

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._wandb is None:
            return
        hp_search = state.is_hyper_param_search
        if not self._initialized or hp_search:
            self.setup(args, state, model, reinit=hp_search, **kwargs)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        # commit last step
        self._wandb.log({})
        if self._log_model and self._initialized and state.is_world_process_zero:
            from .trainer import Trainer

            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                # use run name and ensure it's a valid Artifact name
                artifact_name = re.sub(r"[^a-zA-Z0-9_\.\-]", "", self._wandb.run.name)
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
                artifact = self._wandb.Artifact(name=f"run-{artifact_name}", type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model, reinit=False)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            self._wandb.log(logs, step=state.global_step)


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

    MAX_LOG_SIZE = 100

    def __init__(self):
        assert is_mlflow_available(), "MLflowCallback requires mlflow to be installed. Run `pip install mlflow`."
        import mlflow

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
                whatever is in TrainerArgument's output_dir to the local or remote artifact storage. Using it without a
                remote storage will just copy the files to your artifact location.
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
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(0, len(combined_dict_items), MLflowCallback.MAX_LOG_SIZE):
                self._ml_flow.log_params(dict(combined_dict_items[i : i + MLflowCallback.MAX_LOG_SIZE]))
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
                        "Trainer is attempting to log a value of "
                        '"%s" of type %s for key "%s" as a metric. '
                        "MLflow's log_metric() only accepts float and "
                        "int types so we dropped this attribute.",
                        v,
                        type(v),
                        k,
                    )

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            if self._log_artifacts:
                logger.info("Logging artifacts. This may take time.")
                self._ml_flow.log_artifacts(args.output_dir)
            self._ml_flow.end_run()

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if self._ml_flow.active_run is not None:
            self._ml_flow.end_run(status="KILLED")
