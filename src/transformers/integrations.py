# Integrations with other Python libraries
import math
import os


# Import 3rd-party integrations first:

try:
    # Comet needs to be imported before any ML frameworks
    import comet_ml  # noqa: F401

    # XXX: there should be comet_ml.ensure_configured(), like `wandb`, for now emulate it
    comet_ml.Experiment(project_name="ensure_configured")
    _has_comet = True
except (ImportError, ValueError):
    _has_comet = False

try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except (ImportError, AttributeError):
    _has_wandb = False

try:
    import optuna  # noqa: F401

    _has_optuna = True
except (ImportError):
    _has_optuna = False

try:
    import ray  # noqa: F401

    _has_ray = True
except (ImportError):
    _has_ray = False

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter  # noqa: F401

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False

# No transformer imports above this point

from .file_utils import is_torch_tpu_available
from .trainer_callback import TrainerCallback
from .trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun
from .utils import logging


logger = logging.get_logger(__name__)


# Integration functions:
def is_wandb_available():
    return _has_wandb


def is_comet_available():
    return _has_comet


def is_tensorboard_available():
    return _has_tensorboard


def is_optuna_available():
    return _has_optuna


def is_ray_available():
    return _has_ray


def default_hp_search_backend():
    if is_optuna_available():
        return "optuna"
    elif is_ray_available():
        return "ray"


def run_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
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
        ) and (not trainer.args.do_eval or not trainer.args.evaluate_during_training):
            raise RuntimeError(
                "You are using {cls} as a scheduler but you haven't enabled evaluation during training. "
                "This means your trials will not report intermediate results to Ray Tune, and "
                "can thus not be stopped early or used to exploit other trials parameters. "
                "If this is what you want, do not use {cls}. If you would like to use {cls}, "
                "make sure you pass `do_eval=True` and `evaluate_during_training=True` in the "
                "Trainer `args`.".format(cls=type(kwargs["scheduler"]).__name__)
            )

    analysis = ray.tune.run(_objective, config=trainer.hp_space(None), num_samples=n_trials, **kwargs)
    best_trial = analysis.get_best_trial(metric="objective", mode=direction[:3])
    best_run = BestRun(best_trial.trial_id, best_trial.last_result["objective"], best_trial.config)
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)
    return best_run


class TensorBoardCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `TensorBoard
    <https://www.tensorflow.org/tensorboard>`__.

    Args:
        tb_writer (:obj:`SummaryWriter`, `optional`):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writer=None):
        assert (
            _has_tensorboard
        ), "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or install tensorboardX."
        self.tb_writer = tb_writer

    def on_init_end(self, args, state, control, **kwargs):
        if self.tb_writer is None and state.is_world_process_zero:
            self.tb_writer = SummaryWriter(log_dir=args.logging_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            self.tb_writer.add_hparams(args.to_sanitized_dict(), metric_dict={})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.tb_writer:
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
    A :class:`~transformers.TrainerCallback` that sends the logs to `Weight and Biases
    <https://www.wandb.com/>`__.
    """

    def __init__(self):
        assert _has_wandb, "WandbCallback requires wandb to be installed. Run `pip install wandb`."
        self._initialized = False

    def setup(self, args, state, model):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information
        `here <https://docs.wandb.com/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_WATCH (:obj:`str`, `optional` defaults to :obj:`"gradients"`):
                Can be :obj:`"gradients"`, :obj:`"all"` or :obj:`"false"`. Set to :obj:`"false"` to disable gradient
                logging or :obj:`"all"` to log gradients and parameters.
            WANDB_PROJECT (:obj:`str`, `optional`, defaults to :obj:`"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable wandb entirely.
        """
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}
            if getattr(model, "config", None) is not None:
                combined_dict = {**model.config.to_dict(), **combined_dict}
            wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=combined_dict, name=args.run_name)
            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                wandb.watch(model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps))

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            wandb.log(logs, step=state.global_step)


class CometCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `Comet ML
    <https://www.comet.ml/site/>`__.
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

        For a number of configurable items in the environment,
        see `here <https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables>`__.
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
