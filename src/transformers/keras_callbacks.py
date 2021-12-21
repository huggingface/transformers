import logging
import os
from pathlib import Path
from time import sleep
from typing import Optional, Union

from tensorflow.keras.callbacks import Callback

from huggingface_hub import Repository

from . import IntervalStrategy, PreTrainedTokenizerBase
from .file_utils import get_full_repo_name
from .modelcard import TrainingSummary


logger = logging.getLogger(__name__)


class PushToHubCallback(Callback):
    def __init__(
        self,
        output_dir: Union[str, Path],
        save_strategy: Union[str, IntervalStrategy] = "epoch",
        save_steps: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
        checkpoint: bool = False,
        **model_card_args
    ):
        """
        output_dir (`str`):
            The output directory where the model predictions and checkpoints will be written and synced with the
            repository on the Hub.
        save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"epoch"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                - `"no"`: No save is done during training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`
        save_steps (`int`, *optional*):
            The number of steps between saves when using the "steps" save_strategy.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            The tokenizer used by the model. If supplied, will be uploaded to the repo alongside the weights.
        hub_model_id (`str`, *optional*):
            The name of the repository to keep in sync with the local *output_dir*. It can be a simple model ID in
            which case the model will be pushed in your namespace. Otherwise it should be the whole repository name,
            for instance `"user_name/model"`, which allows you to push to an organization you are a member of with
            `"organization_name/model"`.

            Will default to to the name of `output_dir`.
        hub_token (`str`, *optional*):
            The token to use to push the model to the Hub. Will default to the token in the cache folder obtained with
            `huggingface-cli login`.
        checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to save full training checkpoints (including epoch and optimizer state) to allow training to be
            resumed. Only usable when *save_strategy* is *epoch*.
        """
        super().__init__()
        if checkpoint and save_strategy != "epoch":
            raise ValueError("Cannot save checkpoints when save_strategy is not 'epoch'!")
        if isinstance(save_strategy, str):
            save_strategy = IntervalStrategy(save_strategy.lower())
        self.save_strategy = save_strategy
        if self.save_strategy == IntervalStrategy.STEPS and (not isinstance(save_steps, int) or save_steps <= 0):
            raise ValueError("Please supply a positive integer argument for save_steps when save_strategy == 'steps'!")
        self.save_steps = save_steps
        output_dir = Path(output_dir)
        if hub_model_id is None:
            hub_model_id = output_dir.absolute().name
        if "/" not in hub_model_id:
            hub_model_id = get_full_repo_name(hub_model_id, token=hub_token)

        self.output_dir = output_dir
        self.hub_model_id = hub_model_id
        self.repo = Repository(
            str(self.output_dir),
            clone_from=self.hub_model_id,
            use_auth_token=hub_token if hub_token else True,
        )
        self.tokenizer = tokenizer
        self.last_job = None
        self.checkpoint = checkpoint
        self.training_history = None
        self.model_card_args = model_card_args

    def on_train_begin(self, logs=None):
        # Although we can access model.history, we have no guarantees that the History callback will fire before this
        # one, so we keep track of it here too
        self.training_history = []

    def on_train_batch_end(self, batch, logs=None):
        if self.save_strategy == IntervalStrategy.STEPS and batch + 1 % self.save_steps == 0:
            if self.last_job is not None and not self.last_job.is_done:
                return  # The last upload is still running, don't start another
            self.model.save_pretrained(self.output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.output_dir)
            _, self.last_job = self.repo.push_to_hub(
                commit_message=f"Training in progress steps {batch}", blocking=False
            )

    def on_epoch_end(self, epoch, logs=None):
        if "epoch" not in logs:
            logs["epoch"] = epoch
        self.training_history.append(logs)
        if self.save_strategy == IntervalStrategy.EPOCH:
            if self.last_job is not None and not self.last_job.is_done:
                return  # The last upload is still running, don't start another
            self.model.save_pretrained(self.output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(self.output_dir)
            if self.checkpoint:
                checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
                self.model._save_checkpoint(checkpoint_dir, epoch)
            train_summary = TrainingSummary.from_keras(
                model=self.model,
                model_name=self.hub_model_id,
                keras_history=self.training_history,
                **self.model_card_args,
            )
            model_card = train_summary.to_model_card()
            with (self.output_dir / "README.md").open("w") as f:
                f.write(model_card)
            _, self.last_job = self.repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )

    def on_train_end(self, logs=None):
        if self.last_job is not None and not self.last_job.is_done:
            logger.info("Waiting for existing upload to finish...")
            while not self.last_job.is_done:
                sleep(1)
        self.model.save_pretrained(self.output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.output_dir)
        train_summary = TrainingSummary.from_keras(
            model=self.model, model_name=self.hub_model_id, keras_history=self.training_history, **self.model_card_args
        )
        model_card = train_summary.to_model_card()
        with (self.output_dir / "README.md").open("w") as f:
            f.write(model_card)
        self.repo.push_to_hub(commit_message="End of training", blocking=True)
