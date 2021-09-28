from typing import Optional, Union, Any
from pathlib import Path

from tensorflow.keras.callbacks import Callback
from transformers.file_utils import get_full_repo_name
from huggingface_hub import Repository


class PushToHubCallback(Callback):
    def __init__(
        self,
        output_dir: Union[str, Path],
        save_strategy="epoch",
        save_steps: Optional[int] = None,
        tokenizer: Optional[Any] = None,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
    ):
        super().__init__()
        self.save_strategy = save_strategy
        if self.save_strategy == "steps" and (not isinstance(save_steps, int) or save_steps <= 0):
            raise ValueError("Please supply a positive integer argument for save_steps when save_strategy == 'steps'!")
        self.save_steps = save_steps
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        if hub_model_id is None:
            repo_name = get_full_repo_name(output_dir.name, token=hub_token)
        else:
            repo_name = hub_model_id
        self.output_dir = output_dir
        self.repo = Repository(str(output_dir), clone_from=repo_name)
        self.tokenizer = tokenizer
        self.last_job = None

    def on_train_batch_end(self, batch, logs=None):
        if self.save_strategy == "steps" and batch + 1 % self.save_steps == 0:
            if self.last_job is not None and not self.last_job.is_done():
                return  # The last upload is still running, don't start another
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            _, self.last_job = self.repo.push_to_hub(
                commit_message=f"Training in progress steps {batch}", blocking=False)

    def on_epoch_end(self, epoch, logs=None):
        if self.save_strategy == "epoch":
            if self.last_job is not None and not self.last_job.is_done():
                return  # The last upload is still running, don't start another
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            _, self.last_job = self.repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False)

    def on_train_end(self, logs=None):
        if self.last_job is not None and not self.last_job.is_done():
            self.last_job.process.join()  # Wait for existing upload to finish to finish
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        self.repo.push_to_hub(commit_message=f"End of training", blocking=True)
