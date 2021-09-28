from typing import Optional

from tensorflow.keras.callbacks import Callback


class PushToHubCallback(Callback):
    def __init__(
        self,
        save_strategy="epoch",
        save_steps: Optional[int] = None,
        repo_path_or_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        **push_to_hub_kwargs
    ):
        super().__init__()
        self.save_strategy = save_strategy
        if self.save_strategy == "steps" and (not isinstance(save_steps, int) or save_steps <= 0):
            raise ValueError("Please supply a positive integer argument for save_steps when save_strategy == 'steps'!")
        self.save_steps = save_steps
        if repo_path_or_name is None and repo_url is None:
            raise ValueError("You need to specify a `repo_path_or_name` or a `repo_url`.")
        self.repo_path_or_name = repo_path_or_name
        self.repo_url = repo_url
        self.push_to_hub_kwargs = push_to_hub_kwargs

    def on_train_batch_end(self, batch, logs=None):
        if self.save_strategy == "steps" and batch + 1 % self.save_steps == 0:
            self.model.push_to_hub(
                repo_path_or_name=self.repo_path_or_name, repo_url=self.repo_url, **self.push_to_hub_kwargs
            )

    def on_epoch_end(self, epoch, logs=None):
        if self.save_strategy == "epoch":
            self.model.push_to_hub(
                repo_path_or_name=self.repo_path_or_name, repo_url=self.repo_url, **self.push_to_hub_kwargs
            )
