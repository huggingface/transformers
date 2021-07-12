# coding=utf-8
# Copyright 2020 Hugging Face
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

import collections
import re
import time
from typing import Optional

import IPython.display as disp

from ..trainer_callback import TrainerCallback
from ..trainer_utils import IntervalStrategy


def format_time(t):
    "Format `t` (in seconds) to (h):mm:ss"
    t = int(t)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h != 0 else f"{m:02d}:{s:02d}"


def html_progress_bar(value, total, prefix, label, width=300):
    # docstyle-ignore
    return f"""
    <div>
      {prefix}
      <progress value='{value}' max='{total}' style='width:{width}px; height:20px; vertical-align: middle;'></progress>
      {label}
    </div>
    """


def text_to_html_table(items):
    "Put the texts in `items` in an HTML table."
    html_code = """<table border="1" class="dataframe">\n"""
    html_code += """  <thead>\n    <tr style="text-align: left;">\n"""
    for i in items[0]:
        html_code += f"      <th>{i}</th>\n"
    html_code += "    </tr>\n  </thead>\n  <tbody>\n"
    for line in items[1:]:
        html_code += "    <tr>\n"
        for elt in line:
            elt = f"{elt:.6f}" if isinstance(elt, float) else str(elt)
            html_code += f"      <td>{elt}</td>\n"
        html_code += "    </tr>\n"
    html_code += "  </tbody>\n</table><p>"
    return html_code


class NotebookProgressBar:
    """
    A progress par for display in a notebook.

    Class attributes (overridden by derived classes)

        - **warmup** (:obj:`int`) -- The number of iterations to do at the beginning while ignoring
          :obj:`update_every`.
        - **update_every** (:obj:`float`) -- Since calling the time takes some time, we only do it every presumed
          :obj:`update_every` seconds. The progress bar uses the average time passed up until now to guess the next
          value for which it will call the update.

    Args:
        total (:obj:`int`):
            The total number of iterations to reach.
        prefix (:obj:`str`, `optional`):
            A prefix to add before the progress bar.
        leave (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to leave the progress bar once it's completed. You can always call the
            :meth:`~transformers.utils.notebook.NotebookProgressBar.close` method to make the bar disappear.
        parent (:class:`~transformers.notebook.NotebookTrainingTracker`, `optional`):
            A parent object (like :class:`~transformers.utils.notebook.NotebookTrainingTracker`) that spawns progress
            bars and handle their display. If set, the object passed must have a :obj:`display()` method.
        width (:obj:`int`, `optional`, defaults to 300):
            The width (in pixels) that the bar will take.

    Example::

        import time

        pbar = NotebookProgressBar(100)
        for val in range(100):
            pbar.update(val)
            time.sleep(0.07)
        pbar.update(100)
    """

    warmup = 5
    update_every = 0.2

    def __init__(
        self,
        total: int,
        prefix: Optional[str] = None,
        leave: bool = True,
        parent: Optional["NotebookTrainingTracker"] = None,
        width: int = 300,
    ):
        self.total = total
        self.prefix = "" if prefix is None else prefix
        self.leave = leave
        self.parent = parent
        self.width = width
        self.last_value = None
        self.comment = None
        self.output = None

    def update(self, value: int, force_update: bool = False, comment: str = None):
        """
        The main method to update the progress bar to :obj:`value`.

        Args:

            value (:obj:`int`):
                The value to use. Must be between 0 and :obj:`total`.
            force_update (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force and update of the internal state and display (by default, the bar will wait for
                :obj:`value` to reach the value it predicted corresponds to a time of more than the :obj:`update_every`
                attribute since the last update to avoid adding boilerplate).
            comment (:obj:`str`, `optional`):
                A comment to add on the left of the progress bar.
        """
        self.value = value
        if comment is not None:
            self.comment = comment
        if self.last_value is None:
            self.start_time = self.last_time = time.time()
            self.start_value = self.last_value = value
            self.elapsed_time = self.predicted_remaining = None
            self.first_calls = self.warmup
            self.wait_for = 1
            self.update_bar(value)
        elif value <= self.last_value and not force_update:
            return
        elif force_update or self.first_calls > 0 or value >= min(self.last_value + self.wait_for, self.total):
            if self.first_calls > 0:
                self.first_calls -= 1
            current_time = time.time()
            self.elapsed_time = current_time - self.start_time
            self.average_time_per_item = self.elapsed_time / (value - self.start_value)
            if value >= self.total:
                value = self.total
                self.predicted_remaining = None
                if not self.leave:
                    self.close()
            else:
                self.predicted_remaining = self.average_time_per_item * (self.total - value)
            self.update_bar(value)
            self.last_value = value
            self.last_time = current_time
            self.wait_for = max(int(self.update_every / self.average_time_per_item), 1)

    def update_bar(self, value, comment=None):
        spaced_value = " " * (len(str(self.total)) - len(str(value))) + str(value)
        if self.elapsed_time is None:
            self.label = f"[{spaced_value}/{self.total} : < :"
        elif self.predicted_remaining is None:
            self.label = f"[{spaced_value}/{self.total} {format_time(self.elapsed_time)}"
        else:
            self.label = f"[{spaced_value}/{self.total} {format_time(self.elapsed_time)} < {format_time(self.predicted_remaining)}"
            self.label += f", {1/self.average_time_per_item:.2f} it/s"
        self.label += "]" if self.comment is None or len(self.comment) == 0 else f", {self.comment}]"
        self.display()

    def display(self):
        self.html_code = html_progress_bar(self.value, self.total, self.prefix, self.label, self.width)
        if self.parent is not None:
            # If this is a child bar, the parent will take care of the display.
            self.parent.display()
            return
        if self.output is None:
            self.output = disp.display(disp.HTML(self.html_code), display_id=True)
        else:
            self.output.update(disp.HTML(self.html_code))

    def close(self):
        "Closes the progress bar."
        if self.parent is None and self.output is not None:
            self.output.update(disp.HTML(""))


class NotebookTrainingTracker(NotebookProgressBar):
    """
    An object tracking the updates of an ongoing training with progress bars and a nice table reporting metrics.

    Args:

        num_steps (:obj:`int`): The number of steps during training.
        column_names (:obj:`List[str]`, `optional`):
            The list of column names for the metrics table (will be inferred from the first call to
            :meth:`~transformers.utils.notebook.NotebookTrainingTracker.write_line` if not set).
    """

    def __init__(self, num_steps, column_names=None):
        super().__init__(num_steps)
        self.inner_table = None if column_names is None else [column_names]
        self.child_bar = None

    def display(self):
        self.html_code = html_progress_bar(self.value, self.total, self.prefix, self.label, self.width)
        if self.inner_table is not None:
            self.html_code += text_to_html_table(self.inner_table)
        if self.child_bar is not None:
            self.html_code += self.child_bar.html_code
        if self.output is None:
            self.output = disp.display(disp.HTML(self.html_code), display_id=True)
        else:
            self.output.update(disp.HTML(self.html_code))

    def write_line(self, values):
        """
        Write the values in the inner table.

        Args:
            values (:obj:`Dict[str, float]`): The values to display.
        """
        if self.inner_table is None:
            self.inner_table = [list(values.keys()), list(values.values())]
        else:
            columns = self.inner_table[0]
            if len(self.inner_table) == 1:
                # We give a chance to update the column names at the first iteration
                for key in values.keys():
                    if key not in columns:
                        columns.append(key)
                self.inner_table[0] = columns
            self.inner_table.append([values[c] for c in columns])

    def add_child(self, total, prefix=None, width=300):
        """
        Add a child progress bar displayed under the table of metrics. The child progress bar is returned (so it can be
        easily updated).

        Args:
            total (:obj:`int`): The number of iterations for the child progress bar.
            prefix (:obj:`str`, `optional`): A prefix to write on the left of the progress bar.
            width (:obj:`int`, `optional`, defaults to 300): The width (in pixels) of the progress bar.
        """
        self.child_bar = NotebookProgressBar(total, prefix=prefix, parent=self, width=width)
        return self.child_bar

    def remove_child(self):
        """
        Closes the child progress bar.
        """
        self.child_bar = None
        self.display()


class NotebookProgressCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that displays the progress of training or evaluation, optimized for
    Jupyter Notebooks or Google colab.
    """

    def __init__(self):
        self.training_tracker = None
        self.prediction_bar = None
        self._force_next_update = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.first_column = "Epoch" if args.evaluation_strategy == IntervalStrategy.EPOCH else "Step"
        self.training_loss = 0
        self.last_log = 0
        column_names = [self.first_column] + ["Training Loss"]
        if args.evaluation_strategy != IntervalStrategy.NO:
            column_names.append("Validation Loss")
        self.training_tracker = NotebookTrainingTracker(state.max_steps, column_names)

    def on_step_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if int(state.epoch) == state.epoch else f"{state.epoch:.2f}"
        self.training_tracker.update(
            state.global_step + 1,
            comment=f"Epoch {epoch}/{state.num_train_epochs}",
            force_update=self._force_next_update,
        )
        self._force_next_update = False

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if not isinstance(eval_dataloader.dataset, collections.abc.Sized):
            return
        if self.prediction_bar is None:
            if self.training_tracker is not None:
                self.prediction_bar = self.training_tracker.add_child(len(eval_dataloader))
            else:
                self.prediction_bar = NotebookProgressBar(len(eval_dataloader))
            self.prediction_bar.update(1)
        else:
            self.prediction_bar.update(self.prediction_bar.value + 1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only for when there is no evaluation
        if args.evaluation_strategy == IntervalStrategy.NO and "loss" in logs:
            values = {"Training Loss": logs["loss"]}
            # First column is necessarily Step sine we're not in epoch eval strategy
            values["Step"] = state.global_step
            self.training_tracker.write_line(values)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.training_tracker is not None:
            values = {"Training Loss": "No log", "Validation Loss": "No log"}
            for log in reversed(state.log_history):
                if "loss" in log:
                    values["Training Loss"] = log["loss"]
                    break

            if self.first_column == "Epoch":
                values["Epoch"] = int(state.epoch)
            else:
                values["Step"] = state.global_step
            metric_key_prefix = "eval"
            for k in metrics:
                if k.endswith("_loss"):
                    metric_key_prefix = re.sub(r"\_loss$", "", k)
            _ = metrics.pop("total_flos", None)
            _ = metrics.pop("epoch", None)
            _ = metrics.pop(f"{metric_key_prefix}_runtime", None)
            _ = metrics.pop(f"{metric_key_prefix}_samples_per_second", None)
            for k, v in metrics.items():
                if k == f"{metric_key_prefix}_loss":
                    values["Validation Loss"] = v
                else:
                    splits = k.split("_")
                    name = " ".join([part.capitalize() for part in splits[1:]])
                    values[name] = v
            self.training_tracker.write_line(values)
            self.training_tracker.remove_child()
            self.prediction_bar = None
            # Evaluation takes a long time so we should force the next update.
            self._force_next_update = True

    def on_train_end(self, args, state, control, **kwargs):
        self.training_tracker.update(
            state.global_step, comment=f"Epoch {int(state.epoch)}/{state.num_train_epochs}", force_update=True
        )
        self.training_tracker = None
