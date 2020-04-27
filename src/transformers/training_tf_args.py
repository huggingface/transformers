from dataclasses import dataclass, field
import logging
from typing import Tuple

from .training_args import TrainingArguments
from .file_utils import cached_property, is_tf_available, tf_required

logger = logging.getLogger(__name__)

if is_tf_available():
    import tensorflow as tf


@dataclass
class TFTrainingArguments(TrainingArguments):
    tpu: bool = field(default=False, metadata={"help": "Run the training over TPUs"})
    optimizer_name: str = field(default="adam", metadata={"help": "Name of a Tensorflow optimizer"})
    mode: str = field(default="sequence-classification", metadata={"help": "Type of task, one of \"sequence-classification\", \"token-classification\" "})
    loss_name: str = field(default="SparseCategoricalCrossentropy", metadata={"help": "Name of a Tensorflow loss"})
    eval_steps: int = field(default=1000, metadata={"help": "Run an eval every X steps."})

    @cached_property
    @tf_required
    def _setup_strategy(self) -> Tuple["tf.distribute.Strategy", int]:
        logger.info("Tensorflow: setting up strategy")
        if self.no_cuda or len(tf.config.list_physical_devices('GPU')) == 0:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            n_gpu = 0
            """
            elif len(tf.config.list_physical_devices('GPU')) > 1:
                # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
                strategy = tf.distribute.MirroredStrategy()
                n_gpu = len(tf.config.list_physical_devices('GPU'))
            """
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            n_gpu = 1
        return strategy, n_gpu

    @property
    @tf_required
    def strategy(self) -> "tf.distribute.Strategy":
        return self._setup_strategy[0]

    @property
    @tf_required
    def n_gpu(self) -> int:
        return self._setup_strategy[1]
