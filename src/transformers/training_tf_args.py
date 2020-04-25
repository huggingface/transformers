from dataclasses import dataclass, field
from typing import Tuple
import logging

import TrainingArguments
from .file_utils import cached_property, is_tensorflow_available, tensorflow_required


if is_tensorflow_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class TFTrainingArguments(TrainingArguments):
    tpu: bool = field(default=False)
    optimizer_name: str = field(default="adam")
    mode: str = field(default="classification")
    loss_name: str = field(default="SparseCategoricalCrossentropy")
    metric_name: str = field(default="SparseCategoricalAccuracy")

    @cached_property
    @tensorflow_required
    def _setup_strategy(self) -> Tuple["tf.distribute.Strategy", int]:
        logger.info("Tensorflow: setting up strategy")
        if self.no_cuda or len(tf.config.list_physical_devices('GPU')) == 0:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            n_gpu = 0
        elif len(tf.config.list_physical_devices('GPU')) > 1:
            strategy = strategy = tf.distribute.MirroredStrategy()
            n_gpu = len(tf.config.list_physical_devices('GPU'))
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            n_gpu = 1
        return strategy, n_gpu

    @property
    @tensorflow_required
    def strategy(self) -> "tf.distribute.Strategy":
        return self._setup_devices[0]

    @property
    @tensorflow_required
    def n_gpu(self):
        return self._setup_devices[1]
