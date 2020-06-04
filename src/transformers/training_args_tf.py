import logging
from dataclasses import dataclass, field
from typing import Tuple

from .file_utils import cached_property, is_tf_available, tf_required
from .training_args import TrainingArguments


logger = logging.getLogger(__name__)

if is_tf_available():
    import tensorflow as tf


@dataclass
class TFTrainingArguments(TrainingArguments):
    optimizer_name: str = field(
        default="adam",
        metadata={
            "help": 'Name of a Tensorflow optimizer among "adadelta, adagrad, adam, adamax, ftrl, nadam, rmsprop, sgd, adamw"'
        },
    )
    mode: str = field(
        default="text-classification",
        metadata={"help": 'Type of task, one of "text-classification", "token-classification", "question-answering"'},
    )
    loss_name: str = field(
        default="SparseCategoricalCrossentropy",
        metadata={
            "help": "Name of a Tensorflow loss. For the list see: https://www.tensorflow.org/api_docs/python/tf/keras/losses"
        },
    )
    tpu_name: str = field(
        default=None, metadata={"help": "Name of TPU"},
    )
    end_lr: float = field(
        default=0, metadata={"help": "End learning rate for optimizer"},
    )
    eval_steps: int = field(default=1000, metadata={"help": "Run an evaluation every X steps."})
    debug: bool = field(
        default=False, metadata={"help": "Activate the trace to record computation graphs and profiling information"}
    )

    @cached_property
    @tf_required
    def _setup_strategy(self) -> Tuple["tf.distribute.Strategy", int]:
        logger.info("Tensorflow: setting up strategy")
        gpus = tf.config.list_physical_devices("GPU")

        if self.no_cuda:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        else:
            try:
                if self.tpu_name:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(self.tpu_name)
                else:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            except ValueError:
                tpu = None

            if tpu:
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)

                strategy = tf.distribute.experimental.TPUStrategy(tpu)
            elif len(gpus) == 0:
                strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            elif len(gpus) == 1:
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            elif len(gpus) > 1:
                # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
                strategy = tf.distribute.MirroredStrategy()
            else:
                raise ValueError("Cannot find the proper strategy please check your environment properties.")

        return strategy

    @property
    @tf_required
    def strategy(self) -> "tf.distribute.Strategy":
        return self._setup_strategy

    @property
    @tf_required
    def n_gpu(self) -> int:
        return self._setup_strategy.num_replicas_in_sync
