import warnings
from dataclasses import dataclass, field
from typing import Tuple

from .file_utils import cached_property, is_tf_available, tf_required
from .training_args import TrainingArguments
from .utils import logging


logger = logging.get_logger(__name__)

if is_tf_available():
    import tensorflow as tf


@dataclass
class TFTrainingArguments(TrainingArguments):
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using :class:`~transformers.HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on the command line.

    Parameters:
        output_dir (:obj:`str`):
            The output directory where the model predictions and checkpoints will be written.
        overwrite_output_dir (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, overwrite the content of the output directory. Use this to continue training if
            :obj:`output_dir` points to a checkpoint directory.
        do_train (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run training or not.
        do_eval (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run evaluation on the dev set or not.
        do_predict (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run predictions on the test set or not.
        evaluate_during_training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run evaluation during training at each logging step or not.
        per_device_train_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for training.
        per_device_eval_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for evaluation.
        gradient_accumulation_steps: (:obj:`int`, `optional`, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

            .. warning::

                When using gradient accumulation, one step is counted as one step with backward pass. Therefore,
                logging, evaluation, save will be conducted every ``gradient_accumulation_steps * xxx_step`` training
                examples.
        learning_rate (:obj:`float`, `optional`, defaults to 5e-5):
            The initial learning rate for Adam.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            The weight decay to apply (if not zero).
        adam_epsilon (:obj:`float`, `optional`, defaults to 1e-8):
            Epsilon for the Adam optimizer.
        max_grad_norm (:obj:`float`, `optional`, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        num_train_epochs(:obj:`float`, `optional`, defaults to 3.0):
            Total number of training epochs to perform.
        max_steps (:obj:`int`, `optional`, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides
            :obj:`num_train_epochs`.
        warmup_steps (:obj:`int`, `optional`, defaults to 0):
            Number of steps used for a linear warmup from 0 to :obj:`learning_rate`.
        logging_dir (:obj:`str`, `optional`):
            Tensorboard log directory. Will default to `runs/**CURRENT_DATETIME_HOSTNAME**`.
        logging_first_step (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Wheter to log and evalulate the first :obj:`global_step` or not.
        logging_steps (:obj:`int`, `optional`, defaults to 500):
            Number of update steps between two logs.
        save_steps (:obj:`int`, `optional`, defaults to 500):
            Number of updates steps before two checkpoint saves.
        save_total_limit (:obj:`int`, `optional`):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            :obj:`output_dir`.
        no_cuda (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to not use CUDA even when it is available or not.
        seed (:obj:`int`, `optional`, defaults to 42):
            Random seed for initialization.
        fp16 (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.
        fp16_opt_level (:obj:`str`, `optional`, defaults to 'O1'):
            For :obj:`fp16` training, apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details
            on the `apex documentation <https://nvidia.github.io/apex/amp.html>`__.
        local_rank (:obj:`int`, `optional`, defaults to -1):
            During distributed training, the rank of the process.
        tpu_num_cores (:obj:`int`, `optional`):
            When training on TPU, the mumber of TPU cores (automatically passed by launcher script).
        debug (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Wheter to activate the trace to record computation graphs and profiling information or not.
        dataloader_drop_last (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        eval_steps (:obj:`int`, `optional`, defaults to 1000):
            Number of update steps before two evaluations.
        past_index (:obj:`int`, `optional`, defaults to -1):
            Some models like :doc:`TransformerXL <../model_doc/transformerxl>` or :doc`XLNet <../model_doc/xlnet>` can
            make use of the past hidden states for their predictions. If this argument is set to a positive int, the
            ``Trainer`` will use the corresponding output (usually index 2) as the past state and feed it to the model
            at the next training step under the keyword argument ``mems``.
        tpu_name (:obj:`str`, `optional`):
            The name of the TPU the process is running on.
        run_name (:obj:`str`, `optional`):
            A descriptor for the run. Notably used for wandb logging.
    """

    tpu_name: str = field(
        default=None,
        metadata={"help": "Name of TPU"},
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
        """
        The strategy used for distributed training.
        """
        return self._setup_strategy

    @property
    @tf_required
    def n_replicas(self) -> int:
        """
        The number of replicas (CPUs, GPUs or TPU cores) used in this training.
        """
        return self._setup_strategy.num_replicas_in_sync

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from :obj:`per_gpu_train_batch_size` in distributed training).
        """
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        return per_device_batch_size * self.n_replicas

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
        """
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        return per_device_batch_size * self.n_replicas

    @property
    @tf_required
    def n_gpu(self) -> int:
        """
        The number of replicas (CPUs, GPUs or TPU cores) used in this training.
        """
        warnings.warn(
            "The n_gpu argument is deprecated and will be removed in a future version, use n_replicas instead.",
            FutureWarning,
        )
        return self._setup_strategy.num_replicas_in_sync
