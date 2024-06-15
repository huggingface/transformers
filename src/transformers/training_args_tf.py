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

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .training_args import TrainingArguments
from .utils import cached_property, is_tf_available, logging, requires_backends


logger = logging.get_logger(__name__)

if is_tf_available():
    import tensorflow as tf

    from .modeling_tf_utils import keras


@dataclass
class TFTrainingArguments(TrainingArguments):
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        output_dir (`str`):
            The output directory where the model predictions and checkpoints will be written.
        overwrite_output_dir (`bool`, *optional*, defaults to `False`):
            If `True`, overwrite the content of the output directory. Use this to continue training if `output_dir`
            points to a checkpoint directory.
        do_train (`bool`, *optional*, defaults to `False`):
            Whether to run training or not. This argument is not directly used by [`Trainer`], it's intended to be used
            by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        do_eval (`bool`, *optional*):
            Whether to run evaluation on the validation set or not. Will be set to `True` if `eval_strategy` is
            different from `"no"`. This argument is not directly used by [`Trainer`], it's intended to be used by your
            training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        do_predict (`bool`, *optional*, defaults to `False`):
            Whether to run predictions on the test set or not. This argument is not directly used by [`Trainer`], it's
            intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/huggingface/transformers/tree/main/examples) for more details.
        eval_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
            The evaluation strategy to adopt during training. Possible values are:

                - `"no"`: No evaluation is done during training.
                - `"steps"`: Evaluation is done (and logged) every `eval_steps`.
                - `"epoch"`: Evaluation is done at the end of each epoch.

        per_device_train_batch_size (`int`, *optional*, defaults to 8):
            The batch size per GPU/TPU core/CPU for training.
        per_device_eval_batch_size (`int`, *optional*, defaults to 8):
            The batch size per GPU/TPU core/CPU for evaluation.
        gradient_accumulation_steps (`int`, *optional*, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

            <Tip warning={true}>

            When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging,
            evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training examples.

            </Tip>

        learning_rate (`float`, *optional*, defaults to 5e-5):
            The initial learning rate for Adam.
        weight_decay (`float`, *optional*, defaults to 0):
            The weight decay to apply (if not zero).
        adam_beta1 (`float`, *optional*, defaults to 0.9):
            The beta1 hyperparameter for the Adam optimizer.
        adam_beta2 (`float`, *optional*, defaults to 0.999):
            The beta2 hyperparameter for the Adam optimizer.
        adam_epsilon (`float`, *optional*, defaults to 1e-8):
            The epsilon hyperparameter for the Adam optimizer.
        max_grad_norm (`float`, *optional*, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        num_train_epochs(`float`, *optional*, defaults to 3.0):
            Total number of training epochs to perform.
        max_steps (`int`, *optional*, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
            For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
            `max_steps` is reached.
        warmup_ratio (`float`, *optional*, defaults to 0.0):
            Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
        warmup_steps (`int`, *optional*, defaults to 0):
            Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`.
        logging_dir (`str`, *optional*):
            [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
            *runs/**CURRENT_DATETIME_HOSTNAME***.
        logging_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The logging strategy to adopt during training. Possible values are:

                - `"no"`: No logging is done during training.
                - `"epoch"`: Logging is done at the end of each epoch.
                - `"steps"`: Logging is done every `logging_steps`.

        logging_first_step (`bool`, *optional*, defaults to `False`):
            Whether to log and evaluate the first `global_step` or not.
        logging_steps (`int`, *optional*, defaults to 500):
            Number of update steps between two logs if `logging_strategy="steps"`.
        save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                - `"no"`: No save is done during training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`.

        save_steps (`int`, *optional*, defaults to 500):
            Number of updates steps before two checkpoint saves if `save_strategy="steps"`.
        save_total_limit (`int`, *optional*):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            `output_dir`.
        no_cuda (`bool`, *optional*, defaults to `False`):
            Whether to not use CUDA even when it is available or not.
        seed (`int`, *optional*, defaults to 42):
            Random seed that will be set at the beginning of training.
        fp16 (`bool`, *optional*, defaults to `False`):
            Whether to use 16-bit (mixed) precision training (through NVIDIA Apex) instead of 32-bit training.
        fp16_opt_level (`str`, *optional*, defaults to 'O1'):
            For `fp16` training, Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details on
            the [Apex documentation](https://nvidia.github.io/apex/amp).
        local_rank (`int`, *optional*, defaults to -1):
            During distributed training, the rank of the process.
        tpu_num_cores (`int`, *optional*):
            When training on TPU, the number of TPU cores (automatically passed by launcher script).
        debug (`bool`, *optional*, defaults to `False`):
            Whether to activate the trace to record computation graphs and profiling information or not.
        dataloader_drop_last (`bool`, *optional*, defaults to `False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        eval_steps (`int`, *optional*, defaults to 1000):
            Number of update steps before two evaluations.
        past_index (`int`, *optional*, defaults to -1):
            Some models like [TransformerXL](../model_doc/transformerxl) or :doc*XLNet <../model_doc/xlnet>* can make
            use of the past hidden states for their predictions. If this argument is set to a positive int, the
            `Trainer` will use the corresponding output (usually index 2) as the past state and feed it to the model at
            the next training step under the keyword argument `mems`.
        tpu_name (`str`, *optional*):
            The name of the TPU the process is running on.
        tpu_zone (`str`, *optional*):
            The zone of the TPU the process is running on. If not specified, we will attempt to automatically detect
            from metadata.
        gcp_project (`str`, *optional*):
            Google Cloud Project name for the Cloud TPU-enabled project. If not specified, we will attempt to
            automatically detect from metadata.
        run_name (`str`, *optional*):
            A descriptor for the run. Notably used for wandb logging.
        xla (`bool`, *optional*):
            Whether to activate the XLA compilation or not.
    """

    framework = "tf"
    tpu_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of TPU"},
    )

    tpu_zone: Optional[str] = field(
        default=None,
        metadata={"help": "Zone of TPU"},
    )

    gcp_project: Optional[str] = field(
        default=None,
        metadata={"help": "Name of Cloud TPU-enabled project"},
    )

    poly_power: float = field(
        default=1.0,
        metadata={"help": "Power for the Polynomial decay LR scheduler."},
    )

    xla: bool = field(default=False, metadata={"help": "Whether to activate the XLA compilation or not"})

    @cached_property
    def _setup_strategy(self) -> Tuple["tf.distribute.Strategy", int]:
        requires_backends(self, ["tf"])
        logger.info("Tensorflow: setting up strategy")

        gpus = tf.config.list_physical_devices("GPU")

        # Set to float16 at first
        if self.fp16:
            keras.mixed_precision.set_global_policy("mixed_float16")

        if self.no_cuda:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        else:
            try:
                if self.tpu_name:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
                        self.tpu_name, zone=self.tpu_zone, project=self.gcp_project
                    )
                else:
                    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            except ValueError:
                if self.tpu_name:
                    raise RuntimeError(f"Couldn't connect to TPU {self.tpu_name}!")
                else:
                    tpu = None

            if tpu:
                # Set to bfloat16 in case of TPU
                if self.fp16:
                    keras.mixed_precision.set_global_policy("mixed_bfloat16")

                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)

                strategy = tf.distribute.TPUStrategy(tpu)

            elif len(gpus) == 0:
                strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            elif len(gpus) == 1:
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            elif len(gpus) > 1:
                # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
                strategy = tf.distribute.MirroredStrategy()
            else:
                raise ValueError("Cannot find the proper strategy, please check your environment properties.")

        return strategy

    @property
    def strategy(self) -> "tf.distribute.Strategy":
        """
        The strategy used for distributed training.
        """
        requires_backends(self, ["tf"])
        return self._setup_strategy

    @property
    def n_replicas(self) -> int:
        """
        The number of replicas (CPUs, GPUs or TPU cores) used in this training.
        """
        requires_backends(self, ["tf"])
        return self._setup_strategy.num_replicas_in_sync

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        return False  # TF Logging is handled by Keras not the Trainer

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from `per_gpu_train_batch_size` in distributed training).
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
        The actual batch size for evaluation (may differ from `per_gpu_eval_batch_size` in distributed training).
        """
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        return per_device_batch_size * self.n_replicas

    @property
    def n_gpu(self) -> int:
        """
        The number of replicas (CPUs, GPUs or TPU cores) used in this training.
        """
        requires_backends(self, ["tf"])
        warnings.warn(
            "The n_gpu argument is deprecated and will be removed in a future version, use n_replicas instead.",
            FutureWarning,
        )
        return self._setup_strategy.num_replicas_in_sync
