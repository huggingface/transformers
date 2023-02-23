#!/usr/bin/env python
# coding=utf-8
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
"""
Fine-tuning the Flax library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
import numpy as np
from datasets import DatasetDict, load_dataset, load_metric
from tqdm import tqdm

import flax
import jax
import jax.numpy as jnp
import optax
import transformers
import wandb as wandb
from flax import core, jax_utils, struct, traverse_util
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from huggingface_hub import Repository
from models import FlaxSpeechEncoderDecoderModel
from optax._src import linear_algebra
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    is_tensorboard_available,
)
from transformers.file_utils import get_full_repo_name
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)


@flax.struct.dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    activation_dropout: float = field(
        default=0.1,
        metadata={
            "help": "The hidden activation dropout probability in the embeddings, encoder, and pooler."
        },
    )
    hidden_dropout: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    feat_proj_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The feat proj dropout probability for feature encoder representations."
        },
    )
    mask_time_prob: float = field(
        default=0.1,
        metadata={
            "help": "The spec aug dropout probability for feature encoder representations."
        },
    )
    encoder_add_adapter: bool = field(
        default=True, metadata={"help": "Whether to add an adapter layer between the encoder and decoder."}
    )


@flax.struct.dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to cache directory for saving and loading datasets"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    id_column_name: str = field(
        default="id",
        metadata={"help": "The name of the dataset column containing the id data. Defaults to 'id'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": "Filter audio files in the training set that are longer than `max_duration_in_seconds` seconds"
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files in the training set that are shorter than `min_duration_in_seconds` seconds"}
    )
    max_eval_duration_in_seconds: float = field(
        default=None,
        metadata={
            "help": "Filter audio files in the eval/test set that are longer than `max_duration_in_seconds` seconds"
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    min_target_length: Optional[int] = field(
        default=0,
        metadata={
            "help": "The minimum total sequence length for target text after tokenization. Sequences shorter "
            "than this will be filtered."
        },
    )
    pad_input_to_multiple_of: Optional[int] = field(
        default=24000,
        metadata={
            "help": "If set will pad the input sequence to a multiple of the provided value. "
            "This is important to avoid triggering recompilations on TPU."
        },
    )
    pad_target_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set will pad the target sequence to a multiple of the provided value. "
            "This is important to avoid triggering recompilations on TPU. If unspecified, will default to `max_target_length`, "
            " the equivalent of padding the targets to max length."
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only do data preprocessing and skip training. "
            "This is especially useful when data preprocessing errors out in distributed training due to timeout. "
            "In this case, one should run the preprocessing in a non-distributed setup with `preprocessing_only=True` "
            "so that the cached datasets can consequently be loaded in distributed training"
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    test_split_name: str = field(
        default="test",
        metadata={"help": "The name of the test data set split to use (via the datasets library). Defaults to 'test'"},
    )
    wandb_project: str = field(
        default="flax-speech-recognition-seq2seq",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_name: str = field(
        default=None,
        metadata={"help": "The name of the wandb run."},
    )
    wandb_job_type: str = field(
        default="Seq2Seq",
        metadata={"help": "The name of the wandb job type."},
    )
    log_first_ids: bool = field(
        default=True,
        metadata={
            "help": "Whether to log the first id's from the dataset. Defaults to `True`. If `False`, will log the first id's returned by the grouped length sampler."
        },
    )


# @flax.struct.dataclass
@dataclass
class FlaxSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    precision: str = field(
        default="full",
        metadata={
            "help": "Whether to enable mixed-precision training. If true, the optimizer is stored in half-precision (bfloat16) and computations are executed in half-precision"
            "**Note that this only specifies the dtype of the computation and optimizer state. It does not influence the dtype of model parameters.**"
        },
    )
    matmul_precision: str = field(
        default="default",
        metadata={
            "help": "Default floating-point precision of internal computations used in TPU matrix multiplications and convolutions. "
            "This configuration option controls the default precision for JAX operations that take an optional precision argument (e.g. `lax.conv_general_dilated` and `lax.dot`). "
            "This configuration option does not change the behaviours of such calls with explicit precision arguments; "
            "it only changes the behaviors of calls with no such argument provided. "
            "One of `['highest', 'float32', 'high', 'bfloat16_3x', 'default', 'bfloat16', 'fastest', None]`."
        },
    )
    generation_length_penalty: float = field(
        default=1,
        metadata={
            "help": "Exponential penalty to the length. 1.0 (default) means no penalty. Set to values < 1.0 in order to encourage the model"
            "to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences."
        },
    )
    final_generation_max_length: int = field(
        default=None,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. If unspecified, will default "
            "to the `max_length` value of the model configuration."
        },
    )
    final_generation_num_beams: int = field(
        default=None,
        metadata={
            "help": "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. If unspecified, will default "
            "to the `num_beams` value of the model configuration."
        },
    )

    def __post_init__(self):
        if self.final_generation_max_length is None:
            self.final_generation_max_length = self.generation_max_length
        if self.final_generation_num_beams is None:
            self.final_generation_num_beams = self.generation_num_beams


def to_fp32(t):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t)


def to_bf16(t):
    return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t)


class MixedPrecisionTrainState(struct.PyTreeNode):
    """Train state for use with a single Optax optimizer.
    Adapted from flax train_state https://github.com/google/flax/blob/main/flax/training/train_state.py

    Synopsis::

        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=tx)
        grad_fn = jax.grad(make_loss_fn(state.apply_fn))
        for batch in data:
          grads = grad_fn(state.params, batch)
          state = state.apply_gradients(grads=grads)

    Args:
      step: Counter starts at 0 and is incremented by every call to
        `.apply_gradients()`.
      apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
        convenience to have a shorter params list for the `train_step()` function
        in your training loop.
      params: The parameters to be updated by `tx` and used by `apply_fn`.
      tx: An Optax gradient transformation.
      opt_state: The state for `tx`.
      dropout_rng: PRNG key for stochastic operations.
      bf16: Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training.
    """

    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState
    dropout_rng: jnp.ndarray
    max_grad_norm: Optional[float] = 1.0

    def apply_gradients(self, *, grads, to_dtype, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """

        # clip gradients by global l2 norm
        casted_max_grad_norm = to_dtype(self.max_grad_norm)
        g_norm = linear_algebra.global_norm(grads)
        g_norm = jnp.maximum(casted_max_grad_norm, g_norm)
        grads = jax.tree_map(lambda t: (t / g_norm) * casted_max_grad_norm, grads)

        # perform update step in fp32 and subsequently downcast optimizer states if mixed precision training
        # grads and opt_state in bf16 (need to upcast), params in fp32 (leave as is)
        updates, new_opt_state = self.tx.update(to_fp32(grads), to_fp32(self.opt_state), self.params)

        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=to_dtype(new_opt_state),
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, to_dtype, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        # downcast optimizer state to bf16 if mixed-precision training
        opt_state = tx.init(to_dtype(params)) if tx is not None else None
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def pad_to_max_length(data, tokenizer):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.ones_like(mask, dtype=data.dtype) * tokenizer.pad_token_id
    out[mask] = np.concatenate(data)
    return out


def shift_tokens_right(label_ids: np.array, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift label ids one token to the right.
    """
    shifted_label_ids = np.zeros_like(label_ids)
    shifted_label_ids[:, 1:] = label_ids[:, :-1]
    shifted_label_ids[:, 0] = decoder_start_token_id

    return shifted_label_ids


@flax.struct.dataclass
class FlaxDataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The begin-of-sentence of the decoder.
        input_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        target_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned target sequences (according to the model's padding side and padding index).
            See above for details.
        max_input_length (:obj:`float`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
        pad_input_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the input sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        pad_target_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the target sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Any
    decoder_start_token_id: int
    input_padding: Union[bool, str] = "longest"
    target_padding: Union[bool, str] = "max_length"
    max_input_length: Optional[float] = None
    max_target_length: Optional[int] = None
    pad_input_to_multiple_of: Optional[int] = None
    pad_target_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        input_ids = [feature["input_id"] for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            max_length=self.max_input_length,
            padding=self.input_padding,
            pad_to_multiple_of=self.pad_input_to_multiple_of,
            return_tensors="np",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            pad_to_multiple_of=self.pad_target_to_multiple_of,
            return_tensors="np",
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        labels = labels_batch["input_ids"]
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
            labels_batch.attention_mask = labels_batch.attention_mask[:, 1:]

        decoder_input_ids = shift_tokens_right(labels, self.decoder_start_token_id)

        # replace padding with -100 to ignore correctly when computing the loss
        labels = np.ma.array(labels, mask=np.not_equal(labels_batch.attention_mask, 1))
        labels = labels.filled(fill_value=-100)

        batch["inputs"] = batch.pop("input_values")
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch


def get_grouped_indices(
    dataset, batch_size: int, rng: Optional[List[int]] = None, mega_batch_mult: Optional[int] = None
) -> np.array:
    """
    Adapted from the `get_length_grouped_indices` function in the PyTorch Trainer utils file (https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L486)
    Function that returns a list of indices in which each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted (if a JAX rng is specified)
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    lengths = dataset["input_length"]

    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # We need to use JAX for the random permutation as the PRNG key will be set based on the seed outside of the sampler.
    num_samples = len(lengths)
    indices = jax.random.permutation(rng, np.arange(num_samples)) if rng is not None else np.arange(num_samples)
    indices = np.asarray(indices)

    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [list(sorted(megabatch, key=lambda i: lengths[i], reverse=True)) for megabatch in megabatches]

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length, the longest element is the first
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = np.argmax(megabatch_maximums).item()
    # Switch to put the longest batch in first position
    # (note that this is different to the PT grouped sampler in which we only put the longest element in the first position, and not its batch)
    megabatches[0], megabatches[max_idx] = megabatches[max_idx], megabatches[0]

    megabatches = np.array([i for megabatch in megabatches for i in megabatch])

    return megabatches


def generate_batch_splits(samples_idx: np.ndarray, batch_size: int, drop_last_batch=True) -> np.ndarray:
    """Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by
    the batch size and `drop_last` is `True`, the last incomplete batch is dropped. Else, it is returned."""
    num_samples = len(samples_idx)
    if drop_last_batch:
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        samples_idx = samples_idx.reshape((sections_split, batch_size))
    else:
        sections_split = math.ceil(num_samples / batch_size)
        samples_idx = np.array_split(samples_idx, sections_split)
    return samples_idx


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step, pred_str=None):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)

    if pred_str is not None:
        # write output actual predictions for debugging
        summary_writer.text("eval_predictions", "\n".join(pred_str), step)


def write_wandb_log(metrics, step, prefix=None):
    if jax.process_index() == 0:
        log_metrics = {}
        for k, v in metrics.items():
            if "layer" in k:
                log_metrics[f"{k}/"] = v
            elif prefix is not None:
                log_metrics[f"{prefix}/{k}"] = v
            else:
                log_metrics[k] = v
        wandb.log(log_metrics, step)


def write_wandb_pred(pred_str, label_str, eval_ids, step, prefix="eval", top_ids=None, final_step=True):
    if jax.process_index() == 0:
        top_ids = top_ids if top_ids else eval_ids
        num_beams = len(pred_str)
        # convert str data to a wandb compatible format
        str_data = []
        for id in top_ids:
            if id in eval_ids:
                idx = eval_ids.index(id)
                str_data.append([eval_ids[idx], label_str[idx]] + [pred_str[beam][idx] for beam in range(num_beams)])
        columns = ["id", "label_str"] + [f"beam_{i + 1}" for i in range(num_beams)]
        wandb.log(
            {f"{prefix}/step_{int(step / 1000)}k": wandb.Table(columns=columns, data=str_data[:50])},
            step,
        )
        if final_step:
            str_data = np.array(str_data)
            wandb.log(
                {f"{prefix}/step_{int(step / 1000)}k_all": wandb.Table(columns=columns, data=str_data[:200000])},
                step,
            )
            str_data = str_data[str_data[:, 1] != str_data[:, 2]]
            wandb.log(
                {f"{prefix}/step_{int(step / 1000)}k_incorrect": wandb.Table(columns=columns, data=str_data[:200000])},
                step,
            )


def create_learning_rate_fn(
    num_train_steps: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FlaxSeq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Set the verbosity to info of the Transformers logger.
    # We only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set up wandb run
    if jax.process_index() == 0:
        wandb.init(project=data_args.wandb_project, name=data_args.wandb_name, job_type=data_args.wandb_job_type)

    logger.info("Training/evaluation parameters %s", training_args)

    # Set the default TPU matmul precision and display the number of devices
    jax.config.update("jax_default_matmul_precision", training_args.matmul_precision)
    logger.info(f"JAX devices: {jax.device_count()}, matmul precision: {training_args.matmul_precision}")

    # TODO: 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if training_args.do_predict:
        test_split = data_args.test_split_name.split("+")
        for split in test_split:
            raw_datasets[split] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=split,
                cache_dir=data_args.dataset_cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    if not training_args.do_train and not training_args.do_eval and not training_args.do_predict:
        raise ValueError(
            "Cannot not train, not do evaluation and not do prediction. At least one of "
            "training, evaluation or prediction has to be done."
        )

    # if not training, there is no need to run multiple epochs
    if not training_args.do_train:
        training_args.num_train_epochs = 1

    if data_args.audio_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--text_column_name` to the correct text column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if data_args.log_first_ids and data_args.id_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--id_column_name {data_args.id_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--id_column_name` to the correct id column - one of "
            f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    # 5. Load pretrained model, tokenizer, and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # update config according to training and model args
    config.encoder.update(
        {
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "hidden_dropout": model_args.hidden_dropout,
            "activation_dropout": model_args.activation_dropout,
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "add_adapter": model_args.encoder_add_adapter,
        }
    )
    config.decoder.update(
        {
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "dropout": model_args.hidden_dropout,
            "activation_dropout": model_args.activation_dropout,
        }
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if training_args.precision == "full_mixed":
        dtype = jnp.bfloat16
        training_args.mixed_precision = True
    elif training_args.precision == "half_mixed":
        dtype = jnp.bfloat16
        training_args.mixed_precision = False
    else:
        dtype = jnp.float32
        training_args.mixed_precision = False

    model = FlaxSpeechEncoderDecoderModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=dtype,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # 6. Resample speech dataset ALWAYS
    raw_datasets = raw_datasets.cast_column(data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate))

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = int(data_args.max_duration_in_seconds * feature_extractor.sampling_rate)
    min_input_length = int(data_args.min_duration_in_seconds * feature_extractor.sampling_rate)
    max_eval_input_length = int(data_args.max_eval_duration_in_seconds * feature_extractor.sampling_rate) if data_args.max_eval_duration_in_seconds else None
    max_target_length = data_args.max_target_length
    min_target_length = data_args.min_target_length
    pad_input_to_multiple_of = data_args.pad_input_to_multiple_of
    pad_target_to_multiple_of = data_args.pad_target_to_multiple_of
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    id_column_name = data_args.id_column_name
    model_input_name = feature_extractor.model_input_names[0]
    log_first_ids = data_args.log_first_ids

    if training_args.do_train and data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    if training_args.do_eval and data_args.max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

    if training_args.do_predict and data_args.max_test_samples is not None:
        for split in test_split:
            raw_datasets[split] = raw_datasets[split].select(range(data_args.max_eval_samples))


    def prepare_dataset(batch):
        # Pre-process audio
        sample = batch[audio_column_name]

        # normalise audio (mean, std) to (0, 1)
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["input_id"] = batch[id_column_name] if log_first_ids else None

        input_str = batch[text_column_name]
        # Finally, we tokenize the processed text
        batch["labels"] = tokenizer(input_str).input_ids
        batch["labels_length"] = len(batch["labels"])
        return batch

    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        num_proc=num_workers,
        desc="preprocess train dataset",
    )

    # filter training data with inputs longer than max_input_length
    def is_audio_in_length_range(length):
        return min_input_length < length < max_input_length

    if training_args.do_train:
        vectorized_datasets["train"] = vectorized_datasets["train"].filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_length"],
        )

    if max_eval_input_length is not None:
        # filter training data with inputs longer than max_input_length
        def is_eval_audio_in_length_range(length):
            return min_input_length < length < max_eval_input_length

        if training_args.do_eval:
            vectorized_datasets["eval"] = vectorized_datasets["eval"].filter(
                is_eval_audio_in_length_range,
                num_proc=num_workers,
                input_columns=["input_length"],
            )

        if training_args.do_test:
            for split in test_split:
                vectorized_datasets[split] = vectorized_datasets[split].filter(
                    is_eval_audio_in_length_range,
                    num_proc=num_workers,
                    input_columns=["input_length"],
                )

    # filter data with targets shorter than min_target_length or longer than max_target_length
    def is_labels_in_length_range(length):
        return min_target_length < length < max_target_length

    if training_args.do_train:
        vectorized_datasets["train"] = vectorized_datasets["train"].filter(
            is_labels_in_length_range,
            num_proc=num_workers,
            input_columns=["labels_length"],
        )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    # 8. Load Metrics
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    def compute_metrics(pred_ids: List[List[int]], label_ids: List[List[int]]):
        label_ids = (
            pad_to_max_length(np.array(label_ids, dtype="object"), tokenizer)
            if pad_target_to_multiple_of
            else label_ids
        )

        padded_ids = np.where(np.asarray(label_ids) == -100, tokenizer.pad_token_id, np.asarray(label_ids))
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(padded_ids, skip_special_tokens=True)

        pred_ids = np.array(pred_ids)
        num_beams = pred_ids.shape[1]
        # decode on a beam-by-beam basis
        pred_str = [
            tokenizer.batch_decode(pred_ids[:, beam, :], skip_special_tokens=True)
            for beam in reversed(range(num_beams))
        ]
        # compute word/character error rate for top beam
        wer = wer_metric.compute(predictions=pred_str[0], references=label_str)
        cer = cer_metric.compute(predictions=pred_str[0], references=label_str)

        return {"wer": wer, "cer": cer}, pred_str, label_str

    # 9. Save feature extractor, tokenizer and config
    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    data_collator = FlaxDataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        input_padding="longest",
        target_padding="longest",
        max_target_length=max_target_length,
        pad_input_to_multiple_of=pad_input_to_multiple_of,
        pad_target_to_multiple_of=pad_target_to_multiple_of if pad_target_to_multiple_of else max_target_length,
    )

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run `pip install tensorboard` to enable."
        )

    # 10. Handle the repository creation
    if training_args.push_to_hub:
        with open(os.path.join(training_args.output_dir, ".gitattributes"), "r+") as f:
            git_lfs_extensions = f.read()
            if "*.wandb" not in git_lfs_extensions:
                f.write("*.wandb filter=lfs diff=lfs merge=lfs -text")
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name, token=training_args.hub_token
            )
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(training_args.output_dir, clone_from=repo_name)

    # 11. Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constants
    max_steps = int(training_args.max_steps)
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    batch_size_per_update = train_batch_size * gradient_accumulation_steps
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    to_dtype = to_bf16 if training_args.mixed_precision else to_fp32

    if training_args.do_train:
        num_train_samples = len(vectorized_datasets["train"])
        steps_per_epoch = num_train_samples // batch_size_per_update
        if max_steps > 0:
            num_epochs = -(training_args.max_steps // -steps_per_epoch)
            total_train_steps = max_steps
        else:
            num_epochs = int(training_args.num_train_epochs)
            total_train_steps = steps_per_epoch * num_epochs

        # Create learning rate schedule
        linear_decay_lr_schedule_fn = create_learning_rate_fn(
            total_train_steps,
            training_args.warmup_steps,
            training_args.learning_rate,
        )

        # We use Optax's "masking" functionality to not apply weight decay
        # to bias and LayerNorm scale parameters. decay_mask_fn returns a
        # mask boolean with the same structure as the parameters.
        # The mask is True for parameters that should be decayed.
        # Note that this mask is specifically adapted for FlaxWav2Vec2 and FlaxBart.
        # For FlaxT5, one should correct the layer norm parameter naming
        # accordingly - see `run_t5_mlm_flax.py` e.g.
        def decay_mask_fn(params):
            flat_params = traverse_util.flatten_dict(params)
            layer_norm_params = [
                (name, "scale")
                for name in ["layer_norm", "self_attn_layer_norm", "layernorm_embedding", "final_layer_norm"]
            ]
            flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_params) for path in flat_params}
            return traverse_util.unflatten_dict(flat_mask)

        if training_args.adafactor:
            # Create Adafactor optimizer
            optim = optax.adafactor(
                learning_rate=linear_decay_lr_schedule_fn,
                dtype_momentum=jnp.bfloat16 if training_args.mixed_precision else jnp.float32,
                weight_decay_rate=training_args.weight_decay,
                weight_decay_mask=decay_mask_fn,
            )
        else:
            # Create AdamW optimizer
            optim = optax.adamw(
                learning_rate=linear_decay_lr_schedule_fn,
                b1=training_args.adam_beta1,
                b2=training_args.adam_beta2,
                eps=training_args.adam_epsilon,
                weight_decay=training_args.weight_decay,
                mask=decay_mask_fn,
            )
    else:
        num_epochs = 0
        total_train_steps = 0
        num_train_samples = 0
        optim = None

    # Setup train state
    state = MixedPrecisionTrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optim,
        to_dtype=to_dtype,
        dropout_rng=dropout_rng,
        max_grad_norm=training_args.max_grad_norm,
    )

    # Cross entropy loss
    def loss_fn(logits, labels):
        vocab_size = logits.shape[-1]
        # optax onehot always returns a float32 device array, need to downcast if performing mixed precision training
        onehot_targets = to_dtype(onehot(labels, vocab_size))
        loss = optax.softmax_cross_entropy(logits, onehot_targets)
        # ignore padded tokens from loss, i.e. where labels are not set to -100
        padding = labels >= 0
        loss = loss * padding
        loss = loss.sum()
        num_labels = padding.sum()
        return loss, num_labels

    # Define gradient update step fn
    def train_step(state, batch):
        # only one single rng per grad step, with or without accumulation, as the graph should be identical over one effective training batch
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params, minibatch):
            labels = minibatch.pop("labels")
            logits = state.apply_fn(
                **minibatch,
                params=params,
                dropout_rng=dropout_rng,
                freeze_feature_encoder=model_args.freeze_feature_encoder,
                train=True,
            )[0]
            loss, num_labels = loss_fn(logits, labels)
            return loss, num_labels

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

        if gradient_accumulation_steps == 1:
            (loss, num_labels), grad = grad_fn(to_dtype(state.params), batch)

        # Custom gradient accumulation
        else:
            # add a first dimension over gradient_accumulation_steps for minibatch slices
            batch = jax.tree_map(
                lambda x: x.reshape(
                    gradient_accumulation_steps, training_args.per_device_train_batch_size, *x.shape[1::]
                ),
                batch,
            )

            def accum_minibatch_step(accum_grad, minibatch):
                # compute loss, num labels and grad over minibatch and accumulate
                (loss, num_labels), grad = grad_fn(to_dtype(state.params), minibatch)
                return jax.tree_map(jnp.add, accum_grad, grad), (loss, num_labels)

            # create an initial state for accumulating losses, num labels and gradients
            init_grad = jax.tree_map(jnp.zeros_like, to_dtype(state.params))
            # loop accum minibatch step over the number of gradient accumulation steps
            grad, (loss, num_labels) = jax.lax.scan(accum_minibatch_step, init_grad, batch)

        grad = jax.lax.psum(grad, "batch")
        loss = jax.lax.psum(loss.sum(), "batch")
        total_samples = jax.lax.psum(num_labels.sum(), "batch")
        grad = jax.tree_map(lambda g: g / total_samples, grad)
        loss = jax.tree_map(lambda l: l / total_samples, loss)

        # update state
        new_state = state.apply_gradients(
            grads=grad,
            dropout_rng=new_dropout_rng,
            to_dtype=to_dtype,
        )

        # compute gradient norms over all layers, total encoder, total decoder and global for detailed monitoring
        layer_grad_norm = jax.tree_map(jnp.linalg.norm, grad)
        logs = {
            "layer_grad_norm": layer_grad_norm,
            "encoder_grad_norm": jnp.linalg.norm(jax.tree_util.tree_leaves(layer_grad_norm["encoder"])),
            "decoder_grad_norm": jnp.linalg.norm(jax.tree_util.tree_leaves(layer_grad_norm["decoder"])),
        }
        logs["grad_norm"] = jnp.linalg.norm([logs["encoder_grad_norm"], logs["decoder_grad_norm"]])

        # compute parameter norms over all layers, total encoder, total decoder and global for detailed monitoring
        layer_param_norm = jax.tree_map(jnp.linalg.norm, new_state.params)
        logs["layer_param_norm"] = layer_param_norm
        logs["encoder_param_norm"] = jnp.linalg.norm(jax.tree_util.tree_leaves(layer_param_norm["encoder"]))
        logs["decoder_param_norm"] = jnp.linalg.norm(jax.tree_util.tree_leaves(layer_param_norm["decoder"]))
        logs["param_norm"] = jnp.linalg.norm([logs["encoder_param_norm"], logs["decoder_param_norm"]])

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        metrics.update(logs)

        metrics = jax.lax.pmean(metrics, axis_name="batch")
        # metrics = to_fp32(metrics)

        return new_state, metrics

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss, num_labels = loss_fn(logits, labels)

        total_samples = jax.lax.psum(num_labels, "batch")
        loss = jax.lax.psum(loss, "batch")
        loss = jax.tree_map(lambda l: l / total_samples, loss)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        # metrics = to_fp32(metrics)
        return metrics

    # Define generation function
    gen_kwargs = {
        "max_length": training_args.generation_max_length,
        "num_beams": training_args.generation_num_beams,
        "length_penalty": training_args.generation_length_penalty,
    }
    final_gen_kwargs = {
        "max_length": training_args.final_generation_max_length,
        "num_beams": training_args.final_generation_num_beams,
        "length_penalty": training_args.generation_length_penalty,
    }

    def generate_step(params, batch):
        model.params = params
        output_ids = model.generate(batch["inputs"], **gen_kwargs)
        return output_ids.sequences

    def final_generate_step(params, batch):
        model.params = params
        output_ids = model.generate(batch["inputs"], **final_gen_kwargs)
        return output_ids.sequences

    # Create parallel version of the train and eval step
    if training_args.do_train:
        p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    if training_args.do_eval or training_args.do_predict:
        p_eval_step = jax.pmap(eval_step, "batch")

    if training_args.predict_with_generate:
        p_generate_step = jax.pmap(generate_step, "batch")
        p_final_generate_step = jax.pmap(final_generate_step, "batch")

    def run_evaluation(step, final_step=False):
        if training_args.do_eval:
            # ======================== Evaluating ==============================
            eval_metrics = []
            eval_preds = []
            eval_ids = []
            eval_labels = []

            # Generate eval set by sequentially sampling indices from the eval dataset and grouping by length
            eval_samples_idx = get_grouped_indices(vectorized_datasets["eval"], eval_batch_size)
            eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size, drop_last_batch=False)

            for i, batch_idx in enumerate(tqdm(eval_batch_idx, desc="Evaluating ...", position=2)):
                samples = [vectorized_datasets["eval"][int(idx)] for idx in batch_idx]
                batch = data_collator(samples)
                eval_ids.extend(batch.pop("input_ids"))
                labels = batch["labels"]

                metrics = pad_shard_unpad(p_eval_step, static_return=True)(state.params, batch.data, min_device_batch=per_device_eval_batch_size)
                eval_metrics.append(metrics)

                # generation
                if training_args.predict_with_generate:
                    if not final_step:
                        generated_ids = pad_shard_unpad(p_generate_step)(state.params, batch.data, min_device_batch=per_device_eval_batch_size)
                        eval_preds.extend(
                            jax.device_get(
                                generated_ids.reshape(-1, gen_kwargs["num_beams"], gen_kwargs["max_length"])
                            )
                        )
                    else:
                        generated_ids = pad_shard_unpad(p_final_generate_step)(state.params, batch.data, min_device_batch=per_device_eval_batch_size)
                        eval_preds.extend(
                            jax.device_get(
                                generated_ids.reshape(
                                    -1, final_gen_kwargs["num_beams"], final_gen_kwargs["max_length"]
                                )
                            )
                        )
                    eval_labels.extend(labels)

            # normalize eval metrics
            eval_metrics = get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.mean, eval_metrics)
            eval_metrics = to_fp32(eval_metrics)

            # compute error rate metric and get predicted string (for debugging)
            error_rate_desc = ""
            pred_str = []
            label_str = []
            if training_args.predict_with_generate:
                error_rate_metric, pred_str, label_str = compute_metrics(eval_preds, eval_labels)
                eval_metrics.update(error_rate_metric)
                error_rate_desc = " ".join([f"Eval {key}: {value} |" for key, value in error_rate_metric.items()])

            # Print metrics and update progress bar
            desc = f"Step... ({step}/{total_train_steps} | Eval Loss: {eval_metrics['loss']} | {error_rate_desc})"
            epochs.write(desc)
            epochs.desc = desc

            # Save metrics
            write_wandb_log(eval_metrics, step, prefix="eval")
            write_wandb_pred(
                pred_str,
                label_str,
                eval_ids,
                step,
                top_ids=vectorized_datasets["eval"]["input_id"] if data_args.log_first_ids else None,
                final_step=final_step,
            )
            # if has_tensorboard and jax.process_index() == 0:
            # write_eval_metric(summary_writer, eval_metrics, step, pred_str=pred_str)

    def save_checkpoint(step):
        # save and push checkpoint to the hub
        if jax.process_index() == 0:
            params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
            model.save_pretrained(training_args.output_dir, params=params)
            tokenizer.save_pretrained(training_args.output_dir)
            if training_args.push_to_hub:
                repo.push_to_hub(commit_message=f"{wandb.run.id}: saving weights and logs of step {int(step / 1000)}k", blocking=False)

    # Replicate the train state on each device
    state = state.replicate()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_train_samples}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Num gradient accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {batch_size_per_update}")
    logger.info(f"  Total optimization steps = {total_train_steps}")
    logger.info(f"  Gradient checkpointing: {config.encoder.gradient_checkpointing}")
    logger.info(f"  Use scan: {config.encoder.use_scan}")
    logger.info(f"  Fuse matmuls: {config.encoder.fuse_matmuls}")

    train_time = cur_step = 0
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        if training_args.do_train:
            # ======================== Training ================================
            train_start = time.time()

            # Create sampling rng
            rng, input_rng = jax.random.split(rng)

            # Generate an epoch by randomly shuffling sampling indices from the train dataset and grouping by length
            train_samples_idx = get_grouped_indices(vectorized_datasets["train"], batch_size_per_update, input_rng)
            train_batch_idx = generate_batch_splits(train_samples_idx, batch_size_per_update, drop_last_batch=True)

            # Gather the indices for creating the batch and do a training step
            for step, batch_idx in enumerate(tqdm(train_batch_idx, desc="Training...", position=1), 1):
                samples = [vectorized_datasets["train"][int(idx)] for idx in batch_idx]
                batch = data_collator(samples)
                batch.pop("input_ids")
                batch = shard(batch.data)
                state, train_metric = p_train_step(state, batch)

                cur_step = epoch * (num_train_samples // batch_size_per_update) + step

                if cur_step % training_args.logging_steps == 0:
                    # Save metrics
                    train_metric = unreplicate(train_metric)
                    train_time += time.time() - train_start
                    # need to upcast all device arrays to fp32 for wandb logging (jnp.bfloat16 not supported) -> do this here OR in train_step
                    write_wandb_log(to_fp32(train_metric), cur_step, prefix="train")
                    # we won't log to tensorboard for now (it is fiddly logging param and grad norms on a layer-by-layer basis)
                    # if has_tensorboard and jax.process_index() == 0:
                    # write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                    epochs.write(
                        f"Step... ({cur_step} | Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']}, Gradient Norm: {train_metric['grad_norm']})"
                    )

                if cur_step % total_train_steps == 0:
                    break

                if training_args.eval_steps and cur_step % training_args.eval_steps == 0:
                    # run beam search at each eval step
                    run_evaluation(cur_step, final_step=False)

                if cur_step % training_args.save_steps == 0:
                    save_checkpoint(cur_step)

            if training_args.eval_steps == 0 and (epoch + 1) != num_epochs:
                # run evaluation at the end of the epoch if eval steps are not specified
                run_evaluation(cur_step, final_step=False)
                save_checkpoint(cur_step)

    if training_args.do_train:
        save_checkpoint(cur_step)

    cur_step = max_steps if max_steps > 0 else cur_step  # set step to max steps so that eval happens in alignment with training

    if training_args.do_eval:
        run_evaluation(cur_step, final_step=True)

    # TODO: collapse 'do_predict' into the run_evaluation function
    if training_args.do_predict:
        # ======================== Prediction ==============================
        for split in test_split:
            pred_metrics = []
            pred_generations = []
            pred_ids = []
            pred_labels = []

            # Generate eval set by sequentially sampling indices from the eval dataset and grouping by length
            pred_samples_idx = get_grouped_indices(vectorized_datasets[split], eval_batch_size)
            pred_batch_idx = generate_batch_splits(pred_samples_idx, eval_batch_size, drop_last_batch=False)

            for i, batch_idx in enumerate(tqdm(pred_batch_idx, desc=f"Predicting {split}...", position=2)):
                samples = [vectorized_datasets[split][int(idx)] for idx in batch_idx]
                batch = data_collator(samples)
                pred_ids.extend(batch.pop("input_ids"))
                labels = batch["labels"]

                metrics = pad_shard_unpad(p_eval_step, static_return=True)(state.params, batch.data,
                                                                           min_device_batch=per_device_eval_batch_size)
                pred_metrics.append(metrics)

                # generation
                if training_args.predict_with_generate:
                    generated_ids = pad_shard_unpad(p_final_generate_step)(state.params, batch.data, min_device_batch=per_device_eval_batch_size)
                    pred_generations.extend(
                        jax.device_get(
                            generated_ids.reshape(-1, final_gen_kwargs["num_beams"], final_gen_kwargs["max_length"])
                        )
                    )
                    pred_labels.extend(labels)

            # normalize eval metrics
            pred_metrics = get_metrics(pred_metrics)
            pred_metrics = jax.tree_map(jnp.mean, pred_metrics)
            pred_metrics = to_fp32(pred_metrics)

            # compute error rate metric and get predicted string (for debugging)
            error_rate_desc = ""
            pred_str = []
            label_str = []
            if training_args.predict_with_generate:
                error_rate_metric, pred_str, label_str = compute_metrics(pred_generations, pred_labels)
                pred_metrics.update(error_rate_metric)
                error_rate_desc = " ".join([f"{split} {key}: {value} |" for key, value in error_rate_metric.items()])

            # Print metrics and update progress bar
            desc = f"Step... ({cur_step}/{total_train_steps} | {split} Loss: {pred_metrics['loss']} | {error_rate_desc})"
            epochs.write(desc)
            epochs.desc = desc

            # Save metrics
            write_wandb_log(pred_metrics, cur_step, prefix=split)
            write_wandb_pred(
                pred_str,
                label_str,
                pred_ids,
                cur_step,
                prefix=split,
                top_ids=vectorized_datasets[split]["input_id"] if data_args.log_first_ids else None,
                final_step=True,
            )


if __name__ == "__main__":
    main()
