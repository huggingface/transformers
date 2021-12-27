#!/usr/bin/env python3
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
from datasets import DatasetDict, load_dataset
from tqdm import tqdm

import flax
import jax
import jax.numpy as jnp
import optax
import wandb
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.training.common_utils import get_metrics, shard
from transformers import FlaxWav2Vec2ForPreTraining, HfArgumentParser, Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_flax_wav2vec2 import _compute_mask_indices, _sample_negative_indices


logger = logging.getLogger(__name__)


wandb.init(project="pretraining-wav2vec2-flax")


@flax.struct.dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    max_gumbel_temperature: Optional[float] = field(
        default=2.0, metadata={"help": "Maximum temperature for gumbel softmax."}
    )
    min_gumbel_temperature: Optional[float] = field(
        default=0.1, metadata={"help": "Minimum temperature for gumbel softmax."}
    )
    gumbel_temperature_decay: Optional[float] = field(
        default=0.999995, metadata={"help": "Decay of gumbel temperature during training."}
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@flax.struct.dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    validation_split_name: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the validation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    speech_file_column: Optional[str] = field(
        default="audio",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'file'"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=1024,
        metadata={
            "help": "If set will pad the sequence to a multiple of the provided value. This is important to avoid triggering recompilations on TPU"
        },
    )
    audio_column_name: Optional[str] = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=15.0,
        metadata={
            "help": "Filter audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
    )
    min_duration_in_seconds: Optional[float] = field(
        default=3.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only do data preprocessing and skip training. "
            "This is especially useful when data preprocessing errors out in distributed training due to timeout. "
            "In this case, one should run the preprocessing in a non-distributed setup with `preprocessing_only=True` "
            "so that the cached datasets can consequently be loaded in distributed training"
        },
    )


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@flax.struct.dataclass
class FlaxDataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.FlaxWav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    model: FlaxWav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            max_length=self.max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="np",
        )
        # `input_length` is no longer needed
        batch.pop("input_length")

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        batch_size = batch["input_values"].shape[0]

        sub_attention_mask = None
        if batch["attention_mask"] is not None:
            output_lengths = self.model._get_feat_extract_output_lengths(batch["attention_mask"].sum(-1))
            sub_attention_mask = np.zeros((batch_size, mask_indices_seq_length), dtype=np.int8)

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            sub_attention_mask[(np.arange(sub_attention_mask.shape[0]), output_lengths - 1)] = 1
            sub_attention_mask = np.flip(np.flip(sub_attention_mask, -1).cumsum(-1), -1).astype("bool")

        # sample randomly masked indices
        batch["mask_time_indices"] = _compute_mask_indices(
            (batch_size, mask_indices_seq_length),
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=sub_attention_mask,
            min_masks=2,
        )

        # sample indices to take for negative vectors
        batch["sampled_negative_indices"] = _sample_negative_indices(
            batch["mask_time_indices"].shape,
            self.model.config.num_negatives,
            mask_time_indices=batch["mask_time_indices"],
        )

        # attach sub_attention_mask for logs
        batch["sub_attention_mask"] = sub_attention_mask

        return batch


def configure_logger(model_args: ModelArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    logger.setLevel(logging_level)


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def generate_batch_splits(samples_idx: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    num_samples = len(samples_idx)
    samples_to_remove = num_samples % batch_size

    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]
    sections_split = num_samples // batch_size
    batch_idx = np.split(samples_idx, sections_split)
    return batch_idx


def compute_contrastive_loss(
    quantized_features, transformer_features, negative_indices, mask_time_indices, logits_temp, num_negatives
):
    batch_size, sequence_length, hidden_size = quantized_features.shape

    # take negative vectors from sampled indices
    quantized_negatives = quantized_features.reshape(-1, hidden_size)[negative_indices.reshape(-1)]
    quantized_negatives = quantized_negatives.reshape(
        batch_size, sequence_length, num_negatives, hidden_size
    ).transpose(2, 0, 1, 3)

    target_features = jnp.concatenate([quantized_features[None, :], quantized_negatives], axis=0)
    loss_logits = optax.cosine_similarity(transformer_features, target_features, epsilon=1e-8)
    loss_logits = loss_logits / logits_temp

    neg_is_pos = jax.numpy.abs((quantized_negatives - quantized_features)).sum(-1) < 1e-2
    neg_is_pos = jnp.concatenate([jnp.full((1,) + loss_logits.shape[1:], False), neg_is_pos], axis=0)

    # make sure incorrectly sampled vectors don't contribute to loss
    loss_logits = jnp.where(neg_is_pos, -1e9, loss_logits)

    # => Shape batch_size*sequence_length x [1, num_negatives]
    predictions = loss_logits.transpose(2, 1, 0).reshape(-1, loss_logits.shape[0])
    target_mask = mask_time_indices.transpose(1, 0).flatten()

    contrastive_loss = -jax.nn.log_softmax(predictions)[:, 0] * target_mask
    contrastive_loss = contrastive_loss.sum()

    return contrastive_loss


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    configure_logger(model_args)

    # Downloading and loading a dataset from the hub.
    datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)

    if "validation" not in datasets.keys():
        # make sure only "validation" and "train" keys remain"
        raw_datasets = DatasetDict()
        raw_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"{data_args.train_split_name}[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
        )
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"{data_args.train_split_name}[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
        )
    else:
        # make sure only "validation" and "train" keys remain"
        raw_datasets = DatasetDict()
        raw_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="validation",
            cache_dir=model_args.cache_dir,
        )
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"{data_args.train_split_name}",
            cache_dir=model_args.cache_dir,
        )

    raw_datasets["train"] = raw_datasets["train"].select(range(1000))
    raw_datasets["validation"] = raw_datasets["validation"].select(range(100))

    # pretraining is only supported for "newer" stable layer norm architecture
    # apply_spec_augment has to be True, mask_feature_prob has to be 0.0
    config = Wav2Vec2Config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if not config.do_stable_layer_norm or config.feat_extract_norm != "layer":
        raise ValueError(
            "PreTraining is only supported for ``config.do_stable_layer_norm=True`` and ``config.feat_extract_norm='layer'"
        )

    # only normalized-inputs-training is supported
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, do_normalize=True
    )

    # make sure that dataset decodes audio with correct sampling rate
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    # derive max & min input length for sample rate & max duration
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column_name]

        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], truncate=True, max_length=max_input_length
        )
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])

        return batch

    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        num_proc=num_workers,
        desc="preprocess datasets",
    )

    # filter data that is shorter than min_input_length
    vectorized_datasets = vectorized_datasets.filter(
        lambda d: d > min_input_length,
        num_proc=num_workers,
        input_columns=["input_length"],
    )

    model = FlaxWav2Vec2ForPreTraining(config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype))

    data_collator = FlaxDataCollatorForWav2Vec2Pretraining(
        model=model, feature_extractor=feature_extractor, pad_to_multiple_of=data_args.pad_to_multiple_of
    )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())
    gumbel_rngs = jax.random.split(rng, jax.local_device_count())

    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()

    num_train_steps = len(vectorized_datasets["train"]) // train_batch_size * num_epochs

    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=training_args.learning_rate, transition_steps=training_args.warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=training_args.learning_rate,
        end_value=0,
        transition_steps=num_train_steps - training_args.warmup_steps,
    )
    linear_decay_lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[training_args.warmup_steps]
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in [("layer_norm", "scale"), ("final_layer_norm", "scale")])
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )

    # Setup train state and define training hyper-parameters
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)
    num_negatives = model.config.num_negatives
    contrastive_logits_temperature = model.config.contrastive_logits_temperature
    num_codevectors = model.config.num_codevectors_per_group * model.config.num_codevector_groups
    diversity_loss_weight = model.config.diversity_loss_weight

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng, gumbel_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        gumbel_rng, new_gumbel_rng = jax.random.split(gumbel_rng)

        def loss_fn(params):
            negative_indices = batch.pop("sampled_negative_indices")
            sample_size = batch["mask_time_indices"].sum()
            num_tokens = batch.pop("sub_attention_mask").sum()

            gumbel_temperature = jnp.clip(
                model_args.max_gumbel_temperature * (model_args.gumbel_temperature_decay ** state.step),
                a_min=model_args.min_gumbel_temperature,
            )

            outputs = state.apply_fn(
                **batch,
                gumbel_temperature=gumbel_temperature,
                params=params,
                dropout_rng=dropout_rng,
                gumbel_rng=gumbel_rng,
                train=True,
            )

            contrastive_loss, logs = compute_contrastive_loss(
                outputs.projected_quantized_states,
                outputs.projected_states,
                negative_indices,
                batch["mask_time_indices"],
                contrastive_logits_temperature,
                num_negatives,
            )

            # higher codevector_perplexity leads to lower diversity loss
            diversity_loss = (num_codevectors - outputs.codevector_perplexity) / num_codevectors * sample_size

            loss = contrastive_loss + diversity_loss_weight * diversity_loss

            # add more metrics
            logs["code_ppl"] = outputs.codevector_perplexity * sample_size
            logs["contrast_loss"] = contrastive_loss
            logs["diversity_loss"] = diversity_loss
            logs["sample_size"] = sample_size
            logs["num_input_tokens"] = num_tokens

            return loss, logs

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logs), grad = grad_fn(state.params)

        grad = jax.lax.psum(grad, "batch")
        logs = jax.lax.psum(logs)

        logs_2 = jax.tree_map(jax.lax.psum, logs)

        total_sample_size = logs.pop("sample_size")
        total_num_input_tokens = logs.pop("num_input_tokens")

        # average gradients
        grad = jax.tree_map(lambda g: g / logs["sample_size"], grad)

        # average log values
        logs = jax.tree_map(lambda v: v / logs["sample_size"], logs)

        # compute gradient norm for monitoring
        grad_norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grad)))

        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss, "learn_rate": linear_decay_lr_schedule_fn(state.step), "grad_norm": grad_norm}
        metrics.update(logs)

        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics, new_dropout_rng, new_gumbel_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Define eval fn
    def eval_step(params, batch):
        negative_indices = batch.pop("sampled_negative_indices")

        outputs = model(**batch, params=params, train=False)

        contrastive_loss = compute_contrastive_loss(
            outputs.projected_quantized_states,
            outputs.projected_states,
            negative_indices,
            batch["mask_time_indices"],
            contrastive_logits_temperature,
            num_negatives,
        )

        diversity_loss = (num_codevectors - outputs.codevector_perplexity) / num_codevectors
        loss = contrastive_loss + diversity_loss_weight * diversity_loss

        # summarize metrics
        metrics = {"loss": loss.mean(), "codevector_perplexity": outputs.codevector_perplexity}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return metrics

    p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)

    train_time = 0
    train_metrics = []
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        num_train_samples = len(vectorized_datasets["train"])
        train_samples_idx = jax.random.permutation(input_rng, jnp.arange(num_train_samples))
        train_batch_idx = generate_batch_splits(train_samples_idx, train_batch_size)

        # Gather the indexes for creating the batch and do a training step
        for step, batch_idx in enumerate(tqdm(train_batch_idx, desc="Training...", position=1)):
            samples = [vectorized_datasets["train"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples)
            model_inputs = shard(model_inputs.data)

            # Model forward
            state, train_metric, dropout_rngs, gumbel_rngs = p_train_step(
                state, model_inputs, dropout_rngs, gumbel_rngs
            )
            train_metrics.append(train_metric)

            cur_step = epoch * (num_train_samples // train_batch_size) + step

        #            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
        # Save metrics
        #                train_metric = jax_utils.unreplicate(train_metric)
        #                train_time += time.time() - train_start
        #                if has_tensorboard and jax.process_index() == 0:
        #                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)
        #
        #                epochs.write(
        #                    f"Step... ({cur_step} | Loss: {train_metric['loss'].mean()}, Learning Rate: {train_metric['learning_rate'].mean()})"
        #                )

        #                train_metrics = []

        continue
        # TODO(PVP) Touch this later

        # ======================== Evaluating ==============================
        num_eval_samples = len(vectorized_datasets["validation"])
        eval_samples_idx = jnp.arange(num_eval_samples)
        eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size)

        eval_metrics = []
        for i, batch_idx in enumerate(tqdm(eval_batch_idx, desc="Evaluating ...", position=2)):
            samples = [vectorized_datasets["validation"][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples)

            # Model forward
            model_inputs = shard(model_inputs.data)
            metrics = p_eval_step(state.params, model_inputs)
            eval_metrics.append(metrics)

        # get eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

        # Update progress bar
        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {eval_metrics['loss']}, Perplexity: {eval_metrics['codevector_perplexity']})"
        )

        # Save metrics
        if has_tensorboard and jax.process_index() == 0:
            cur_step = epoch * (len(vectorized_datasets["train"]) // train_batch_size)
            write_eval_metric(summary_writer, eval_metrics, cur_step)

        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0:
            params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
            model.save_pretrained(training_args.output_dir, params=params, push_to_hub=training_args.push_to_hub)


if __name__ == "__main__":
    main()
