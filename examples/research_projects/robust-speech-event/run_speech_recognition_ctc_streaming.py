#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

""" Fine-tuning a 🤗 Transformers CTC model for automatic speech recognition in streaming mode"""

import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import datasets
import numpy as np
import torch
from datasets import IterableDatasetDict, interleave_datasets, load_dataset, load_metric
from torch.utils.data import IterableDataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2Processor,
    set_seed,
)
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risk.
check_min_version("4.17.0.dev0")

require_version("datasets>=1.18.2", "To fix: pip install 'datasets>=1.18.2'")


logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    attention_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    feat_proj_dropout: float = field(default=0.0, metadata={"help": "The dropout ratio for the projected features."})
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector"
                "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
                "vectors will be masked along the time axis."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": (
                "Probability of each feature vector along the feature axis to be chosen as the start of the vectorspan"
                " to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature"
                " bins will be masked along the time axis."
            )
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: Optional[str] = field(
        default="mean", metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: str = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: str = field(
        default="train+validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to "
                "'train+validation'"
            )
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'test'"
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
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    shuffle_buffer_size: Optional[int] = field(
        default=500,
        metadata={
            "help": (
                "The number of streamed examples to download before shuffling them. The large the buffer, "
                "the closer it is to real offline shuffling."
            )
        },
    )
    chars_to_ignore: Optional[List[str]] = list_field(
        default=None,
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    eval_metrics: List[str] = list_field(
        default=["wer"],
        metadata={"help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds."},
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "If :obj:`True`, will use the token generated when running"
                ":obj:`huggingface-cli login` as HTTP bearer authorization for remote files."
            )
        },
    )
    phoneme_language: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The target language that should be used be"
                " passed to the tokenizer for tokenization. Note that"
                " this is only relevant if the model classifies the"
                " input audio to a sequence of phoneme sequences."
            )
        },
    )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
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
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = []
        label_features = []
        for feature in features:
            if self.max_length and feature["input_values"].shape[-1] > self.max_length:
                continue
            input_features.append({"input_values": feature["input_values"]})
            label_features.append({"input_ids": feature["labels"]})

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


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

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 1. First, let's load the dataset
    raw_datasets = IterableDatasetDict()
    raw_column_names = {}

    def load_streaming_dataset(split, sampling_rate, **kwargs):
        if "+" in split:
            dataset_splits = [load_dataset(split=split_name, **kwargs) for split_name in split.split("+")]
            # `features` and `cast_column` won't be available after interleaving, so we'll use them here
            features = dataset_splits[0].features
            # make sure that the dataset decodes audio with a correct sampling rate
            dataset_splits = [
                dataset.cast_column(data_args.audio_column_name, datasets.features.Audio(sampling_rate=sampling_rate))
                for dataset in dataset_splits
            ]

            interleaved_dataset = interleave_datasets(dataset_splits)
            return interleaved_dataset, features
        else:
            dataset = load_dataset(split=split, **kwargs)
            features = dataset.features
            # make sure that the dataset decodes audio with a correct sampling rate
            dataset = dataset.cast_column(
                data_args.audio_column_name, datasets.features.Audio(sampling_rate=sampling_rate)
            )
            return dataset, features

    # `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_auth_token=data_args.use_auth_token
    )

    if training_args.do_train:
        raw_datasets["train"], train_features = load_streaming_dataset(
            path=data_args.dataset_name,
            name=data_args.dataset_config_name,
            split=data_args.train_split_name,
            use_auth_token=data_args.use_auth_token,
            streaming=True,
            sampling_rate=feature_extractor.sampling_rate,
        )
        raw_column_names["train"] = list(train_features.keys())

        if data_args.audio_column_name not in raw_column_names["train"]:
            raise ValueError(
                f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'."
                " Make sure to set `--audio_column_name` to the correct audio column - one of"
                f" {', '.join(raw_column_names['train'])}."
            )

        if data_args.text_column_name not in raw_column_names["train"]:
            raise ValueError(
                f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--text_column_name` to the correct text column - one of "
                f"{', '.join(raw_column_names['train'])}."
            )

        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].take(range(data_args.max_train_samples))

    if training_args.do_eval:
        raw_datasets["eval"], eval_features = load_streaming_dataset(
            path=data_args.dataset_name,
            name=data_args.dataset_config_name,
            split=data_args.eval_split_name,
            use_auth_token=data_args.use_auth_token,
            streaming=True,
            sampling_rate=feature_extractor.sampling_rate,
        )
        raw_column_names["eval"] = list(eval_features.keys())

        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = raw_datasets["eval"].take(range(data_args.max_eval_samples))

    # 2. We remove some special characters from the datasets
    # that make training complicated and do not help in transcribing the speech
    # E.g. characters, such as `,` and `.` do not really have an acoustic characteristic
    # that could be easily picked up by the model
    chars_to_ignore_regex = (
        f'[{"".join(data_args.chars_to_ignore)}]' if data_args.chars_to_ignore is not None else None
    )
    text_column_name = data_args.text_column_name

    def remove_special_characters(batch):
        if chars_to_ignore_regex is not None:
            batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[text_column_name]).lower() + " "
        else:
            batch["target_text"] = batch[text_column_name].lower() + " "
        return batch

    with training_args.main_process_first(desc="dataset map special characters removal"):
        for split, dataset in raw_datasets.items():
            raw_datasets[split] = dataset.map(
                remove_special_characters,
            ).remove_columns([text_column_name])

    # 3. Next, let's load the config as we might need it to create
    # the tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_auth_token=data_args.use_auth_token
    )

    # 4. Now we can instantiate the tokenizer and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if tokenizer_name_or_path is None:
        raise ValueError(
            "Tokenizer has to be created before training in streaming mode. Please specify --tokenizer_name_or_path"
        )
    # load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        config=config,
        use_auth_token=data_args.use_auth_token,
    )

    # adapt config
    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_args.activation_dropout,
        }
    )

    # create model
    model = AutoModelForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        use_auth_token=data_args.use_auth_token,
    )

    # freeze encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # 5. Now we preprocess the datasets including loading the audio, resampling and normalization
    audio_column_name = data_args.audio_column_name

    # `phoneme_language` is only relevant if the model is fine-tuned on phoneme classification
    phoneme_language = data_args.phoneme_language

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column_name]

        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])

        # encode targets
        additional_kwargs = {}
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language

        batch["labels"] = tokenizer(batch["target_text"], **additional_kwargs).input_ids
        return batch

    vectorized_datasets = IterableDatasetDict()
    with training_args.main_process_first(desc="dataset map preprocessing"):
        for split, dataset in raw_datasets.items():
            vectorized_datasets[split] = (
                dataset.map(prepare_dataset)
                .remove_columns(raw_column_names[split] + ["target_text"])
                .with_format("torch")
            )
            if split == "train":
                vectorized_datasets[split] = vectorized_datasets[split].shuffle(
                    buffer_size=data_args.shuffle_buffer_size,
                    seed=training_args.seed,
                )

    # 6. Next, we can prepare the training.
    # Let's use word error rate (WER) as our evaluation metric,
    # instantiate a data collator and the trainer

    # Define evaluation metrics during training, *i.e.* word error rate, character error rate
    eval_metrics = {metric: load_metric(metric) for metric in data_args.eval_metrics}

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

        return metrics

    # Now save everything to be able to create a single processor later
    if is_main_process(training_args.local_rank):
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)

    try:
        processor = AutoProcessor.from_pretrained(training_args.output_dir)
    except (OSError, KeyError):
        warnings.warn(
            "Loading a processor from a feature extractor config that does not"
            " include a `processor_class` attribute is deprecated and will be removed in v5. Please add the following "
            " attribute to your `preprocessor_config.json` file to suppress this warning: "
            " `'processor_class': 'Wav2Vec2Processor'`",
            FutureWarning,
        )
        processor = Wav2Vec2Processor.from_pretrained(training_args.output_dir)

    # Instantiate custom data collator
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    data_collator = DataCollatorCTCWithPadding(processor=processor, max_length=max_input_length)

    # trainer callback to reinitialize and reshuffle the streamable datasets at the beginning of each epoch
    class ShuffleCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
            if isinstance(train_dataloader.dataset, IterableDatasetShard):
                pass  # set_epoch() is handled by the Trainer
            elif isinstance(train_dataloader.dataset, IterableDataset):
                train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        tokenizer=processor,
        callbacks=[ShuffleCallback()],
    )

    # 7. Finally, we can start training

    # Training
    if training_args.do_train:

        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        if data_args.max_train_samples:
            metrics["train_samples"] = data_args.max_train_samples

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        if data_args.max_eval_samples:
            metrics["eval_samples"] = data_args.max_eval_samples

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    config_name = data_args.dataset_config_name if data_args.dataset_config_name is not None else "na"
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "speech-recognition",
        "tags": ["automatic-speech-recognition", data_args.dataset_name],
        "dataset_args": (
            f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split:"
            f" {data_args.eval_split_name}"
        ),
        "dataset": f"{data_args.dataset_name.upper()} - {config_name.upper()}",
    }
    if "common_voice" in data_args.dataset_name:
        kwargs["language"] = config_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()
