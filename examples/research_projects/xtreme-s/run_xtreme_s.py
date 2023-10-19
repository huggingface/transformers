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

""" Fine-tuning a 🤗 Transformers pretrained speech model on the XTREME-S benchmark tasks"""

import json
import logging
import os
import re
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import datasets
import numpy as np
import torch
from datasets import DatasetDict, load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.18.0.dev0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")


logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


TASK_TO_TARGET_COLUMN_NAME = {
    "fleurs-asr": "transcription",
    "fleurs-lang_id": "lang_id",
    "mls": "transcription",
    "voxpopuli": "transcription",
    "covost2": "translation",
    "minds14": "intent_class",
    "babel": "transcription",
}


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
        metadata={
            "help": "Where do you want to store the pretrained models and datasets downloaded from huggingface.co"
        },
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
    ctc_zero_infinity: bool = field(
        default=False,
        metadata={"help": "Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`."},
    )
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
        default="google/xtreme_s",
        metadata={"help": "The name of the dataset to use (via the datasets library). Defaults to 'google/xtreme_s'"},
    )
    task: str = field(
        default=None,
        metadata={
            "help": (
                "The task name of the benchmark to use (via the datasets library). Should be on of: "
                "'fleurs-asr', 'mls', 'voxpopuli', 'covost2', 'minds14', 'fleurs-lang_id', 'babel'."
            )
        },
    )
    language: str = field(
        default="all",
        metadata={"help": "The language id as defined in the datasets config name or `all` for all languages."},
    )
    language_group: str = field(
        default=None,
        metadata={
            "help": (
                "The language group to select a subset of languages to train on. "
                "This option is only used the 'fleurs-asr' task. Should be one of: "
                "'western_european_we', 'eastern_european_ee', 'central_asia_middle_north_african_cmn', "
                "'sub_saharan_african_ssa', 'south_asian_sa', 'south_east_asian_sea', 'chinese_japanase_korean_cjk'."
            )
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training dataset split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the evaluation dataset split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    predict_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the prediction dataset split to use (via the datasets library). Defaults to 'test'"
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    target_column_name: str = field(
        default=None,
        metadata={
            "help": (
                "The name of the dataset column containing the target data (transcription/translation/label). If None,"
                " the name will be inferred from the task. Defaults to None."
            )
        },
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    chars_to_ignore: Optional[List[str]] = list_field(
        default=', ? . ! - ; : " “ % ‘ ” �'.split(" "),
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={
            "help": (
                "Filter audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
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
    unk_token: str = field(
        default="[UNK]",
        metadata={"help": "The unk token for the tokenizer"},
    )
    pad_token: str = field(
        default="[PAD]",
        metadata={"help": "The padding token for the tokenizer"},
    )
    word_delimiter_token: str = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer"},
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
    per_lang_metrics: bool = field(
        default=True,
        metadata={
            "help": (
                "If `True`, compute the test metrics separately for each language, and average the results. "
                "If `False` compute the average test metrics in a single pass for all languages at once."
            )
        },
    )


@dataclass
class SpeechDataCollatorWithPadding:
    processor: AutoProcessor
    decoder_start_token_id: Optional[int] = None
    padding: Union[bool, str] = "longest"
    pad_labels: Optional[int] = True
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if self.pad_labels:
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.pad(
                labels=label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (
                self.decoder_start_token_id is not None
                and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item()
            ):
                labels = labels[:, 1:]

            batch["labels"] = labels
        else:
            batch["labels"] = torch.tensor([feature["labels"] for feature in features])

        return batch


def create_vocabulary_from_data(
    datasets: DatasetDict,
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        all_text = " ".join(batch["target_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets["train"].column_names,
    )

    # take union of all unique characters in each dataset
    vocab_set = (
        (set(vocabs["train"]["vocab"][0]) if "train" in vocabs else set())
        | (set(vocabs["eval"]["vocab"][0]) if "eval" in vocabs else set())
        | (set(vocabs["predict"]["vocab"][0]) if "predict" in vocabs else set())
    )

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_set))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
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
    raw_datasets = DatasetDict()
    task_name = data_args.task
    lang_id = data_args.language

    if task_name is None:
        raise ValueError(
            "Set --task should be set to '<xtreme_s_task>' (e.g. 'fleurs-asr', 'mls', 'covost2', 'minds14') "
        )
    if lang_id is None:
        raise ValueError(
            "Set --language should be set to the language id of the sub dataset "
            "config to be used (e.g. 'pl', 'en.tr', 'fr-FR') or 'all'"
            " for multi-lingual fine-tuning."
        )
    if data_args.language_group is not None:
        if data_args.task != "fleurs-asr":
            raise ValueError("--language_group should only be used with --task=fleurs-asr")
        if data_args.language != "all":
            raise ValueError("--language_group should only be used with --language=all")

    if data_args.target_column_name is None:
        target_column_name = TASK_TO_TARGET_COLUMN_NAME[task_name]
    else:
        target_column_name = data_args.target_column_name

    # here we differentiate between tasks with text as the target and classification tasks
    is_text_target = target_column_name in ("transcription", "translation")

    config_name = ".".join([task_name.split("-")[0], lang_id])

    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            config_name,
            split=data_args.train_split_name,
            use_auth_token=data_args.use_auth_token,
            cache_dir=model_args.cache_dir,
        )

        if data_args.audio_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'."
                " Make sure to set `--audio_column_name` to the correct audio column - one of"
                f" {', '.join(raw_datasets['train'].column_names)}."
            )

        if target_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--target_column_name {target_column_name} not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--target_column_name` to the correct text column - one of "
                f"{', '.join(raw_datasets['train'].column_names)}."
            )

        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            data_args.dataset_name,
            config_name,
            split=data_args.eval_split_name,
            use_auth_token=data_args.use_auth_token,
            cache_dir=model_args.cache_dir,
        )

        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        raw_datasets["predict"] = load_dataset(
            data_args.dataset_name,
            config_name,
            split=data_args.predict_split_name,
            use_auth_token=data_args.use_auth_token,
            cache_dir=model_args.cache_dir,
        )

        if data_args.max_predict_samples is not None:
            raw_datasets["predict"] = raw_datasets["predict"].select(range(data_args.max_predict_samples))

    lang_list = next(iter(raw_datasets.values())).features["lang_id"].names
    if not is_text_target:
        label_list = next(iter(raw_datasets.values())).features[target_column_name].names
        num_labels = len(label_list)

    num_workers = data_args.preprocessing_num_workers

    lang_group = data_args.language_group
    if lang_group is not None:
        with training_args.main_process_first(desc="language group filter"):
            lang_group_id = next(iter(raw_datasets.values())).features["lang_group_id"].str2int(lang_group)
            raw_datasets = raw_datasets.filter(
                lambda lang_group: lang_group == lang_group_id,
                num_proc=num_workers,
                input_columns=["lang_group_id"],
            )

    # 2. We remove some special characters from the datasets
    # that make training complicated and do not help in transcribing the speech
    # E.g. characters, such as `,` and `.` do not really have an acoustic characteristic
    # that could be easily picked up by the model
    chars_to_ignore_regex = (
        f'[{"".join(data_args.chars_to_ignore)}]' if data_args.chars_to_ignore is not None else None
    )

    def remove_special_characters(batch):
        if chars_to_ignore_regex is not None:
            batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[target_column_name]).lower() + " "
        else:
            batch["target_text"] = batch[target_column_name].lower() + " "
        return batch

    if is_text_target:
        with training_args.main_process_first(desc="dataset map special characters removal"):
            raw_datasets = raw_datasets.map(
                remove_special_characters,
                remove_columns=[target_column_name],
                desc="remove special characters from datasets",
            )

        # save special tokens for tokenizer
        word_delimiter_token = data_args.word_delimiter_token
        unk_token = data_args.unk_token
        pad_token = data_args.pad_token

    # 3. Next, let's load the config as we might need it to create
    # the tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_auth_token=data_args.use_auth_token
    )

    if is_text_target:
        # 4. (Optional, for ASR and translation) If no tokenizer file is defined,
        # we create the vocabulary of the model by extracting all unique characters from
        # the training and evaluation datasets
        # We need to make sure that only first rank saves vocabulary
        # make sure all processes wait until vocab is created
        tokenizer_name_or_path = model_args.tokenizer_name_or_path
        tokenizer_kwargs = {}
        if tokenizer_name_or_path is None:
            # save vocab in training output dir
            tokenizer_name_or_path = training_args.output_dir

            vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")

            with training_args.main_process_first():
                if training_args.overwrite_output_dir and os.path.isfile(vocab_file):
                    os.remove(vocab_file)

            with training_args.main_process_first(desc="dataset map vocabulary creation"):
                if not os.path.isfile(vocab_file):
                    os.makedirs(tokenizer_name_or_path, exist_ok=True)
                    vocab_dict = create_vocabulary_from_data(
                        raw_datasets,
                        word_delimiter_token=word_delimiter_token,
                        unk_token=unk_token,
                        pad_token=pad_token,
                    )

                    # save vocab dict to be loaded into tokenizer
                    with open(vocab_file, "w") as file:
                        json.dump(vocab_dict, file)

            # if tokenizer has just been created
            # it is defined by `tokenizer_class` if present in config else by `model_type`
            if not config.is_encoder_decoder:
                tokenizer_kwargs = {
                    "config": config if config.tokenizer_class is not None else None,
                    "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
                    "unk_token": unk_token,
                    "pad_token": pad_token,
                    "word_delimiter_token": word_delimiter_token,
                }
            else:
                tokenizer_kwargs = {}

    # 5. Now we can instantiate the feature extractor, tokenizer and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load feature_extractor and tokenizer
    if is_text_target:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_auth_token=data_args.use_auth_token,
            **tokenizer_kwargs,
        )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_auth_token=data_args.use_auth_token
    )

    # adapt config
    # (speech translation requires pre-configured seq2seq models)
    if task_name != "covost2":
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
                "ctc_zero_infinity": model_args.ctc_zero_infinity,
                "ctc_loss_reduction": model_args.ctc_loss_reduction,
                "activation_dropout": model_args.activation_dropout,
            }
        )
        if training_args.do_train:
            if is_text_target:
                config.pad_token_id = tokenizer.pad_token_id
                config.vocab_size = len(tokenizer)
            else:
                label_to_id = {v: i for i, v in enumerate(label_list)}
                config.label2id = label_to_id
                config.id2label = {id: label for label, id in label_to_id.items()}
                config.num_labels = num_labels

    # create model
    if target_column_name == "transcription":
        model = AutoModelForCTC.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            config=config,
            use_auth_token=data_args.use_auth_token,
        )
    elif config.is_encoder_decoder:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            config=config,
            use_auth_token=data_args.use_auth_token,
        )
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    else:
        model = AutoModelForAudioClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            config=config,
            use_auth_token=data_args.use_auth_token,
        )

    # freeze encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # 6. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

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

    # `phoneme_language` is only relevant if the model is fine-tuned on phoneme classification
    phoneme_language = data_args.phoneme_language

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column_name]

        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch["input_values"] = inputs.input_values[0]
        batch["length"] = len(batch["input_values"])

        # encode targets
        additional_kwargs = {}
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language

        if is_text_target:
            batch["labels"] = tokenizer(batch["target_text"], **additional_kwargs).input_ids
        else:
            batch["labels"] = batch[target_column_name]

        batch["lang"] = batch["lang_id"]

        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=num_workers,
            desc="preprocess datasets",
        )

        if training_args.do_train:

            def is_audio_in_length_range(length):
                return length > min_input_length and length < max_input_length

            # filter data that is shorter than min_input_length
            vectorized_datasets["train"] = vectorized_datasets["train"].filter(
                is_audio_in_length_range,
                num_proc=num_workers,
                input_columns=["length"],
            )

    # 7. Next, we can prepare for the training step.
    # Let's use the appropriate XTREME-S evaluation metric,
    # instantiate a data collator and the trainer

    # Define evaluation metrics during training, *i.e.* word error rate, character error rate
    eval_metric = load_metric("xtreme_s", task_name)

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return

    def asr_logits_argmax(logits, labels):
        return logits.argmax(dim=-1)

    def compute_asr_metric(pred):
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred.predictions)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        metric = eval_metric.compute(predictions=pred_str, references=label_str)
        return metric

    def compute_classification_metric(pred):
        pred_ids = np.argmax(pred.predictions, axis=1)
        metric = eval_metric.compute(predictions=pred_ids, references=pred.label_ids)
        return metric

    # Now save everything to be able to create a single processor later
    if is_main_process(training_args.local_rank):
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(training_args.output_dir)
        if is_text_target:
            tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)
    # wait until configs are saved in the main process before loading the processor
    if training_args.local_rank != -1:
        torch.distributed.barrier()

    if is_text_target:
        processor = AutoProcessor.from_pretrained(training_args.output_dir)
    else:
        processor = AutoFeatureExtractor.from_pretrained(training_args.output_dir)

    # Instantiate custom data collator
    data_collator = SpeechDataCollatorWithPadding(processor=processor, pad_labels=is_text_target)

    # Initialize Trainer
    if target_column_name == "translation":
        trainer = Seq2SeqTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            preprocess_logits_for_metrics=asr_logits_argmax if training_args.predict_with_generate else None,
            compute_metrics=compute_asr_metric if training_args.predict_with_generate else None,
            train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
            eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
            tokenizer=feature_extractor,
        )
    else:
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            preprocess_logits_for_metrics=asr_logits_argmax if is_text_target else None,
            compute_metrics=compute_asr_metric if is_text_target else compute_classification_metric,
            train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
            eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
            tokenizer=feature_extractor,
        )

    # 8. Finally, we can start training

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
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation on the test set
    results = {}
    if training_args.do_predict:
        logger.info(f"*** Evaluating on the `{data_args.predict_split_name}` set ***")
        if data_args.per_lang_metrics:
            # separate the `test` dataset into language-specific subsets and compute metrics for each of them
            metrics = {}
            average_metrics = defaultdict(list)
            for lang_id in range(len(lang_list)):
                lang_name = lang_list[lang_id]
                with training_args.main_process_first(desc="per-language dataset filter"):
                    lang_dataset = vectorized_datasets["predict"].filter(
                        lambda lang: lang == lang_id,
                        num_proc=num_workers,
                        input_columns=["lang"],
                    )
                lang_metrics = trainer.evaluate(lang_dataset)
                redundant_metrics = ["eval_runtime", "eval_samples_per_second", "eval_steps_per_second", "eval_epoch"]
                for metric_name, value in lang_metrics.items():
                    average_metrics[metric_name].append(value)
                    if metric_name not in redundant_metrics:
                        metrics[f"{metric_name}_{lang_name}"] = value
            for metric_name, value in average_metrics.items():
                metrics[metric_name] = np.mean(value)
        else:
            metrics = trainer.evaluate(vectorized_datasets["predict"])
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(vectorized_datasets["predict"])
        )
        metrics["predict_samples"] = min(max_predict_samples, len(vectorized_datasets["predict"]))

        # make sure that the `predict` metrics end up in the log history for the model card
        trainer.log(OrderedDict(sorted(metrics.items())))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": task_name,
        "tags": [task_name, data_args.dataset_name],
        "dataset_args": (
            f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split:"
            f" {data_args.eval_split_name}, Predict split: {data_args.predict_split_name}"
        ),
        "dataset": f"{data_args.dataset_name.upper()} - {config_name.upper()}",
        "language": data_args.language,
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()
