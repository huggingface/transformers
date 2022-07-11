#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""

import logging
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import datasets
import numpy as np
import tensorflow as tf
from datasets import ClassLabel, load_dataset, load_metric

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TFAutoModelForTokenClassification,
    TFTrainingArguments,
    create_optimizer,
    set_seed,
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# region Command-line arguments
@dataclass
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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_length: Optional[int] = field(default=256, metadata={"help": "Max length (in tokens) for truncation/padding"})
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
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
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
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
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


# endregion


# region Data generator
def sample_generator(dataset, tokenizer, shuffle, pad_to_multiple_of=None):
    # Trim off the last partial batch if present
    if shuffle:
        sample_ordering = np.random.permutation(len(dataset))
    else:
        sample_ordering = np.arange(len(dataset))
    for sample_idx in sample_ordering:
        example = dataset[int(sample_idx)]
        # Handle dicts with proper padding and conversion to tensor.
        example = tokenizer.pad(example, return_tensors="np", pad_to_multiple_of=pad_to_multiple_of)
        if tokenizer.pad_token_id is not None:
            example["labels"][example["attention_mask"] == 0] = -100
        example = {key: tf.convert_to_tensor(arr) for key, arr in example.items()}

        yield example, example["labels"]  # TF needs some kind of labels, even if we don't use them
    return


# endregion


# region Helper functions
def dataset_to_tf(dataset, tokenizer, total_batch_size, num_epochs, shuffle):
    train_generator = partial(sample_generator, dataset, tokenizer, shuffle=shuffle)
    train_signature = {
        feature: tf.TensorSpec(shape=(None,), dtype=tf.int64)
        for feature in dataset.features
        if feature != "special_tokens_mask"
    }
    # This may need to be changed depending on your particular model or tokenizer!
    padding_values = {key: tf.convert_to_tensor(0, dtype=tf.int64) for key in dataset.features}
    padding_values["labels"] = tf.convert_to_tensor(-100, dtype=tf.int64)
    if tokenizer.pad_token_id is not None:
        padding_values["input_ids"] = tf.convert_to_tensor(tokenizer.pad_token_id, dtype=tf.int64)
    train_signature["labels"] = train_signature["input_ids"]
    train_signature = (train_signature, train_signature["labels"])
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    tf_dataset = (
        tf.data.Dataset.from_generator(train_generator, output_signature=train_signature)
        .with_options(options)
        .padded_batch(
            batch_size=total_batch_size,
            drop_remainder=True,
            padding_values=(padding_values, np.array(0, dtype=np.int64)),
        )
        .repeat(int(num_epochs))
    )
    return tf_dataset


# endregion


def main():
    # region Argument Parsing
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_ner", model_args, data_args, framework="tensorflow")
    # endregion

    # region Setup logging
    # we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)
    # endregion

    # region Loading datasets
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)
    # endregion

    # region Load config and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, num_labels=num_labels)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    # endregion

    # region Preprocessing the raw datasets
    # First we tokenize all the texts.
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=data_args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # endregion

    with training_args.strategy.scope():
        # region Initialize model
        if model_args.model_name_or_path:
            model = TFAutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
            )
        else:
            logger.info("Training new model from scratch")
            model = TFAutoModelForTokenClassification.from_config(config)

        model.resize_token_embeddings(len(tokenizer))
        # endregion

        # region Create TF datasets
        num_replicas = training_args.strategy.num_replicas_in_sync
        total_train_batch_size = training_args.per_device_train_batch_size * num_replicas
        train_batches_per_epoch = len(train_dataset) // total_train_batch_size
        tf_train_dataset = dataset_to_tf(
            train_dataset,
            tokenizer,
            total_batch_size=total_train_batch_size,
            num_epochs=training_args.num_train_epochs,
            shuffle=True,
        )
        total_eval_batch_size = training_args.per_device_eval_batch_size * num_replicas
        eval_batches_per_epoch = len(eval_dataset) // total_eval_batch_size
        tf_eval_dataset = dataset_to_tf(
            eval_dataset,
            tokenizer,
            total_batch_size=total_eval_batch_size,
            num_epochs=training_args.num_train_epochs,
            shuffle=False,
        )

        # endregion

        # region Optimizer, loss and compilation
        optimizer, lr_schedule = create_optimizer(
            init_lr=training_args.learning_rate,
            num_train_steps=int(training_args.num_train_epochs * train_batches_per_epoch),
            num_warmup_steps=training_args.warmup_steps,
            adam_beta1=training_args.adam_beta1,
            adam_beta2=training_args.adam_beta2,
            adam_epsilon=training_args.adam_epsilon,
            weight_decay_rate=training_args.weight_decay,
        )

        def dummy_loss(y_true, y_pred):
            return tf.reduce_mean(y_pred)

        model.compile(loss={"loss": dummy_loss}, optimizer=optimizer)
        # endregion

        # Metrics
        metric = load_metric("seqeval")

        def get_labels(y_pred, y_true):
            # Transform predictions and references tensos to numpy arrays

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
                for pred, gold_label in zip(y_pred, y_true)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
                for pred, gold_label in zip(y_pred, y_true)
            ]
            return true_predictions, true_labels

        def compute_metrics():
            results = metric.compute()
            if data_args.return_entity_level_metrics:
                # Unpack nested dictionaries
                final_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        for n, v in value.items():
                            final_results[f"{key}_{n}"] = v
                    else:
                        final_results[key] = value
                return final_results
            else:
                return {
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    "accuracy": results["overall_accuracy"],
                }

        # endregion

        # region Training
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size = {total_train_batch_size}")
        # Only show the progress bar once on each machine.
        model.fit(
            tf_train_dataset,
            validation_data=tf_eval_dataset,
            epochs=int(training_args.num_train_epochs),
            steps_per_epoch=train_batches_per_epoch,
            validation_steps=eval_batches_per_epoch,
        )
        # endregion

        # region Predictions
        # For predictions, we preload the entire validation set - note that if you have a really giant validation
        # set, you might need to change this!
        eval_inputs = {key: tf.ragged.constant(eval_dataset[key]).to_tensor() for key in eval_dataset.features}
        predictions = model.predict(eval_inputs, batch_size=training_args.per_device_eval_batch_size)["logits"]
        predictions = tf.math.argmax(predictions, axis=-1)
        labels = np.array(eval_inputs["labels"])
        labels[np.array(eval_inputs["attention_mask"]) == 0] = -100
        preds, refs = get_labels(predictions, labels)
        metric.add_batch(
            predictions=preds,
            references=refs,
        )
        eval_metric = compute_metrics()
        logger.info("Evaluation metrics:")
        for key, val in eval_metric.items():
            logger.info(f"{key}: {val:.4f}")
        # endregion

    # We don't do predictions in the strategy scope because there are some issues in there right now.
    # They'll get fixed eventually, promise!

    if training_args.output_dir is not None:
        model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
