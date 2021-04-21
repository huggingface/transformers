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
""" Fine-tuning the library models for sequence classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PretrainedConfig,
    TFAutoModelForSequenceClassification,
    TrainingArguments,
    set_seed,
)
from transformers.file_utils import CONFIG_NAME, TF2_WEIGHTS_NAME


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Reduce the amount of console output from TF
import tensorflow as tf  # noqa: E402


logger = logging.getLogger(__name__)


# region Helper classes
class DataSequence(tf.keras.utils.Sequence):
    # We use a Sequence object to load the data. Although it's completely possible to load your data as Numpy/TF arrays
    # and pass those straight to the Model, this constrains you in a couple of ways. Most notably, it requires all
    # the data to be padded to the length of the longest input example, and it also requires the whole dataset to be
    # loaded into memory. If these aren't major problems for you, you can skip the sequence object in your own code!
    def __init__(self, dataset, non_label_column_names, batch_size, labels, shuffle=True):
        super().__init__()
        # Retain all of the columns not present in the original data - these are the ones added by the tokenizer
        self.data = {
            key: dataset[key]
            for key in dataset.features.keys()
            if key not in non_label_column_names and key != "label"
        }
        data_lengths = {len(array) for array in self.data.values()}
        assert len(data_lengths) == 1, "Dataset arrays differ in length!"
        self.data_length = data_lengths.pop()
        self.num_batches = ceil(self.data_length / batch_size)
        if labels:
            self.labels = np.array(dataset["label"])
            assert len(self.labels) == self.data_length, "Labels not the same length as input arrays!"
        else:
            self.labels = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            # Shuffle the data order
            self.permutation = np.random.permutation(self.data_length)
        else:
            self.permutation = None

    def on_epoch_end(self):
        # If we're shuffling, reshuffle the data order after each epoch
        if self.shuffle:
            self.permutation = np.random.permutation(self.data_length)

    def __getitem__(self, item):
        # Note that this yields a batch, not a single sample
        batch_start = item * self.batch_size
        batch_end = (item + 1) * self.batch_size
        if self.shuffle:
            data_indices = self.permutation[batch_start:batch_end]
        else:
            data_indices = np.arange(batch_start, batch_end)
        # We want to pad the data as little as possible, so we only pad each batch
        # to the maximum length within that batch. We do that by stacking the variable-
        # length inputs into a ragged tensor and then densifying it.
        batch_input = {
            key: tf.ragged.constant([data[i] for i in data_indices]).to_tensor() for key, data in self.data.items()
        }
        if self.labels is None:
            return batch_input
        else:
            batch_labels = self.labels[data_indices]
            return batch_input, batch_labels

    def __len__(self):
        return self.num_batches


class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)


# endregion

# region Command-line arguments
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
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

    def __post_init__(self):
        train_extension = self.train_file.split(".")[-1].lower() if self.train_file is not None else None
        validation_extension = (
            self.validation_file.split(".")[-1].lower() if self.validation_file is not None else None
        )
        test_extension = self.test_file.split(".")[-1].lower() if self.test_file is not None else None
        extensions = {train_extension, validation_extension, test_extension}
        extensions.discard(None)
        assert len(extensions) != 0, "Need to supply at least one of --train_file, --validation_file or --test_file!"
        assert len(extensions) == 1, "All input files should have the same file extension, either csv or json!"
        assert "csv" in extensions or "json" in extensions, "Input files should have either .csv or .json extensions!"
        self.input_file_extension = extensions.pop()


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
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


# endregion


def main():
    # region Argument parsing
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
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # endregion

    # region Checkpoints
    # Detecting last checkpoint.
    checkpoint = None
    if len(os.listdir(training_args.output_dir)) > 0 and not training_args.overwrite_output_dir:
        if (output_dir / CONFIG_NAME).is_file() and (output_dir / TF2_WEIGHTS_NAME).is_file():
            checkpoint = output_dir
            logger.info(
                f"Checkpoint detected, resuming training from checkpoint in {training_args.output_dir}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        else:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to continue regardless."
            )

    # endregion

    # region Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    logger.info(f"Training/evaluation parameters {training_args}")
    # endregion

    # region Loading data
    # For CSV/JSON files, this script will use the 'label' field as the label and the 'sentence1' and optionally
    # 'sentence2' fields as inputs if they exist. If not, the first two fields not named label are used if at least two
    # columns are provided. Note that the term 'sentence' can be slightly misleading, as they often contain more than
    # a single grammatical sentence, when the task requires it.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file, "test": data_args.test_file}
    data_files = {key: file for key, file in data_files.items() if file is not None}

    for key in data_files.keys():
        logger.info(f"Loading a local file for {key}: {data_files[key]}")

    if data_args.input_file_extension == "csv":
        # Loading a dataset from local csv files
        datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
    else:
        # Loading a dataset from local json files
        datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    # endregion

    # region Label preprocessing
    # If you've passed us a training set, we try to infer your labels from it
    if "train" in datasets:
        # By default we assume that if your label column looks like a float then you're doing regression,
        # and if not then you're doing classification. This is something you may want to change!
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    # If you haven't passed a training set, we read label info from the saved model (this happens later)
    else:
        num_labels = None
        label_list = None
        is_regression = None
    # endregion

    # region Load pretrained model and tokenizer
    # Set seed before initializing model
    set_seed(training_args.seed)
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if checkpoint is not None:
        config_path = training_args.output_dir
    elif model_args.config_name:
        config_path = model_args.config_name
    else:
        config_path = model_args.model_name_or_path
    if num_labels is not None:
        config = AutoConfig.from_pretrained(
            config_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = AutoConfig.from_pretrained(
            config_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if checkpoint is None:
        model_path = model_args.model_name_or_path
    else:
        model_path = checkpoint
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # endregion

    # region Optimizer, loss and compilation
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=training_args.learning_rate,
        beta_1=training_args.adam_beta1,
        beta_2=training_args.adam_beta2,
        epsilon=training_args.adam_epsilon,
        clipnorm=training_args.max_grad_norm,
    )
    if is_regression:
        loss = tf.keras.losses.MeanSquaredError()
        metrics = []
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # endregion

    # region Dataset preprocessing
    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    column_names = {col for cols in datasets.column_names.values() for col in cols}
    non_label_column_names = [name for name in column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    elif "sentence1" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", None
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Ensure that our labels match the model's, if it has some pre-specified
    if "train" in datasets:
        if not is_regression and model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
            label_name_to_id = model.config.label2id
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                label_to_id = label_name_to_id  # Use the model's labels
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
                label_to_id = {v: i for i, v in enumerate(label_list)}
        elif not is_regression:
            label_to_id = {v: i for i, v in enumerate(label_list)}
        else:
            label_to_id = None
        # Now we've established our label2id, let's overwrite the model config with it.
        model.config.label2id = label_to_id
        if model.config.label2id is not None:
            model.config.id2label = {id: label for label, id in label_to_id.items()}
        else:
            model.config.id2label = None
    else:
        label_to_id = model.config.label2id  # Just load the data from the model

    if "validation" in datasets and model.config.label2id is not None:
        validation_label_list = datasets["validation"].unique("label")
        for val_label in validation_label_list:
            assert val_label in label_to_id, f"Label {val_label} is in the validation set but not the training set!"

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs
        if model.config.label2id is not None and "label" in examples:
            result["label"] = [(model.config.label2id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    if "train" in datasets:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # Log a few random samples from the training set so we can see that it's working as expected:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if "validation" in datasets:
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if "test" in datasets:
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # endregion

    # region Training
    if "train" in datasets:
        training_dataset = DataSequence(
            train_dataset, non_label_column_names, batch_size=training_args.per_device_train_batch_size, labels=True
        )
        if "validation" in datasets:
            eval_dataset = DataSequence(
                eval_dataset, non_label_column_names, batch_size=training_args.per_device_eval_batch_size, labels=True
            )
        else:
            eval_dataset = None

        callbacks = [SavePretrainedCallback(output_dir=training_args.output_dir)]
        model.fit(
            training_dataset, validation_data=eval_dataset, epochs=training_args.num_train_epochs, callbacks=callbacks
        )
    elif "validation" in datasets:
        # If there's a validation dataset but no training set, just evaluate the metrics
        eval_dataset = DataSequence(
            eval_dataset, non_label_column_names, batch_size=training_args.per_device_eval_batch_size, labels=True
        )
        logger.info("Computing metrics on validation data...")
        if is_regression:
            loss = model.evaluate(eval_dataset)
            logger.info(f"Loss: {loss:.5f}")
        else:
            loss, accuracy = model.evaluate(eval_dataset)
            logger.info(f"Loss: {loss:.5f}, Accuracy: {accuracy * 100:.4f}%")
    # endregion

    # region Prediction
    if "test" in datasets:
        logger.info("Doing predictions on test dataset...")

        test_dataset = DataSequence(
            test_dataset, non_label_column_names, batch_size=training_args.per_device_eval_batch_size, labels=False
        )
        predictions = model.predict(test_dataset)["logits"]
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
        output_test_file = os.path.join(training_args.output_dir, "test_results.txt")
        with open(output_test_file, "w") as writer:
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                if is_regression:
                    writer.write(f"{index}\t{item:3.3f}\n")
                else:
                    item = model.config.id2label[item]
                    writer.write(f"{index}\t{item}\n")
        logger.info(f"Wrote predictions to {output_test_file}!")
    # endregion


if __name__ == "__main__":
    main()
