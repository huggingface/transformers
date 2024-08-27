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
"""Fine-tuning the library models for sequence classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset
from packaging.version import parse

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PretrainedConfig,
    PushToHubCallback,
    TFAutoModelForSequenceClassification,
    TFTrainingArguments,
    create_optimizer,
    set_seed,
)
from transformers.utils import CONFIG_NAME, TF2_WEIGHTS_NAME, send_example_telemetry


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Reduce the amount of console output from TF
import tensorflow as tf  # noqa: E402


try:
    import tf_keras as keras
except (ModuleNotFoundError, ImportError):
    import keras

    if parse(keras.__version__).major > 2:
        raise ValueError(
            "Your currently installed version of Keras is Keras 3, but this is not yet supported in "
            "Transformers. Please install the backwards-compatible tf-keras package with "
            "`pip install tf-keras`."
        )


logger = logging.getLogger(__name__)


# region Helper classes
class SavePretrainedCallback(keras.callbacks.Callback):
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
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. "
                "Data will always be padded when using TPUs."
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
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of test examples to this "
                "value if set."
            )
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
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )


# endregion


def main():
    # region Argument parsing
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_text_classification", model_args, data_args, framework="tensorflow")

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
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
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
        datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        # Loading a dataset from local json files
        datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.
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

    # region Load model config and tokenizer
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
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        config = AutoConfig.from_pretrained(
            config_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
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

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Ensure that our labels match the model's, if it has some pre-specified
    if "train" in datasets:
        if not is_regression and config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
            label_name_to_id = config.label2id
            if sorted(label_name_to_id.keys()) == sorted(label_list):
                label_to_id = label_name_to_id  # Use the model's labels
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: "
                    f"model labels: {sorted(label_name_to_id.keys())}, dataset labels:"
                    f" {sorted(label_list)}.\nIgnoring the model labels as a result.",
                )
                label_to_id = {v: i for i, v in enumerate(label_list)}
        elif not is_regression:
            label_to_id = {v: i for i, v in enumerate(label_list)}
        else:
            label_to_id = None
        # Now we've established our label2id, let's overwrite the model config with it.
        config.label2id = label_to_id
        if config.label2id is not None:
            config.id2label = {id: label for label, id in label_to_id.items()}
        else:
            config.id2label = None
    else:
        label_to_id = config.label2id  # Just load the data from the model

    if "validation" in datasets and config.label2id is not None:
        validation_label_list = datasets["validation"].unique("label")
        for val_label in validation_label_list:
            assert val_label in label_to_id, f"Label {val_label} is in the validation set but not the training set!"

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, max_length=max_seq_length, truncation=True)

        # Map labels to IDs
        if config.label2id is not None and "label" in examples:
            result["label"] = [(config.label2id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    # endregion

    with training_args.strategy.scope():
        # region Load pretrained model
        # Set seed before initializing model
        set_seed(training_args.seed)
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        if checkpoint is None:
            model_path = model_args.model_name_or_path
        else:
            model_path = checkpoint
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        # endregion

        # region Convert data to a tf.data.Dataset
        dataset_options = tf.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        num_replicas = training_args.strategy.num_replicas_in_sync

        tf_data = {}
        max_samples = {
            "train": data_args.max_train_samples,
            "validation": data_args.max_val_samples,
            "test": data_args.max_test_samples,
        }
        for key in ("train", "validation", "test"):
            if key not in datasets:
                tf_data[key] = None
                continue
            if (
                (key == "train" and not training_args.do_train)
                or (key == "validation" and not training_args.do_eval)
                or (key == "test" and not training_args.do_predict)
            ):
                tf_data[key] = None
                continue
            if key in ("train", "validation"):
                assert "label" in datasets[key].features, f"Missing labels from {key} data!"
            if key == "train":
                shuffle = True
                batch_size = training_args.per_device_train_batch_size * num_replicas
            else:
                shuffle = False
                batch_size = training_args.per_device_eval_batch_size * num_replicas
            samples_limit = max_samples[key]
            dataset = datasets[key]
            if samples_limit is not None:
                dataset = dataset.select(range(samples_limit))

            # model.prepare_tf_dataset() wraps a Hugging Face dataset in a tf.data.Dataset which is ready to use in
            # training. This is the recommended way to use a Hugging Face dataset when training with Keras. You can also
            # use the lower-level dataset.to_tf_dataset() method, but you will have to specify things like column names
            # yourself if you use this method, whereas they are automatically inferred from the model input names when
            # using model.prepare_tf_dataset()
            # For more info see the docs:
            # https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset
            # https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.to_tf_dataset

            data = model.prepare_tf_dataset(
                dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                tokenizer=tokenizer,
            )
            data = data.with_options(dataset_options)
            tf_data[key] = data
        # endregion

        # region Optimizer, loss and compilation

        if training_args.do_train:
            num_train_steps = len(tf_data["train"]) * training_args.num_train_epochs
            if training_args.warmup_steps > 0:
                num_warmup_steps = training_args.warmup_steps
            elif training_args.warmup_ratio > 0:
                num_warmup_steps = int(num_train_steps * training_args.warmup_ratio)
            else:
                num_warmup_steps = 0

            optimizer, schedule = create_optimizer(
                init_lr=training_args.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                adam_beta1=training_args.adam_beta1,
                adam_beta2=training_args.adam_beta2,
                adam_epsilon=training_args.adam_epsilon,
                weight_decay_rate=training_args.weight_decay,
                adam_global_clipnorm=training_args.max_grad_norm,
            )
        else:
            optimizer = "sgd"  # Just use any default
        if is_regression:
            metrics = []
        else:
            metrics = ["accuracy"]
        # Transformers models compute the right loss for their task by default when labels are passed, and will
        # use this for training unless you specify your own loss function in compile().
        model.compile(optimizer=optimizer, metrics=metrics)
        # endregion

        # region Preparing push_to_hub and model card
        push_to_hub_model_id = training_args.push_to_hub_model_id
        model_name = model_args.model_name_or_path.split("/")[-1]
        if not push_to_hub_model_id:
            push_to_hub_model_id = f"{model_name}-finetuned-text-classification"

        model_card_kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}

        if training_args.push_to_hub:
            callbacks = [
                PushToHubCallback(
                    output_dir=training_args.output_dir,
                    hub_model_id=push_to_hub_model_id,
                    hub_token=training_args.push_to_hub_token,
                    tokenizer=tokenizer,
                    **model_card_kwargs,
                )
            ]
        else:
            callbacks = []
        # endregion

        # region Training and validation
        if tf_data["train"] is not None:
            model.fit(
                tf_data["train"],
                validation_data=tf_data["validation"],
                epochs=int(training_args.num_train_epochs),
                callbacks=callbacks,
            )
        if tf_data["validation"] is not None:
            logger.info("Computing metrics on validation data...")
            if is_regression:
                loss = model.evaluate(tf_data["validation"])
                logger.info(f"Eval loss: {loss:.5f}")
            else:
                loss, accuracy = model.evaluate(tf_data["validation"])
                logger.info(f"Eval loss: {loss:.5f}, Eval accuracy: {accuracy * 100:.4f}%")
            if training_args.output_dir is not None:
                output_eval_file = os.path.join(training_args.output_dir, "all_results.json")
                eval_dict = {"eval_loss": loss}
                if not is_regression:
                    eval_dict["eval_accuracy"] = accuracy
                with open(output_eval_file, "w") as writer:
                    writer.write(json.dumps(eval_dict))
        # endregion

        # region Prediction
        if tf_data["test"] is not None:
            logger.info("Doing predictions on test dataset...")
            predictions = model.predict(tf_data["test"])["logits"]
            predicted_class = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            output_test_file = os.path.join(training_args.output_dir, "test_results.txt")
            with open(output_test_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predicted_class):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = config.id2label[item]
                        writer.write(f"{index}\t{item}\n")
            logger.info(f"Wrote predictions to {output_test_file}!")
        # endregion

        if training_args.output_dir is not None and not training_args.push_to_hub:
            # If we're not pushing to hub, at least save a local copy when we're done
            model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
