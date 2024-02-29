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
Fine-tuning the library models for causal language modeling (GPT-2, GPT-Neo...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own clm task. Pointers for this are left as comments.

import json

# region Imports
import logging
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Optional

import datasets
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split

import transformers
from transformers import (
    CONFIG_MAPPING,
    CONFIG_NAME,
    TF2_WEIGHTS_NAME,
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PushToHubCallback,
    TFAutoModelForCausalLM,
    TFTrainingArguments,
    create_optimizer,
    set_seed,
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/tensorflow/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(TF_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
# endregion


# region Command-line arguments
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
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
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
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

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
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
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


# endregion


def main():
    # region Argument Parsing
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args, framework="tensorflow")

    # Sanity checks
    if data_args.dataset_name is None and data_args.train_file is None and data_args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if data_args.train_file is not None:
            extension = data_args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if data_args.validation_file is not None:
            extension = data_args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if training_args.output_dir is not None:
        training_args.output_dir = Path(training_args.output_dir)
        os.makedirs(training_args.output_dir, exist_ok=True)
    # endregion

    # region Checkpoints
    # Detecting last checkpoint.
    checkpoint = None
    if len(os.listdir(training_args.output_dir)) > 0 and not training_args.overwrite_output_dir:
        config_path = training_args.output_dir / CONFIG_NAME
        weights_path = training_args.output_dir / TF2_WEIGHTS_NAME
        if config_path.is_file() and weights_path.is_file():
            checkpoint = training_args.output_dir
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

    # region Setup logging
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    # endregion

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # region Load datasets
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.
    # endregion

    # region Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, token=model_args.token, trust_remote_code=model_args.trust_remote_code
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, token=model_args.token, trust_remote_code=model_args.trust_remote_code
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, token=model_args.token, trust_remote_code=model_args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # endregion

    # region Dataset preprocessing
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    if data_args.validation_file is not None:
        eval_dataset = lm_datasets["validation"]
    else:
        logger.info(
            f"Validation file not found: using {data_args.validation_split_percentage}% of the dataset as validation"
            " as provided in data_args"
        )
        train_indices, val_indices = train_test_split(
            list(range(len(train_dataset))), test_size=data_args.validation_split_percentage / 100
        )

        eval_dataset = train_dataset.select(val_indices)
        train_dataset = train_dataset.select(train_indices)

    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # endregion

    with training_args.strategy.scope():
        # region Prepare model
        if checkpoint is not None:
            model = TFAutoModelForCausalLM.from_pretrained(
                checkpoint, config=config, token=model_args.token, trust_remote_code=model_args.trust_remote_code
            )
        elif model_args.model_name_or_path:
            model = TFAutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
        else:
            logger.info("Training new model from scratch")
            model = TFAutoModelForCausalLM.from_config(
                config, token=model_args.token, trust_remote_code=model_args.trust_remote_code
            )

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embeddings = model.get_input_embeddings()

        # Matt: This is a temporary workaround as we transition our models to exclusively using Keras embeddings.
        #       As soon as the transition is complete, all embeddings should be keras.Embeddings layers, and
        #       the weights will always be in embeddings.embeddings.
        if hasattr(embeddings, "embeddings"):
            embedding_size = embeddings.embeddings.shape[0]
        else:
            embedding_size = embeddings.weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        # endregion

        # region TF Dataset preparation
        num_replicas = training_args.strategy.num_replicas_in_sync
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        # model.prepare_tf_dataset() wraps a Hugging Face dataset in a tf.data.Dataset which is ready to use in
        # training. This is the recommended way to use a Hugging Face dataset when training with Keras. You can also
        # use the lower-level dataset.to_tf_dataset() method, but you will have to specify things like column names
        # yourself if you use this method, whereas they are automatically inferred from the model input names when
        # using model.prepare_tf_dataset()
        # For more info see the docs:
        # https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset
        # https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.to_tf_dataset

        tf_train_dataset = model.prepare_tf_dataset(
            train_dataset,
            shuffle=True,
            batch_size=num_replicas * training_args.per_device_train_batch_size,
        ).with_options(options)

        tf_eval_dataset = model.prepare_tf_dataset(
            eval_dataset,
            shuffle=False,
            batch_size=num_replicas * training_args.per_device_eval_batch_size,
            drop_remainder=True,
        ).with_options(options)
        # endregion

        # region Optimizer and loss
        num_train_steps = len(tf_train_dataset) * int(training_args.num_train_epochs)
        if training_args.warmup_steps > 0:
            num_warmup_steps = training_args.warmup_steps
        elif training_args.warmup_ratio > 0:
            num_warmup_steps = int(num_train_steps * training_args.warmup_ratio)
        else:
            num_warmup_steps = 0

        # Bias and layernorm weights are automatically excluded from the decay
        optimizer, lr_schedule = create_optimizer(
            init_lr=training_args.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            adam_beta1=training_args.adam_beta1,
            adam_beta2=training_args.adam_beta2,
            adam_epsilon=training_args.adam_epsilon,
            weight_decay_rate=training_args.weight_decay,
            adam_global_clipnorm=training_args.max_grad_norm,
        )

        # Transformers models compute the right loss for their task by default when labels are passed, and will
        # use this for training unless you specify your own loss function in compile().
        model.compile(optimizer=optimizer, jit_compile=training_args.xla)
        # endregion

        # region Preparing push_to_hub and model card
        push_to_hub_model_id = training_args.push_to_hub_model_id
        model_name = model_args.model_name_or_path.split("/")[-1]
        if not push_to_hub_model_id:
            if data_args.dataset_name is not None:
                push_to_hub_model_id = f"{model_name}-finetuned-{data_args.dataset_name}"
            else:
                push_to_hub_model_id = f"{model_name}-finetuned-clm"

        model_card_kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            model_card_kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                model_card_kwargs["dataset_args"] = data_args.dataset_config_name
                model_card_kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                model_card_kwargs["dataset"] = data_args.dataset_name

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
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size = {training_args.per_device_train_batch_size * num_replicas}")

        # For long training runs, you may wish to use the PushToHub() callback here to save intermediate checkpoints
        # to the Hugging Face Hub rather than just pushing the finished model.
        # See https://huggingface.co/docs/transformers/main_classes/keras_callbacks#transformers.PushToHubCallback

        history = model.fit(
            tf_train_dataset,
            validation_data=tf_eval_dataset,
            epochs=int(training_args.num_train_epochs),
            callbacks=callbacks,
        )
        train_loss = history.history["loss"][-1]
        try:
            train_perplexity = math.exp(train_loss)
        except OverflowError:
            train_perplexity = math.inf
        logger.info(f"  Final train loss: {train_loss:.3f}")
        logger.info(f"  Final train perplexity: {train_perplexity:.3f}")
        validation_loss = history.history["val_loss"][-1]
        try:
            validation_perplexity = math.exp(validation_loss)
        except OverflowError:
            validation_perplexity = math.inf
        logger.info(f"  Final validation loss: {validation_loss:.3f}")
        logger.info(f"  Final validation perplexity: {validation_perplexity:.3f}")

        if training_args.output_dir is not None:
            output_eval_file = os.path.join(training_args.output_dir, "all_results.json")
            results_dict = {}
            results_dict["train_loss"] = train_loss
            results_dict["train_perplexity"] = train_perplexity
            results_dict["eval_loss"] = validation_loss
            results_dict["eval_perplexity"] = validation_perplexity
            with open(output_eval_file, "w") as writer:
                writer.write(json.dumps(results_dict))
        # endregion

    if training_args.output_dir is not None and not training_args.push_to_hub:
        # If we're not pushing to hub, at least save a local copy when we're done
        model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
