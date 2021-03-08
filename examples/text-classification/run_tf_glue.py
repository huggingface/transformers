#!/usr/bin/env python
# coding=utf-8
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
""" Fine-tuning the library models for sequence classification."""


import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizer,
    TFAutoModelForSequenceClassification,
    TFTrainer,
    TFTrainingArguments,
    glue_compute_metrics,
    glue_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
)
from transformers.utils import logging as hf_logging


hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()


class Split(Enum):
    train = "train"
    dev = "validation"
    test = "test"


def get_tfds(
    task_name: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: Optional[int] = None,
    mode: Split = Split.train,
    data_dir: str = None,
):
    if task_name == "mnli-mm" and mode == Split.dev:
        tfds_name = "mnli_mismatched"
    elif task_name == "mnli-mm" and mode == Split.train:
        tfds_name = "mnli"
    elif task_name == "mnli" and mode == Split.dev:
        tfds_name = "mnli_matched"
    elif task_name == "sst-2":
        tfds_name = "sst2"
    elif task_name == "sts-b":
        tfds_name = "stsb"
    else:
        tfds_name = task_name

    ds, info = tfds.load("glue/" + tfds_name, split=mode.value, with_info=True, data_dir=data_dir)
    ds = glue_convert_examples_to_features(ds, tokenizer, max_seq_length, task_name)
    ds = ds.apply(tf.data.experimental.assert_cardinality(info.splits[mode.value].num_examples))

    return ds


logger = logging.getLogger(__name__)


@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: Optional[str] = field(default=None, metadata={"help": "The input/output data dir for TFDS."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


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
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, GlueDataTrainingArguments, TFTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(
        "n_replicas: %s, distributed training: %s, 16-bits training: %s",
        training_args.n_replicas,
        bool(training_args.n_replicas > 1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    try:
        num_labels = glue_tasks_num_labels["mnli" if data_args.task_name == "mnli-mm" else data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    with training_args.strategy.scope():
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_pt=bool(".bin" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # Get datasets
    train_dataset = (
        get_tfds(
            task_name=data_args.task_name,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            data_dir=data_args.data_dir,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        get_tfds(
            task_name=data_args.task_name,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            mode=Split.dev,
            data_dir=data_args.data_dir,
        )
        if training_args.do_eval
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    # Initialize our Trainer
    trainer = TFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")

            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

            results.update(result)

    return results


if __name__ == "__main__":
    main()
