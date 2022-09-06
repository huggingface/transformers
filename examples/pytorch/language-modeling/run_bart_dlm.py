#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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
"""Pretraining models for denoising language modeling on a text file or a dataset.
"""
import hashlib
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

import datasets
from datasets import load_dataset

import evaluate
import transformers
from denoising_collator import DataCollatorForBartDenoisingLM
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
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
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
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
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
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
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    spacy_model: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "By default, an English NLTK punct model is used for sentence splitting. If you give a spacy_model"
                " name instead, we'll use that for sentence splitting. Note that spaCy and the chosen model have"
                " to be installed."
            )
        },
    )
    no_sentence_splitting: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "By default, an English NLTK punct model is used for sentence splitting, or with 'spacy_model' you"
                " can specify a spaCy model for splitting. If you decide to do sentence splitting yourself, you"
                " can disable any sentence splitting in this script with this flag. Note that sentences need to be"
                " split and then joined together again with the tokenizer's padding token."
            )
        },
    )
    mask_ratio: float = field(
        default=0.3, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    random_ratio: float = field(
        default=0.1, metadata={"help": "The probability with which to (randomly) replace a token by a random token"}
    )
    insert_ratio: float = field(
        default=0.0,
        metadata={
            "help": (
                "The probability with which to (randomly) insert noise. Will add `insert_ratio * input.numel()` noise"
            )
        },
    )
    rotate_ratio: float = field(
        default=0.0,
        metadata={
            "help": "The probability with which to (randomly) add rolling noise (i.e., shifted tokens in order)"
        },
    )
    permute_sentence_ratio: float = field(
        default=1.0, metadata={"help": "Ratio of sentences to be permuted in each document"}
    )
    poisson_lambda: float = field(
        default=3.5, metadata={"help": "Mean of Poisson distribution used to generate span-lengths to be masked"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in {"csv", "json", "txt"}:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in {"csv", "json", "txt"}:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")


def clean_fingerprint(fingerprint: str):
    return re.sub(r"[<>:/\\|?*.]", "", fingerprint)


def hash_fingerprint(text: Optional[str], length: int = 49) -> Union[None, str]:
    if text is None:
        return None
    # Using a length of 49: max. length in datasets for fingerprint is 64
    # and by using multiple workers, an additional fingerprint like `_00000_of_00004`
    # is added to the end. So we just have room for 49
    return str(int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 10**length)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        loaded_ds = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if "validation" not in loaded_ds.keys():
            loaded_ds["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            loaded_ds["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        extension = None
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        loaded_ds = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in loaded_ds.keys():
            loaded_ds["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            loaded_ds["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    ############################
    # Load tokenizer and model #
    ############################
    # TOKENIZER
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = BartTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        model_args.tokenizer_name = model_args.model_name_or_path
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # CONFIG
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = BartConfig.from_pretrained(model_args.config_name, vocab_size=len(tokenizer), **config_kwargs)
    elif model_args.model_name_or_path:
        config = BartConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = BartConfig()
        logger.warning("You are instantiating a new config instance from scratch.")

    # MODEL
    if model_args.model_name_or_path:
        model = BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        config.vocab_size = len(tokenizer)
        model = BartForConditionalGeneration(config)

    model.resize_token_embeddings(len(tokenizer))

    #######################
    # Preprocess datasets #
    #######################
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = loaded_ds["train"].column_names
    else:
        column_names = loaded_ds["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Set max length
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Caching is not working well with spaCy so we use explicit fingerprints
    # Looping over splits because we cannot use new_fingerprint on DatasetDict
    for k in loaded_ds.keys():
        fingerprint = (
            f"{k}@{data_args.dataset_name}{data_args.dataset_config_name}" if data_args.dataset_name else None
        )

        if not data_args.no_sentence_splitting:
            # Do sentence splitting
            sentence_tokenizer = None
            if data_args.spacy_model:
                import spacy

                spacy.prefer_gpu()
                # Only load the parser (depparse) which will set sentence boundaries
                sentence_tokenizer = spacy.load(
                    data_args.spacy_model, exclude=["tagger", "ner", "lemmatizer", "textcat"]
                )
            else:
                import nltk

                # Use Punkt Sentence Tokenizer to divide a document into a list of sentences
                nltk.download("punkt")
                sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

            def sentence_split(examples):
                if data_args.spacy_model:
                    docs = sentence_tokenizer.pipe(examples["text"])
                    doc_sents = [map(str, doc.sents) for doc in docs]
                else:
                    doc_sents = [[s for s in sentence_tokenizer.tokenize(t)] for t in examples["text"]]

                # use pad token as end of sentence indicator
                new_texts = [
                    f"{tokenizer.bos_token}{tokenizer.pad_token.join(sents)}{tokenizer.eos_token}"
                    for sents in doc_sents
                ]
                return {"text": new_texts}

            with training_args.main_process_first(desc="Sentence splitting"):
                # If using spaCy, we don't run multiple workers here but pass that to spacy's pipe
                fingerprint = clean_fingerprint(f"{fingerprint}+ss{data_args.spacy_model}") if fingerprint else None
                loaded_ds[k] = loaded_ds[k].map(
                    sentence_split,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Sentence splitting",
                    new_fingerprint=hash_fingerprint(fingerprint),
                )
                del sentence_tokenizer

        # Tokenize (subword) every text, then concatenate them together before splitting them in smaller parts.
        # Attention masks will be added in the collator
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], add_special_tokens=False, return_attention_mask=False)

        with training_args.main_process_first(desc="Subword tokenization"):
            fingerprint = clean_fingerprint(f"{fingerprint}+tok{model_args.tokenizer_name}") if fingerprint else None
            loaded_ds[k] = loaded_ds[k].map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=text_column_name,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenizing",
                new_fingerprint=hash_fingerprint(fingerprint),
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        with training_args.main_process_first(desc="Grouping"):
            fingerprint = clean_fingerprint(f"{fingerprint}+group{max_seq_length}") if fingerprint else None
            loaded_ds[k] = loaded_ds[k].map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping in blocks of {max_seq_length}",
                new_fingerprint=hash_fingerprint(fingerprint),
            )

    train_dataset = None
    if training_args.do_train:
        if "train" not in loaded_ds:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = loaded_ds["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in loaded_ds:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = loaded_ds["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Data collator will take care of randomly masking the tokens/spans, permuting sentences, adding noise
    data_collator = DataCollatorForBartDenoisingLM(
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        mask_ratio=data_args.mask_ratio,
        random_ratio=data_args.random_ratio,
        insert_ratio=data_args.insert_ratio,
        rotate_ratio=data_args.rotate_ratio,
        permute_sentence_ratio=data_args.permute_sentence_ratio,
        poisson_lambda=data_args.poisson_lambda,
    )

    # Some trainer-specific submethods that may be relevant:
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text2text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
