""" Compress a 🤗 Transformers model for sequence classification on GLUE. by PaddleSlim"""
import argparse
import logging
import os
import random

import datasets
import numpy as np
from datasets import load_dataset

import evaluate
import paddle
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from paddle.io import DataLoader
from paddleslim.auto_compression.compressor import AutoCompression
from paddleslim.common import load_config as load_slim_config
from transformers import AutoTokenizer, default_data_collator
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--config_path", type=str, default=None, help="path of compression strategy config.", required=True
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


def paddle_data_collator(features):
    batch = default_data_collator(features, return_tensors="np")
    return batch


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    for data in eval_dataloader():
        logits = exe.run(
            compiled_test_program,
            feed={
                test_feed_names[0]: data[0]["x0"],
                test_feed_names[1]: data[0]["x1"],
                test_feed_names[2]: data[0]["x2"],
            },
            fetch_list=test_fetch_list,
        )
        predictions = np.argmax(np.array(logits[0]), axis=-1) if not is_regression else np.squeeze(np.array(logits[0]))
        predictions, references = accelerator.gather((predictions, np.squeeze(data[0]["labels"])))
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    eval_metric = metric.compute()
    if isinstance(eval_metric, dict):
        res = list(eval_metric.values())[0]
    elif isinstance(eval_metric, list) or isinstance(eval_metric, tuple):
        res = eval_metric[0]
    else:
        res = eval_metric
    return res

def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("auto_compress_glue", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    global accelerator
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the metric function
    global metric
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    global is_regression
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if not is_regression:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism

    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download vocab.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding="max_length", max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    # If padding was already done ot max length, we use the default data collator that will just convert everything
    # to tensors.
    data_collator = paddle_data_collator

    paddle.enable_static()

    def reader_wrapper(reader, input_name=["x0", "x1", "x2"]):
        def gen():
            feed_data = {}
            data_names = list(reader.dataset[0].keys())
            for data in reader:
                for idx in range(len(input_name)):
                    feed_data[input_name[idx]] = data[data_names[idx]]
                if "labels" in feed_data:
                    feed_data["labels"] = np.array(feed_data["labels"]).reshape(-1, 1)
                yield [feed_data]

        return gen

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=0,
        shuffle=True,
    )
    train_dataloader = reader_wrapper(train_dataloader, ["x0", "x2", "x1"])

    global eval_dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        num_workers=0,
        shuffle=False,
    )
    eval_dataloader = reader_wrapper(eval_dataloader, ["x0", "x2", "x1", "labels"])

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    def apply_decay_param_fun(name):
        if name.find("bias") > -1:
            return True
        elif name.find("b_0") > -1:
            return True
        elif name.find("norm") > -1:
            return True
        else:
            return False

    config = load_slim_config(args.config_path)
    if "TrainConfig" in config:
        config["TrainConfig"]["optimizer_builder"]["apply_decay_param_fun"] = apply_decay_param_fun

    ac = AutoCompression(
        model_dir=args.model_name_or_path,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams",
        save_dir=args.output_dir,
        config=config,
        train_dataloader=train_dataloader,
        eval_callback=eval_function,
        eval_dataloader=eval_dataloader,
    )
    ac.compress()
    #ac.export_onnx()


if __name__ == "__main__":
    main()
