# Copyright 2018 The HuggingFace Inc. team.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""


import os
import tensorflow as tf
import tensorflow_datasets
import logging

from transformers import (
    BertTokenizer,
    TFBertForQuestionAnswering,
    BertConfig,
    XLNetTokenizer,
    TFXLNetForQuestionAnsweringSimple,
    XLNetConfig,
    XLMTokenizer,
    TFXLMForQuestionAnsweringSimple,
    XLMConfig,
    DistilBertTokenizer,
    TFDistilBertForQuestionAnswering,
    DistilBertConfig,
)
import argparse

from transformers import squad_convert_examples_to_features, SquadV1Processor, SquadV2Processor

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, XLNetConfig, XLMConfig, DistilBertConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, TFBertForQuestionAnswering, BertTokenizer),
    "xlnet": (XLNetConfig, TFXLNetForQuestionAnsweringSimple, XLNetTokenizer),
    "xlm": (XLMConfig, TFXLMForQuestionAnsweringSimple, XLMTokenizer),
    "distilbert": (DistilBertConfig, TFDistilBertForQuestionAnswering, DistilBertTokenizer),
}


def train():
    ""


def evaluate():
    ""


def load_and_cache_examples(args, data, tokenizer, evaluate=False):
    features_output_dir = os.path.join(args.output_dir, "features")
    cached_features_file = os.path.join(
        features_output_dir,
        "cached_{}_{}_{}.tfrecord".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    if not os.path.exists(cached_features_file) or args.overwrite_cache:
        if args.version_2_with_negative:
            processor = SquadV2Processor()
        else:
            processor = SquadV1Processor()

        if args.data_dir:
            directory = args.directory
            examples = processor.get_dev_examples(directory) if evaluate else processor.get_train_examples(directory)
        else:
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            examples = processor.get_examples_from_dataset(tfds.load("squad"), evaluate=evaluate)

        logger.info("Converting examples to features")
        dataset = squad_convert_examples_to_features(examples, tokenizer, args.max_seq_length)

        if not os.path.exists(features_output_dir):
            os.makedirs(features_output_dir)

        with tf.compat.v1.python_io.TFRecordWriter(cached_features_file) as tfwriter:
            for feature in dataset:
                example, result = feature
                feature_key_value_pair = {
                    "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=example["input_ids"])),
                    "attention_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=example["attention_mask"])),
                    "token_type_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=example["token_type_ids"])),
                    "start_position": tf.train.Feature(int64_list=tf.train.Int64List(value=result["start_position"])),
                    "end_position": tf.train.Feature(int64_list=tf.train.Int64List(value=result["end_position"])),
                    "cls_index": tf.train.Feature(int64_list=tf.train.Int64List(value=result["cls_index"])),
                    "p_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=result["p_mask"])),
                }
                features = tf.train.Features(feature=feature_key_value_pair)
                example = tf.train.Example(features=features)

                tfwriter.write(example.SerializeToString())

        logger.info("Features saved to cache")

    features = {
        "input_ids": tf.io.FixedLenFeature([args.max_seq_length], tf.int64),
        "attention_mask": tf.io.FixedLenFeature([args.max_seq_length], tf.int64),
        "token_type_ids": tf.io.FixedLenFeature([args.max_seq_length], tf.int64),
        "start_position": tf.io.FixedLenFeature([], tf.int64),
        "end_position": tf.io.FixedLenFeature([], tf.int64),
        "cls_index": tf.io.FixedLenFeature([], tf.int64),
        "p_mask": tf.io.FixedLenFeature([args.max_seq_length], tf.int64),
    }

    def select_data_from_record(record):
        record = tf.io.parse_single_example(record, features)
        x = {
            "input_ids": record["input_ids"],
            "attention_mask": record["attention_mask"],
            "token_type_ids": record["token_type_ids"],
        }
        y = {
            "start_position": record["start_position"],
            "end_position": record["end_position"],
            "cls_index": record["cls_index"],
            "p_mask": record["p_mask"],
        }
        return (x, y)

    dataset = tf.data.TFRecordDataset(cached_features_file)
    dataset = dataset.map(select_data_from_record)

    logger.info("Created dataset %s from TFRecord" % "dev" if evaluate else "train")
    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=False,
        help="The input data directory containing the .json files. If no data dir is specified, uses tensorflow_datasets to load the data."
        + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )

    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--valid_batch_size", default=8, type=int, help="Batch size per GPU/CPU for validation during training."
    )
    parser.add_argument(
        "--test_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation after training."
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument("--xla", action="store_true", help="Whether to use XLA (Accelerated Linear Algebra).")
    parser.add_argument("--amp", action="store_true", help="Whether to use AMP (Automatic Mixed Precision).")
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Whether to force download the weights from S3 (useful if the file is corrupted).",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if os.path.exists(args.output_dir) and args.do_train:
        if not args.overwrite_output_dir and bool(
            [file for file in os.listdir(args.output_dir) if "features" not in file]
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args.output_dir
                )
            )

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
