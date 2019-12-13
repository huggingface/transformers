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
    TFBertForSequenceClassification,
    BertConfig,
    XLNetTokenizer,
    TFXLNetForSequenceClassification,
    XLNetConfig,
    XLMTokenizer,
    TFXLMForSequenceClassification,
    XLMConfig,
    RobertaTokenizer,
    TFRobertaForSequenceClassification,
    RobertaConfig,
    DistilBertTokenizer,
    TFDistilBertForSequenceClassification,
    DistilBertConfig,
)
import argparse

from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig, DistilBertConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, TFBertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, TFXLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, TFXLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, TFRobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, TFDistilBertForSequenceClassification, DistilBertTokenizer),
}


def train():
    ""


def evaluate():
    ""


def load_and_cache_examples():
    ""


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



