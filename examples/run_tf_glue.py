# coding=utf-8
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


import argparse
import logging
import os

import tensorflow as tf
import tensorflow_datasets

from transformers import (
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    TFBertForSequenceClassification,
    TFDistilBertForSequenceClassification,
    TFRobertaForSequenceClassification,
    TFXLMForSequenceClassification,
    TFXLNetForSequenceClassification,
    XLMConfig,
    XLMTokenizer,
    XLNetConfig,
    XLNetTokenizer,
)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


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


def load_and_cache_examples(args, data, task, tokenizer, split):
    if task == "mnli" and split == "validation":
        split = "validation_matched"

    features_output_dir = os.path.join(args.output_dir, "features")
    cached_features_file = os.path.join(
        features_output_dir,
        "cached_{}_{}_{}_{}.tfrecord".format(
            split, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length), str(task)
        ),
    )

    if not os.path.exists(cached_features_file) or args.overwrite_cache:
        logger.info("Converting examples to features")
        dataset = convert_examples_to_features(data[split], tokenizer, args.max_seq_length, task)

        if not os.path.exists(features_output_dir):
            os.makedirs(features_output_dir)

        with tf.compat.v1.python_io.TFRecordWriter(cached_features_file) as tfwriter:
            for feature in dataset:
                example, label = feature
                feature_key_value_pair = {
                    "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=example["input_ids"])),
                    "attention_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=example["attention_mask"])),
                    "token_type_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=example["token_type_ids"])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                }
                features = tf.train.Features(feature=feature_key_value_pair)
                example = tf.train.Example(features=features)

                tfwriter.write(example.SerializeToString())

        logger.info("Features saved to cache")

    features = {
        "input_ids": tf.io.FixedLenFeature([args.max_seq_length], tf.int64),
        "attention_mask": tf.io.FixedLenFeature([args.max_seq_length], tf.int64),
        "token_type_ids": tf.io.FixedLenFeature([args.max_seq_length], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def select_data_from_record(record):
        record = tf.io.parse_single_example(record, features)
        x = {
            "input_ids": record["input_ids"],
            "attention_mask": record["attention_mask"],
            "token_type_ids": record["token_type_ids"],
        }
        y = record["label"]
        return (x, y)

    dataset = tf.data.TFRecordDataset(cached_features_file)
    dataset = dataset.map(select_data_from_record)

    logger.info("Created dataset %s from TFRecord" % split)
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
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
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

    TASK = args.task_name.lower()

    if TASK not in processors:
        raise ValueError("Task not found: %s" % (TASK))

    if TASK == "sst-2":
        TFDS_TASK = "sst2"
    elif TASK == "sts-b":
        TFDS_TASK = "stsb"
    else:
        TFDS_TASK = TASK

    num_labels = len(processors[TASK]().get_labels())
    print(num_labels)

    tf.config.optimizer.set_jit(args.xla)
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.amp})

    # Load tokenizer and model from pretrained model/vocabulary. Specify the number of labels to classify (2+: classification, 1: regression)
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
        force_download=args.force_download,
    )

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        force_download=args.force_download,
    )

    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        force_download=args.force_download,
    )

    # Load dataset via TensorFlow Datasets
    data, info = tensorflow_datasets.load("glue/%s" % TFDS_TASK, with_info=True)

    # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, epsilon=args.adam_epsilon)

    if args.amp:
        # loss scaling is currently required when using mixed precision
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

    if num_labels == 1:
        loss = tf.keras.losses.MeanSquaredError()
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
    model.compile(optimizer=opt, loss=loss, metrics=[metric])

    class save_model(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Saving model at epoch {}".format(epoch))
            output_dir = os.path.join(args.output_dir, "checkpoint-epoch-{}".format(epoch))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.model.save_pretrained(output_dir)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, data=data, task=TASK, tokenizer=tokenizer, split="train")
        train_dataset = train_dataset.batch(args.train_batch_size).repeat(args.num_train_epochs)
        train_examples = info.splits["train"].num_examples / args.train_batch_size

        validation_identifier = "validation_mismatched" if TASK == "mnli" else "validation"
        valid_dataset = load_and_cache_examples(
            args, data=data, task=TASK, tokenizer=tokenizer, split=validation_identifier
        )
        valid_dataset = valid_dataset.batch(args.valid_batch_size)
        valid_examples = info.splits[validation_identifier].num_examples / args.valid_batch_size

        history = model.fit(
            train_dataset,
            steps_per_epoch=train_examples,
            epochs=args.num_train_epochs,
            validation_data=valid_dataset if args.evaluate_during_training else None,
            validation_steps=valid_examples if args.evaluate_during_training else None,
            callbacks=[save_model()],
        )

    if args.do_eval:
        test_dataset = load_and_cache_examples(args, data=data, task=TASK, tokenizer=tokenizer, split="test")
        test_dataset = test_dataset.batch(args.test_batch_size)
        test_examples = info.splits["test"].num_examples / args.test_batch_size

        results = model.evaluate(test_dataset, steps=test_examples)


if __name__ == "__main__":
    main()
