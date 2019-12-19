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
from typing import Dict

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

from seqeval import metrics
from fastprogress import master_bar, progress_bar
from transformers import create_optimizer, GradientAccumulator
from transformers import squad_convert_examples_to_features, SquadV1Processor, SquadV2Processor
import datetime
import math

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


def train(args, strategy, train_dataset, tokenizer, model, num_train_examples, train_batch_size):
    if args.max_steps > 0:
        num_train_steps = args.max_steps * args.gradient_accumulation_steps
        args.num_train_epochs = 1
    else:
        num_train_steps = (
            math.ceil(num_train_examples / train_batch_size)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    writer = tf.summary.create_file_writer("/tmp/mylogs")

    with strategy.scope():
        loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        optimizer = create_optimizer(args.learning_rate, num_train_steps, args.warmup_steps)

        if args.amp:
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

        loss_metric = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
        gradient_accumulator = GradientAccumulator()

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", num_train_examples)
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per device = %d", args.per_device_train_batch_size)
    logging.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        train_batch_size * args.gradient_accumulation_steps,
    )
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total training steps = %d", num_train_steps)

    model.summary()

    @tf.function
    def apply_gradients():
        grads_and_vars = []

        for gradient, variable in zip(gradient_accumulator.gradients, model.trainable_variables):
            if gradient is not None:
                scaled_gradient = gradient / (args.n_device * args.gradient_accumulation_steps)
                grads_and_vars.append((scaled_gradient, variable))
            else:
                grads_and_vars.append((gradient, variable))

        optimizer.apply_gradients(grads_and_vars, args.max_grad_norm)
        gradient_accumulator.reset()

    # @tf.function
    def train_step(train_features, train_labels):
        def step_fn(train_features, train_labels):
            if args.model_type in ["distilbert", "roberta"]:
                del train_features['token_type_ids']


            with tf.GradientTape() as tape:
                start_logits, end_logits = model(train_features)
                start_logits = tf.multiply(start_logits, tf.dtypes.cast((train_features['attention_mask']), tf.float32))
                end_logits = tf.multiply(end_logits, tf.dtypes.cast((train_features['attention_mask']), tf.float32))
                start_loss = loss_fct(train_labels['start_position'], start_logits)
                end_loss = loss_fct(train_labels['end_position'], end_logits)
                total_loss = (start_loss + end_loss) / 2

                loss = tf.reduce_sum(total_loss) * (1.0 / train_batch_size)
                grads = tape.gradient(loss, model.trainable_variables)

                gradient_accumulator(grads)

            return total_loss

        per_example_losses = strategy.experimental_run_v2(step_fn, args=(train_features, train_labels))
        mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)

        return mean_loss

    current_time = datetime.datetime.now()
    train_iterator = master_bar(range(args.num_train_epochs))
    global_step = 0
    logging_loss = 0.0

    for epoch in train_iterator:
        epoch_iterator = progress_bar(
            train_dataset, total=num_train_steps, parent=train_iterator, display=args.n_device > 1
        )
        step = 1

        with strategy.scope():
            for train_features, train_labels in epoch_iterator:
                loss = train_step(train_features, train_labels)

                if step % args.gradient_accumulation_steps == 0:
                    strategy.experimental_run_v2(apply_gradients)

                    loss_metric(loss)

                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        if (
                            args.n_device == 1 and args.evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            y_true, y_pred, eval_loss = evaluate(
                                args, strategy, model, tokenizer, labels, pad_token_label_id, mode="dev"
                            )
                            report = metrics.classification_report(y_true, y_pred, digits=4)

                            logging.info("Eval at step " + str(global_step) + "\n" + report)
                            logging.info("eval_loss: " + str(eval_loss))

                            precision = metrics.precision_score(y_true, y_pred)
                            recall = metrics.recall_score(y_true, y_pred)
                            f1 = metrics.f1_score(y_true, y_pred)

                            with writer.as_default():
                                tf.summary.scalar("eval_loss", eval_loss, global_step)
                                tf.summary.scalar("precision", precision, global_step)
                                tf.summary.scalar("recall", recall, global_step)
                                tf.summary.scalar("f1", f1, global_step)

                        lr = optimizer.learning_rate
                        learning_rate = lr(step)

                        with writer.as_default():
                            tf.summary.scalar("lr", learning_rate, global_step)
                            tf.summary.scalar(
                                "loss", (loss_metric.result() - logging_loss) / args.logging_steps, global_step
                            )

                        logging_loss = loss_metric.result()

                    with writer.as_default():
                        tf.summary.scalar("loss", loss_metric.result(), step=step)

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        model.save_pretrained(output_dir)
                        logging.info("Saving model checkpoint to %s", output_dir)

                train_iterator.child.comment = f"loss : {loss_metric.result()}"
                step += 1

        train_iterator.write(f"loss epoch {epoch + 1}: {loss_metric.result()}")

        loss_metric.reset_states()

    logging.info("  Training took time = {}".format(datetime.datetime.now() - current_time))


def evaluate():
    ""


def load_and_cache_examples(args, tokenizer, evaluate=False):
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
                logger.warning("tensorflow_datasets does not handle version 2 of SQuAD.")

            examples = processor.get_examples_from_dataset(tfds.load("squad"), evaluate=evaluate)
            examples = examples

        logger.info("Converting examples to features")
        dataset = squad_convert_examples_to_features(
            examples,
            tokenizer,
            args.max_seq_length,
            args.doc_stride,
            args.max_query_length,
            is_training=not evaluate,
            return_dataset="tf",
        )

        if not os.path.exists(features_output_dir):
            os.makedirs(features_output_dir)

        with tf.compat.v1.python_io.TFRecordWriter(cached_features_file) as tfwriter:
            for feature in dataset:
                example, result = feature
                feature_key_value_pair = {
                    "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=example["input_ids"])),
                    "attention_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=example["attention_mask"])),
                    "token_type_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=example["token_type_ids"])),
                    "start_position": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[result["start_position"]])
                    ),
                    "end_position": tf.train.Feature(int64_list=tf.train.Int64List(value=[result["end_position"]])),
                    "cls_index": tf.train.Feature(int64_list=tf.train.Int64List(value=[result["cls_index"]])),
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
        return x, y

    dataset = tf.data.TFRecordDataset(cached_features_file)
    dataset = dataset.map(select_data_from_record)

    logger.info("Created dataset %s from TFRecord" % "dev" if evaluate else "train")
    return dataset, len(list(dataset.__iter__()))


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

    parser.add_argument(
        "--per_device_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for validation during training.",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation after training.",
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
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
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

    parser.add_argument(
        "--tpu",
        default=None,
        help="The Cloud TPU to use for training. This should be either the name "
        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
        "url.",
    )
    parser.add_argument("--num_tpu_cores", default="8", help="Total number of TPU cores to use.")
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma separated list of gpus devices. If only one, switch to single gpu strategy, if None takes all the gpus available.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        help="Linear warmup over warmup_steps.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, help="Max gradient norm."
    )
    parser.add_argument("--logging_steps", default=50, help="Log every X updates.")
    parser.add_argument("--save_steps", default=50, help="Save checkpoint every X updates.")
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

        if args.amp:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    if args.tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        args.n_device = args.num_tpu_cores
    elif len(args.gpus.split(",")) > 1:
        args.n_device = len([f"/gpu:{gpu}" for gpu in args.gpus.split(",")])
        strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{gpu}" for gpu in args.gpus.split(",")])
    elif args.no_cuda:
        args.n_device = 1
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    else:
        args.n_device = len(args.gpus.split(","))
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:" + args.gpus.split(",")[0])

    logging.warning(
        "n_device: %s, distributed training: %s, 16-bits training: %s",
        args.n_device,
        bool(args.n_device > 1),
        args.amp,
    )

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    logging.info("Training/evaluation parameters %s", args)

    if args.do_train:
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        with strategy.scope():
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_pt=bool(".bin" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            model.layers[-1].activation = tf.keras.activations.softmax

        train_batch_size = args.per_device_train_batch_size * args.n_device
        train_dataset, num_train_examples = load_and_cache_examples(args, tokenizer, evaluate=False)

        train_dataset = train_dataset.batch(train_batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=train_batch_size)
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)

        train(
            args, strategy, train_dataset, tokenizer, model, num_train_examples, train_batch_size,
        )

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logging.info("Saving model to %s", args.output_dir)

        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
