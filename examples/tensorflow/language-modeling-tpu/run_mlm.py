#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

"""Script for training a masked language model on TPU."""

import argparse
import logging
import os
import re

import tensorflow as tf
from packaging.version import parse

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PushToHubCallback,
    TFAutoModelForMaskedLM,
    create_optimizer,
)


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

AUTO = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Train a masked language model on TPU.")
    parser.add_argument(
        "--pretrained_model_config",
        type=str,
        default="FacebookAI/roberta-base",
        help="The model config to use. Note that we don't copy the model's weights, only the config!",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="unigram-tokenizer-wikitext",
        help="The name of the tokenizer to load. We use the pretrained tokenizer to initialize the model's vocab size.",
    )

    parser.add_argument(
        "--per_replica_batch_size",
        type=int,
        default=8,
        help="Batch size per TPU core.",
    )

    parser.add_argument(
        "--no_tpu",
        action="store_true",
        help="If set, run on CPU and don't try to initialize a TPU. Useful for debugging on non-TPU instances.",
    )

    parser.add_argument(
        "--tpu_name",
        type=str,
        help="Name of TPU resource to initialize. Should be blank on Colab, and 'local' on TPU VMs.",
        default="local",
    )

    parser.add_argument(
        "--tpu_zone",
        type=str,
        help="Google cloud zone that TPU resource is located in. Only used for non-Colab TPU nodes.",
    )

    parser.add_argument(
        "--gcp_project", type=str, help="Google cloud project name. Only used for non-Colab TPU nodes."
    )

    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Use mixed-precision bfloat16 for training. This is the recommended lower-precision format for TPU.",
    )

    parser.add_argument(
        "--train_dataset",
        type=str,
        help="Path to training dataset to load. If the path begins with `gs://`"
        " then the dataset will be loaded from a Google Cloud Storage bucket.",
    )

    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=2**18,  # Default corresponds to a 1GB buffer for seq_len 512
        help="Size of the shuffle buffer (in samples)",
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        help="Path to evaluation dataset to load. If the path begins with `gs://`"
        " then the dataset will be loaded from a Google Cloud Storage bucket.",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to train for.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use for training.",
    )

    parser.add_argument(
        "--weight_decay_rate",
        type=float,
        default=1e-3,
        help="Weight decay rate to use for training.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length of tokenized sequences. Should match the setting used in prepare_tfrecord_shards.py",
    )

    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Fraction of tokens to mask during training.",
    )

    parser.add_argument("--output_dir", type=str, required=True, help="Path to save model checkpoints to.")
    parser.add_argument("--hub_model_id", type=str, help="Model ID to upload to on the Hugging Face Hub.")

    args = parser.parse_args()
    return args


def initialize_tpu(args):
    try:
        if args.tpu_name:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
                args.tpu_name, zone=args.tpu_zone, project=args.gcp_project
            )
        else:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    except ValueError:
        raise RuntimeError(
            "Couldn't connect to TPU! Most likely you need to specify --tpu_name, --tpu_zone, or "
            "--gcp_project. When running on a TPU VM, use --tpu_name local."
        )

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    return tpu


def count_samples(file_list):
    num_samples = 0
    for file in file_list:
        filename = file.split("/")[-1]
        sample_count = re.search(r"-\d+-(\d+)\.tfrecord", filename).group(1)
        sample_count = int(sample_count)
        num_samples += sample_count

    return num_samples


def prepare_dataset(records, decode_fn, mask_fn, batch_size, shuffle, shuffle_buffer_size=None):
    num_samples = count_samples(records)
    dataset = tf.data.Dataset.from_tensor_slices(records)
    if shuffle:
        dataset = dataset.shuffle(len(dataset))
    dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=AUTO)
    # TF can't infer the total sample count because it doesn't read all the records yet, so we assert it here
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(num_samples))
    dataset = dataset.map(decode_fn, num_parallel_calls=AUTO)
    if shuffle:
        assert shuffle_buffer_size is not None
        dataset = dataset.shuffle(args.shuffle_buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(mask_fn, num_parallel_calls=AUTO)
    dataset = dataset.prefetch(AUTO)
    return dataset


def main(args):
    if not args.no_tpu:
        tpu = initialize_tpu(args)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

    if args.bfloat16:
        keras.mixed_precision.set_global_policy("mixed_bfloat16")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = AutoConfig.from_pretrained(args.pretrained_model_config)
    config.vocab_size = tokenizer.vocab_size

    training_records = tf.io.gfile.glob(os.path.join(args.train_dataset, "*.tfrecord"))
    if not training_records:
        raise ValueError(f"No .tfrecord files found in {args.train_dataset}.")
    eval_records = tf.io.gfile.glob(os.path.join(args.eval_dataset, "*.tfrecord"))
    if not eval_records:
        raise ValueError(f"No .tfrecord files found in {args.eval_dataset}.")

    num_train_samples = count_samples(training_records)

    steps_per_epoch = num_train_samples // (args.per_replica_batch_size * strategy.num_replicas_in_sync)
    total_train_steps = steps_per_epoch * args.num_epochs

    with strategy.scope():
        model = TFAutoModelForMaskedLM.from_config(config)
        model(model.dummy_inputs)  # Pass some dummy inputs through the model to ensure all the weights are built
        optimizer, schedule = create_optimizer(
            num_train_steps=total_train_steps,
            num_warmup_steps=total_train_steps // 20,
            init_lr=args.learning_rate,
            weight_decay_rate=args.weight_decay_rate,
        )

        # Transformers models compute the right loss for their task by default when labels are passed, and will
        # use this for training unless you specify your own loss function in compile().
        model.compile(optimizer=optimizer, metrics=["accuracy"])

    def decode_fn(example):
        features = {
            "input_ids": tf.io.FixedLenFeature(dtype=tf.int64, shape=(args.max_length,)),
            "attention_mask": tf.io.FixedLenFeature(dtype=tf.int64, shape=(args.max_length,)),
        }
        return tf.io.parse_single_example(example, features)

    # Many of the data collators in Transformers are TF-compilable when return_tensors == "tf", so we can
    # use their methods in our data pipeline.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_probability, mlm=True, return_tensors="tf"
    )

    def mask_with_collator(batch):
        # TF really needs an isin() function
        special_tokens_mask = (
            ~tf.cast(batch["attention_mask"], tf.bool)
            | (batch["input_ids"] == tokenizer.cls_token_id)
            | (batch["input_ids"] == tokenizer.sep_token_id)
        )
        batch["input_ids"], batch["labels"] = data_collator.tf_mask_tokens(
            batch["input_ids"],
            vocab_size=len(tokenizer),
            mask_token_id=tokenizer.mask_token_id,
            special_tokens_mask=special_tokens_mask,
        )
        return batch

    batch_size = args.per_replica_batch_size * strategy.num_replicas_in_sync

    train_dataset = prepare_dataset(
        training_records,
        decode_fn=decode_fn,
        mask_fn=mask_with_collator,
        batch_size=batch_size,
        shuffle=True,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )

    eval_dataset = prepare_dataset(
        eval_records,
        decode_fn=decode_fn,
        mask_fn=mask_with_collator,
        batch_size=batch_size,
        shuffle=False,
    )

    callbacks = []
    if args.hub_model_id:
        callbacks.append(
            PushToHubCallback(output_dir=args.output_dir, hub_model_id=args.hub_model_id, tokenizer=tokenizer)
        )

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=args.num_epochs,
        callbacks=callbacks,
    )

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
