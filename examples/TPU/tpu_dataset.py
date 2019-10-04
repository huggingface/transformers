# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
""" === Under active development === Dataset load
"""


import tensorflow as tf
from transformers import glue_convert_examples_to_features, glue_processors


def create_dataset(tokenizer,
                   file_path,
                   seq_length,
                   batch_size,
                   is_training=True,
                   drop_remainder=False
):
    processor = glue_processors["mnli"]()
    examples = processor.get_dev_examples(file_path)
    features = glue_convert_examples_to_features(examples, tokenizer, seq_length, 'mnli')

    all_input_ids = tf.constant([f.input_ids for f in features])
    all_attention_masks = tf.constant([f.attention_mask for f in features])
    all_token_type_ids = tf.constant([f.token_type_ids for f in features])
    all_labels = tf.constant([f.label for f in features])

    dataset = tf.data.Dataset.from_tensor_slices(({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "token_type_ids": all_token_type_ids
    }, all_labels))

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(1024)

    return dataset


if __name__ == "__main__":
    from transformers import BertTokenizer

    train_data_path = "/home/lysandre/transformers/examples/TPU/glue_data/MNLI"
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    seq_length = 128
    batch_size = 32

    create_dataset(
        tokenizer, train_data_path, seq_length, batch_size
    )
