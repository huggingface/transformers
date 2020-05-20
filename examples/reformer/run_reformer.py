# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import logging
import os
import sys

import numpy as np
import torch

import nlp

from transformers import (
    ReformerConfig,
    ReformerModelWithLMHead,
    DataCollator,
    ReformerTokenizer,
)
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)


class ReformerCollator(DataCollator):
    def __init__(self, max_roll_length):
        self.max_roll_length = 3

    # From the official notebook: "Normally we would have a dataset with many examples, but for this demonstration we fit a language model on the single novel only. We don't want the model to just memorize the dataset by encoding the words in its position embeddings, so at each training iteration we will randomly select how much padding to put before the text vs. after it"
    def collate_batch(self, features):
        # get random shift int
        random_shift_length = torch.randint(self.max_roll_length, (1,)).item()

        # shift input and mask
        rolled_input_ids = torch.roll(
            features[0]["input_ids"], random_shift_length
        ).unsqueeze(0)
        rolled_attention_mask = torch.roll(
            features[0]["attention_mask"], random_shift_length
        ).unsqueeze(0)

        # return dict having the correct argument naming
        return {
            "input_ids": rolled_input_ids,  # BS x SEQ_LEN
            "labels": rolled_input_ids,  # BS x SEQ_LEN
            "attention_mask": rolled_attention_mask,  # BS x SEQ_LEN
        }


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(TrainingArguments)
    sequence_length = 2 ** 13  # 524288

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        training_args = parser.parse_args_into_dataclasses()[0]

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
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = {
        "attention_head_size": 64,
        "attn_layers": ["local", "lsh", "local", "lsh", "local", "lsh"],
        "axial_pos_embds": True,
        "sinusoidal_pos_embds": False,
        "axial_pos_embds_dim": [64, 192],
        # "axial_pos_shape": [512, 1024],
        "axial_pos_shape": [2 ** 6, 2 ** 7],
        "lsh_attn_chunk_length": 64,
        "local_attn_chunk_length": 64,
        "feed_forward_size": 512,
        "hidden_act": "relu",
        "hidden_size": 256,
        "is_decoder": True,
        "max_position_embeddings": 2 ** 13,
        "num_attention_heads": 2,
        "num_buckets": [64, 128],
        "num_hashes": 1,
        "vocab_size": 320,
        "lsh_attention_probs_dropout_prob": 0.0,
        "lsh_num_chunks_before": 1,
        "lsh_num_chunks_after": 0,
        "local_num_chunks_before": 1,
        "local_num_chunks_after": 0,
        "local_attention_probs_dropout_prob": 0.05,
        "hidden_dropout_prob": 0.05,
    }

    config = ReformerConfig(**config)
    model = ReformerModelWithLMHead(config)

    # Get datasets
    dataset = nlp.load_dataset("crime_and_punish", split="train[:1%]")
    tokenizer = ReformerTokenizer.from_pretrained(
        "google/reformer-crime-and-punishment"
    )

    # define our map function to reduce the dataset to one sample
    def flatten_and_tokenize(batch):
        all_input_text = ["".join(batch["line"])]
        input_ids_dict = tokenizer.batch_encode_plus(
            all_input_text, pad_to_max_length=True, max_length=sequence_length
        )

        # duplicate data 8 times to have have 8 examples in dataset
        for key in input_ids_dict.keys():
            input_ids_dict[key] = [8 * [x] for x in input_ids_dict[key]][0]

        return input_ids_dict

    # # reduce the dataset and set batch_size to all inputs
    dataset = dataset.map(
        flatten_and_tokenize, batched=True, batch_size=-1, remove_columns=["line"]
    )

    # prepare dataset to be in torch format
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def compute_metrics(pred):
        non_padded_indices = pred.label_ids != -100

        # correctly shift labels and pred as it's done in forward()
        labels = pred.label_ids[..., 1:][non_padded_indices[..., 1:]]
        pred = np.argmax(pred.predictions[:, :-1], axis=-1)[
            non_padded_indices[..., :-1]
        ]

        acc = np.mean(np.asarray(pred == labels), dtype=np.float)
        return {"accuracy": acc}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    return


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
