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

import os
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import torch
from filelock import FileLock
from torch.utils.data import Dataset

from ...tokenization_utils_base import PreTrainedTokenizerBase
from ...utils import check_torch_load_is_safe, logging
from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
from ..processors.utils import InputFeatures


logger = logging.get_logger(__name__)


@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command
    line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: list[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        warnings.warn(
            "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
            "library. You can have a look at this example script for pointers: "
            "https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py",
            FutureWarning,
        )
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            f"cached_{mode.value}_{tokenizer.__class__.__name__}_{args.max_seq_length}_{args.task_name}",
        )
        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__.__name__ in (
            "RobertaTokenizer",
            "RobertaTokenizerFast",
            "XLMRobertaTokenizer",
            "BartTokenizer",
            "BartTokenizerFast",
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                check_torch_load_is_safe()
                self.features = torch.load(cached_features_file, weights_only=True)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list
