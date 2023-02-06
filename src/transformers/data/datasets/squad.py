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
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data import Dataset

from ...models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from ..processors.squad import SquadFeatures, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features


logger = logging.get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class SquadDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    model_type: str = field(
        default=None, metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_TYPES)}
    )
    data_dir: str = field(
        default=None, metadata={"help": "The input data dir. Should contain the .json files for the SQuAD task."}
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
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_query_length: int = field(
        default=64,
        metadata={
            "help": (
                "The maximum number of tokens for the question. Questions longer than this will "
                "be truncated to this length."
            )
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, the SQuAD examples contain some that do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}
    )
    n_best_size: int = field(
        default=20, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}
    )
    lang_id: int = field(
        default=0,
        metadata={
            "help": (
                "language id of input for language-specific xlm models (see"
                " tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)"
            )
        },
    )
    threads: int = field(default=1, metadata={"help": "multiple threads for converting example to features"})


class Split(Enum):
    train = "train"
    dev = "dev"


class SquadDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    args: SquadDataTrainingArguments
    features: List[SquadFeatures]
    mode: Split
    is_language_sensitive: bool

    def __init__(
        self,
        args: SquadDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        is_language_sensitive: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        dataset_format: Optional[str] = "pt",
    ):
        self.args = args
        self.is_language_sensitive = is_language_sensitive
        self.processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        self.mode = mode
        # Load data features from cache or dataset file
        version_tag = "v2" if args.version_2_with_negative else "v1"
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            f"cached_{mode.value}_{tokenizer.__class__.__name__}_{args.max_seq_length}_{version_tag}",
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.old_features = torch.load(cached_features_file)

                # Legacy cache files have only features, while new cache files
                # will have dataset and examples also.
                self.features = self.old_features["features"]
                self.dataset = self.old_features.get("dataset", None)
                self.examples = self.old_features.get("examples", None)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

                if self.dataset is None or self.examples is None:
                    logger.warning(
                        f"Deleting cached file {cached_features_file} will allow dataset and examples to be cached in"
                        " future run"
                    )
            else:
                if mode == Split.dev:
                    self.examples = self.processor.get_dev_examples(args.data_dir)
                else:
                    self.examples = self.processor.get_train_examples(args.data_dir)

                self.features, self.dataset = squad_convert_examples_to_features(
                    examples=self.examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=mode == Split.train,
                    threads=args.threads,
                    return_dataset=dataset_format,
                )

                start = time.time()
                torch.save(
                    {"features": self.features, "dataset": self.dataset, "examples": self.examples},
                    cached_features_file,
                )
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Convert to Tensors and build dataset
        feature = self.features[i]

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        cls_index = torch.tensor(feature.cls_index, dtype=torch.long)
        p_mask = torch.tensor(feature.p_mask, dtype=torch.float)
        is_impossible = torch.tensor(feature.is_impossible, dtype=torch.float)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
            del inputs["token_type_ids"]

        if self.args.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": cls_index, "p_mask": p_mask})
            if self.args.version_2_with_negative:
                inputs.update({"is_impossible": is_impossible})
            if self.is_language_sensitive:
                inputs.update({"langs": (torch.ones(input_ids.shape, dtype=torch.int64) * self.args.lang_id)})

        if self.mode == Split.train:
            start_positions = torch.tensor(feature.start_position, dtype=torch.long)
            end_positions = torch.tensor(feature.end_position, dtype=torch.long)
            inputs.update({"start_positions": start_positions, "end_positions": end_positions})

        return inputs
