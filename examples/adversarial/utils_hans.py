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

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import tqdm

from filelock import FileLock
from transformers import (
    BartTokenizer,
    BartTokenizerFast,
    DataProcessor,
    PreTrainedTokenizer,
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizer,
    is_tf_available,
    is_torch_available,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        pairID: (Optional) string. Unique identifier for the pair of sentences.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    pairID: Optional[str] = None


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
        pairID: (Optional) Unique identifier for the pair of sentences.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    pairID: Optional[int] = None


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class HansDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            evaluate: bool = False,
        ):
            processor = hans_processors[task]()

            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}".format(
                    "dev" if evaluate else "train",
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                    task,
                ),
            )
            label_list = processor.get_labels()
            if tokenizer.__class__ in (
                RobertaTokenizer,
                RobertaTokenizerFast,
                XLMRobertaTokenizer,
                BartTokenizer,
                BartTokenizerFast,
            ):
                # HACK(label indices are swapped in RoBERTa pretrained model)
                label_list[1], label_list[2] = label_list[2], label_list[1]
            self.label_list = label_list

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")

                    examples = (
                        processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
                    )

                    logger.info("Training examples: %s", len(examples))
                    self.features = hans_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

        def get_labels(self):
            return self.label_list


if is_tf_available():
    import tensorflow as tf

    class TFHansDataset:
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = 128,
            overwrite_cache=False,
            evaluate: bool = False,
        ):
            processor = hans_processors[task]()
            label_list = processor.get_labels()
            if tokenizer.__class__ in (
                RobertaTokenizer,
                RobertaTokenizerFast,
                XLMRobertaTokenizer,
                BartTokenizer,
                BartTokenizerFast,
            ):
                # HACK(label indices are swapped in RoBERTa pretrained model)
                label_list[1], label_list[2] = label_list[2], label_list[1]
            self.label_list = label_list

            examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
            self.features = hans_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)

            def gen():
                for (ex_index, ex) in tqdm.tqdm(enumerate(self.features), desc="convert examples to features"):
                    if ex_index % 10000 == 0:
                        logger.info("Writing example %d of %d" % (ex_index, len(examples)))

                    yield (
                        {
                            "example_id": 0,
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                        },
                        ex.label,
                    )

            self.dataset = tf.data.Dataset.from_generator(
                gen,
                (
                    {
                        "example_id": tf.int32,
                        "input_ids": tf.int32,
                        "attention_mask": tf.int32,
                        "token_type_ids": tf.int32,
                    },
                    tf.int64,
                ),
                (
                    {
                        "example_id": tf.TensorShape([]),
                        "input_ids": tf.TensorShape([None, None]),
                        "attention_mask": tf.TensorShape([None, None]),
                        "token_type_ids": tf.TensorShape([None, None]),
                    },
                    tf.TensorShape([]),
                ),
            )

        def get_dataset(self):
            return self.dataset

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

        def get_labels(self):
            return self.label_list


class HansProcessor(DataProcessor):
    """Processor for the HANS data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "heuristics_train_set.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "heuristics_evaluation_set.txt")), "dev")

    def get_labels(self):
        """See base class.
        Note that we follow the standard three labels for MNLI
        (see :class:`~transformers.data.processors.utils.MnliProcessor`)
        but the HANS evaluation groups `contradiction` and `neutral` into `non-entailment` (label 0) while
        `entailment` is label 1."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[5]
            text_b = line[6]
            pairID = line[7][2:] if line[7].startswith("ex") else line[7]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, pairID=pairID))
        return examples


def hans_convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples.
        max_length: Maximum example length.
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method.
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``.

    Returns:
        A list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
        )

        label = label_map[example.label] if example.label in label_map else 0

        pairID = int(example.pairID)

        features.append(InputFeatures(**inputs, label=label, pairID=pairID))

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info(f"guid: {example}")
        logger.info(f"features: {features[i]}")

    return features


hans_tasks_num_labels = {
    "hans": 3,
}

hans_processors = {
    "hans": HansProcessor,
}
