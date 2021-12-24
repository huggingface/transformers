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
""" XNLI utils (dataset loading and evaluation) """


import os

from ...utils import logging
from .utils import DataProcessor, InputExample


logger = logging.get_logger(__name__)


class XnliProcessor(DataProcessor):
    """
    Processor for the XNLI dataset. Adapted from
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207
    """

    def __init__(self, language, train_language=None):
        self.language = language
        self.train_language = train_language

    def get_train_examples(self, data_dir):
        """See base class."""
        lg = self.language if self.train_language is None else self.train_language
        lines = self._read_tsv(os.path.join(data_dir, f"XNLI-MT-1.0/multinli/multinli.train.{lg}.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"train-{i}"
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2] == "contradictory" else line[2]
            if not isinstance(text_a, str):
                raise ValueError(f"Training input {text_a} is not a string")
            if not isinstance(text_b, str):
                raise ValueError(f"Training input {text_b} is not a string")
            if not isinstance(label, str):
                raise ValueError(f"Training label {label} is not a string")
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            language = line[0]
            if language != self.language:
                continue
            guid = f"test-{i}"
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            if not isinstance(text_a, str):
                raise ValueError(f"Training input {text_a} is not a string")
            if not isinstance(text_b, str):
                raise ValueError(f"Training input {text_b} is not a string")
            if not isinstance(label, str):
                raise ValueError(f"Training label {label} is not a string")
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]


xnli_processors = {
    "xnli": XnliProcessor,
}

xnli_output_modes = {
    "xnli": "classification",
}

xnli_tasks_num_labels = {
    "xnli": 3,
}
