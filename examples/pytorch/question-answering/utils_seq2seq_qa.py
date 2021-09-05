# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Post-processing utilities for question answering.
"""
import logging
from typing import List, Tuple


logger = logging.getLogger(__name__)


def prepreocess_sqaud_batch(
    examples,
    question_column: str,
    context_column: str,
    answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]

    def generate_input(_question, _context):
        return " ".join(["question:", _question, "context:", _context])

    inputs = [generate_input(question, context) for question, context in zip(questions, contexts)]
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets
