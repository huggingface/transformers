# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
import unittest
import inspect

from transformers import is_torch_available
from .utils import require_torch, torch_device

if is_torch_available():
    from transformers import (
        AlbertConfig,
        AlbertModel,
        AlbertForMaskedLM,
        AlbertForSequenceClassification,
        AlbertForTokenClassification,
        AlbertForQuestionAnswering,
        BertConfig,
        BertModel,
        BertForMaskedLM,
        BertForNextSentencePrediction,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertForTokenClassification,
        BertForMultipleChoice,
        DistilBertConfig,
        DistilBertModel,
        DistilBertForMaskedLM,
        DistilBertForTokenClassification,
        DistilBertForQuestionAnswering,
        DistilBertForSequenceClassification,
        RobertaForMaskedLM,
        RobertaForSequenceClassification,
        RobertaForTokenClassification,
        RobertaForQuestionAnswering,
        AlbertForMaskedLM,
        AlbertForSequenceClassification,
        AlbertForTokenClassification,
        AlbertForQuestionAnswering,

    )


import ast
import astor



@require_torch
class ConsistencyModelTest(unittest.TestCase):
    def check_equiv(self, f1, f2):
        src1 = inspect.getsource(f1)
        src2 = inspect.getsource(f2)
        ast1 = ast.parse("class Temp: \n" + src1).body[0].body[0].body[1:]
        ast2 = ast.parse("class Temp: \n" + src2).body[0].body[0].body[1:]
        for l1, l2 in zip(ast1, ast2):
            s1 = ast.dump(l1)
            s2 = ast.dump(l2)
            if s1 != s2:
                print("l1", astor.code_gen.to_source(l1))
                print("l2", astor.code_gen.to_source(l2))
                self.assertTrue(False)


    def test_sequence_classification(self):
        self.check_equiv(RobertaForTokenClassification.forward,
                         BertForTokenClassification.forward)
        self.check_equiv(RobertaForQuestionAnswering.forward,
                         BertForQuestionAnswering.forward)
        self.check_equiv(RobertaForTokenClassification.forward,
                         BertForTokenClassification.forward)
        self.check_equiv(AlbertForTokenClassification.forward,
                         BertForTokenClassification.forward)


        # self.check_equiv(DistilBertForQuestionAnswering.forward,
        #                  BertForQuestionAnswering.forward)

        # self.check_equiv(RobertaForMaskedLM.forward,
        #                  BertForMaskedLM.forward)
