# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
import sys
import unittest


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(git_repo_path, "utils"))

import get_test_info  # noqa: E402
from get_test_info import (  # noqa: E402
    get_model_to_test_mapping,
    get_model_to_tester_mapping,
    get_test_to_tester_mapping,
)


BERT_TEST_FILE = os.path.join("tests", "models", "bert", "test_modeling_bert.py")
BLIP_TEST_FILE = os.path.join("tests", "models", "blip", "test_modeling_blip.py")


class GetTestInfoTester(unittest.TestCase):
    def test_get_test_to_tester_mapping(self):
        bert_test_tester_mapping = get_test_to_tester_mapping(BERT_TEST_FILE)
        blip_test_tester_mapping = get_test_to_tester_mapping(BLIP_TEST_FILE)

        EXPECTED_BERT_MAPPING = {"BertModelTest": "BertModelTester"}

        EXPECTED_BLIP_MAPPING = {
            "BlipModelTest": "BlipModelTester",
            "BlipTextImageModelTest": "BlipTextImageModelsModelTester",
            "BlipTextModelTest": "BlipTextModelTester",
            "BlipTextRetrievalModelTest": "BlipTextRetrievalModelTester",
            "BlipVQAModelTest": "BlipVQAModelTester",
            "BlipVisionModelTest": "BlipVisionModelTester",
        }

        self.assertEqual(get_test_info.to_json(bert_test_tester_mapping), EXPECTED_BERT_MAPPING)
        self.assertEqual(get_test_info.to_json(blip_test_tester_mapping), EXPECTED_BLIP_MAPPING)

    def test_get_model_to_test_mapping(self):
        bert_model_test_mapping = get_model_to_test_mapping(BERT_TEST_FILE)
        blip_model_test_mapping = get_model_to_test_mapping(BLIP_TEST_FILE)

        EXPECTED_BERT_MAPPING = {
            "BertForMaskedLM": ["BertModelTest"],
            "BertForMultipleChoice": ["BertModelTest"],
            "BertForNextSentencePrediction": ["BertModelTest"],
            "BertForPreTraining": ["BertModelTest"],
            "BertForQuestionAnswering": ["BertModelTest"],
            "BertForSequenceClassification": ["BertModelTest"],
            "BertForTokenClassification": ["BertModelTest"],
            "BertLMHeadModel": ["BertModelTest"],
            "BertModel": ["BertModelTest"],
        }

        EXPECTED_BLIP_MAPPING = {
            "BlipForConditionalGeneration": ["BlipTextImageModelTest"],
            "BlipForImageTextRetrieval": ["BlipTextRetrievalModelTest"],
            "BlipForQuestionAnswering": ["BlipVQAModelTest"],
            "BlipModel": ["BlipModelTest"],
            "BlipTextModel": ["BlipTextModelTest"],
            "BlipVisionModel": ["BlipVisionModelTest"],
        }

        self.assertEqual(get_test_info.to_json(bert_model_test_mapping), EXPECTED_BERT_MAPPING)
        self.assertEqual(get_test_info.to_json(blip_model_test_mapping), EXPECTED_BLIP_MAPPING)

    def test_get_model_to_tester_mapping(self):
        bert_model_tester_mapping = get_model_to_tester_mapping(BERT_TEST_FILE)
        blip_model_tester_mapping = get_model_to_tester_mapping(BLIP_TEST_FILE)

        EXPECTED_BERT_MAPPING = {
            "BertForMaskedLM": ["BertModelTester"],
            "BertForMultipleChoice": ["BertModelTester"],
            "BertForNextSentencePrediction": ["BertModelTester"],
            "BertForPreTraining": ["BertModelTester"],
            "BertForQuestionAnswering": ["BertModelTester"],
            "BertForSequenceClassification": ["BertModelTester"],
            "BertForTokenClassification": ["BertModelTester"],
            "BertLMHeadModel": ["BertModelTester"],
            "BertModel": ["BertModelTester"],
        }

        EXPECTED_BLIP_MAPPING = {
            "BlipForConditionalGeneration": ["BlipTextImageModelsModelTester"],
            "BlipForImageTextRetrieval": ["BlipTextRetrievalModelTester"],
            "BlipForQuestionAnswering": ["BlipVQAModelTester"],
            "BlipModel": ["BlipModelTester"],
            "BlipTextModel": ["BlipTextModelTester"],
            "BlipVisionModel": ["BlipVisionModelTester"],
        }

        self.assertEqual(get_test_info.to_json(bert_model_tester_mapping), EXPECTED_BERT_MAPPING)
        self.assertEqual(get_test_info.to_json(blip_model_tester_mapping), EXPECTED_BLIP_MAPPING)
