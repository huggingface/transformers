# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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


import json
import os
import tempfile
import unittest

from transformers.modelcard import ModelCard


class ModelCardTester(unittest.TestCase):
    def setUp(self):
        self.inputs_dict = {
            "model_details": {
                "Organization": "testing",
                "Model date": "today",
                "Model version": "v2.1, Developed by Test Corp in 2019.",
                "Architecture": "Convolutional Neural Network.",
            },
            "metrics": "BLEU and ROUGE-1",
            "evaluation_data": {
                "Datasets": {"BLEU": "My-great-dataset-v1", "ROUGE-1": "My-short-dataset-v2.1"},
                "Preprocessing": "See details on https://arxiv.org/pdf/1810.03993.pdf",
            },
            "training_data": {
                "Dataset": "English Wikipedia dump dated 2018-12-01",
                "Preprocessing": (
                    "Using SentencePiece vocabulary of size 52k tokens. See details on"
                    " https://arxiv.org/pdf/1810.03993.pdf"
                ),
            },
            "quantitative_analyses": {"BLEU": 55.1, "ROUGE-1": 76},
        }

    def test_model_card_common_properties(self):
        modelcard = ModelCard.from_dict(self.inputs_dict)
        self.assertTrue(hasattr(modelcard, "model_details"))
        self.assertTrue(hasattr(modelcard, "intended_use"))
        self.assertTrue(hasattr(modelcard, "factors"))
        self.assertTrue(hasattr(modelcard, "metrics"))
        self.assertTrue(hasattr(modelcard, "evaluation_data"))
        self.assertTrue(hasattr(modelcard, "training_data"))
        self.assertTrue(hasattr(modelcard, "quantitative_analyses"))
        self.assertTrue(hasattr(modelcard, "ethical_considerations"))
        self.assertTrue(hasattr(modelcard, "caveats_and_recommendations"))

    def test_model_card_to_json_string(self):
        modelcard = ModelCard.from_dict(self.inputs_dict)
        obj = json.loads(modelcard.to_json_string())
        for key, value in self.inputs_dict.items():
            self.assertEqual(obj[key], value)

    def test_model_card_to_json_file(self):
        model_card_first = ModelCard.from_dict(self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = os.path.join(tmpdirname, "modelcard.json")
            model_card_first.to_json_file(filename)
            model_card_second = ModelCard.from_json_file(filename)

        self.assertEqual(model_card_second.to_dict(), model_card_first.to_dict())

    def test_model_card_from_and_save_pretrained(self):
        model_card_first = ModelCard.from_dict(self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_card_first.save_pretrained(tmpdirname)
            model_card_second = ModelCard.from_pretrained(tmpdirname)

        self.assertEqual(model_card_second.to_dict(), model_card_first.to_dict())
