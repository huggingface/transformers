# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import unittest

from transformers import RobertaTokenizer, RobertaTokenizerFast
from transformers.testing_utils import require_spacy, require_tokenizers, require_torch, slow
from transformers.utils import cached_property, is_spacy_available, is_torch_available


if is_spacy_available() and is_torch_available():
    from transformers import GreaseLMFeatureExtractor, GreaseLMProcessor


@require_tokenizers
class GreaseLMProcessorTest(unittest.TestCase):
    tokenizer_class = RobertaTokenizer
    rust_tokenizer_class = RobertaTokenizerFast

    def setUp(self):
        pass


@require_torch
@require_spacy
class GreaseLMProcessorIntegrationTests(unittest.TestCase):
    @cached_property
    def get_tokenizers(self):
        slow_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        fast_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
        return [slow_tokenizer, fast_tokenizer]

    @slow
    def test_processor_common_sense_qa(self):
        feature_extractor = GreaseLMFeatureExtractor.from_pretrained("Xikun/greaselm-csqa")
        tokenizers = self.get_tokenizers
        csqa_example1 = {
            "answerKey": "A",
            "id": "1afa02df02c908a558b4036e80242fac",
            "question": {
                "question_concept": "revolving door",
                "choices": [
                    {"label": "A", "text": "bank"},
                    {"label": "B", "text": "library"},
                    {"label": "C", "text": "department store"},
                    {"label": "D", "text": "mall"},
                    {"label": "E", "text": "new york"},
                ],
                "stem": (
                    "A revolving door is convenient for two direction travel, but it also serves as a security measure"
                    " at a what?"
                ),
            },
        }

        csqa_example2 = {
            "answerKey": "A",
            "id": "a7ab086045575bb497933726e4e6ad28",
            "question": {
                "question_concept": "people",
                "choices": [
                    {"label": "A", "text": "complete job"},
                    {"label": "B", "text": "learn from each other"},
                    {"label": "C", "text": "kill animals"},
                    {"label": "D", "text": "wear hats"},
                    {"label": "E", "text": "talk to each other"},
                ],
                "stem": "What do people aim to do at work?",
            },
        }

        # multiple example batch
        for tokenizer in tokenizers:
            processor = GreaseLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            self.verify_batch(processor, [csqa_example1, csqa_example2], num_choices=5)

        # single example batch
        for tokenizer in tokenizers:
            processor = GreaseLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            self.verify_batch(processor, [csqa_example1], num_choices=5)

    @slow
    def test_processor_openbook_qa(self):
        feature_extractor = GreaseLMFeatureExtractor.from_pretrained("Xikun/greaselm-obqa")
        tokenizers = self.get_tokenizers

        obqa_example1 = {
            "id": "7-97",
            "question": {
                "stem": "A person can grow cabbage in January with the help of what product?",
                "choices": [
                    {"text": "Green house", "label": "A"},
                    {"text": "Parliament", "label": "B"},
                    {"text": "Congress", "label": "C"},
                    {"text": "White house", "label": "D"},
                ],
            },
            "answerKey": "A",
        }
        obqa_example2 = {
            "id": "1953",
            "question": {
                "stem": "Which organism cannot specialize?",
                "choices": [
                    {"text": "mammal", "label": "A"},
                    {"text": "plant", "label": "B"},
                    {"text": "fish", "label": "C"},
                    {"text": "protozoa", "label": "D"},
                ],
            },
            "answerKey": "D",
        }

        # multiple example batch
        for tokenizer in tokenizers:
            processor = GreaseLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            self.verify_batch(processor, [obqa_example1, obqa_example2], num_choices=4)

        # single example batch
        for tokenizer in tokenizers:
            processor = GreaseLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            self.verify_batch(processor, [obqa_example1], num_choices=4)

    def verify_batch(self, processor, batch_input, num_choices=5):
        batch_length = len(batch_input)
        batch_encoded = processor(batch_input)
        assert len(batch_encoded.keys()) == 12  #
        features = [
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "special_tokens_mask",
            "labels",
            "concept_ids",
            "node_type_ids",
            "node_scores",
            "adj_lengths",
            "special_nodes_mask",
            "edge_index",
            "edge_type",
        ]
        self.assertListEqual(list(batch_encoded.keys()), features)
        # check LM features, graph feature are checked in GreaseLMFeatureExtractionTest
        seq_len = processor.max_seq_length
        expected_shape = {
            "input_ids": (batch_length, num_choices, seq_len),
            "token_type_ids": (batch_length, num_choices, seq_len),
            "attention_mask": (batch_length, num_choices, seq_len),
            "special_tokens_mask": (batch_length, num_choices, seq_len),
            "labels": (batch_length,),
        }
        for key, value in expected_shape.items():
            self.assertEqual(batch_encoded[key].shape, value)
