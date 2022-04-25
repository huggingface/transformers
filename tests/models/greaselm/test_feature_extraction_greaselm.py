# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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
from pathlib import Path
from typing import Union

from transformers.testing_utils import require_spacy, require_torch, slow
from transformers.utils import is_spacy_available, is_torch_available

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin


if is_spacy_available() and is_torch_available():
    from transformers import GreaseLMFeatureExtractor


class GreaseLMFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        cpnet_vocab_path: Union[Path, str] = "concept.txt",
        pattern_path: Union[Path, str] = "matcher_patterns.json",
        pruned_graph_path: Union[Path, str] = "conceptnet.en.pruned.graph",
        score_model: Union[Path, str] = "distilroberta-base",
        device: str = "cpu",
        cxt_node_connects_all: bool = False,
    ):
        self.cpnet_vocab_path = cpnet_vocab_path
        self.pattern_path = pattern_path
        self.pruned_graph_path = pruned_graph_path
        self.score_model = score_model
        self.device = device
        self.cxt_node_connects_all = cxt_node_connects_all

    def prepare_feat_extract_dict(self):
        return {
            "cpnet_vocab_path": self.cpnet_vocab_path,
            "pattern_path": self.pattern_path,
            "pruned_graph_path": self.pruned_graph_path,
            "score_model": self.score_model,
            "device": self.device,
            "cxt_node_connects_all": self.cxt_node_connects_all,
        }


@require_torch
@require_spacy
class GreaseLMFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):
    feature_extraction_class = GreaseLMFeatureExtractor if (is_spacy_available() and is_torch_available()) else None

    def setUp(self):
        self.feature_extract_tester = GreaseLMFeatureExtractionTester()
        self.fe = GreaseLMFeatureExtractor.from_pretrained("Xikun/greaselm-csqa", device="cpu")

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_from_and_save_pretrained(self):
        # can't really test this use case from FeatureExtractionSavingTestMixin as we need actual remote files
        # locally resolved, not in temp dir
        pass

    def test_feat_extract_properties(self):
        feature_extractor = GreaseLMFeatureExtractor(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "cpnet_vocab_path"))
        self.assertTrue(hasattr(feature_extractor, "pattern_path"))
        self.assertTrue(hasattr(feature_extractor, "pruned_graph_path"))
        self.assertTrue(hasattr(feature_extractor, "score_model"))
        self.assertTrue(hasattr(feature_extractor, "device"))
        self.assertTrue(hasattr(feature_extractor, "cxt_node_connects_all"))

    @slow
    def test_common_sense_qa_feature_extraction(self):
        self.fe.start()  # start (if not already started) the feature extractor, takes ~ 2min
        csqa_example = {
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

        num_choices = 5
        max_node_num = 200

        # Prepare inputs
        entailed_example = self.fe.convert_commonsenseqa_to_entailment(csqa_example)
        graph_representation = self.fe([csqa_example], [entailed_example], num_choices=num_choices)

        # Verify the extracted graph representation
        expected_shape = {
            "concept_ids": (1, num_choices, max_node_num),
            "node_type_ids": (1, num_choices, max_node_num),
            "node_scores": (1, num_choices, max_node_num, 1),
            "adj_lengths": (1, num_choices),
            "special_nodes_mask": (1, num_choices, max_node_num),
            "edge_index": num_choices,
            "edge_type": num_choices,
        }

        self.assertEqual(len(graph_representation), len(expected_shape.keys()))
        self.assertTrue(all([k in graph_representation for k in expected_shape.keys()]))
        for key, value in expected_shape.items():
            if key == "edge_index" or key == "edge_type":
                self.assertEqual(len(graph_representation[key][0]), value)
            else:
                self.assertEqual(graph_representation[key].shape, value)

    @slow
    def test_openbook_qa_feature_extraction(self):
        self.fe.start()  # start (if not already started) the feature extractor, takes ~ 2min
        obqa_example = {
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

        num_choices = 4
        max_node_num = 200

        # Prepare inputs
        entailed_example = self.fe.convert_openbookqa_to_entailment(obqa_example)
        graph_representation = self.fe([obqa_example], [entailed_example], num_choices=num_choices)

        # Verify the extracted graph representation
        expected_shape = {
            "concept_ids": (1, num_choices, max_node_num),
            "node_type_ids": (1, num_choices, max_node_num),
            "node_scores": (1, num_choices, max_node_num, 1),
            "adj_lengths": (1, num_choices),
            "special_nodes_mask": (1, num_choices, max_node_num),
            "edge_index": num_choices,
            "edge_type": num_choices,
        }

        self.assertEqual(len(graph_representation), len(expected_shape.keys()))
        self.assertTrue(all([k in graph_representation for k in expected_shape.keys()]))
        for key, value in expected_shape.items():
            if key == "edge_index" or key == "edge_type":
                self.assertEqual(len(graph_representation[key][0]), value)
            else:
                self.assertEqual(graph_representation[key].shape, value)
