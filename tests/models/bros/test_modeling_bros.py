# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Bros model. """


import unittest

from transformers import BrosConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        BrosForTokenClassification,
        BrosModel,
    )
    from transformers.models.bros.modeling_bros import (
        BROS_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


class BrosModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
        range_bbox=1000,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.range_bbox = range_bbox

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        bbox = ids_tensor([self.batch_size, self.seq_length, 8], 1)
        # Ensure that bbox is legal
        for i in range(bbox.shape[0]):
            for j in range(bbox.shape[1]):
                if bbox[i, j, 3] < bbox[i, j, 1]:
                    t = bbox[i, j, 3]
                    bbox[i, j, 3] = bbox[i, j, 1]
                    bbox[i, j, 1] = t
                if bbox[i, j, 2] < bbox[i, j, 0]:
                    t = bbox[i, j, 2]
                    bbox[i, j, 2] = bbox[i, j, 0]
                    bbox[i, j, 0] = t

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, bbox, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return BrosConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(
        self, config, input_ids, bbox, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BrosModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, bbox=bbox, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, bbox=bbox, token_type_ids=token_type_ids)
        result = model(input_ids, bbox=bbox)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_token_classification(
        self, config, input_ids, bbox, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = BrosForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids, bbox=bbox, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            bbox,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_torch
class BrosModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            BrosForTokenClassification,
            BrosModel,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = () if is_torch_available() else ()

    def setUp(self):
        self.model_tester = BrosModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BrosConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in BROS_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BrosModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


def prepare_bros_batch_inputs():
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    labels = torch.tensor(
        [[-100, 6, 6, 6, 6, 5, 5, 5, 6, 6, 6, 6, -100], [-100, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, -100]]
    )

    bbox = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.5989650711513583,
                    0.654,
                    0.6377749029754204,
                    0.654,
                    0.6377749029754204,
                    0.667,
                    0.5989650711513583,
                    0.667,
                ],
                [
                    0.5989650711513583,
                    0.654,
                    0.6377749029754204,
                    0.654,
                    0.6377749029754204,
                    0.667,
                    0.5989650711513583,
                    0.667,
                ],
                [
                    0.6377749029754204,
                    0.654,
                    0.6804657179818887,
                    0.654,
                    0.6804657179818887,
                    0.667,
                    0.6377749029754204,
                    0.667,
                ],
                [
                    0.6377749029754204,
                    0.654,
                    0.6804657179818887,
                    0.654,
                    0.6804657179818887,
                    0.667,
                    0.6377749029754204,
                    0.667,
                ],
                [
                    0.1203104786545925,
                    0.683,
                    0.24708926261319533,
                    0.683,
                    0.24708926261319533,
                    0.698,
                    0.1203104786545925,
                    0.698,
                ],
                [
                    0.1203104786545925,
                    0.683,
                    0.24708926261319533,
                    0.683,
                    0.24708926261319533,
                    0.698,
                    0.1203104786545925,
                    0.698,
                ],
                [
                    0.1203104786545925,
                    0.683,
                    0.24708926261319533,
                    0.683,
                    0.24708926261319533,
                    0.698,
                    0.1203104786545925,
                    0.698,
                ],
                [
                    0.26002587322121606,
                    0.683,
                    0.3053040103492885,
                    0.683,
                    0.3053040103492885,
                    0.698,
                    0.26002587322121606,
                    0.698,
                ],
                [
                    0.26002587322121606,
                    0.683,
                    0.3053040103492885,
                    0.683,
                    0.3053040103492885,
                    0.698,
                    0.26002587322121606,
                    0.698,
                ],
                [
                    0.31565329883570503,
                    0.683,
                    0.35058214747736094,
                    0.683,
                    0.35058214747736094,
                    0.696,
                    0.31565329883570503,
                    0.696,
                ],
                [
                    0.31565329883570503,
                    0.683,
                    0.35058214747736094,
                    0.683,
                    0.35058214747736094,
                    0.696,
                    0.31565329883570503,
                    0.696,
                ],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.6788511749347258,
                    0.563,
                    0.7245430809399478,
                    0.563,
                    0.7245430809399478,
                    0.58,
                    0.6788511749347258,
                    0.58,
                ],
                [
                    0.7362924281984334,
                    0.566,
                    0.7728459530026109,
                    0.566,
                    0.7728459530026109,
                    0.579,
                    0.7362924281984334,
                    0.579,
                ],
                [
                    0.7806788511749347,
                    0.566,
                    0.8224543080939948,
                    0.566,
                    0.8224543080939948,
                    0.579,
                    0.7806788511749347,
                    0.579,
                ],
                [
                    0.8289817232375979,
                    0.567,
                    0.8785900783289817,
                    0.567,
                    0.8785900783289817,
                    0.578,
                    0.8289817232375979,
                    0.578,
                ],
                [
                    0.206266318537859,
                    0.581,
                    0.2898172323759791,
                    0.581,
                    0.2898172323759791,
                    0.595,
                    0.206266318537859,
                    0.595,
                ],
                [
                    0.3028720626631854,
                    0.581,
                    0.3263707571801567,
                    0.581,
                    0.3263707571801567,
                    0.594,
                    0.3028720626631854,
                    0.594,
                ],
                [
                    0.3368146214099217,
                    0.581,
                    0.35900783289817234,
                    0.581,
                    0.35900783289817234,
                    0.595,
                    0.3368146214099217,
                    0.595,
                ],
                [
                    0.3368146214099217,
                    0.581,
                    0.35900783289817234,
                    0.581,
                    0.35900783289817234,
                    0.595,
                    0.3368146214099217,
                    0.595,
                ],
                [
                    0.36161879895561355,
                    0.581,
                    0.38511749347258484,
                    0.581,
                    0.38511749347258484,
                    0.595,
                    0.36161879895561355,
                    0.595,
                ],
                [
                    0.36161879895561355,
                    0.581,
                    0.38511749347258484,
                    0.581,
                    0.38511749347258484,
                    0.595,
                    0.36161879895561355,
                    0.595,
                ],
                [
                    0.38511749347258484,
                    0.581,
                    0.4073107049608355,
                    0.581,
                    0.4073107049608355,
                    0.595,
                    0.38511749347258484,
                    0.595,
                ],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    input_ids = torch.tensor(
        [
            [101, 6640, 1011, 8732, 1012, 14017, 12083, 15148, 1006, 2156, 2061, 2361, 102],
            [101, 2199, 15805, 2001, 2109, 2612, 1997, 1016, 1012, 1015, 1013, 4413, 102],
        ]
    )

    return input_ids, bbox, attention_mask, labels


@require_torch
class BrosModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = BrosModel.from_pretrained("naver-clova-ocr/bros-base-uncased")
        # input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])

        input_ids, bbox, attention_mask, labels = prepare_bros_batch_inputs()

        with torch.no_grad():
            output = model(input_ids, bbox, attention_mask=attention_mask)[0]

        # TODO need to write expected shape and slice
        # and check values

    def test_inference_token_classification(self):
        ...

    # TODO add other tests
