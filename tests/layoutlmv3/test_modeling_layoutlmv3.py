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
""" Testing suite for the PyTorch LayoutLMv3 model. """

import unittest

from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import is_torch_available

from ..test_configuration_common import ConfigTester
from ..test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        LayoutLMv3Config,
        LayoutLMv3ForQuestionAnswering,
        LayoutLMv3ForSequenceClassification,
        LayoutLMv3ForTokenClassification,
        LayoutLMv3Model,
    )
    from transformers.models.layoutlmv3.modeling_layoutlmv3 import LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST


class LayoutLMv3ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        image_size=4,
        patch_size=2,
        text_seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=36,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        coordinate_size=6,
        shape_size=6,
        num_labels=3,
        num_choices=4,
        scope=None,
        range_bbox=1000,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.text_seq_length = text_seq_length
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
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.range_bbox = range_bbox

        # LayoutLMv3's sequence length equals the number of text tokens + number of patches + 1 (we add 1 for the CLS token)
        self.seq_length = text_seq_length + (image_size // patch_size) ** 2 + 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.text_seq_length], self.vocab_size)

        bbox = ids_tensor([self.batch_size, self.text_seq_length, 4], self.range_bbox)
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

        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.text_seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.text_seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = LayoutLMv3Config(
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
            initializer_range=self.initializer_range,
            coordinate_size=self.coordinate_size,
            shape_size=self.shape_size,
            input_size=self.image_size,
            patch_size=self.patch_size,
        )

        return config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels, token_labels

    def create_and_check_model(
        self, config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        model = LayoutLMv3Model(config=config)
        model.to(torch_device)
        model.eval()

        result = model(
            input_ids, bbox=bbox, pixel_values=pixel_values, attention_mask=input_mask, token_type_ids=token_type_ids
        )
        result = model(input_ids, bbox=bbox, pixel_values=pixel_values, token_type_ids=token_type_ids)
        result = model(input_ids, bbox=bbox, pixel_values=pixel_values)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = LayoutLMv3ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = LayoutLMv3ForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self, config, input_ids, bbox, pixel_values, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        model = LayoutLMv3ForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            bbox,
            pixel_values,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_torch
class LayoutLMv3ModelTest(ModelTesterMixin, unittest.TestCase):

    test_pruning = False
    test_torchscript = False
    test_mismatched_shapes = False

    all_model_classes = (
        (
            LayoutLMv3Model,
            LayoutLMv3ForSequenceClassification,
            LayoutLMv3ForTokenClassification,
            LayoutLMv3ForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = LayoutLMv3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LayoutLMv3Config, hidden_size=37)

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

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = LayoutLMv3Model.from_pretrained(model_name)
            self.assertIsNotNone(model)


def prepare_layoutlmv3_batch_inputs():
    # Here we prepare a batch of 2 sequences to test a LayoutLMv3 forward pass on:
    # fmt: off
    input_ids = torch.tensor([[101,1019,1014,1016,1037,12849,4747,1004,14246,2278,5439,4524,5002,2930,2193,2930,4341,3208,1005,1055,2171,2848,11300,3531,102],[101,4070,4034,7020,1024,3058,1015,1013,2861,1013,6070,19274,2772,6205,27814,16147,16147,4343,2047,10283,10969,14389,1012,2338,102]])  # noqa: E231
    bbox = torch.tensor([[[0,0,0,0],[423,237,440,251],[427,272,441,287],[419,115,437,129],[961,885,992,912],[256,38,330,58],[256,38,330,58],[336,42,353,57],[360,39,401,56],[360,39,401,56],[411,39,471,59],[479,41,528,59],[533,39,630,60],[67,113,134,131],[141,115,209,132],[68,149,133,166],[141,149,187,164],[195,148,287,165],[195,148,287,165],[195,148,287,165],[295,148,349,165],[441,149,492,166],[497,149,546,164],[64,201,125,218],[1000,1000,1000,1000]],[[0,0,0,0],[662,150,754,166],[665,199,742,211],[519,213,554,228],[519,213,554,228],[134,433,187,454],[130,467,204,480],[130,467,204,480],[130,467,204,480],[130,467,204,480],[130,467,204,480],[314,469,376,482],[504,684,582,706],[941,825,973,900],[941,825,973,900],[941,825,973,900],[941,825,973,900],[610,749,652,765],[130,659,168,672],[176,657,237,672],[238,657,312,672],[443,653,628,672],[443,653,628,672],[716,301,825,317],[1000,1000,1000,1000]]])  # noqa: E231
    pixel_values = torch.randn((2,3,224,224))  # noqa: E231
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],])  # noqa: E231
    token_type_ids = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])  # noqa: E231
    # fmt: on

    return input_ids, bbox, pixel_values, attention_mask, token_type_ids


@require_torch
class LayoutLMv3ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to(torch_device)

        (
            input_ids,
            bbox,
            pixel_values,
            attention_mask,
            token_type_ids,
        ) = prepare_layoutlmv3_batch_inputs()

        # forward pass
        outputs = model(
            input_ids=input_ids.to(torch_device),
            bbox=bbox.to(torch_device),
            pixel_values=pixel_values.to(torch_device),
            attention_mask=attention_mask.to(torch_device),
            token_type_ids=token_type_ids.to(torch_device),
        )

        print(outputs.keys())

        raise NotImplementedError("To do")
