# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch VisualBERT model. """


import copy
import unittest

from tests.test_modeling_common import floats_tensor
from transformers import is_torch_available, MODEL_FOR_MULTIPLE_CHOICE_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, MODEL_FOR_MASKED_LM_MAPPING, MODEL_FOR_PRETRAINING_MAPPING
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask

import numpy as np
import random

if is_torch_available():
    import torch

    from transformers import VisualBertConfig, VisualBertForMultipleChoice, VisualBertModel, VisualBertForFlickr, VisualBertForNLVR, VisualBertForPreTraining, VisualBertForVQA, VisualBertForVQAAdvanced, VisualBertForMultipleChoice
    from transformers.models.visual_bert.modeling_visual_bert import VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST


class VisualBertModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        visual_seq_length=5,
        is_training=True,
        use_attention_mask=True,
        use_visual_attention_mask=True,
        use_token_type_ids=True,
        use_visual_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        visual_embedding_dim=20,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.visual_seq_length = visual_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_visual_attention_mask = use_visual_attention_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_visual_token_type_ids = use_visual_token_type_ids
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
        self.visual_embedding_dim = visual_embedding_dim
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config(self):
        return VisualBertConfig(
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
            visual_embedding_dim=self.visual_embedding_dim,
            num_labels=self.num_labels,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        visual_embeds = floats_tensor([self.batch_size, self.visual_seq_length, self.visual_embedding_dim])

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        visual_attention_mask = None
        if self.use_visual_attention_mask:
            visual_attention_mask = random_attention_mask([self.batch_size, self.visual_seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        visual_token_type_ids = None
        if self.use_visual_token_type_ids:
            visual_token_type_ids = ids_tensor([self.batch_size, self.visual_seq_length], self.type_vocab_size)

        config = self.prepare_config()
        return config, {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "visual_embeds": visual_embeds,
            "visual_token_type_ids" : visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
        }

    def prepare_config_and_inputs_for_pretraining(self):
        masked_lm_labels = None
        sentence_image_labels = None

        if self.use_labels:
            masked_lm_labels = ids_tensor([self.batch_size, self.seq_length + self.visual_seq_length], self.vocab_size)
            sentence_image_labels = ids_tensor([self.batch_size, ], self.type_sequence_label_size)

        config, input_dict = self.prepare_config_and_inputs_for_common()

        input_dict.update({"labels": masked_lm_labels, "sentence_image_labels": sentence_image_labels})

        return config, input_dict

    def prepare_config_and_inputs_for_multiple_choice(self):
        input_ids = ids_tensor([self.batch_size, self.num_choices, self.seq_length], self.vocab_size)
        visual_embeds = floats_tensor([self.batch_size, self.num_choices, self.visual_seq_length, self.visual_embedding_dim])

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = random_attention_mask([self.batch_size, self.num_choices, self.seq_length])

        visual_attention_mask = None
        if self.use_visual_attention_mask:
            visual_attention_mask = random_attention_mask([self.batch_size, self.num_choices, self.visual_seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.num_choices, self.seq_length], self.type_vocab_size)

        visual_token_type_ids = None
        if self.use_visual_token_type_ids:
            visual_token_type_ids = ids_tensor([self.batch_size, self.num_choices, self.visual_seq_length], self.type_vocab_size)

        labels = None

        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.prepare_config()
        return config, {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "visual_embeds": visual_embeds,
            "visual_token_type_ids" : visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
            "labels": labels
        }

    def prepare_config_and_inputs_for_vqa(self):
        vqa_labels = None

        if self.use_labels:
            vqa_labels = ids_tensor([self.batch_size, self.num_labels], self.num_labels)

        config, input_dict = self.prepare_config_and_inputs_for_common()

        input_dict.update({"labels": vqa_labels})
        return config, input_dict

    def prepare_config_and_inputs_for_nlvr(self):
        nlvr_labels = None

        if self.use_labels:
            nlvr_labels = ids_tensor([self.batch_size], self.num_labels)

        config, input_dict = self.prepare_config_and_inputs_for_common()

        input_dict.update({"labels": nlvr_labels})
        return config, input_dict

    def prepare_config_and_inputs_for_flickr(self):
        flickr_position = torch.cat((ids_tensor([self.batch_size, self.seq_length], self.visual_seq_length), torch.ones(self.batch_size, self.visual_seq_length, dtype=torch.long, device=torch_device) * -1), dim=-1)
        flickr_labels = None
        if self.use_labels:
            flickr_labels = ids_tensor([self.batch_size, self.seq_length + self.visual_seq_length, self.visual_seq_length], 2)

        config, input_dict = self.prepare_config_and_inputs_for_common()

        input_dict.update({"flickr_position": flickr_position, "labels": flickr_labels})
        return config, input_dict

    def create_and_check_model(
        self, config, input_dict
    ):
        model = VisualBertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length + self.visual_seq_length, self.hidden_size))

    def create_and_check_for_pretraining(
        self, config, input_dict
    ):
        model = VisualBertForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(result.prediction_logits.shape, (self.batch_size, self.seq_length + self.visual_seq_length, self.vocab_size))

    def create_and_check_for_vqa(
        self, config, input_dict
    ):
        model = VisualBertForVQA(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_dict
    ):
        model = VisualBertForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def create_and_check_for_nlvr(
        self, config, input_dict
    ):
        model = VisualBertForNLVR(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_flickr(
        self, config, input_dict
    ):
        model = VisualBertForFlickr(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length + self.visual_seq_length, self.visual_seq_length))


@require_torch
class VisualBertModelTest(unittest.TestCase):

    all_model_classes = (
        (
            VisualBertModel,
            VisualBertForMultipleChoice,
            VisualBertForNLVR,
            VisualBertForFlickr,
            VisualBertForVQA,
            VisualBertForVQAAdvanced,
            VisualBertForPreTraining
        )
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = VisualBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VisualBertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_pretraining()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_model_for_vqa(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_vqa()
        self.model_tester.create_and_check_for_vqa(*config_and_inputs)

    def test_model_for_nlvr(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_nlvr()
        self.model_tester.create_and_check_for_nlvr(*config_and_inputs)

    def test_model_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_multiple_choice()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_model_for_flickr(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_flickr()
        self.model_tester.create_and_check_for_flickr(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = VisualBertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class VisualBertModelIntegrationTest(unittest.TestCase):

    @slow
    def test_inference_vqa_coco_pre(self):
        model = VisualBertForPreTraining.from_pretrained("gchhablani/visualbert-vqa-coco-pre")

        input_ids = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(1, -1)
        token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1]).reshape(1, -1)
        visual_embeds = torch.ones(size=(1, 10, 2048), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.int32)
        attention_mask = torch.tensor([1] * 6).reshape(1, -1)
        visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)

        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids, visual_embeds=visual_embeds,
                       visual_attention_mask=visual_attention_mask,
                       visual_token_type_ids=visual_token_type_ids)

        vocab_size = 30522

        expected_shape = torch.Size((1, 16, vocab_size))
        self.assertEqual(output.prediction_logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[-5.1858, -5.1903, -4.9142],
              [-6.2214, -5.9238, -5.8381],
                [-6.3027, -5.9939, -5.9297]]]
        )

        self.assertTrue(torch.allclose(output.prediction_logits[:, :3, :3], expected_slice, atol=1e-4))

        expected_shape_2 = torch.Size((1, 2))
        self.assertEqual(output.seq_relationship_logits.shape, expected_shape_2)

        expected_slice_2 = torch.tensor(
            [[0.7393, 0.1754]]
        )

        self.assertTrue(torch.allclose(output.seq_relationship_logits, expected_slice_2, atol=1e-4))

    @slow
    def test_inference_vqa_pre(self):
        model = VisualBertForPreTraining.from_pretrained("gchhablani/visualbert-vqa-pre")

        input_ids = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(1, -1)
        token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1]).reshape(1, -1)
        visual_embeds = torch.ones(size=(1, 10, 2048), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.int32)
        attention_mask = torch.tensor([1] * 6).reshape(1, -1)
        visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)

        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids, visual_embeds=visual_embeds,
                       visual_attention_mask=visual_attention_mask,
                       visual_token_type_ids=visual_token_type_ids)

        vocab_size = 30522

        expected_shape = torch.Size((1, 16, vocab_size))
        self.assertEqual(output.prediction_logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[-6.2381, -6.5230, -6.7078],
              [-6.7028, -7.0870, -7.2368],
                [-6.8588, -7.3453, -7.3635]]]
        )

        self.assertTrue(torch.allclose(output.prediction_logits[:, :3, :3], expected_slice, atol=1e-4))

        expected_shape_2 = torch.Size((1, 2))
        self.assertEqual(output.seq_relationship_logits.shape, expected_shape_2)

        expected_slice_2 = torch.tensor(
            [[-0.5355, 2.0733]]
        )

        self.assertTrue(torch.allclose(output.seq_relationship_logits, expected_slice_2, atol=1e-4))

    @slow
    def test_inference_nlvr_coco_pre(self):
        model = VisualBertForPreTraining.from_pretrained("gchhablani/visualbert-nlvr2-coco-pre")

        input_ids = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(1, -1)
        token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1]).reshape(1, -1)
        visual_embeds = torch.ones(size=(1, 10, 1024), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.int32)
        attention_mask = torch.tensor([1] * 6).reshape(1, -1)
        visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)

        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids, visual_embeds=visual_embeds,
                       visual_attention_mask=visual_attention_mask,
                       visual_token_type_ids=visual_token_type_ids)

        vocab_size = 30522

        expected_shape = torch.Size((1, 16, vocab_size))
        self.assertEqual(output.prediction_logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[-6.6872, -6.7688, -6.5785],
              [-7.3175, -7.3025, -7.2098],
                [-7.1086, -7.0725, -6.9211]]]
        )

        self.assertTrue(torch.allclose(output.prediction_logits[:, :3, :3], expected_slice, atol=1e-4))

        expected_shape_2 = torch.Size((1, 2))
        self.assertEqual(output.seq_relationship_logits.shape, expected_shape_2)

        expected_slice_2 = torch.tensor(
            [[0.1609, 0.5207]]
        )

        self.assertTrue(torch.allclose(output.seq_relationship_logits, expected_slice_2, atol=1e-4))

    @slow
    def test_inference_nlvr_pre(self):
        model = VisualBertForPreTraining.from_pretrained("gchhablani/visualbert-nlvr2-pre")

        input_ids = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(1, -1)
        token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1]).reshape(1, -1)
        visual_embeds = torch.ones(size=(1, 10, 1024), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.int32)
        attention_mask = torch.tensor([1] * 6).reshape(1, -1)
        visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)

        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids, visual_embeds=visual_embeds,
                       visual_attention_mask=visual_attention_mask,
                       visual_token_type_ids=visual_token_type_ids)

        vocab_size = 30522

        expected_shape = torch.Size((1, 16, vocab_size))
        self.assertEqual(output.prediction_logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[-11.9891, -11.4899, -11.2978],
              [-8.6170, -8.4391, -8.3391],
                [-7.2381, -7.3971, -6.8986]]]
        )

        self.assertTrue(torch.allclose(output.prediction_logits[:, :3, :3], expected_slice, atol=1e-4))

        expected_shape_2 = torch.Size((1, 2))
        self.assertEqual(output.seq_relationship_logits.shape, expected_shape_2)

        expected_slice_2 = torch.tensor(
            [[0.5289, -0.0404]]
        )

        self.assertTrue(torch.allclose(output.seq_relationship_logits, expected_slice_2, atol=1e-4))

    @slow
    def test_inference_vcr_coco_pre(self):
        model = VisualBertForPreTraining.from_pretrained("gchhablani/visualbert-vcr-coco-pre")

        input_ids = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(1, -1)
        token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1]).reshape(1, -1)
        visual_embeds = torch.ones(size=(1, 10, 512), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.int32)
        attention_mask = torch.tensor([1] * 6).reshape(1, -1)
        visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)

        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids, visual_embeds=visual_embeds,
                       visual_attention_mask=visual_attention_mask,
                       visual_token_type_ids=visual_token_type_ids)

        vocab_size = 30522

        expected_shape = torch.Size((1, 16, vocab_size))
        self.assertEqual(output.prediction_logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[-8.9958, -10.4880, -9.5806],
              [-8.2890, -9.6861, -8.9888],
                [-8.2512, -9.5826, -8.9046]]]
        )

        self.assertTrue(torch.allclose(output.prediction_logits[:, :3, :3], expected_slice, atol=1e-4))

        expected_shape_2 = torch.Size((1, 2))
        self.assertEqual(output.seq_relationship_logits.shape, expected_shape_2)

        expected_slice_2 = torch.tensor(
            [[0.7907, 0.1728]]
        )

        self.assertTrue(torch.allclose(output.seq_relationship_logits, expected_slice_2, atol=1e-4))

    @slow
    def test_inference_vcr_pre(self):
        model = VisualBertForPreTraining.from_pretrained("gchhablani/visualbert-vcr-pre")

        input_ids = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(1, -1)
        token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1]).reshape(1, -1)
        visual_embeds = torch.ones(size=(1, 10, 512), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.int32)
        attention_mask = torch.tensor([1] * 6).reshape(1, -1)
        visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)

        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids, visual_embeds=visual_embeds,
                       visual_attention_mask=visual_attention_mask,
                       visual_token_type_ids=visual_token_type_ids)

        vocab_size = 30522

        expected_shape = torch.Size((1, 16, vocab_size))
        self.assertEqual(output.prediction_logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[-7.8357, -8.7081, -9.1610],
              [-8.3819, -9.0302, -9.0794],
                [-8.1837, -8.8348, -8.9581]]]
        )

        self.assertTrue(torch.allclose(output.prediction_logits[:, :3, :3], expected_slice, atol=1e-4))

        expected_shape_2 = torch.Size((1, 2))
        self.assertEqual(output.seq_relationship_logits.shape, expected_shape_2)

        expected_slice_2 = torch.tensor(
            [[3.4079, -2.7999]]
        )

        self.assertTrue(torch.allclose(output.seq_relationship_logits, expected_slice_2, atol=1e-4))

    @slow
    def test_inference_vqa(self):
        model = VisualBertForVQA.from_pretrained("gchhablani/visualbert-vqa")

        input_ids = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(1, -1)
        token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1]).reshape(1, -1)
        visual_embeds = torch.ones(size=(1, 10, 2048), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.int32)
        attention_mask = torch.tensor([1] * 6).reshape(1, -1)
        visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)

        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids, visual_embeds=visual_embeds,
                       visual_attention_mask=visual_attention_mask,
                       visual_token_type_ids=visual_token_type_ids)

        vocab_size = 30522

        expected_shape = torch.Size((1, 3129))
        self.assertEqual(output.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-8.9898, 3.0803, -1.8016, 2.4542, -8.3420, -2.0224, -3.3124,
              -4.4139, -3.1491, -3.8997]]
        )

        self.assertTrue(torch.allclose(output.logits[:, :10], expected_slice, atol=1e-4))

    @slow
    def test_inference_nlvr(self):
        model = VisualBertForNLVR.from_pretrained("gchhablani/visualbert-nlvr2")

        input_ids = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(1, -1)
        token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1]).reshape(1, -1)
        visual_embeds = torch.ones(size=(1, 10, 1024), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.int32)
        attention_mask = torch.tensor([1] * 6).reshape(1, -1)
        visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)

        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids, visual_embeds=visual_embeds,
                       visual_attention_mask=visual_attention_mask,
                       visual_token_type_ids=visual_token_type_ids)

        vocab_size = 30522

        expected_shape = torch.Size((1, 2))
        self.assertEqual(output.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-1.1436, 0.8900]]
        )

        self.assertTrue(torch.allclose(output.logits, expected_slice, atol=1e-4))

    @slow
    def test_inference_vcr(self):
        model = VisualBertForMultipleChoice.from_pretrained("gchhablani/visualbert-vcr")

        input_ids = torch.tensor([[[1, 2, 3, 4, 5, 6] for i in range(4)]])
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.ones_like(input_ids)

        visual_embeds = torch.ones(size=(1, 4, 10, 512), dtype=torch.float32) * 0.5
        visual_token_type_ids = torch.ones(size=(1, 4, 10), dtype=torch.int32)
        visual_attention_mask = torch.ones_like(visual_token_type_ids)

        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids, visual_embeds=visual_embeds,
                       visual_attention_mask=visual_attention_mask,
                       visual_token_type_ids=visual_token_type_ids)

        vocab_size = 30522

        expected_shape = torch.Size((1, 4))
        self.assertEqual(output.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-7.7697, -7.7697, -7.7697, -7.7697]]
        )

        self.assertTrue(torch.allclose(output.logits, expected_slice, atol=1e-4))
