# coding=utf-8
# Copyright 2018 The Microsoft Research Asia LayoutLMv2 Team Authors, The Hugging Face Team.
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
import copy
from detectron2.structures.image_list import ImageList

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        LayoutLMv2Config,
        LayoutLMv2ForTokenClassification,
        LayoutLMv2Model,
    )

def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key:
            setattr(configs_no_init, key, 1e-10)
    return configs_no_init


class LayoutLMv2ModelTester:
    """You can also import this e.g from .test_modeling_layoutlmv2 import LayoutLMv2ModelTester"""

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=50,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=768,
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

        bbox = ids_tensor([self.batch_size, self.seq_length, 4], self.range_bbox)
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
        
        image = ImageList(torch.zeros(self.batch_size, 3, 256, 256), 256)
        position_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

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

        config = LayoutLMv2Config(
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
        )

        return config, input_ids, bbox, image, token_type_ids, position_ids, input_mask, sequence_labels, token_labels, choice_labels

    def create_and_check_model(
        self, config, input_ids, bbox, image, token_type_ids, position_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LayoutLMv2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=input_mask, token_type_ids=token_type_ids, position_ids=position_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_token_classification(
        self, config, input_ids, bbox, image, token_type_ids, position_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = LayoutLMv2ForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=input_mask, token_type_ids=token_type_ids, position_ids=position_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            bbox,
            image,
            token_type_ids,
            position_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "bbox": bbox,
            "image": image,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_torch
class LayoutLMv2ModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            LayoutLMv2Model,
            LayoutLMv2ForTokenClassification,
        )
        if is_torch_available()
        else None
    )

    def setUp(self):
        self.model_tester = LayoutLMv2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LayoutLMv2Config, hidden_size=768)

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

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "visual.backbone" in name:
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # loss is at first position
                if "labels" in inputs_dict:
                    correct_outlen += 1  # loss is added to beginning
                # Question Answering model returns start_logits and end_logits
                if model_class in get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                    correct_outlen += 1  # start_logits and end_logits instead of only 1 output
                if "past_key_values" in outputs:
                    correct_outlen += 1  # past_key_values have been returned

                self.assertEqual(out_len, correct_outlen)

                # decoder attentions
                decoder_attentions = outputs.decoder_attentions
                self.assertIsInstance(decoder_attentions, (list, tuple))
                self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
                )

                # cross attentions
                cross_attentions = outputs.cross_attentions
                self.assertIsInstance(cross_attentions, (list, tuple))
                self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(cross_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length,
                        encoder_key_length,
                    ],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            # the image features are concatinated onto the sequence
            image_feature_pool_shape = config.image_feature_pool_shape
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length + image_feature_pool_shape[0]**2, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)
    


def prepare_layoutlmv2_batch_inputs():
    # Here we prepare a batch of 2 sequences to test a LayoutLMv2 forward pass on:
    # fmt: off
    input_ids = torch.tensor([[101,1019,1014,1016,1037,12849,4747,1004,14246,2278,5439,4524,5002,2930,2193,2930,4341,3208,1005,1055,2171,2848,11300,3531,102],[101,4070,4034,7020,1024,3058,1015,1013,2861,1013,6070,19274,2772,6205,27814,16147,16147,4343,2047,10283,10969,14389,1012,2338,102]],device=torch_device)  # noqa: E231
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],],device=torch_device)  # noqa: E231
    bbox = torch.tensor([[[0,0,0,0],[423,237,440,251],[427,272,441,287],[419,115,437,129],[961,885,992,912],[256,38,330,58],[256,38,330,58],[336,42,353,57],[360,39,401,56],[360,39,401,56],[411,39,471,59],[479,41,528,59],[533,39,630,60],[67,113,134,131],[141,115,209,132],[68,149,133,166],[141,149,187,164],[195,148,287,165],[195,148,287,165],[195,148,287,165],[295,148,349,165],[441,149,492,166],[497,149,546,164],[64,201,125,218],[1000,1000,1000,1000]],[[0,0,0,0],[662,150,754,166],[665,199,742,211],[519,213,554,228],[519,213,554,228],[134,433,187,454],[130,467,204,480],[130,467,204,480],[130,467,204,480],[130,467,204,480],[130,467,204,480],[314,469,376,482],[504,684,582,706],[941,825,973,900],[941,825,973,900],[941,825,973,900],[941,825,973,900],[610,749,652,765],[130,659,168,672],[176,657,237,672],[238,657,312,672],[443,653,628,672],[443,653,628,672],[716,301,825,317],[1000,1000,1000,1000]]],device=torch_device)  # noqa: E231
    token_type_ids = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],device=torch_device)  # noqa: E231
    # these are sequence labels (i.e. at the token level)
    labels = torch.tensor([[-100,10,10,10,9,1,-100,7,7,-100,7,7,4,2,5,2,8,8,-100,-100,5,0,3,2,-100],[-100,12,12,12,-100,12,10,-100,-100,-100,-100,10,12,9,-100,-100,-100,10,10,10,9,12,-100,10,-100]],device=torch_device)  # noqa: E231
    image = ImageList(torch.zeros(13, 3, 256, 256), 256)
    position_ids = token_type_ids
    # fmt: on

    return input_ids, attention_mask, bbox, image, token_type_ids, position_ids, labels


@require_torch
class LayoutLMv2ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_forward_pass_no_head(self):
        model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased").to(torch_device)

        input_ids, attention_mask, bbox, image, token_type_ids, position_ids, labels = prepare_layoutlmv2_batch_inputs()

        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)

        # test the sequence output on [0, :3, :3]
        expected_slice = torch.tensor(
            [[0.1785, -0.1947, -0.0425], [-0.3254, -0.2807, 0.2553], [-0.5391, -0.3322, 0.3364]],
            device=torch_device,
        )

        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-3))

        # test the pooled output on [1, :3]
        expected_slice = torch.tensor([-0.6580, -0.0214, 0.8552], device=torch_device)

        self.assertTrue(torch.allclose(outputs.pooler_output[1, :3], expected_slice, atol=1e-3))

    @slow
    def test_forward_pass_token_classification(self):
        # initialize model with randomly initialized token classification head
        model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=3).to(
            torch_device
        )

        input_ids, attention_mask, bbox, image, token_type_ids, position_ids, labels = prepare_layoutlmv2_batch_inputs()

        # forward pass
        outputs = model(
            input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )

        # test the loss calculation to be around 2.65
        # expected_loss = torch.tensor(2.65, device=torch_device)

        # The loss is currently somewhat random and can vary between 0.1-0.3 atol.
        # self.assertTrue(torch.allclose(outputs.loss, expected_loss, atol=0.1))

        # test the shape of the logits
        logits = outputs.logits
        expected_shape = torch.Size((2, 25, 13))
        self.assertEqual(logits.shape, expected_shape)
