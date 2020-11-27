# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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


import random
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_generation_utils import GenerationTesterMixin
from .test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        XLNetConfig,
        XLNetForMultipleChoice,
        XLNetForQuestionAnswering,
        XLNetForQuestionAnsweringSimple,
        XLNetForSequenceClassification,
        XLNetForTokenClassification,
        XLNetLMHeadModel,
        XLNetModel,
    )
    from transformers.models.xlnet.modeling_xlnet import XLNET_PRETRAINED_MODEL_ARCHIVE_LIST


class XLNetModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        mem_len=10,
        clamp_len=-1,
        reuse_len=15,
        is_training=True,
        use_labels=True,
        vocab_size=99,
        cutoffs=[10, 50, 80],
        hidden_size=32,
        num_attention_heads=4,
        d_inner=128,
        num_hidden_layers=5,
        type_sequence_label_size=2,
        untie_r=True,
        bi_data=False,
        same_length=False,
        initializer_range=0.05,
        seed=1,
        type_vocab_size=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=5,
        num_choices=4,
    ):
        self.parent = parent
        self.batch_size = 14
        self.seq_length = 7
        self.mem_len = 10
        # self.key_len = seq_length + mem_len
        self.clamp_len = -1
        self.reuse_len = 15
        self.is_training = True
        self.use_labels = True
        self.vocab_size = 99
        self.cutoffs = [10, 50, 80]
        self.hidden_size = 32
        self.num_attention_heads = 4
        self.d_inner = 128
        self.num_hidden_layers = 5
        self.type_sequence_label_size = 2
        self.untie_r = True
        self.bi_data = False
        self.same_length = False
        self.initializer_range = 0.05
        self.seed = 1
        self.type_vocab_size = 2
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 5
        self.num_choices = 4

    def prepare_config_and_inputs(self):
        input_ids_1 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids_2 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        segment_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
        input_mask = random_attention_mask([self.batch_size, self.seq_length])

        input_ids_q = ids_tensor([self.batch_size, self.seq_length + 1], self.vocab_size)
        perm_mask = torch.zeros(
            self.batch_size,
            self.seq_length + 1,
            self.seq_length + 1,
            dtype=torch.float,
            device=torch_device,
        )
        perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
        target_mapping = torch.zeros(
            self.batch_size,
            1,
            self.seq_length + 1,
            dtype=torch.float,
            device=torch_device,
        )
        target_mapping[:, 0, -1] = 1.0  # predict last token

        sequence_labels = None
        lm_labels = None
        is_impossible_labels = None
        token_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            is_impossible_labels = ids_tensor([self.batch_size], 2).float()
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        config = XLNetConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            n_head=self.num_attention_heads,
            d_inner=self.d_inner,
            n_layer=self.num_hidden_layers,
            untie_r=self.untie_r,
            mem_len=self.mem_len,
            clamp_len=self.clamp_len,
            same_length=self.same_length,
            reuse_len=self.reuse_len,
            bi_data=self.bi_data,
            initializer_range=self.initializer_range,
            num_labels=self.type_sequence_label_size,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        return (
            config,
            input_ids_1,
            input_ids_2,
            input_ids_q,
            perm_mask,
            input_mask,
            target_mapping,
            segment_ids,
            lm_labels,
            sequence_labels,
            is_impossible_labels,
            token_labels,
        )

    def set_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def create_and_check_xlnet_base_model(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
        lm_labels,
        sequence_labels,
        is_impossible_labels,
        token_labels,
    ):
        model = XLNetModel(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids_1, input_mask=input_mask)
        result = model(input_ids_1, attention_mask=input_mask)
        result = model(input_ids_1, token_type_ids=segment_ids)
        result = model(input_ids_1)

        config.mem_len = 0
        model = XLNetModel(config)
        model.to(torch_device)
        model.eval()
        base_model_output = model(input_ids_1)
        self.parent.assertEqual(len(base_model_output), 2)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertListEqual(
            [mem.shape for mem in result.mems],
            [(self.seq_length, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
        )

    def create_and_check_use_mems_train(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
        lm_labels,
        sequence_labels,
        is_impossible_labels,
        token_labels,
    ):
        model = XLNetForSequenceClassification(config)
        model.to(torch_device)
        model.train()

        train_size = input_ids_1.shape[0]

        batch_size = 4
        for i in range(train_size // batch_size + 1):
            input_ids = input_ids_1[i : (i + 1) * batch_size]
            labels = sequence_labels[i : (i + 1) * batch_size]
            outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
            self.parent.assertIsNone(outputs.mems)
            self.parent.assertIsNotNone(outputs.loss)

    def create_and_check_xlnet_model_use_mems(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
        lm_labels,
        sequence_labels,
        is_impossible_labels,
        token_labels,
    ):
        model = XLNetModel(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        causal_mask = torch.ones(
            input_ids_1.shape[0],
            input_ids_1.shape[1],
            input_ids_1.shape[1],
            dtype=torch.float,
            device=torch_device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=0)
        outputs_cache = model(input_ids_1, use_mems=True, perm_mask=causal_mask)
        outputs_no_cache = model(input_ids_1, use_mems=False, perm_mask=causal_mask)
        outputs_conf = model(input_ids_1)

        self.parent.assertTrue(len(outputs_cache) == len(outputs_conf))
        self.parent.assertTrue(len(outputs_cache) == len(outputs_no_cache) + 1)

        output, mems = outputs_cache.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and token_type_ids
        next_input_ids = torch.cat([input_ids_1, next_tokens], dim=-1)

        # causal mask
        causal_mask = torch.ones(
            input_ids_1.shape[0],
            input_ids_1.shape[1] + 1,
            input_ids_1.shape[1] + 1,
            dtype=torch.float,
            device=torch_device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=0)
        single_mask = torch.ones(input_ids_1.shape[0], 1, 1, dtype=torch.float, device=torch_device)

        # second forward pass
        output_from_no_past = model(next_input_ids, perm_mask=causal_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, mems=mems, perm_mask=single_mask)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_xlnet_base_model_with_att_output(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
        lm_labels,
        sequence_labels,
        is_impossible_labels,
        token_labels,
    ):
        model = XLNetModel(config)
        model.to(torch_device)
        model.eval()

        attentions = model(input_ids_1, target_mapping=target_mapping, output_attentions=True)["attentions"]

        self.parent.assertEqual(len(attentions), config.n_layer)
        self.parent.assertIsInstance(attentions[0], tuple)
        self.parent.assertEqual(len(attentions[0]), 2)
        self.parent.assertTrue(attentions[0][0].shape, attentions[0][0].shape)

    def create_and_check_xlnet_lm_head(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
        lm_labels,
        sequence_labels,
        is_impossible_labels,
        token_labels,
    ):
        model = XLNetLMHeadModel(config)
        model.to(torch_device)
        model.eval()

        result1 = model(input_ids_1, token_type_ids=segment_ids, labels=lm_labels)

        result2 = model(input_ids_2, token_type_ids=segment_ids, labels=lm_labels, mems=result1.mems)

        _ = model(input_ids_q, perm_mask=perm_mask, target_mapping=target_mapping)

        self.parent.assertEqual(result1.loss.shape, ())
        self.parent.assertEqual(result1.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertListEqual(
            [mem.shape for mem in result1.mems],
            [(self.seq_length, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
        )

        self.parent.assertEqual(result2.loss.shape, ())
        self.parent.assertEqual(result2.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertListEqual(
            [mem.shape for mem in result2.mems],
            [(self.mem_len, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
        )

    def create_and_check_xlnet_qa(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
        lm_labels,
        sequence_labels,
        is_impossible_labels,
        token_labels,
    ):
        model = XLNetForQuestionAnswering(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids_1)

        result_with_labels = model(
            input_ids_1,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            cls_index=sequence_labels,
            is_impossible=is_impossible_labels,
            p_mask=input_mask,
        )

        result_with_labels = model(
            input_ids_1,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            cls_index=sequence_labels,
            is_impossible=is_impossible_labels,
        )

        total_loss, mems = result_with_labels.to_tuple()

        result_with_labels = model(
            input_ids_1,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )

        total_loss, mems = result_with_labels.to_tuple()

        self.parent.assertEqual(result_with_labels.loss.shape, ())
        self.parent.assertEqual(result.start_top_log_probs.shape, (self.batch_size, model.config.start_n_top))
        self.parent.assertEqual(result.start_top_index.shape, (self.batch_size, model.config.start_n_top))
        self.parent.assertEqual(
            result.end_top_log_probs.shape, (self.batch_size, model.config.start_n_top * model.config.end_n_top)
        )
        self.parent.assertEqual(
            result.end_top_index.shape, (self.batch_size, model.config.start_n_top * model.config.end_n_top)
        )
        self.parent.assertEqual(result.cls_logits.shape, (self.batch_size,))
        self.parent.assertListEqual(
            [mem.shape for mem in result.mems],
            [(self.seq_length, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
        )

    def create_and_check_xlnet_token_classif(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
        lm_labels,
        sequence_labels,
        is_impossible_labels,
        token_labels,
    ):
        model = XLNetForTokenClassification(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids_1)
        result = model(input_ids_1, labels=token_labels)

        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.type_sequence_label_size))
        self.parent.assertListEqual(
            [mem.shape for mem in result.mems],
            [(self.seq_length, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
        )

    def create_and_check_xlnet_sequence_classif(
        self,
        config,
        input_ids_1,
        input_ids_2,
        input_ids_q,
        perm_mask,
        input_mask,
        target_mapping,
        segment_ids,
        lm_labels,
        sequence_labels,
        is_impossible_labels,
        token_labels,
    ):
        model = XLNetForSequenceClassification(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids_1)
        result = model(input_ids_1, labels=sequence_labels)

        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))
        self.parent.assertListEqual(
            [mem.shape for mem in result.mems],
            [(self.seq_length, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids_1,
            input_ids_2,
            input_ids_q,
            perm_mask,
            input_mask,
            target_mapping,
            segment_ids,
            lm_labels,
            sequence_labels,
            is_impossible_labels,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids_1}
        return config, inputs_dict


@require_torch
class XLNetModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            XLNetModel,
            XLNetLMHeadModel,
            XLNetForTokenClassification,
            XLNetForSequenceClassification,
            XLNetForQuestionAnswering,
            XLNetForQuestionAnsweringSimple,
            XLNetForMultipleChoice,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (
        (XLNetLMHeadModel,) if is_torch_available() else ()
    )  # TODO (PVP): Check other models whether language generation is also applicable
    test_pruning = False

    # XLNet has 2 QA models -> need to manually set the correct labels for one of them here
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "XLNetForQuestionAnswering":
                inputs_dict["start_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
                inputs_dict["end_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = XLNetModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XLNetConfig, d_inner=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_xlnet_base_model(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_base_model(*config_and_inputs)

    def test_xlnet_base_model_use_mems(self):
        # checking that in auto-regressive mode, :obj:`use_mems` gives the same results
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_model_use_mems(*config_and_inputs)

    def test_seq_classification_use_mems_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_use_mems_train(*config_and_inputs)

    def test_xlnet_base_model_with_att_output(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_base_model_with_att_output(*config_and_inputs)

    def test_xlnet_lm_head(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_lm_head(*config_and_inputs)

    def test_xlnet_sequence_classif(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_sequence_classif(*config_and_inputs)

    def test_xlnet_token_classif(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_token_classif(*config_and_inputs)

    def test_xlnet_qa(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_qa(*config_and_inputs)

    def test_retain_grad_hidden_states_attentions(self):
        # xlnet cannot keep gradients in attentions or hidden states
        return

    @slow
    def test_model_from_pretrained(self):
        for model_name in XLNET_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = XLNetModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class XLNetModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_xlnet_base_cased(self):
        model = XLNetLMHeadModel.from_pretrained("xlnet-base-cased")
        model.to(torch_device)
        input_ids = torch.tensor(
            [
                [
                    67,
                    2840,
                    19,
                    18,
                    1484,
                    20,
                    965,
                    29077,
                    8719,
                    1273,
                    21,
                    45,
                    273,
                    17,
                    10,
                    15048,
                    28,
                    27511,
                    21,
                    4185,
                    11,
                    41,
                    2444,
                    9,
                    32,
                    1025,
                    20,
                    8719,
                    26,
                    23,
                    673,
                    966,
                    19,
                    29077,
                    20643,
                    27511,
                    20822,
                    20643,
                    19,
                    17,
                    6616,
                    17511,
                    18,
                    8978,
                    20,
                    18,
                    777,
                    9,
                    19233,
                    1527,
                    17669,
                    19,
                    24,
                    673,
                    17,
                    28756,
                    150,
                    12943,
                    4354,
                    153,
                    27,
                    442,
                    37,
                    45,
                    668,
                    21,
                    24,
                    256,
                    20,
                    416,
                    22,
                    2771,
                    4901,
                    9,
                    12943,
                    4354,
                    153,
                    51,
                    24,
                    3004,
                    21,
                    28142,
                    23,
                    65,
                    20,
                    18,
                    416,
                    34,
                    24,
                    2958,
                    22947,
                    9,
                    1177,
                    45,
                    668,
                    3097,
                    13768,
                    23,
                    103,
                    28,
                    441,
                    148,
                    48,
                    20522,
                    19,
                    12943,
                    4354,
                    153,
                    12860,
                    34,
                    18,
                    326,
                    27,
                    17492,
                    684,
                    21,
                    6709,
                    9,
                    8585,
                    123,
                    266,
                    19,
                    12943,
                    4354,
                    153,
                    6872,
                    24,
                    3004,
                    20,
                    18,
                    9225,
                    2198,
                    19,
                    12717,
                    103,
                    22,
                    401,
                    24,
                    6348,
                    9,
                    12943,
                    4354,
                    153,
                    1068,
                    2768,
                    2286,
                    19,
                    33,
                    104,
                    19,
                    176,
                    24,
                    9313,
                    19,
                    20086,
                    28,
                    45,
                    10292,
                    9,
                    4,
                    3,
                ]
            ],
            dtype=torch.long,
            device=torch_device,
        )
        #  In 1991, the remains of Russian Tsar Nicholas II and his family
        #  (except for Alexei and Maria) are discovered.
        #  The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
        #  remainder of the story. 1883 Western Siberia,
        #  a young Grigori Rasputin is asked by his father and a group of men to perform magic.
        #  Rasputin has a vision and denounces one of the men as a horse thief. Although his
        #  father initially slaps him for making such an accusation, Rasputin watches as the
        #  man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
        #  the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
        #  with people, even a bishop, begging for his blessing. """

        expected_output_ids = [
            67,
            2840,
            19,
            18,
            1484,
            20,
            965,
            29077,
            8719,
            1273,
            21,
            45,
            273,
            17,
            10,
            15048,
            28,
            27511,
            21,
            4185,
            11,
            41,
            2444,
            9,
            32,
            1025,
            20,
            8719,
            26,
            23,
            673,
            966,
            19,
            29077,
            20643,
            27511,
            20822,
            20643,
            19,
            17,
            6616,
            17511,
            18,
            8978,
            20,
            18,
            777,
            9,
            19233,
            1527,
            17669,
            19,
            24,
            673,
            17,
            28756,
            150,
            12943,
            4354,
            153,
            27,
            442,
            37,
            45,
            668,
            21,
            24,
            256,
            20,
            416,
            22,
            2771,
            4901,
            9,
            12943,
            4354,
            153,
            51,
            24,
            3004,
            21,
            28142,
            23,
            65,
            20,
            18,
            416,
            34,
            24,
            2958,
            22947,
            9,
            1177,
            45,
            668,
            3097,
            13768,
            23,
            103,
            28,
            441,
            148,
            48,
            20522,
            19,
            12943,
            4354,
            153,
            12860,
            34,
            18,
            326,
            27,
            17492,
            684,
            21,
            6709,
            9,
            8585,
            123,
            266,
            19,
            12943,
            4354,
            153,
            6872,
            24,
            3004,
            20,
            18,
            9225,
            2198,
            19,
            12717,
            103,
            22,
            401,
            24,
            6348,
            9,
            12943,
            4354,
            153,
            1068,
            2768,
            2286,
            19,
            33,
            104,
            19,
            176,
            24,
            9313,
            19,
            20086,
            28,
            45,
            10292,
            9,
            4,
            3,
            19,
            12943,
            4354,
            153,
            27,
            442,
            22,
            2771,
            4901,
            9,
            69,
            27,
            442,
            22,
            2771,
            24,
            11335,
            20,
            18,
            9225,
            2198,
            9,
            69,
            27,
            442,
            22,
            2771,
            24,
            11335,
            20,
            18,
            9225,
            2198,
            9,
            69,
            27,
            442,
            22,
            2771,
        ]
        #  In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria)
        #  are discovered. The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich,
        #  narrates the remainder of the story. 1883 Western Siberia, a young Grigori Rasputin
        #  is asked by his father and a group of men to perform magic. Rasputin has a vision and
        #  denounces one of the men as a horse thief. Although his father initially slaps
        #  him for making such an accusation, Rasputin watches as the man is chased outside and beaten.
        #  Twenty years later, Rasputin sees a vision of the Virgin Mary, prompting him to become a priest.
        #  Rasputin quickly becomes famous, with people, even a bishop, begging for his blessing.
        #  <sep><cls>, Rasputin is asked to perform magic. He is asked to perform a ritual of the Virgin Mary.
        #  He is asked to perform a ritual of the Virgin Mary. He is asked to perform

        output_ids = model.generate(input_ids, max_length=200, do_sample=False)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
