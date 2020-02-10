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

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor
from .utils import CACHE_DIR, require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import (
        XLNetConfig,
        XLNetModel,
        XLNetLMHeadModel,
        XLNetForSequenceClassification,
        XLNetForTokenClassification,
        XLNetForQuestionAnswering,
    )
    from transformers.modeling_xlnet import XLNET_PRETRAINED_MODEL_ARCHIVE_MAP


@require_torch
class XLNetModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            XLNetModel,
            XLNetLMHeadModel,
            XLNetForTokenClassification,
            XLNetForSequenceClassification,
            XLNetForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    test_pruning = False

    class XLNetModelTester(object):
        def __init__(
            self,
            parent,
            batch_size=13,
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
        ):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.mem_len = mem_len
            # self.key_len = seq_length + mem_len
            self.clamp_len = clamp_len
            self.reuse_len = reuse_len
            self.is_training = is_training
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.cutoffs = cutoffs
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.d_inner = d_inner
            self.num_hidden_layers = num_hidden_layers
            self.bi_data = bi_data
            self.untie_r = untie_r
            self.same_length = same_length
            self.initializer_range = initializer_range
            self.seed = seed
            self.type_vocab_size = type_vocab_size
            self.type_sequence_label_size = type_sequence_label_size

        def prepare_config_and_inputs(self):
            input_ids_1 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            input_ids_2 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            segment_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
            input_mask = ids_tensor([self.batch_size, self.seq_length], 2).float()

            input_ids_q = ids_tensor([self.batch_size, self.seq_length + 1], self.vocab_size)
            perm_mask = torch.zeros(
                self.batch_size, self.seq_length + 1, self.seq_length + 1, dtype=torch.float, device=torch_device
            )
            perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
            target_mapping = torch.zeros(
                self.batch_size, 1, self.seq_length + 1, dtype=torch.float, device=torch_device
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

            _, _ = model(input_ids_1, input_mask=input_mask)
            _, _ = model(input_ids_1, attention_mask=input_mask)
            _, _ = model(input_ids_1, token_type_ids=segment_ids)
            outputs, mems_1 = model(input_ids_1)

            result = {
                "mems_1": mems_1,
                "outputs": outputs,
            }

            config.mem_len = 0
            model = XLNetModel(config)
            model.to(torch_device)
            model.eval()
            no_mems_outputs = model(input_ids_1)
            self.parent.assertEqual(len(no_mems_outputs), 1)

            self.parent.assertListEqual(
                list(result["outputs"].size()), [self.batch_size, self.seq_length, self.hidden_size]
            )
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
            )

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

            _, _, attentions = model(input_ids_1, target_mapping=target_mapping)

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

            loss_1, all_logits_1, mems_1 = model(input_ids_1, token_type_ids=segment_ids, labels=lm_labels)

            loss_2, all_logits_2, mems_2 = model(
                input_ids_2, token_type_ids=segment_ids, labels=lm_labels, mems=mems_1
            )

            logits, _ = model(input_ids_q, perm_mask=perm_mask, target_mapping=target_mapping)

            result = {
                "loss_1": loss_1,
                "mems_1": mems_1,
                "all_logits_1": all_logits_1,
                "loss_2": loss_2,
                "mems_2": mems_2,
                "all_logits_2": all_logits_2,
            }

            self.parent.assertListEqual(list(result["loss_1"].size()), [])
            self.parent.assertListEqual(
                list(result["all_logits_1"].size()), [self.batch_size, self.seq_length, self.vocab_size]
            )
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
            )

            self.parent.assertListEqual(list(result["loss_2"].size()), [])
            self.parent.assertListEqual(
                list(result["all_logits_2"].size()), [self.batch_size, self.seq_length, self.vocab_size]
            )
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_2"]),
                [[self.mem_len, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
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

            outputs = model(input_ids_1)
            start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits, mems = outputs

            outputs = model(
                input_ids_1,
                start_positions=sequence_labels,
                end_positions=sequence_labels,
                cls_index=sequence_labels,
                is_impossible=is_impossible_labels,
                p_mask=input_mask,
            )

            outputs = model(
                input_ids_1,
                start_positions=sequence_labels,
                end_positions=sequence_labels,
                cls_index=sequence_labels,
                is_impossible=is_impossible_labels,
            )

            total_loss, mems = outputs

            outputs = model(input_ids_1, start_positions=sequence_labels, end_positions=sequence_labels)

            total_loss, mems = outputs

            result = {
                "loss": total_loss,
                "start_top_log_probs": start_top_log_probs,
                "start_top_index": start_top_index,
                "end_top_log_probs": end_top_log_probs,
                "end_top_index": end_top_index,
                "cls_logits": cls_logits,
                "mems": mems,
            }

            self.parent.assertListEqual(list(result["loss"].size()), [])
            self.parent.assertListEqual(
                list(result["start_top_log_probs"].size()), [self.batch_size, model.config.start_n_top]
            )
            self.parent.assertListEqual(
                list(result["start_top_index"].size()), [self.batch_size, model.config.start_n_top]
            )
            self.parent.assertListEqual(
                list(result["end_top_log_probs"].size()),
                [self.batch_size, model.config.start_n_top * model.config.end_n_top],
            )
            self.parent.assertListEqual(
                list(result["end_top_index"].size()),
                [self.batch_size, model.config.start_n_top * model.config.end_n_top],
            )
            self.parent.assertListEqual(list(result["cls_logits"].size()), [self.batch_size])
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
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

            logits, mems_1 = model(input_ids_1)
            loss, logits, mems_1 = model(input_ids_1, labels=token_labels)

            result = {
                "loss": loss,
                "mems_1": mems_1,
                "logits": logits,
            }

            self.parent.assertListEqual(list(result["loss"].size()), [])
            self.parent.assertListEqual(
                list(result["logits"].size()), [self.batch_size, self.seq_length, self.type_sequence_label_size]
            )
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
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

            logits, mems_1 = model(input_ids_1)
            loss, logits, mems_1 = model(input_ids_1, labels=sequence_labels)

            result = {
                "loss": loss,
                "mems_1": mems_1,
                "logits": logits,
            }

            self.parent.assertListEqual(list(result["loss"].size()), [])
            self.parent.assertListEqual(
                list(result["logits"].size()), [self.batch_size, self.type_sequence_label_size]
            )
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers,
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

    def setUp(self):
        self.model_tester = XLNetModelTest.XLNetModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XLNetConfig, d_inner=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_xlnet_base_model(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_base_model(*config_and_inputs)

    def test_xlnet_base_model_with_att_output(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config_and_inputs[0].output_attentions = True
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

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(XLNET_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = XLNetModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.assertIsNotNone(model)
