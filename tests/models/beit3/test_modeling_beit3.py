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
import inspect
import unittest

from transformers import VisualBertConfig, is_torch_available
from transformers.models.beit_3.configuration_beit_3 import Beit3Config
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor,_config_zero_init
from ...test_pipeline_mixin import PipelineTesterMixin
import numpy as np

if is_torch_available():
    import torch

    from transformers import (
        BEiT3Model,
        BEiT3ForVisualReasoning,
        BEiT3ForImageClassification,
        BEiT3ForCaptioning,
        BEiT3ForVisualQuestionAnswering,
        BEiT3ForImageTextRetrieval,
    )
    from transformers.models.visual_bert.modeling_visual_bert import VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST


class Beit3ModelTester:
    def __init__(
        self,
        embed_dim=768,
        attention_heads=12,
        hidden_size=3072,
        num_hidden_layers=3,
        normalize_before=True,
        activation_fn="gelu",
        dropout=0.0,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        deepnorm=False,
        subln=True,
        bert_init=False,
        multiway=True,
        max_source_positions=1024,
        layernorm_eps=1e-5,
        vocab_size=64010,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_labels=2,
        batch_size=1,
        seq_length=7,
        use_labels=True,
        is_training=True,
    ):

        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_heads = attention_heads
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.normalize_before = normalize_before
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.deepnorm = deepnorm
        self.subln = subln
        self.bert_init = bert_init
        self.multiway = multiway
        self.max_source_positions = max_source_positions
        self.layernorm_eps = layernorm_eps
        # Text
        self.vocab_size = vocab_size
        # Vision
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_labels = num_labels
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_labels = use_labels
        self.is_training= is_training

    def get_config(self):
        return Beit3Config(
            embed_dim=self.embed_dim,
            attention_heads=self.attention_heads,
            hidden_size=self.hidden_size,
            layers=self.num_hidden_layers,
            normalize_before=self.normalize_before,
            activation_fn=self.activation_fn,
            dropout=self.dropout,
            drop_path_rate=self.drop_path_rate,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            deepnorm=self.deepnorm,
            subln=self.subln,
            bert_init=self.bert_init,
            multiway=self.multiway,
            max_source_positions=self.max_source_positions,
            layernorm_eps=self.layernorm_eps,
            vocab_size=self.vocab_size,
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            num_labels=self.num_labels,
        )

        # return Beit3Config()

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, self.in_chans, self.img_size, self.img_size])
        padding_mask = torch.zeros((self.batch_size, self.seq_length))
        return self.get_config(), {"input_ids":input_ids, "pixel_values":pixel_values,"padding_mask":padding_mask}

    def prepare_config_and_inputs_for_visual_reasoning(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_value1 = floats_tensor([self.batch_size, self.in_chans, self.img_size, self.img_size])
        pixel_value2 = floats_tensor([self.batch_size, self.in_chans, self.img_size , self.img_size])
        padding_mask = torch.zeros((self.batch_size,self.seq_length))
        config = self.get_config()
        model_input = {
            "input_ids": input_ids,
            "pixel_values1": pixel_value1,
            "pixel_values2": pixel_value2,
            "padding_mask": padding_mask
        }
        labels = ids_tensor([self.batch_size],self.num_labels)
        if self.use_labels:
            model_input["labels"] = labels

        return config, model_input

    def prepare_config_and_inputs_for_image_classification(self):
        pixel_value = floats_tensor([self.batch_size, self.in_chans, self.img_size, self.img_size])
        config = self.seq_length
        labels = torch.zeros(
                    self.batch_size, dtype=torch.long, device=torch_device
                )
        model_input = {"pixel_values": pixel_value}
        if self.use_labels:
            model_input["labels"]= labels
        return config, model_input

    def prepare_config_and_inputs_for_captioning(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, self.in_chans, self.img_size, self.img_size])
        padding_mask = torch.zeros((self.batch_size, self.seq_length))

        config = self.get_config()
        return config, {}

    def prepare_config_and_inputs_for_visual_question_answering(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, self.in_chans, self.img_size, self.img_size])
        padding_mask = torch.zeros((self.batch_size, self.seq_length))
        return self.get_config(), {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "padding_mask":padding_mask
        }

    def prepare_config_and_inputs_for_text_retrieval(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, self.in_chans, self.img_size, self.img_size])
        padding_mask = torch.zeros((self.batch_size, self.seq_length))
        return self.get_config(), {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "padding_mask":padding_mask
        }

    def create_and_check_model(self, config, input_dict):
        model = BEiT3Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        # self.parent.assertEqual(
        #     result.last_hidden_state.shape,
        #     (self.batch_size, self.seq_length + self.visual_seq_length, self.hidden_size),
        # )

    def create_and_check_for_visual_reasoning(self, config, input_dict):
        model = BEiT3ForVisualReasoning(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))
    #
    # def create_and_check_for_multiple_choice(self, config, input_dict):
    #     model = VisualBertForMultipleChoice(config=config)
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(**input_dict)
    #     self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))
    #
    # def create_and_check_for_nlvr(self, config, input_dict):
    #     model = VisualBertForVisualReasoning(config=config)
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(**input_dict)
    #     self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))
    #
    # def create_and_check_for_flickr(self, config, input_dict):
    #     model = VisualBertForRegionToPhraseAlignment(config=config)
    #     model.to(torch_device)
    #     model.eval()
    #     result = model(**input_dict)
    #     self.parent.assertEqual(
    #         result.logits.shape, (self.batch_size, self.seq_length + self.visual_seq_length, self.visual_seq_length)
    #     )


@require_torch
class Beit3ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            # BEiT3Model,
            BEiT3ForVisualReasoning,
            BEiT3ForImageTextRetrieval,
            BEiT3ForVisualQuestionAnswering,
            BEiT3ForImageClassification,
            # BEiT3ForCaptioning,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": BEiT3Model,
            "image-classification": BEiT3ForImageClassification,
        }
        if is_torch_available()
        else {}
    )
    test_torchscript = False
    test_pruning = False
    test_attention_outputs = False
    has_attentions = False
    test_inputs_embeds = False
    test_head_masking = False

    # def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
    #     inputs_dict = copy.deepcopy(inputs_dict)
    #     if model_class == VisualBertForMultipleChoice:
    #         for key in inputs_dict.keys():
    #             value = inputs_dict[key]
    #             if isinstance(value, torch.Tensor) and value.ndim > 1:
    #                 if key != "visual_embeds":
    #                     inputs_dict[key] = (
    #                         inputs_dict[key].unsqueeze(1).expand(-1, self.model_tester.num_choices, -1).contiguous()
    #                     )
    #                 else:
    #                     inputs_dict[key] = (
    #                         inputs_dict[key]
    #                         .unsqueeze(1)
    #                         .expand(-1, self.model_tester.num_choices, -1, self.model_tester.visual_embedding_dim)
    #                         .contiguous()
    #                     )
    #
    #     elif model_class == VisualBertForRegionToPhraseAlignment:
    #         total_length = self.model_tester.seq_length + self.model_tester.visual_seq_length
    #         batch_size = self.model_tester.batch_size
    #         inputs_dict["region_to_phrase_position"] = torch.zeros(
    #             (batch_size, total_length),
    #             dtype=torch.long,
    #             device=torch_device,
    #         )
    #
    #     if return_labels:
    #         if model_class == VisualBertForMultipleChoice:
    #             inputs_dict["labels"] = torch.zeros(
    #                 self.model_tester.batch_size, dtype=torch.long, device=torch_device
    #             )
    #         elif model_class == VisualBertForPreTraining:
    #             total_length = self.model_tester.seq_length + self.model_tester.visual_seq_length
    #             batch_size = self.model_tester.batch_size
    #             inputs_dict["labels"] = torch.zeros(
    #                 (batch_size, total_length),
    #                 dtype=torch.long,
    #                 device=torch_device,
    #             )
    #             inputs_dict["sentence_image_labels"] = torch.zeros(
    #                 self.model_tester.batch_size, dtype=torch.long, device=torch_device
    #             )
    #
    #         # Flickr expects float labels
    #         elif model_class == VisualBertForRegionToPhraseAlignment:
    #             batch_size = self.model_tester.batch_size
    #             total_length = self.model_tester.seq_length + self.model_tester.visual_seq_length
    #
    #             inputs_dict["labels"] = torch.ones(
    #                 (
    #                     batch_size,
    #                     total_length,
    #                     self.model_tester.visual_seq_length,
    #                 ),
    #                 dtype=torch.float,
    #                 device=torch_device,
    #             )
    #
    #         # VQA expects float labels
    #         elif model_class == VisualBertForQuestionAnswering:
    #             inputs_dict["labels"] = torch.ones(
    #                 (self.model_tester.batch_size, self.model_tester.num_labels),
    #                 dtype=torch.float,
    #                 device=torch_device,
    #             )
    #
    #         elif model_class == VisualBertForVisualReasoning:
    #             inputs_dict["labels"] = torch.zeros(
    #                 (self.model_tester.batch_size), dtype=torch.long, device=torch_device
    #             )
    #
    #     return inputs_dict

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict_to_return = None
        if model_class.__name__ == "BEiT3ForVisualReasoning":
            # inputs_dict_to_return =  self.model_tester.prepare_config_and_inputs_for_visual_reasoning()[1]
            inputs_dict_to_return = {}
            if return_labels:
                inputs_dict_to_return["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            inputs_dict_to_return['pixel_values1'] = inputs_dict["pixel_values"]
            inputs_dict_to_return['pixel_values2'] = inputs_dict["pixel_values"]
            inputs_dict_to_return['padding_mask'] = inputs_dict["padding_mask"]
            inputs_dict_to_return['input_ids'] = inputs_dict["input_ids"]
            return inputs_dict_to_return
        elif model_class.__name__ == "BEiT3ForImageClassification":
            inputs_dict_to_return =  self.model_tester.prepare_config_and_inputs_for_image_classification()[1]
            if return_labels:
                inputs_dict_to_return["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            inputs_dict_to_return.update(inputs_dict)
            del inputs_dict_to_return["input_ids"]
            del inputs_dict_to_return["padding_mask"]
            return inputs_dict_to_return
        elif model_class.__name__ == "BEiT3ForImageTextRetrieval":
            inputs_dict_to_return =  self.model_tester.prepare_config_and_inputs_for_text_retrieval()[1]
        elif model_class.__name__ == "BEiT3ForVisualQuestionAnswering":
            inputs_dict_to_return =  self.model_tester.prepare_config_and_inputs_for_visual_question_answering()[1]
            inputs_dict_to_return["labels"] = torch.ones(
                (self.model_tester.batch_size, self.model_tester.num_labels),
                dtype=torch.float,
                device=torch_device,
            )
        elif model_class.__name__ == "BEiT3ForCaptioning":
            inputs_dict_to_return =  self.model_tester.prepare_config_and_inputs_for_captioning()[1]

        inputs_dict_to_return.update(inputs_dict)
        return inputs_dict_to_return

    def setUp(self):
        self.model_tester = Beit3ModelTester()
        self.config_tester = ConfigTester(self, config_class=VisualBertConfig, hidden_size=37)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]

            if model_class.__name__ == "BEiT3ForImageClassification":
                # signature.parameters is an OrderedDict => so arg_names order is deterministic

                expected_arg_names = ["pixel_values"]
                self.assertListEqual(arg_names[:1], expected_arg_names)
            else:
                expected_arg_names = ["input_ids"]
                self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            if model_class.__name__ == "BEiT3ForImageTextRetrieval":
                return
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
            total_seq_length = ((self.model_tester.img_size // self.model_tester.patch_size)**2) + 1
            if model_class.__name__ != "BEiT3ForImageClassification":
                total_seq_length = total_seq_length + seq_length
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [total_seq_length, self.model_tester.embed_dim],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # override as the `logit_scale` parameter initilization is different for Blip
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # check if `logit_scale` is initilized as per the original implementation
                    if name == "logit_scale":
                        self.assertAlmostEqual(
                            param.data.item(),
                            np.log(1 / 0.07),
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

#     def test_attention_outputs(self):
#         config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
#         config.return_dict = True
#
#         seq_len = getattr(self.model_tester, "seq_length", None)
#         visual_seq_len = getattr(self.model_tester, "visual_seq_length", None)
#
#         encoder_seq_length = (seq_len if seq_len is not None else 0) + (
#             visual_seq_len if visual_seq_len is not None else 0
#         )
#         encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
#         chunk_length = getattr(self.model_tester, "chunk_length", None)
#         if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
#             encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes
#
#         for model_class in self.all_model_classes:
#             inputs_dict["output_attentions"] = True
#             inputs_dict["output_hidden_states"] = False
#             config.return_dict = True
#             model = model_class(config)
#             model.to(torch_device)
#             model.eval()
#             with torch.no_grad():
#                 outputs = model(**self._prepare_for_class(inputs_dict, model_class))
#             attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
#             self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
#
#             # check that output_attentions also work using config
#             del inputs_dict["output_attentions"]
#             config.output_attentions = True
#             model = model_class(config)
#             model.to(torch_device)
#             model.eval()
#             with torch.no_grad():
#                 outputs = model(**self._prepare_for_class(inputs_dict, model_class))
#             attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
#             self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
#
#             if chunk_length is not None:
#                 self.assertListEqual(
#                     list(attentions[0].shape[-4:]),
#                     [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
#                 )
#             else:
#                 self.assertListEqual(
#                     list(attentions[0].shape[-3:]),
#                     [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
#                 )
#             out_len = len(outputs)
#
#             # Check attention is always last and order is fine
#             inputs_dict["output_attentions"] = True
#             inputs_dict["output_hidden_states"] = True
#             model = model_class(config)
#             model.to(torch_device)
#             model.eval()
#             with torch.no_grad():
#                 outputs = model(**self._prepare_for_class(inputs_dict, model_class))
#
#             if hasattr(self.model_tester, "num_hidden_states_types"):
#                 added_hidden_states = self.model_tester.num_hidden_states_types
#             elif self.is_encoder_decoder:
#                 added_hidden_states = 2
#             else:
#                 added_hidden_states = 1
#             self.assertEqual(out_len + added_hidden_states, len(outputs))
#
#             self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
#
#             self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
#             if chunk_length is not None:
#                 self.assertListEqual(
#                     list(self_attentions[0].shape[-4:]),
#                     [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
#                 )
#             else:
#                 self.assertListEqual(
#                     list(self_attentions[0].shape[-3:]),
#                     [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
#                 )
#
#     def test_hidden_states_output(self):
#         def check_hidden_states_output(inputs_dict, config, model_class):
#             model = model_class(config)
#             model.to(torch_device)
#             model.eval()
#
#             with torch.no_grad():
#                 outputs = model(**self._prepare_for_class(inputs_dict, model_class))
#
#             hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states
#
#             expected_num_layers = getattr(
#                 self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
#             )
#             self.assertEqual(len(hidden_states), expected_num_layers)
#
#             if hasattr(self.model_tester, "encoder_seq_length"):
#                 seq_length = self.model_tester.encoder_seq_length
#                 if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
#                     seq_length = seq_length * self.model_tester.chunk_length
#             else:
#                 seq_length = self.model_tester.seq_length + self.model_tester.visual_seq_length
#
#             self.assertListEqual(
#                 list(hidden_states[0].shape[-2:]),
#                 [seq_length, self.model_tester.hidden_size],
#             )
#
#         config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
#
#         for model_class in self.all_model_classes:
#             inputs_dict["output_hidden_states"] = True
#             check_hidden_states_output(inputs_dict, config, model_class)
#
#             # check that output_hidden_states also work using config
#             del inputs_dict["output_hidden_states"]
#             config.output_hidden_states = True
#
#             check_hidden_states_output(inputs_dict, config, model_class)
#
#     def test_config(self):
#         self.config_tester.run_common_tests()
#
#     def test_model(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
#         self.model_tester.create_and_check_model(*config_and_inputs)
#
#     def test_model_various_embeddings(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
#         for type in ["absolute", "relative_key", "relative_key_query"]:
#             config_and_inputs[0].position_embedding_type = type
#             self.model_tester.create_and_check_model(*config_and_inputs)
#
#     def test_model_for_pretraining(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs_for_pretraining()
#         self.model_tester.create_and_check_for_pretraining(*config_and_inputs)
#
#     def test_model_for_vqa(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs_for_vqa()
#         self.model_tester.create_and_check_for_vqa(*config_and_inputs)
#
#     def test_model_for_nlvr(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs_for_nlvr()
#         self.model_tester.create_and_check_for_nlvr(*config_and_inputs)
#
#     def test_model_for_multiple_choice(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs_for_multiple_choice()
#         self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)
#
#     def test_model_for_flickr(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs_for_flickr()
#         self.model_tester.create_and_check_for_flickr(*config_and_inputs)
#
#     @slow
#     def test_model_from_pretrained(self):
#         for model_name in VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
#             model = VisualBertModel.from_pretrained(model_name)
#             self.assertIsNotNone(model)
#
#
# @require_torch
# class VisualBertModelIntegrationTest(unittest.TestCase):
#     @slow
#     def test_inference_vqa_coco_pre(self):
#         model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
#
#         input_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long).reshape(1, -1)
#         token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long).reshape(1, -1)
#         visual_embeds = torch.ones(size=(1, 10, 2048), dtype=torch.float32) * 0.5
#         visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.long)
#         attention_mask = torch.tensor([1] * 6).reshape(1, -1)
#         visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)
#
#         with torch.no_grad():
#             output = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 visual_embeds=visual_embeds,
#                 visual_attention_mask=visual_attention_mask,
#                 visual_token_type_ids=visual_token_type_ids,
#             )
#
#         vocab_size = 30522
#
#         expected_shape = torch.Size((1, 16, vocab_size))
#         self.assertEqual(output.prediction_logits.shape, expected_shape)
#
#         expected_slice = torch.tensor(
#             [[[-5.1858, -5.1903, -4.9142], [-6.2214, -5.9238, -5.8381], [-6.3027, -5.9939, -5.9297]]]
#         )
#
#         self.assertTrue(torch.allclose(output.prediction_logits[:, :3, :3], expected_slice, atol=1e-4))
#
#         expected_shape_2 = torch.Size((1, 2))
#         self.assertEqual(output.seq_relationship_logits.shape, expected_shape_2)
#
#         expected_slice_2 = torch.tensor([[0.7393, 0.1754]])
#
#         self.assertTrue(torch.allclose(output.seq_relationship_logits, expected_slice_2, atol=1e-4))
#
#     @slow
#     def test_inference_vqa(self):
#         model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")
#
#         input_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long).reshape(1, -1)
#         token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long).reshape(1, -1)
#         visual_embeds = torch.ones(size=(1, 10, 2048), dtype=torch.float32) * 0.5
#         visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.long)
#         attention_mask = torch.tensor([1] * 6).reshape(1, -1)
#         visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)
#
#         with torch.no_grad():
#             output = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 visual_embeds=visual_embeds,
#                 visual_attention_mask=visual_attention_mask,
#                 visual_token_type_ids=visual_token_type_ids,
#             )
#
#         # vocab_size = 30522
#
#         expected_shape = torch.Size((1, 3129))
#         self.assertEqual(output.logits.shape, expected_shape)
#
#         expected_slice = torch.tensor(
#             [[-8.9898, 3.0803, -1.8016, 2.4542, -8.3420, -2.0224, -3.3124, -4.4139, -3.1491, -3.8997]]
#         )
#
#         self.assertTrue(torch.allclose(output.logits[:, :10], expected_slice, atol=1e-4))
#
#     @slow
#     def test_inference_nlvr(self):
#         model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2")
#
#         input_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long).reshape(1, -1)
#         token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long).reshape(1, -1)
#         visual_embeds = torch.ones(size=(1, 10, 1024), dtype=torch.float32) * 0.5
#         visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.long)
#         attention_mask = torch.tensor([1] * 6).reshape(1, -1)
#         visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)
#
#         with torch.no_grad():
#             output = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 visual_embeds=visual_embeds,
#                 visual_attention_mask=visual_attention_mask,
#                 visual_token_type_ids=visual_token_type_ids,
#             )
#
#         # vocab_size = 30522
#
#         expected_shape = torch.Size((1, 2))
#         self.assertEqual(output.logits.shape, expected_shape)
#
#         expected_slice = torch.tensor([[-1.1436, 0.8900]])
#
#         self.assertTrue(torch.allclose(output.logits, expected_slice, atol=1e-4))
#
#     @slow
#     def test_inference_vcr(self):
#         model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vcr")
#
#         input_ids = torch.tensor([[[1, 2, 3, 4, 5, 6] for i in range(4)]], dtype=torch.long)
#         attention_mask = torch.ones_like(input_ids)
#         token_type_ids = torch.ones_like(input_ids)
#
#         visual_embeds = torch.ones(size=(1, 4, 10, 512), dtype=torch.float32) * 0.5
#         visual_token_type_ids = torch.ones(size=(1, 4, 10), dtype=torch.long)
#         visual_attention_mask = torch.ones_like(visual_token_type_ids)
#
#         with torch.no_grad():
#             output = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 visual_embeds=visual_embeds,
#                 visual_attention_mask=visual_attention_mask,
#                 visual_token_type_ids=visual_token_type_ids,
#             )
#
#         # vocab_size = 30522
#
#         expected_shape = torch.Size((1, 4))
#         self.assertEqual(output.logits.shape, expected_shape)
#
#         expected_slice = torch.tensor([[-7.7697, -7.7697, -7.7697, -7.7697]])
#
#         self.assertTrue(torch.allclose(output.logits, expected_slice, atol=1e-4))
