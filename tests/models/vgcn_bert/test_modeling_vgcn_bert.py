# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import os
import tempfile
import unittest

import numpy as np
from numpy.random import default_rng

from transformers import VGCNBertConfig, is_torch_available
from transformers.models.vgcn_bert.modeling_graph import WordGraph
from transformers.testing_utils import require_torch, require_torch_gpu, slow, torch_device

from ...test_configuration_common import ConfigTester

# from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from .test_modeling_vgcn_bert_common import ModelTesterMixin, ids_tensor, random_attention_mask

rng = default_rng()

if is_torch_available():
    import torch

    from transformers import (
        VGCNBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        VGCNBertForMaskedLM,
        VGCNBertForMultipleChoice,
        VGCNBertForQuestionAnswering,
        VGCNBertForSequenceClassification,
        VGCNBertForTokenClassification,
        VGCNBertModel,
    )


def prepare_fake_wgraphs_and_tokenizer_id_maps(graph1_size: int, graph2_szie: int, vocab_size: int, mode="random"):
    def _make_wgraph(graph_size):
        edge_num = int(graph_size**2 * 0.1)
        adj = torch.sparse_coo_tensor(
            indices=torch.randint(0, graph_size - 1, (edge_num * 2,)).view(2, -1)
            if mode == "random"
            else torch.arange(0, edge_num).expand(2, -1),
            values=torch.rand(edge_num) if mode == "random" else torch.ones(edge_num),
            size=(graph_size - 1, graph_size - 1),
        )
        dense_adj = adj.to_dense()
        dense_adj.fill_diagonal_(1.0)
        adj = dense_adj.to_sparse_coo()
        # zero padding
        indices = adj.indices() + 1
        values = adj.values()
        padded_adj = torch.sparse_coo_tensor(indices=indices, values=values, size=(graph_size, graph_size))
        return padded_adj.coalesce()

    def _make_wgraph_id_to_tokenizer_id_map(graph_size):
        wgraph_id_to_tokenizer_id_map = {0: 0}
        # 0 is reserved for padding
        rand_selected_tokenizer_ids = (
            (rng.choice(vocab_size - 2, (graph_size - 1,), replace=False) + 1)
            if mode == "random"
            else torch.arange(1, graph_size)
        )
        wgraph_id_to_tokenizer_id_map.update({i: j for i, j in zip(range(1, graph_size), rand_selected_tokenizer_ids)})
        return wgraph_id_to_tokenizer_id_map

    assert graph1_size < vocab_size and graph2_szie < vocab_size
    return [_make_wgraph(graph1_size), _make_wgraph(graph2_szie)], [
        _make_wgraph_id_to_tokenizer_id_map(graph1_size),
        _make_wgraph_id_to_tokenizer_id_map(graph2_szie),
    ]


class VGCNBertModelTester(object):
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
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
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
        vgcn_graph_embds_dim=2,
        vgcn_hidden_dim=16,
        vgcn_activation=None,
        vgcn_dropout=0.1,
        vgcn_weight_init_mode="transparent",
        vgcn_vocab1_size=10,
        vgcn_vocab2_size=20,
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
        self.vgcn_graph_embds_dim = vgcn_graph_embds_dim
        self.vgcn_hidden_dim = vgcn_hidden_dim
        self.vgcn_activation = vgcn_activation
        self.vgcn_dropout = vgcn_dropout
        self.vgcn_weight_init_mode = vgcn_weight_init_mode
        self.vgcn_vocab1_size = vgcn_vocab1_size
        self.vgcn_vocab2_size = vgcn_vocab2_size

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def prepare_wgraphs_and_tokenizer_id_maps(self):
        return prepare_fake_wgraphs_and_tokenizer_id_maps(
            self.vgcn_vocab1_size, self.vgcn_vocab2_size, self.vocab_size
        )

    def get_config(self):
        return VGCNBertConfig(
            vgcn_graph_embds_dim=self.vgcn_graph_embds_dim,
            vgcn_hidden_dim=self.vgcn_hidden_dim,
            vgcn_activation=self.vgcn_activation,
            vgcn_dropout=self.vgcn_dropout,
            vgcn_weight_init_mode=self.vgcn_weight_init_mode,
            vocab_size=self.vocab_size,
            dim=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            hidden_dim=self.intermediate_size,
            hidden_act=self.hidden_act,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def create_and_check_vgcn_bert_model(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        wgraphs,
        wgraph_id_to_tokenizer_id_maps,
    ):
        model = VGCNBertModel(
            config=config, wgraphs=wgraphs, wgraph_id_to_tokenizer_id_maps=wgraph_id_to_tokenizer_id_maps
        )
        model.to(torch_device)
        model.eval()
        result = model(input_ids, input_mask)
        result = model(input_ids)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length + self.vgcn_graph_embds_dim, self.hidden_size),
        )

    def create_and_check_vgcn_bert_for_masked_lm(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        wgraphs,
        wgraph_id_to_tokenizer_id_maps,
    ):
        model = VGCNBertForMaskedLM(
            config=config, wgraphs=wgraphs, wgraph_id_to_tokenizer_id_maps=wgraph_id_to_tokenizer_id_maps
        )
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_vgcn_bert_for_question_answering(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        wgraphs,
        wgraph_id_to_tokenizer_id_maps,
    ):
        model = VGCNBertForQuestionAnswering(
            config=config, wgraphs=wgraphs, wgraph_id_to_tokenizer_id_maps=wgraph_id_to_tokenizer_id_maps
        )
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids, attention_mask=input_mask, start_positions=sequence_labels, end_positions=sequence_labels
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_vgcn_bert_for_sequence_classification(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        wgraphs,
        wgraph_id_to_tokenizer_id_maps,
    ):
        config.num_labels = self.num_labels
        model = VGCNBertForSequenceClassification(
            config=config, wgraphs=wgraphs, wgraph_id_to_tokenizer_id_maps=wgraph_id_to_tokenizer_id_maps
        )
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_vgcn_bert_for_token_classification(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        wgraphs,
        wgraph_id_to_tokenizer_id_maps,
    ):
        config.num_labels = self.num_labels
        model = VGCNBertForTokenClassification(
            config=config, wgraphs=wgraphs, wgraph_id_to_tokenizer_id_maps=wgraph_id_to_tokenizer_id_maps
        )
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_vgcn_bert_for_multiple_choice(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        wgraphs,
        wgraph_id_to_tokenizer_id_maps,
    ):
        config.num_choices = self.num_choices
        model = VGCNBertForMultipleChoice(
            config=config, wgraphs=wgraphs, wgraph_id_to_tokenizer_id_maps=wgraph_id_to_tokenizer_id_maps
        )
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            labels=choice_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def prepare_config_and_wgraphs_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        wgraphs, tokenizer_id_maps = self.prepare_wgraphs_and_tokenizer_id_maps()
        wgraphs_and_tokenizer_id_maps = {"wgraphs": wgraphs, "wgraph_id_to_tokenizer_id_maps": tokenizer_id_maps}
        (config, input_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, wgraphs_and_tokenizer_id_maps, inputs_dict


@require_torch
class VGCNBertModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            VGCNBertModel,
            VGCNBertForMaskedLM,
            VGCNBertForMultipleChoice,
            VGCNBertForQuestionAnswering,
            VGCNBertForSequenceClassification,
            VGCNBertForTokenClassification,
        )
        if is_torch_available()
        else None
    )
    fx_compatible = False
    test_pruning = True
    test_resize_embeddings = True
    test_resize_position_embeddings = True

    def setUp(self):
        self.model_tester = VGCNBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VGCNBertConfig, dim=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_vgcn_bert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        wgraphs_and_tokenizer_id_maps = self.model_tester.prepare_wgraphs_and_tokenizer_id_maps()
        self.model_tester.create_and_check_vgcn_bert_model(*config_and_inputs, *wgraphs_and_tokenizer_id_maps)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        wgraphs_and_tokenizer_id_maps = self.model_tester.prepare_wgraphs_and_tokenizer_id_maps()
        self.model_tester.create_and_check_vgcn_bert_for_masked_lm(*config_and_inputs, *wgraphs_and_tokenizer_id_maps)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        wgraphs_and_tokenizer_id_maps = self.model_tester.prepare_wgraphs_and_tokenizer_id_maps()
        self.model_tester.create_and_check_vgcn_bert_for_question_answering(
            *config_and_inputs, *wgraphs_and_tokenizer_id_maps
        )

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        wgraphs_and_tokenizer_id_maps = self.model_tester.prepare_wgraphs_and_tokenizer_id_maps()
        self.model_tester.create_and_check_vgcn_bert_for_sequence_classification(
            *config_and_inputs, *wgraphs_and_tokenizer_id_maps
        )

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        wgraphs_and_tokenizer_id_maps = self.model_tester.prepare_wgraphs_and_tokenizer_id_maps()
        self.model_tester.create_and_check_vgcn_bert_for_token_classification(
            *config_and_inputs, *wgraphs_and_tokenizer_id_maps
        )

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        wgraphs_and_tokenizer_id_maps = self.model_tester.prepare_wgraphs_and_tokenizer_id_maps()
        self.model_tester.create_and_check_vgcn_bert_for_multiple_choice(
            *config_and_inputs, *wgraphs_and_tokenizer_id_maps
        )

    # TODO: upload vgcn-bert model weights file to hub
    @slow
    def test_model_from_pretrained(self):
        for model_name in VGCNBERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = VGCNBertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    @require_torch_gpu
    def test_torchscript_device_change(self):
        (
            config,
            wgraphs_and_tokenizer_id_maps,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_wgraphs_and_inputs_for_common()
        for model_class in self.all_model_classes:
            # BertForMultipleChoice behaves incorrectly in JIT environments.
            if model_class == VGCNBertForMultipleChoice:
                return

            config.torchscript = True
            model = model_class(config=config, **wgraphs_and_tokenizer_id_maps)

            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            traced_model = torch.jit.trace(
                model, (inputs_dict["input_ids"].to("cpu"), inputs_dict["attention_mask"].to("cpu"))
            )

            with tempfile.TemporaryDirectory() as tmp:
                torch.jit.save(traced_model, os.path.join(tmp, "traced_model.pt"))
                loaded = torch.jit.load(os.path.join(tmp, "traced_model.pt"), map_location=torch_device)
                # TODO: correct this failure
                loaded(inputs_dict["input_ids"].to(torch_device), inputs_dict["attention_mask"].to(torch_device))


@require_torch
class VGCNBertModelIntergrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head_absolute_embedding(self):
        # model_path = "zhibinlu/vgcn-distilbert-base-uncased"
        model_path = "/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased"
        config = VGCNBertConfig.from_pretrained(model_path)
        wgraphs, tokenizer_id_maps = prepare_fake_wgraphs_and_tokenizer_id_maps(
            200, 300, config.vocab_size, mode="absolute"
        )
        model = VGCNBertModel.from_pretrained(model_path, wgraphs, tokenizer_id_maps)
        # or
        # model = VGCNBertModel.from_pretrained(model_path)
        # model.set_wgraphs(wgraphs,tokenizer_id_maps)

        input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = torch.Size((1, input_ids.size()[1] + config.vgcn_graph_embds_dim, 768))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor(
            [[[-0.0080, 0.4347, 0.1202], [-0.0221, 0.4312, 0.1239], [-0.0298, 0.4366, 0.1328]]]
        )

        self.assertTrue(torch.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))
