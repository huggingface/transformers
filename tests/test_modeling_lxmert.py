# coding=utf-8
# Copyright 2018 LXMERT Authors.
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

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch
    from transformers import (
        LxmertConfig,
        LxmertModel,
        LxmertForPretraining,
        LxmertForQuestionAnswering,
    )
    from transformers.modeling_lxmert import LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST

    #


class LxmertModelTester:
    """You can also import this e.g from .test_modeling_bart import BartModelTester """

    def __init__(
        self,
        parent,
        vocab_size=300,
        hidden_size=28,
        num_attention_heads=2,
        num_labels=2,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        num_qa_labels=30,
        num_object_labels=16,
        num_attr_labels=4,
        num_visual_features=10,
        l_layers=2,
        x_layers=1,
        r_layers=1,
        visual_feat_dim=128,
        visual_pos_dim=4,
        visual_loss_normalizer=6.67,
        seq_length=20,
        batch_size=4,
        is_training=True,
        task_matched=True,
        task_mask_lm=True,
        task_obj_predict=True,
        task_qa=True,
        visual_obj_loss=True,
        visual_attr_loss=True,
        visual_feat_loss=True,
        use_token_type_ids=True,
        use_lang_mask=True,
        output_attentions=False,
        scope=None,
    ):
        self.parent = parent
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_labels = num_labels
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.num_qa_labels = num_qa_labels
        self.num_object_labels = num_object_labels
        self.num_attr_labels = num_attr_labels
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers
        self.visual_feat_dim = visual_feat_dim
        self.visual_pos_dim = visual_pos_dim
        self.visual_loss_normalizer = visual_loss_normalizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_lang_mask = use_lang_mask
        self.task_matched = task_matched
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict
        self.task_qa = task_qa
        self.visual_obj_loss = visual_obj_loss
        self.visual_attr_loss = visual_attr_loss
        self.visual_feat_loss = visual_feat_loss
        self.num_visual_features = num_visual_features
        self.use_token_type_ids = use_token_type_ids
        self.output_attentions = output_attentions
        self.scope = scope

    def prepare_config_and_inputs(self):

        output_attentions = self.output_attentions
        input_ids = ids_tensor([self.batch_size, self.seq_length], vocab_size=self.vocab_size)
        visual_feats = torch.rand(self.batch_size, self.num_visual_features, self.visual_feat_dim)
        bounding_boxes = torch.rand(self.batch_size, self.num_visual_features, 4)

        input_mask = None
        if self.use_lang_mask:
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)
        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
        obj_labels = None
        if self.task_obj_predict:
            obj_labels = {}
        if self.visual_attr_loss and self.task_obj_predict:
            obj_labels["attr"] = (
                ids_tensor([self.batch_size, self.num_visual_features], self.num_attr_labels),
                ids_tensor([self.batch_size, self.num_visual_features], self.num_attr_labels),
            )
        if self.visual_feat_loss and self.task_obj_predict:
            obj_labels["feat"] = (
                ids_tensor(
                    [self.batch_size, self.num_visual_features, self.visual_feat_dim], self.num_visual_features
                ),
                ids_tensor([self.batch_size, self.num_visual_features], self.num_visual_features),
            )
        if self.visual_obj_loss and self.task_obj_predict:
            obj_labels["obj"] = (
                ids_tensor([self.batch_size, self.num_visual_features], self.num_object_labels),
                ids_tensor([self.batch_size, self.num_visual_features], self.num_object_labels),
            )
        ans = None
        if self.task_qa:
            ans = ids_tensor([self.batch_size], self.num_qa_labels)
        masked_lm_labels = None
        if self.task_mask_lm:
            masked_lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        matched_labels = None
        if self.task_matched:
            matched_labels = ids_tensor([self.batch_size], self.num_labels)

        config = LxmertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_labels=self.num_labels,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            num_qa_labels=self.num_qa_labels,
            num_object_labels=self.num_object_labels,
            num_attr_labels=self.num_attr_labels,
            l_layers=self.l_layers,
            x_layers=self.x_layers,
            r_layers=self.r_layers,
            visual_feat_dim=self.visual_feat_dim,
            visual_pos_dim=self.visual_pos_dim,
            visual_loss_normalizer=self.visual_loss_normalizer,
            task_matched=self.task_matched,
            task_mask_lm=self.task_mask_lm,
            task_obj_predict=self.task_obj_predict,
            task_qa=self.task_qa,
            visual_obj_loss=self.visual_obj_loss,
            visual_attr_loss=self.visual_attr_loss,
            visual_feat_loss=self.visual_feat_loss,
        )

        return (
            config,
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids,
            input_mask,
            obj_labels,
            masked_lm_labels,
            matched_labels,
            ans,
            output_attentions,
        )

    def create_and_check_lxmert_model(
        self,
        config,
        input_ids,
        visual_feats,
        bounding_boxes,
        token_type_ids,
        input_mask,
        obj_labels,
        masked_lm_labels,
        matched_labels,
        ans,
        output_attentions,
    ):
        model = LxmertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            output_attentions=output_attentions,
        )
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            output_attentions=not output_attentions,
        )
        result = model(input_ids, (visual_feats, bounding_boxes), return_dict=False)
        result = model(input_ids, (visual_feats, bounding_boxes), return_dict=True)

        self.parent.assertEqual(result.last_hidden_state_l.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(
            result.last_hidden_state_v.shape, (self.batch_size, self.num_visual_features, self.hidden_size)
        )
        self.parent.assertEqual(result.pooled_output_x_encoder.shape, (self.batch_size, self.hidden_size))

    def create_and_check_lxmert_for_question_answering(
        self,
        config,
        input_ids,
        visual_feats,
        bounding_boxes,
        token_type_ids,
        input_mask,
        obj_labels,
        masked_lm_labels,
        matched_labels,
        ans,
        output_attentions,
    ):
        model = LxmertForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            ans,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
            output_attentions=output_attentions,
            return_dict=True,
        )
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            ans,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            output_attentions=not output_attentions,
            return_dict=False,
        )
        result = model(input_ids, (visual_feats, bounding_boxes), ans)
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            ans,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            output_attentions=output_attentions,
        )
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            ans,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            output_attentions=not output_attentions,
            return_dict=True,
        )

        self.parent.assertEqual(result.question_answering_score.shape, (self.batch_size, self.num_qa_labels))

    def create_and_check_lxmert_for_pretraining(
        self,
        config,
        input_ids,
        visual_feats,
        bounding_boxes,
        token_type_ids,
        input_mask,
        obj_labels,
        masked_lm_labels,
        matched_labels,
        ans,
        output_attentions,
    ):
        model = LxmertForPretraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
            obj_labels=obj_labels,
            matched_label=matched_labels,
            ans=ans,
            output_attentions=output_attentions,
            return_dict=True,
        )
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
            output_attentions=not output_attentions,
            return_dict=False,
        )
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
        )
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            obj_labels=obj_labels,
        )
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            matched_labels=matched_labels,
        )
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            ans=ans,
        )
        result = model(
            input_ids,
            (visual_feats, bounding_boxes),
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
            obj_labels=obj_labels,
            matched_label=matched_labels,
            ans=ans,
            output_attentions=not output_attentions,
            return_dict=True,
        )

        self.parent.assertEqual(result.prediction_logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids,
            input_mask,
            obj_labels,
            masked_lm_labels,
            matched_labels,
            ans,
            output_attentions,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "visual_feats": (visual_feats, bounding_boxes),
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
            "output_attentions": output_attentions,
        }

        return config, inputs_dict


@require_torch
class LxmertModelTest(unittest.TestCase):

    all_model_classes = (LxmertModel, LxmertForPretraining) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = LxmertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LxmertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_lxmert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lxmert_model(*config_and_inputs)

    def test_lxmert_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lxmert_for_question_answering(*config_and_inputs)

    def test_lxmert_pretraning(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lxmert_for_pretraining(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = LxmertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
