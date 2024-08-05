# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from transformers import LxmertConfig, is_tf_available
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers.models.lxmert.modeling_tf_lxmert import TFLxmertForPreTraining, TFLxmertModel


class TFLxmertModelTester:
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
        batch_size=8,
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
        output_hidden_states=False,
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
        self.output_hidden_states = output_hidden_states
        self.scope = scope
        self.num_hidden_layers = {"vision": r_layers, "cross_encoder": x_layers, "language": l_layers}

    def prepare_config_and_inputs(self):
        output_attentions = self.output_attentions
        input_ids = ids_tensor([self.batch_size, self.seq_length], vocab_size=self.vocab_size)
        visual_feats = tf.random.uniform((self.batch_size, self.num_visual_features, self.visual_feat_dim))
        bounding_boxes = tf.random.uniform((self.batch_size, self.num_visual_features, 4))

        input_mask = None
        if self.use_lang_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
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
        matched_label = None
        if self.task_matched:
            matched_label = ids_tensor([self.batch_size], self.num_labels)

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
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
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
            matched_label,
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
        matched_label,
        ans,
        output_attentions,
    ):
        model = TFLxmertModel(config=config)
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            output_attentions=output_attentions,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            output_attentions=not output_attentions,
        )
        result = model(input_ids, visual_feats, bounding_boxes, return_dict=False)
        result = model(input_ids, visual_feats, bounding_boxes, return_dict=True)

        self.parent.assertEqual(result.language_output.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(
            result.vision_output.shape, (self.batch_size, self.num_visual_features, self.hidden_size)
        )
        self.parent.assertEqual(result.pooled_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self, return_obj_labels=False):
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
            matched_label,
            ans,
            output_attentions,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "visual_feats": visual_feats,
            "visual_pos": bounding_boxes,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }

        if return_obj_labels:
            inputs_dict["obj_labels"] = obj_labels
        else:
            config.task_obj_predict = False

        return config, inputs_dict

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
        matched_label,
        ans,
        output_attentions,
    ):
        model = TFLxmertForPreTraining(config=config)
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
            obj_labels=obj_labels,
            matched_label=matched_label,
            ans=ans,
            output_attentions=output_attentions,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
            output_attentions=not output_attentions,
            return_dict=False,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            obj_labels=obj_labels,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            matched_label=matched_label,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            ans=ans,
        )
        result = model(
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels,
            obj_labels=obj_labels,
            matched_label=matched_label,
            ans=ans,
            output_attentions=not output_attentions,
        )

        self.parent.assertEqual(result.prediction_logits.shape, (self.batch_size, self.seq_length, self.vocab_size))


@require_tf
class TFLxmertModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TFLxmertModel, TFLxmertForPreTraining) if is_tf_available() else ()
    pipeline_model_mapping = {"feature-extraction": TFLxmertModel} if is_tf_available() else {}
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFLxmertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LxmertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_lxmert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lxmert_model(*config_and_inputs)

    def test_lxmert_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lxmert_for_pretraining(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in ["unc-nlp/lxmert-base-uncased"]:
            model = TFLxmertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        encoder_seq_length = (
            self.model_tester.encoder_seq_length
            if hasattr(self.model_tester, "encoder_seq_length")
            else self.model_tester.seq_length
        )
        encoder_key_length = (
            self.model_tester.key_length if hasattr(self.model_tester, "key_length") else encoder_seq_length
        )

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            language_attentions, vision_attentions, cross_encoder_attentions = (outputs[-3], outputs[-2], outputs[-1])

            self.assertEqual(model.config.output_hidden_states, False)

            self.assertEqual(len(language_attentions), self.model_tester.num_hidden_layers["language"])
            self.assertEqual(len(vision_attentions), self.model_tester.num_hidden_layers["vision"])
            self.assertEqual(len(cross_encoder_attentions), self.model_tester.num_hidden_layers["cross_encoder"])

            attentions = [language_attentions, vision_attentions, cross_encoder_attentions]
            attention_shapes = [
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.num_visual_features,
                    self.model_tester.num_visual_features,
                ],
                [self.model_tester.num_attention_heads, encoder_key_length, self.model_tester.num_visual_features],
            ]

            for attention, attention_shape in zip(attentions, attention_shapes):
                self.assertListEqual(list(attention[0].shape[-3:]), attention_shape)
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))

            # 2 hidden states were added
            self.assertEqual(out_len + 2, len(outputs))
            language_attentions, vision_attentions, cross_encoder_attentions = (outputs[-3], outputs[-2], outputs[-1])
            self.assertEqual(len(language_attentions), self.model_tester.num_hidden_layers["language"])
            self.assertEqual(len(vision_attentions), self.model_tester.num_hidden_layers["vision"])
            self.assertEqual(len(cross_encoder_attentions), self.model_tester.num_hidden_layers["cross_encoder"])

            attentions = [language_attentions, vision_attentions, cross_encoder_attentions]
            attention_shapes = [
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.num_visual_features,
                    self.model_tester.num_visual_features,
                ],
                [self.model_tester.num_attention_heads, encoder_key_length, self.model_tester.num_visual_features],
            ]

            for attention, attention_shape in zip(attentions, attention_shapes):
                self.assertListEqual(list(attention[0].shape[-3:]), attention_shape)

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_hidden_states_output(config, inputs_dict, model_class):
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            language_hidden_states, vision_hidden_states = outputs[-2], outputs[-1]

            self.assertEqual(len(language_hidden_states), self.model_tester.num_hidden_layers["language"] + 1)
            self.assertEqual(len(vision_hidden_states), self.model_tester.num_hidden_layers["vision"] + 1)

            seq_length = self.model_tester.seq_length
            num_visual_features = self.model_tester.num_visual_features

            self.assertListEqual(
                list(language_hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )
            self.assertListEqual(
                list(vision_hidden_states[0].shape[-2:]),
                [num_visual_features, self.model_tester.hidden_size],
            )

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(config, inputs_dict, model_class)

            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(config, inputs_dict, model_class)

    def prepare_pt_inputs_from_tf_inputs(self, tf_inputs_dict):
        import torch

        pt_inputs_dict = {}
        for key, value in tf_inputs_dict.items():
            if isinstance(value, dict):
                pt_inputs_dict[key] = self.prepare_pt_inputs_from_tf_inputs(value)
            elif isinstance(value, (list, tuple)):
                pt_inputs_dict[key] = (self.prepare_pt_inputs_from_tf_inputs(iter_value) for iter_value in value)
            elif isinstance(key, bool):
                pt_inputs_dict[key] = value
            elif key == "input_values":
                pt_inputs_dict[key] = torch.from_numpy(value.numpy()).to(torch.float32)
            elif key == "pixel_values":
                pt_inputs_dict[key] = torch.from_numpy(value.numpy()).to(torch.float32)
            elif key == "input_features":
                pt_inputs_dict[key] = torch.from_numpy(value.numpy()).to(torch.float32)
            # other general float inputs
            elif tf_inputs_dict[key].dtype.is_floating:
                pt_inputs_dict[key] = torch.from_numpy(value.numpy()).to(torch.float32)
            else:
                pt_inputs_dict[key] = torch.from_numpy(value.numpy()).to(torch.long)

        return pt_inputs_dict

    def test_save_load(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(
                return_obj_labels="PreTraining" in model_class.__name__
            )

            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                after_outputs = model(self._prepare_for_class(inputs_dict, model_class))

                self.assert_outputs_same(after_outputs, outputs)


@require_tf
class TFLxmertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = TFLxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        input_ids = tf.constant([[101, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 102]])

        num_visual_features = 10
        _, visual_feats = np.random.seed(0), np.random.rand(1, num_visual_features, model.config.visual_feat_dim)
        _, visual_pos = np.random.seed(0), np.random.rand(1, num_visual_features, 4)
        visual_feats = tf.convert_to_tensor(visual_feats, dtype=tf.float32)
        visual_pos = tf.convert_to_tensor(visual_pos, dtype=tf.float32)
        output = model(input_ids, visual_feats=visual_feats, visual_pos=visual_pos)[0]
        expected_shape = [1, 11, 768]
        self.assertEqual(expected_shape, output.shape)
        expected_slice = tf.constant(
            [
                [
                    [0.24170142, -0.98075, 0.14797261],
                    [1.2540525, -0.83198136, 0.5112344],
                    [1.4070463, -1.1051831, 0.6990401],
                ]
            ]
        )
        tf.debugging.assert_near(output[:, :3, :3], expected_slice, atol=1e-4)
