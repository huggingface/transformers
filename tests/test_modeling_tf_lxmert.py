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

import os
import tempfile
import unittest

from transformers import LxmertConfig, is_tf_available
from transformers.testing_utils import require_tf, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers.models.lxmert.modeling_tf_lxmert import TFLxmertForPreTraining, TFLxmertModel


class TFLxmertModelTester(object):
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
class TFLxmertModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (TFLxmertModel, TFLxmertForPreTraining) if is_tf_available() else ()
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

    def test_pt_tf_model_equivalence(self):
        from transformers import is_torch_available

        if not is_torch_available():
            return

        import torch

        import transformers

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(
                return_obj_labels="PreTraining" in model_class.__name__
            )

            pt_model_class_name = model_class.__name__[2:]  # Skip the "TF" at the beginning
            pt_model_class = getattr(transformers, pt_model_class_name)

            config.output_hidden_states = True
            config.task_obj_predict = False

            tf_model = model_class(config)
            pt_model = pt_model_class(config)

            # Check we can load pt model in tf and vice-versa with model => model functions

            tf_model = transformers.load_pytorch_model_in_tf2_model(
                tf_model, pt_model, tf_inputs=self._prepare_for_class(inputs_dict, model_class)
            )
            pt_model = transformers.load_tf2_model_in_pytorch_model(pt_model, tf_model)

            # Check predictions on first output (logits/hidden-states) are close enought given low-level computational differences
            pt_model.eval()

            # Delete obj labels as we want to compute the hidden states and not the loss

            if "obj_labels" in inputs_dict:
                del inputs_dict["obj_labels"]

            def torch_type(key):
                if key in ("visual_feats", "visual_pos"):
                    return torch.float32
                else:
                    return torch.long

            def recursive_numpy_convert(iterable):
                return_dict = {}
                for key, value in iterable.items():
                    if isinstance(value, dict):
                        return_dict[key] = recursive_numpy_convert(value)
                    else:
                        if isinstance(value, (list, tuple)):
                            return_dict[key] = (
                                torch.from_numpy(iter_value.numpy()).to(torch_type(key)) for iter_value in value
                            )
                        else:
                            return_dict[key] = torch.from_numpy(value.numpy()).to(torch_type(key))
                return return_dict

            pt_inputs_dict = recursive_numpy_convert(self._prepare_for_class(inputs_dict, model_class))

            # need to rename encoder-decoder "inputs" for PyTorch
            if "inputs" in pt_inputs_dict and self.is_encoder_decoder:
                pt_inputs_dict["input_ids"] = pt_inputs_dict.pop("inputs")

            with torch.no_grad():
                pto = pt_model(**pt_inputs_dict)
            tfo = tf_model(self._prepare_for_class(inputs_dict, model_class), training=False)
            tf_hidden_states = tfo[0].numpy()
            pt_hidden_states = pto[0].numpy()

            import numpy as np

            tf_nans = np.copy(np.isnan(tf_hidden_states))
            pt_nans = np.copy(np.isnan(pt_hidden_states))

            pt_hidden_states[tf_nans] = 0
            tf_hidden_states[tf_nans] = 0
            pt_hidden_states[pt_nans] = 0
            tf_hidden_states[pt_nans] = 0

            max_diff = np.amax(np.abs(tf_hidden_states - pt_hidden_states))
            # Debug info (remove when fixed)
            if max_diff >= 2e-2:
                print("===")
                print(model_class)
                print(config)
                print(inputs_dict)
                print(pt_inputs_dict)
            self.assertLessEqual(max_diff, 6e-2)

            # Check we can load pt model in tf and vice-versa with checkpoint => model functions
            with tempfile.TemporaryDirectory() as tmpdirname:
                import os

                pt_checkpoint_path = os.path.join(tmpdirname, "pt_model.bin")
                torch.save(pt_model.state_dict(), pt_checkpoint_path)
                tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(tf_model, pt_checkpoint_path)

                tf_checkpoint_path = os.path.join(tmpdirname, "tf_model.h5")
                tf_model.save_weights(tf_checkpoint_path)
                pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path)

            # Check predictions on first output (logits/hidden-states) are close enought given low-level computational differences
            pt_model.eval()
            pt_inputs_dict = dict(
                (name, torch.from_numpy(key.numpy()).to(torch.long))
                for name, key in self._prepare_for_class(inputs_dict, model_class).items()
            )

            for key, value in pt_inputs_dict.items():
                if key in ("visual_feats", "visual_pos"):
                    pt_inputs_dict[key] = value.to(torch.float32)
                else:
                    pt_inputs_dict[key] = value.to(torch.long)

            with torch.no_grad():
                pto = pt_model(**pt_inputs_dict)
            tfo = tf_model(self._prepare_for_class(inputs_dict, model_class))
            tfo = tfo[0].numpy()
            pto = pto[0].numpy()
            tf_nans = np.copy(np.isnan(tfo))
            pt_nans = np.copy(np.isnan(pto))

            pto[tf_nans] = 0
            tfo[tf_nans] = 0
            pto[pt_nans] = 0
            tfo[pt_nans] = 0

            max_diff = np.amax(np.abs(tfo - pto))
            self.assertLessEqual(max_diff, 6e-2)

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

    def test_compile_tf_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common(
                return_obj_labels="PreTraining" in model_class.__name__
            )

            input_ids = tf.keras.Input(
                batch_shape=(self.model_tester.batch_size, self.model_tester.seq_length),
                name="input_ids",
                dtype="int32",
            )
            visual_feats = tf.keras.Input(
                batch_shape=(
                    self.model_tester.batch_size,
                    self.model_tester.num_visual_features,
                    self.model_tester.visual_feat_dim,
                ),
                name="visual_feats",
                dtype="int32",
            )
            visual_pos = tf.keras.Input(
                batch_shape=(self.model_tester.batch_size, self.model_tester.num_visual_features, 4),
                name="visual_pos",
                dtype="int32",
            )

            # Prepare our model
            model = model_class(config)

            # Let's load it from the disk to be sure we can use pretrained weights
            with tempfile.TemporaryDirectory() as tmpdirname:
                outputs = model(self._prepare_for_class(inputs_dict, model_class))  # build the model
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)

            outputs_dict = model(input_ids, visual_feats, visual_pos)
            hidden_states = outputs_dict[0]

            # Add a dense layer on top to test integration with other keras modules
            outputs = tf.keras.layers.Dense(2, activation="softmax", name="outputs")(hidden_states)

            # Compile extended model
            extended_model = tf.keras.Model(inputs=[input_ids, visual_feats, visual_pos], outputs=[outputs])
            extended_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        list_lm_models = [TFLxmertForPreTraining]

        for model_class in self.all_model_classes:
            model = model_class(config)
            assert isinstance(model.get_input_embeddings(), tf.keras.layers.Layer)

            if model_class in list_lm_models:
                x = model.get_output_embeddings()
                assert isinstance(x, tf.keras.layers.Layer)
                name = model.get_bias()
                assert isinstance(name, dict)
                for k, v in name.items():
                    assert isinstance(v, tf.Variable)
            else:
                x = model.get_output_embeddings()
                assert x is None
                name = model.get_bias()
                assert name is None

    def test_saved_model_creation(self):
        # This test is too long (>30sec) and makes fail the CI
        pass

    @slow
    def test_saved_model_creation_extended(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        if hasattr(config, "use_cache"):
            config.use_cache = True

        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", self.model_tester.seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            class_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config)
            num_out = len(model(class_inputs_dict))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=True)
                saved_model_dir = os.path.join(tmpdirname, "saved_model", "1")
                model = tf.keras.models.load_model(saved_model_dir)
                outputs = model(class_inputs_dict)
                language_hidden_states = outputs["language_hidden_states"]
                vision_hidden_states = outputs["vision_hidden_states"]
                language_attentions = outputs["language_attentions"]
                vision_attentions = outputs["vision_attentions"]
                cross_encoder_attentions = outputs["cross_encoder_attentions"]

                self.assertEqual(len(outputs), num_out)

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
