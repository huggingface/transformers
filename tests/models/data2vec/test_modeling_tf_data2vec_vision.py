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
""" Testing suite for the TensorFlow Data2VecVision model. """

from __future__ import annotations

import collections.abc
import inspect
import unittest

import numpy as np

from transformers import Data2VecVisionConfig
from transformers.file_utils import cached_property, is_tf_available, is_vision_available
from transformers.testing_utils import require_tf, require_vision, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TFData2VecVisionForImageClassification,
        TFData2VecVisionForSemanticSegmentation,
        TFData2VecVisionModel,
    )
    from transformers.modeling_tf_utils import keras

if is_vision_available():
    from PIL import Image

    from transformers import BeitImageProcessor


class TFData2VecVisionModelTester:
    def __init__(
        self,
        parent,
        vocab_size=100,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
        out_indices=[0, 1, 2, 3],
    ):
        self.parent = parent
        self.vocab_size = 100
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.out_indices = out_indices
        self.num_labels = num_labels

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        pixel_labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            pixel_labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels, pixel_labels

    def get_config(self):
        return Data2VecVisionConfig(
            vocab_size=self.vocab_size,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            out_indices=self.out_indices,
        )

    def create_and_check_model(self, config, pixel_values, labels, pixel_labels):
        model = TFData2VecVisionModel(config=config)
        result = model(pixel_values, training=False)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (
            self.image_size
            if isinstance(self.image_size, collections.abc.Iterable)
            else (self.image_size, self.image_size)
        )
        patch_size = (
            self.patch_size
            if isinstance(self.image_size, collections.abc.Iterable)
            else (self.patch_size, self.patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.type_sequence_label_size
        model = TFData2VecVisionForImageClassification(config)

        result = model(pixel_values, labels=labels, training=False)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def create_and_check_for_image_segmentation(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.num_labels
        model = TFData2VecVisionForSemanticSegmentation(config)
        result = model(pixel_values, training=False)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size * 2, self.image_size * 2)
        )
        result = model(pixel_values, labels=pixel_labels)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size * 2, self.image_size * 2)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels, pixel_labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_keras_fit(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, _, _ = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values, "labels": tf.zeros((self.batch_size))}
        return config, inputs_dict


@require_tf
class TFData2VecVisionModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Data2VecVision does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (TFData2VecVisionModel, TFData2VecVisionForImageClassification, TFData2VecVisionForSemanticSegmentation)
        if is_tf_available()
        else ()
    )
    pipeline_model_mapping = (
        {"feature-extraction": TFData2VecVisionModel, "image-classification": TFData2VecVisionForImageClassification}
        if is_tf_available()
        else {}
    )

    test_pruning = False
    test_onnx = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = TFData2VecVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Data2VecVisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Data2VecVision does not use inputs_embeds")
    def test_inputs_embeds(self):
        # Data2VecVision does not use inputs_embeds
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (keras.layers.Layer))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, keras.layers.Layer))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_segmentation(*config_and_inputs)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        # in Data2VecVision, the seq_len equals the number of patches + 1 (we add 1 for the [CLS] token)
        image_size = (
            self.model_tester.image_size
            if isinstance(self.model_tester.image_size, collections.abc.Iterable)
            else (self.model_tester.image_size, self.model_tester.image_size)
        )
        patch_size = (
            self.model_tester.patch_size
            if isinstance(self.model_tester.patch_size, collections.abc.Iterable)
            else (self.model_tester.patch_size, self.model_tester.patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 1
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)

            self.assertEqual(out_len + 1, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            # Data2VecVision has a different seq_length
            image_size = (
                self.model_tester.image_size
                if isinstance(self.model_tester.image_size, collections.abc.Iterable)
                else (self.model_tester.image_size, self.model_tester.image_size)
            )
            patch_size = (
                self.model_tester.patch_size
                if isinstance(self.model_tester.patch_size, collections.abc.Iterable)
                else (self.model_tester.patch_size, self.model_tester.patch_size)
            )
            num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
            seq_length = num_patches + 1

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # Overriding this method since the base method won't be compatible with Data2VecVision.
    @slow
    def test_keras_fit(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            # Since `TFData2VecVisionModel` cannot operate with the default `fit()` method.
            if model_class.__name__ != "TFData2VecVisionModel":
                model = model_class(config)
                if getattr(model, "hf_compute_loss", None):
                    # Test that model correctly compute the loss with kwargs
                    _, prepared_for_class = self.model_tester.prepare_config_and_inputs_for_keras_fit()

                    label_names = {"labels"}
                    self.assertGreater(len(label_names), 0, msg="No matching label names found!")
                    labels = {key: val for key, val in prepared_for_class.items() if key in label_names}
                    inputs_minus_labels = {
                        key: val for key, val in prepared_for_class.items() if key not in label_names
                    }
                    self.assertGreater(len(inputs_minus_labels), 0)
                    model.compile(optimizer=keras.optimizers.SGD(0.0), run_eagerly=True)

                    # Make sure the model fits without crashing regardless of where we pass the labels
                    history1 = model.fit(
                        prepared_for_class,
                        validation_data=prepared_for_class,
                        steps_per_epoch=1,
                        validation_steps=1,
                        shuffle=False,
                    )
                    val_loss1 = history1.history["val_loss"][0]
                    history2 = model.fit(
                        inputs_minus_labels,
                        labels,
                        validation_data=(inputs_minus_labels, labels),
                        steps_per_epoch=1,
                        validation_steps=1,
                        shuffle=False,
                    )
                    val_loss2 = history2.history["val_loss"][0]
                    self.assertTrue(np.allclose(val_loss1, val_loss2, atol=1e-2, rtol=1e-3))

    def check_pt_tf_outputs(self, tf_outputs, pt_outputs, model_class, tol=2e-4, name="outputs", attributes=None):
        # We override with a slightly higher tol value, as semseg models tend to diverge a bit more
        super().check_pt_tf_outputs(tf_outputs, pt_outputs, model_class, tol, name, attributes)

    # Overriding this method since the base method won't be compatible with Data2VecVision.
    def test_loss_computation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            # Since `TFData2VecVisionModel` won't have labels against which we
            # could compute loss.
            if model_class.__name__ != "TFData2VecVisionModel":
                model = model_class(config)
                if getattr(model, "hf_compute_loss", None):
                    # The number of elements in the loss should be the same as the number of elements in the label
                    _, prepared_for_class = self.model_tester.prepare_config_and_inputs_for_keras_fit()
                    added_label = prepared_for_class[
                        sorted(prepared_for_class.keys() - inputs_dict.keys(), reverse=True)[0]
                    ]
                    loss_size = tf.size(added_label)

                    # Test that model correctly compute the loss with kwargs
                    possible_input_names = {"input_ids", "pixel_values", "input_features"}
                    input_name = possible_input_names.intersection(set(prepared_for_class)).pop()
                    model_input = prepared_for_class.pop(input_name)

                    loss = model(model_input, **prepared_for_class)[0]
                    self.assertEqual(loss.shape, [loss_size])

                    # Test that model correctly compute the loss with a dict
                    _, prepared_for_class = self.model_tester.prepare_config_and_inputs_for_keras_fit()
                    loss = model(**prepared_for_class)[0]
                    self.assertEqual(loss.shape, [loss_size])

                    # Test that model correctly compute the loss with a tuple
                    label_keys = prepared_for_class.keys() - inputs_dict.keys()
                    signature = inspect.signature(model.call).parameters
                    signature_names = list(signature.keys())

                    # Create a dictionary holding the location of the tensors in the tuple
                    tuple_index_mapping = {0: input_name}
                    for label_key in label_keys:
                        label_key_index = signature_names.index(label_key)
                        tuple_index_mapping[label_key_index] = label_key
                    sorted_tuple_index_mapping = sorted(tuple_index_mapping.items())
                    # Initialize a list with their default values, update the values and convert to a tuple
                    list_input = []

                    for name in signature_names:
                        if name != "kwargs":
                            list_input.append(signature[name].default)

                    for index, value in sorted_tuple_index_mapping:
                        list_input[index] = prepared_for_class[value]

                    tuple_input = tuple(list_input)

                    # Send to model
                    loss = model(tuple_input[:-1])[0]

                    self.assertEqual(loss.shape, [loss_size])

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/data2vec-vision-base-ft1k"
        model = TFData2VecVisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_tf
@require_vision
class TFData2VecVisionModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            BeitImageProcessor.from_pretrained("facebook/data2vec-vision-base-ft1k") if is_vision_available() else None
        )

    @slow
    def test_inference_image_classification_head_imagenet_1k(self):
        model = TFData2VecVisionForImageClassification.from_pretrained("facebook/data2vec-vision-base-ft1k")

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="tf")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = tf.convert_to_tensor([1, 1000])
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = tf.convert_to_tensor([0.3277, -0.1395, 0.0911])

        tf.debugging.assert_near(logits[0, :3], expected_slice, atol=1e-4)

        expected_top2 = [model.config.label2id[i] for i in ["remote control, remote", "tabby, tabby cat"]]
        self.assertEqual(tf.nn.top_k(outputs.logits[0], 2).indices.numpy().tolist(), expected_top2)
