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
"""Testing suite for the TensorFlow SegFormer model."""

from __future__ import annotations

import inspect
import unittest
from typing import List, Tuple

from transformers import SegformerConfig
from transformers.file_utils import is_tf_available, is_vision_available
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import numpy as np
    import tensorflow as tf

    from transformers import TFSegformerForImageClassification, TFSegformerForSemanticSegmentation, TFSegformerModel

if is_vision_available():
    from PIL import Image

    from transformers import SegformerImageProcessor


class TFSegformerConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "hidden_sizes"))
        self.parent.assertTrue(hasattr(config, "num_attention_heads"))
        self.parent.assertTrue(hasattr(config, "num_encoder_blocks"))


class TFSegformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=64,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[1, 1, 1, 1],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[8, 8, 16, 16],
        downsampling_rates=[1, 4, 8, 16],
        num_attention_heads=[1, 1, 2, 2],
        is_training=True,
        use_labels=True,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.sr_ratios = sr_ratios
        self.depths = depths
        self.hidden_sizes = hidden_sizes
        self.downsampling_rates = downsampling_rates
        self.num_attention_heads = num_attention_heads
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()
        return config, pixel_values, labels

    def get_config(self):
        return SegformerConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            num_encoder_blocks=self.num_encoder_blocks,
            depths=self.depths,
            hidden_sizes=self.hidden_sizes,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            num_labels=self.num_labels,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = TFSegformerModel(config=config)
        result = model(pixel_values, training=False)
        expected_height = expected_width = self.image_size // (self.downsampling_rates[-1] * 2)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.hidden_sizes[-1], expected_height, expected_width)
        )

    def create_and_check_for_image_segmentation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = TFSegformerForSemanticSegmentation(config)
        result = model(pixel_values, training=False)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size // 4, self.image_size // 4)
        )
        result = model(pixel_values, labels=labels, training=False)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size // 4, self.image_size // 4)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_keras_fit(self, for_segmentation: bool = False):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, seg_labels = config_and_inputs
        if for_segmentation:
            inputs_dict = {"pixel_values": pixel_values, "labels": seg_labels}
        else:
            inputs_dict = {"pixel_values": pixel_values, "labels": tf.zeros((self.batch_size))}
        return config, inputs_dict


@require_tf
class TFSegformerModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (TFSegformerModel, TFSegformerForImageClassification, TFSegformerForSemanticSegmentation)
        if is_tf_available()
        else ()
    )
    pipeline_model_mapping = (
        {"feature-extraction": TFSegformerModel, "image-classification": TFSegformerForImageClassification}
        if is_tf_available()
        else {}
    )

    test_head_masking = False
    test_onnx = False
    test_pruning = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = TFSegformerModelTester(self)
        self.config_tester = TFSegformerConfigTester(self, config_class=SegformerConfig, has_text_modality=False)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip("SegFormer does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("SegFormer does not have get_input_embeddings method and get_output_embeddings methods")
    def test_model_common_attributes(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions

            expected_num_attentions = sum(self.model_tester.depths)
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions

            self.assertEqual(len(attentions), expected_num_attentions)

            # verify the first attentions (first block, first layer)
            expected_seq_len = (self.model_tester.image_size // 4) ** 2
            expected_reduced_seq_len = (self.model_tester.image_size // (4 * self.model_tester.sr_ratios[0])) ** 2
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads[0], expected_seq_len, expected_reduced_seq_len],
            )

            # verify the last attentions (last block, last layer)
            expected_seq_len = (self.model_tester.image_size // 32) ** 2
            expected_reduced_seq_len = (self.model_tester.image_size // (32 * self.model_tester.sr_ratios[-1])) ** 2
            self.assertListEqual(
                list(attentions[-1].shape[-3:]),
                [self.model_tester.num_attention_heads[-1], expected_seq_len, expected_reduced_seq_len],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 1, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), expected_num_attentions)
            # verify the first attentions (first block, first layer)
            expected_seq_len = (self.model_tester.image_size // 4) ** 2
            expected_reduced_seq_len = (self.model_tester.image_size // (4 * self.model_tester.sr_ratios[0])) ** 2
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads[0], expected_seq_len, expected_reduced_seq_len],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = self.model_tester.num_encoder_blocks
            self.assertEqual(len(hidden_states), expected_num_layers)

            # verify the first hidden states (first block)
            self.assertListEqual(
                list(hidden_states[0].shape[-3:]),
                [
                    self.model_tester.hidden_sizes[0],
                    self.model_tester.image_size // 4,
                    self.model_tester.image_size // 4,
                ],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            tuple_output = model(tuple_inputs, return_dict=False, **additional_kwargs)
            dict_output = model(dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

            def recursive_check(tuple_object, dict_object):
                if isinstance(tuple_object, (List, Tuple)):
                    for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                        recursive_check(tuple_iterable_value, dict_iterable_value)
                elif tuple_object is None:
                    return
                else:
                    self.assertTrue(
                        all(tf.equal(tuple_object, dict_object)),
                        msg=(
                            "Tuple and dict output are not equal. Difference:"
                            f" {tf.math.reduce_max(tf.abs(tuple_object - dict_object))}"
                        ),
                    )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            if self.has_attentions:
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

            # todo: incorporate label support for semantic segmentation in `test_modeling_tf_common.py`.

    @unittest.skipIf(
        not is_tf_available() or len(tf.config.list_physical_devices("GPU")) == 0,
        reason="TF does not support backprop for grouped convolutions on CPU.",
    )
    def test_dataset_conversion(self):
        super().test_dataset_conversion()

    def check_keras_fit_results(self, val_loss1, val_loss2, atol=2e-1, rtol=2e-1):
        self.assertTrue(np.allclose(val_loss1, val_loss2, atol=atol, rtol=rtol))

    @unittest.skipIf(
        not is_tf_available() or len(tf.config.list_physical_devices("GPU")) == 0,
        reason="TF does not support backprop for grouped convolutions on CPU.",
    )
    @slow
    def test_keras_fit(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            # Since `TFSegformerModel` cannot operate with the default `fit()` method.
            if model_class.__name__ != "TFSegformerModel":
                model = model_class(config)
                if getattr(model, "hf_compute_loss", None):
                    super().test_keras_fit()

    def test_loss_computation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def apply(model):
            for_segmentation = True if model_class.__name__ == "TFSegformerForSemanticSegmentation" else False
            # The number of elements in the loss should be the same as the number of elements in the label
            _, prepared_for_class = self.model_tester.prepare_config_and_inputs_for_keras_fit(
                for_segmentation=for_segmentation
            )
            added_label = prepared_for_class[sorted(prepared_for_class.keys() - inputs_dict.keys(), reverse=True)[0]]
            loss_size = tf.size(added_label)

            # Test that model correctly compute the loss with kwargs
            possible_input_names = {"input_ids", "pixel_values", "input_features"}
            input_name = possible_input_names.intersection(set(prepared_for_class)).pop()
            model_input = prepared_for_class.pop(input_name)

            loss = model(model_input, **prepared_for_class)[0]

            if model_class.__name__ == "TFSegformerForSemanticSegmentation":
                # Semantic segmentation loss is computed similarly as
                # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_tf_utils.py#L210.
                self.assertEqual(loss.shape, (1,))
            else:
                self.assertEqual(loss.shape, [loss_size])

            # Test that model correctly compute the loss with a dict
            _, prepared_for_class = self.model_tester.prepare_config_and_inputs_for_keras_fit(
                for_segmentation=for_segmentation
            )
            loss = model(**prepared_for_class)[0]

            if model_class.__name__ == "TFSegformerForSemanticSegmentation":
                self.assertEqual(loss.shape, (1,))
            else:
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
            if model_class.__name__ == "TFSegformerForSemanticSegmentation":
                self.assertEqual(loss.shape, (1,))
            else:
                self.assertEqual(loss.shape, [loss_size])

        for model_class in self.all_model_classes:
            # Since `TFSegformerModel` won't have labels against which we
            # could compute loss.
            if model_class.__name__ != "TFSegformerModel":
                model = model_class(config)
                apply(model)

    def check_pt_tf_outputs(self, tf_outputs, pt_outputs, model_class, tol=2e-4, name="outputs", attributes=None):
        # We override with a slightly higher tol value, as semseg models tend to diverge a bit more
        super().check_pt_tf_outputs(tf_outputs, pt_outputs, model_class, tol, name, attributes)

    @slow
    def test_model_from_pretrained(self):
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        model = TFSegformerModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_tf
class TFSegformerModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_image_segmentation_ade(self):
        # only resize + normalize
        image_processor = SegformerImageProcessor(
            image_scale=(512, 512), keep_ratio=False, align=False, do_random_crop=False
        )
        model = TFSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        image = prepare_img()
        encoded_inputs = image_processor(images=image, return_tensors="tf")
        pixel_values = encoded_inputs.pixel_values

        outputs = model(pixel_values, training=False)

        expected_shape = tf.TensorShape((1, model.config.num_labels, 128, 128))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = tf.constant(
            [
                [[-4.6310, -5.5232, -6.2356], [-5.1921, -6.1444, -6.5996], [-5.4424, -6.2790, -6.7574]],
                [[-12.1391, -13.3122, -13.9554], [-12.8732, -13.9352, -14.3563], [-12.9438, -13.8226, -14.2513]],
                [[-12.5134, -13.4686, -14.4915], [-12.8669, -14.4343, -14.7758], [-13.2523, -14.5819, -15.0694]],
            ]
        )
        tf.debugging.assert_near(outputs.logits[0, :3, :3, :3], expected_slice, atol=1e-4)

    @slow
    def test_inference_image_segmentation_city(self):
        # only resize + normalize
        image_processor = SegformerImageProcessor(
            image_scale=(512, 512), keep_ratio=False, align=False, do_random_crop=False
        )
        model = TFSegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
        )

        image = prepare_img()
        encoded_inputs = image_processor(images=image, return_tensors="tf")
        pixel_values = encoded_inputs.pixel_values

        outputs = model(pixel_values, training=False)

        expected_shape = tf.TensorShape((1, model.config.num_labels, 128, 128))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = tf.constant(
            [
                [[-13.5748, -13.9111, -12.6500], [-14.3500, -15.3683, -14.2328], [-14.7532, -16.0424, -15.6087]],
                [[-17.1651, -15.8725, -12.9653], [-17.2580, -17.3718, -14.8223], [-16.6058, -16.8783, -16.7452]],
                [[-3.6456, -3.0209, -1.4203], [-3.0797, -3.1959, -2.0000], [-1.8757, -1.9217, -1.6997]],
            ]
        )
        tf.debugging.assert_near(outputs.logits[0, :3, :3, :3], expected_slice, atol=1e-1)
