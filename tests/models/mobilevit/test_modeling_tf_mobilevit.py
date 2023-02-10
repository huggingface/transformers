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
""" Testing suite for the TensorFlow MobileViT model. """


import inspect
import unittest

from transformers import MobileViTConfig
from transformers.file_utils import is_tf_available, is_vision_available
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor


if is_tf_available():
    import numpy as np
    import tensorflow as tf

    from transformers import TFMobileViTForImageClassification, TFMobileViTForSemanticSegmentation, TFMobileViTModel
    from transformers.models.mobilevit.modeling_tf_mobilevit import TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import MobileViTFeatureExtractor


class TFMobileViTConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "hidden_sizes"))
        self.parent.assertTrue(hasattr(config, "neck_hidden_sizes"))
        self.parent.assertTrue(hasattr(config, "num_attention_heads"))


class TFMobileViTModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        patch_size=2,
        num_channels=3,
        last_hidden_size=640,
        num_attention_heads=4,
        hidden_act="silu",
        conv_kernel_size=3,
        output_stride=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        is_training=True,
        use_labels=True,
        num_labels=10,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.last_hidden_size = last_hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.conv_kernel_size = conv_kernel_size
        self.output_stride = output_stride
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.use_labels = use_labels
        self.is_training = is_training
        self.num_labels = num_labels
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        pixel_labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)
            pixel_labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels, pixel_labels

    def get_config(self):
        return MobileViTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act,
            conv_kernel_size=self.conv_kernel_size,
            output_stride=self.output_stride,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            classifier_dropout_prob=self.classifier_dropout_prob,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values, labels, pixel_labels):
        model = TFMobileViTModel(config=config)
        result = model(pixel_values, training=False)
        expected_height = expected_width = self.image_size // self.output_stride
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.last_hidden_size, expected_height, expected_width)
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.num_labels
        model = TFMobileViTForImageClassification(config)
        result = model(pixel_values, labels=labels, training=False)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.num_labels
        model = TFMobileViTForSemanticSegmentation(config)
        expected_height = expected_width = self.image_size // self.output_stride

        result = model(pixel_values, training=False)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, expected_height, expected_width)
        )

        result = model(pixel_values, labels=pixel_labels, training=False)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, expected_height, expected_width)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels, pixel_labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_tf
class MobileViTModelTest(TFModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as MobileViT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (TFMobileViTModel, TFMobileViTForImageClassification, TFMobileViTForSemanticSegmentation)
        if is_tf_available()
        else ()
    )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFMobileViTModelTester(self)
        self.config_tester = TFMobileViTConfigTester(self, config_class=MobileViTConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="MobileViT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="MobileViT does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="MobileViT does not output attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Test was written for TF 1.x and isn't really relevant here")
    def test_compile_tf_model(self):
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

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_stages = 5
            self.assertEqual(len(hidden_states), expected_num_stages)

            # MobileViT's feature maps are of shape (batch_size, num_channels, height, width)
            # with the width and height being successively divided by 2.
            divisor = 2
            for i in range(len(hidden_states)):
                self.assertListEqual(
                    list(hidden_states[i].shape[-2:]),
                    [self.model_tester.image_size // divisor, self.model_tester.image_size // divisor],
                )
                divisor *= 2

            self.assertEqual(self.model_tester.output_stride, divisor // 2)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_for_semantic_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation(*config_and_inputs)

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
            # Since `TFMobileViTModel` cannot operate with the default `fit()` method.
            if model_class.__name__ != "TFMobileViTModel":
                model = model_class(config)
                if getattr(model, "hf_compute_loss", None):
                    super().test_keras_fit()

    # The default test_loss_computation() uses -100 as a proxy ignore_index
    # to test masked losses. Overridding to avoid -100 since semantic segmentation
    #  models use `semantic_loss_ignore_index` from the config.
    def test_loss_computation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            # set an ignore index to correctly test the masked loss used in
            # `TFMobileViTForSemanticSegmentation`.
            if model_class.__name__ != "TFMobileViTForSemanticSegmentation":
                config.semantic_loss_ignore_index = 5

            model = model_class(config)
            if getattr(model, "hf_compute_loss", None):
                # The number of elements in the loss should be the same as the number of elements in the label
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                added_label = prepared_for_class[
                    sorted(list(prepared_for_class.keys() - inputs_dict.keys()), reverse=True)[0]
                ]
                expected_loss_size = added_label.shape.as_list()[:1]

                # Test that model correctly compute the loss with kwargs
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                possible_input_names = {"input_ids", "pixel_values", "input_features"}
                input_name = possible_input_names.intersection(set(prepared_for_class)).pop()
                model_input = prepared_for_class.pop(input_name)

                loss = model(model_input, **prepared_for_class)[0]
                self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])

                # Test that model correctly compute the loss when we mask some positions
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                possible_input_names = {"input_ids", "pixel_values", "input_features"}
                input_name = possible_input_names.intersection(set(prepared_for_class)).pop()
                model_input = prepared_for_class.pop(input_name)
                if "labels" in prepared_for_class:
                    labels = prepared_for_class["labels"].numpy()
                    if len(labels.shape) > 1 and labels.shape[1] != 1:
                        # labels[0] = -100
                        prepared_for_class["labels"] = tf.convert_to_tensor(labels)
                        loss = model(model_input, **prepared_for_class)[0]
                        self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])
                        self.assertTrue(not np.any(np.isnan(loss.numpy())))

                # Test that model correctly compute the loss with a dict
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)
                loss = model(prepared_for_class)[0]
                self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])

                # Test that model correctly compute the loss with a tuple
                prepared_for_class = self._prepare_for_class(inputs_dict.copy(), model_class, return_labels=True)

                # Get keys that were added with the _prepare_for_class function
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

                self.assertTrue(loss.shape.as_list() == expected_loss_size or loss.shape.as_list() == [1])

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFMobileViTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_tf
class TFMobileViTModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_image_classification_head(self):
        model = TFMobileViTForImageClassification.from_pretrained("apple/mobilevit-xx-small")

        feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-xx-small")
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="tf")

        # forward pass
        outputs = model(**inputs, training=False)

        # verify the logits
        expected_shape = tf.TensorShape((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = tf.constant([-1.9364, -1.2327, -0.4653])

        tf.debugging.assert_near(outputs.logits[0, :3], expected_slice, atol=1e-4, rtol=1e-04)

    @slow
    def test_inference_semantic_segmentation(self):
        # `from_pt` will be removed
        model = TFMobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-xx-small")

        feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/deeplabv3-mobilevit-xx-small")

        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="tf")

        # forward pass
        outputs = model(inputs.pixel_values, training=False)
        logits = outputs.logits

        # verify the logits
        expected_shape = tf.TensorShape((1, 21, 32, 32))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = tf.constant(
            [
                [[6.9713, 6.9786, 7.2422], [7.2893, 7.2825, 7.4446], [7.6580, 7.8797, 7.9420]],
                [[-10.6869, -10.3250, -10.3471], [-10.4228, -9.9868, -9.7132], [-11.0405, -11.0221, -10.7318]],
                [[-3.3089, -2.8539, -2.6740], [-3.2706, -2.5621, -2.5108], [-3.2534, -2.6615, -2.6651]],
            ]
        )

        tf.debugging.assert_near(logits[0, :3, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
