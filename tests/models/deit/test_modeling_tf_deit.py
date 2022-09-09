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
""" Testing suite for the TensorFlow DeiT model. """


import inspect
import unittest

import numpy as np

from transformers import DeiTConfig
from transformers.testing_utils import require_tf, require_vision, slow
from transformers.utils import cached_property, is_tf_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TFDeiTForImageClassification,
        TFDeiTForImageClassificationWithTeacher,
        TFDeiTForMaskedImageModeling,
        TFDeiTModel,
    )
    from transformers.models.deit.modeling_tf_deit import TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import DeiTFeatureExtractor


class TFDeiTModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
        encoder_stride=2,
    ):
        self.parent = parent
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
        self.encoder_stride = encoder_stride

        # in DeiT, the seq length equals the number of patches + 2 (we add 2 for the [CLS] and distilation tokens)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 2

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DeiTConfig(
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
            encoder_stride=self.encoder_stride,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = TFDeiTModel(config=config)
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_image_modeling(self, config, pixel_values, labels):
        model = TFDeiTForMaskedImageModeling(config=config)
        result = model(pixel_values)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_channels, self.image_size, self.image_size)
        )

        # test greyscale images
        config.num_channels = 1
        model = TFDeiTForMaskedImageModeling(config)

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, 1, self.image_size, self.image_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.type_sequence_label_size
        model = TFDeiTForImageClassification(config)
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        # test greyscale images
        config.num_channels = 1
        model = TFDeiTForImageClassification(config)

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_tf
class TFDeiTModelTest(TFModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_tf_common.py, as DeiT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (
            TFDeiTModel,
            TFDeiTForImageClassification,
            TFDeiTForImageClassificationWithTeacher,
            TFDeiTForMaskedImageModeling,
        )
        if is_tf_available()
        else ()
    )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFDeiTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DeiTConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="DeiT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (tf.keras.layers.Layer))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, tf.keras.layers.Dense))

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

    def test_for_masked_image_modeling(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_image_modeling(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    # special case for DeiTForImageClassificationWithTeacher model
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "DeiTForImageClassificationWithTeacher":
                del inputs_dict["labels"]

        return inputs_dict

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFDeiTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_tf
@require_vision
class DeiTModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return (
            DeiTFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-224")
            if is_vision_available()
            else None
        )

    @slow
    def test_inference_image_classification_head(self):
        model = TFDeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="tf")

        # forward pass
        outputs = model(**inputs)

        # verify the logits
        expected_shape = tf.TensorShape((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = tf.constant([-1.0266, 0.1912, -1.2861])

        self.assertTrue(np.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
