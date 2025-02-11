# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the TensorFlow SwiftFormer model."""

import inspect
import unittest

from transformers import SwiftFormerConfig
from transformers.testing_utils import (
    require_tf,
    require_vision,
    slow,
)
from transformers.utils import cached_property, is_tf_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import TFSwiftFormerForImageClassification, TFSwiftFormerModel
    from transformers.modeling_tf_utils import keras


if is_vision_available():
    from PIL import Image

    from transformers import ViTImageProcessor


class TFSwiftFormerModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        image_size=224,
        num_labels=2,
        layer_depths=[3, 3, 6, 4],
        embed_dims=[48, 56, 112, 220],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_labels = num_labels
        self.image_size = image_size
        self.layer_depths = layer_depths
        self.embed_dims = embed_dims

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return SwiftFormerConfig(
            depths=self.layer_depths,
            embed_dims=self.embed_dims,
            mlp_ratio=4,
            downsamples=[True, True, True, True],
            hidden_act="gelu",
            num_labels=self.num_labels,
            down_patch_size=3,
            down_stride=2,
            down_pad=1,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_layer_scale=True,
            layer_scale_init_value=1e-5,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = TFSwiftFormerModel(config=config)
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.embed_dims[-1], 7, 7))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = TFSwiftFormerForImageClassification(config)
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

        model = TFSwiftFormerForImageClassification(config)

        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        (config, pixel_values, labels) = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_tf
class TFSwiftFormerModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SwiftFormer does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (TFSwiftFormerModel, TFSwiftFormerForImageClassification) if is_tf_available() else ()

    pipeline_model_mapping = (
        {"feature-extraction": TFSwiftFormerModel, "image-classification": TFSwiftFormerForImageClassification}
        if is_tf_available()
        else {}
    )

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False
    test_onnx = False
    from_pretrained_id = "MBZUAI/swiftformer-xs"

    def setUp(self):
        self.model_tester = TFSwiftFormerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=SwiftFormerConfig,
            has_text_modality=False,
            hidden_size=37,
            num_attention_heads=12,
            num_hidden_layers=12,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="TFSwiftFormer does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, keras.layers.Dense))

    # Copied from transformers.tests.models.deit.test_modeling_tf_deit.py
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

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model = TFSwiftFormerModel.from_pretrained(self.from_pretrained_id)
        self.assertIsNotNone(model)

    @unittest.skip(reason="TFSwiftFormer does not output attentions")
    def test_attention_outputs(self):
        pass

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_stages = 8
            self.assertEqual(len(hidden_states), expected_num_stages)

            # SwiftFormer's feature maps are of shape (batch_size, embed_dims, height, width)
            # with the width and height being successively divided by 2, after every 2 blocks
            for i in range(len(hidden_states)):
                self.assertEqual(
                    hidden_states[i].shape,
                    tf.TensorShape(
                        [
                            self.model_tester.batch_size,
                            self.model_tester.embed_dims[i // 2],
                            (self.model_tester.image_size // 4) // 2 ** (i // 2),
                            (self.model_tester.image_size // 4) // 2 ** (i // 2),
                        ]
                    ),
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_tf
@require_vision
class TFSwiftFormerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return ViTImageProcessor.from_pretrained("MBZUAI/swiftformer-xs") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = TFSwiftFormerForImageClassification.from_pretrained("MBZUAI/swiftformer-xs")

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="tf")

        # forward pass
        outputs = model(**inputs)

        # verify the logits
        expected_shape = tf.TensorShape((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = tf.constant([[-2.1703e00, 2.1107e00, -2.0811e00]])
        tf.debugging.assert_near(outputs.logits[0, :3], expected_slice, atol=1e-4)
