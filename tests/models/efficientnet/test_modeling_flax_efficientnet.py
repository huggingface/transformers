# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import inspect
import unittest

from transformers import EfficientNetConfig, is_flax_available
from transformers.testing_utils import require_flax, slow
from transformers.utils import cached_property, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_flax_common import FlaxModelTesterMixin, floats_tensor


if is_flax_available():
    import jax
    import jax.numpy as jnp

    from transformers.models.efficientnet.modeling_flax_efficientnet import (
        FlaxEfficientNetForImageClassification,
        FlaxEfficientNetModel,
    )

if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class FlaxEfficientNetModelTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=1,
        image_size=224,
        num_channels=3,
        kernel_sizes=[3, 3, 5],
        in_channels=[32, 16, 24],
        out_channels=[16, 24, 40],
        strides=[1, 1, 2],
        num_block_repeats=[1, 1, 2],
        expand_ratios=[1, 6, 6],
        is_training=True,
        use_labels=True,
        intermediate_size=37,
        hidden_act="gelu",
        num_labels=10,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.num_block_repeats = num_block_repeats
        self.expand_ratios = expand_ratios
        self.is_training = is_training
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.use_labels = use_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return EfficientNetConfig(
            num_channels=self.num_channels,
            kernel_sizes=self.kernel_sizes,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=self.strides,
            num_block_repeats=self.num_block_repeats,
            expand_ratios=self.expand_ratios,
            hidden_act=self.hidden_act,
            num_labels=self.num_labels,
        )

    def create_and_check_model(self, config, pixel_values):
        model = FlaxEfficientNetModel(config=config)
        result = model(pixel_values)

        # expected last hidden states: B, C, H // 4, W // 4
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, config.hidden_dim, self.image_size // 4, self.image_size // 4),
        )

    def create_and_check_for_image_classification(self, config, pixel_values):
        config.num_labels = self.num_labels
        model = FlaxEfficientNetForImageClassification(config=config)
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_flax
class FlaxResNetModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxEfficientNetModel, FlaxEfficientNetForImageClassification) if is_flax_available() else ()

    is_encoder_decoder = False
    test_head_masking = False
    has_attentions = False

    def setUp(self) -> None:
        self.model_tester = FlaxEfficientNetModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=EfficientNetConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        return

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @unittest.skip(reason="EfficientNet does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EfficientNet does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="EfficientNet does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.__call__)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            # EfficientNet's feature maps are of shape (batch_size, num_channels, height, width)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.image_size // 2, self.model_tester.image_size // 2],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_jit_compilation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                model = model_class(config)

                @jax.jit
                def model_jitted(pixel_values, **kwargs):
                    return model(pixel_values=pixel_values, **kwargs)

                with self.subTest("JIT Enabled"):
                    jitted_outputs = model_jitted(**prepared_inputs_dict).to_tuple()

                with self.subTest("JIT Disabled"):
                    with jax.disable_jit():
                        outputs = model_jitted(**prepared_inputs_dict).to_tuple()

                self.assertEqual(len(outputs), len(jitted_outputs))
                for jitted_output, output in zip(jitted_outputs, outputs):
                    self.assertEqual(jitted_output.shape, output.shape)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_flax
class FlaxEfficientNetNetModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return AutoImageProcessor.from_pretrained("google/efficientnet-b7") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = FlaxEfficientNetForImageClassification.from_pretrained("Shubhamai/efficientnet-b7")

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="np")

        outputs = model(**inputs)

        # verify the logits
        expected_shape = (1, 1000)
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = jnp.array([0.0001, 0.0002, 0.0002])

        self.assertTrue(jnp.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
