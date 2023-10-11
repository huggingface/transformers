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

import inspect
import unittest

from transformers import ResNetConfig, is_flax_available
from transformers.testing_utils import require_flax, slow
from transformers.utils import cached_property, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_flax_common import FlaxModelTesterMixin, floats_tensor


if is_flax_available():
    import jax
    import jax.numpy as jnp

    from transformers.models.resnet.modeling_flax_resnet import FlaxResNetForImageClassification, FlaxResNetModel

if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class FlaxResNetModelTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=3,
        image_size=32,
        num_channels=3,
        embeddings_size=10,
        hidden_sizes=[10, 20, 30, 40],
        depths=[1, 1, 2, 1],
        is_training=True,
        use_labels=True,
        hidden_act="relu",
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.embeddings_size = embeddings_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.scope = scope
        self.num_stages = len(hidden_sizes)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return ResNetConfig(
            num_channels=self.num_channels,
            embeddings_size=self.embeddings_size,
            hidden_sizes=self.hidden_sizes,
            depths=self.depths,
            hidden_act=self.hidden_act,
            num_labels=self.num_labels,
            image_size=self.image_size,
        )

    def create_and_check_model(self, config, pixel_values):
        model = FlaxResNetModel(config=config)
        result = model(pixel_values)

        # Output shape (b, c, h, w)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.hidden_sizes[-1], self.image_size // 32, self.image_size // 32),
        )

    def create_and_check_for_image_classification(self, config, pixel_values):
        config.num_labels = self.num_labels
        model = FlaxResNetForImageClassification(config=config)
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_flax
class FlaxResNetModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (FlaxResNetModel, FlaxResNetForImageClassification) if is_flax_available() else ()

    is_encoder_decoder = False
    test_head_masking = False
    has_attentions = False

    def setUp(self) -> None:
        self.model_tester = FlaxResNetModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ResNetConfig, has_text_modality=False)

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

    @unittest.skip(reason="ResNet does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="ResNet does not support input and output embeddings")
    def test_model_common_attributes(self):
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

            expected_num_stages = self.model_tester.num_stages
            self.assertEqual(len(hidden_states), expected_num_stages + 1)

    @unittest.skip(reason="ResNet does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

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
class FlaxResNetModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained("microsoft/resnet-50") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = FlaxResNetForImageClassification.from_pretrained("microsoft/resnet-50")

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="np")

        outputs = model(**inputs)

        # verify the logits
        expected_shape = (1, 1000)
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = jnp.array([-11.1069, -9.7877, -8.3777])

        self.assertTrue(jnp.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
