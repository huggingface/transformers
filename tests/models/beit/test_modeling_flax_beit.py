# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import numpy as np

from transformers import BeitConfig
from transformers.testing_utils import require_flax, require_vision, slow
from transformers.utils import cached_property, is_flax_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_flax_common import FlaxModelTesterMixin, floats_tensor, ids_tensor


if is_flax_available():
    import jax

    from transformers import FlaxBeitForImageClassification, FlaxBeitForMaskedImageModeling, FlaxBeitModel

if is_vision_available():
    from PIL import Image

    from transformers import BeitImageProcessor


class FlaxBeitModelTester(unittest.TestCase):
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
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        num_labels=3,
    ):
        self.parent = parent
        self.vocab_size = vocab_size
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

        # in BeiT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = BeitConfig(
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
        )

        return config, pixel_values, labels

    def create_and_check_model(self, config, pixel_values, labels):
        model = FlaxBeitModel(config=config)
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_lm(self, config, pixel_values, labels):
        model = FlaxBeitForMaskedImageModeling(config=config)
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length - 1, self.vocab_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.type_sequence_label_size
        model = FlaxBeitForImageClassification(config=config)
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        # test greyscale images
        config.num_channels = 1
        model = FlaxBeitForImageClassification(config)

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_flax
class FlaxBeitModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (FlaxBeitModel, FlaxBeitForImageClassification, FlaxBeitForMaskedImageModeling) if is_flax_available() else ()
    )

    def setUp(self) -> None:
        self.model_tester = FlaxBeitModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BeitConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    # We need to override this test because Beit's forward signature is different than text models.
    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.__call__)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    # We need to override this test because Beit expects pixel_values instead of input_ids
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

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained("microsoft/beit-base-patch16-224")
            outputs = model(np.ones((1, 3, 224, 224)))
            self.assertIsNotNone(outputs)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_vision
@require_flax
class FlaxBeitModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224") if is_vision_available() else None

    @slow
    def test_inference_masked_image_modeling_head(self):
        model = FlaxBeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

        image_processor = self.default_image_processor
        image = prepare_img()
        pixel_values = image_processor(images=image, return_tensors="np").pixel_values

        # prepare bool_masked_pos
        bool_masked_pos = np.ones((1, 196), dtype=bool)

        # forward pass
        outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        logits = outputs.logits

        # verify the logits
        expected_shape = (1, 196, 8192)
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = np.array(
            [[-3.2437, 0.5072, -13.9174], [-3.2456, 0.4948, -13.9401], [-3.2033, 0.5121, -13.8550]]
        )

        self.assertTrue(np.allclose(logits[bool_masked_pos][:3, :3], expected_slice, atol=1e-2))

    @slow
    def test_inference_image_classification_head_imagenet_1k(self):
        model = FlaxBeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="np")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = (1, 1000)
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = np.array([-1.2385, -1.0987, -1.0108])

        self.assertTrue(np.allclose(logits[0, :3], expected_slice, atol=1e-4))

        expected_class_idx = 281
        self.assertEqual(logits.argmax(-1).item(), expected_class_idx)

    @slow
    def test_inference_image_classification_head_imagenet_22k(self):
        model = FlaxBeitForImageClassification.from_pretrained("microsoft/beit-large-patch16-224-pt22k-ft22k")

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="np")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = (1, 21841)
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = np.array([1.6881, -0.2787, 0.5901])

        self.assertTrue(np.allclose(logits[0, :3], expected_slice, atol=1e-4))

        expected_class_idx = 2396
        self.assertEqual(logits.argmax(-1).item(), expected_class_idx)
