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
"""Testing suite for the PyTorch Pvt model."""

import unittest

from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import (
    Expectations,
    require_accelerate,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import PvtConfig, PvtForImageClassification, PvtImageProcessor, PvtModel
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES


if is_vision_available():
    from PIL import Image


class PvtConfigTester(ConfigTester):
    def run_common_tests(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "hidden_sizes"))
        self.parent.assertTrue(hasattr(config, "num_encoder_blocks"))


class PvtModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=64,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[16, 32, 64, 128],
        downsampling_rates=[1, 4, 8, 16],
        num_attention_heads=[1, 2, 4, 8],
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
        return PvtConfig(
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
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = PvtModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertIsNotNone(result.last_hidden_state)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
class PvtModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PvtModel, PvtForImageClassification) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-feature-extraction": PvtModel, "image-classification": PvtForImageClassification}
        if is_torch_available()
        else {}
    )

    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_torchscript = False
    has_attentions = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = PvtModelTester(self)
        self.config_tester = PvtConfigTester(self, config_class=PvtConfig)

    def test_batching_equivalence(self, atol=1e-4, rtol=1e-4):
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Pvt does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Pvt does not have get_input_embeddings method and get_output_embeddings methods")
    def test_model_get_set_embeddings(self):
        pass

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for name, param in model.named_parameters():
                self.assertTrue(
                    -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                    msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = sum(self.model_tester.depths) + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            # verify the first hidden states (first block)
            self.assertListEqual(
                list(hidden_states[0].shape[-3:]),
                [
                    self.model_tester.batch_size,
                    (self.model_tester.image_size // 4) ** 2,
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

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            if model_class.__name__ in MODEL_MAPPING_NAMES.values():
                continue
            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    @slow
    def test_model_from_pretrained(self):
        model_name = "Zetatech/pvt-tiny-224"
        model = PvtModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_torch
class PvtModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_image_classification(self):
        # only resize + normalize
        image_processor = PvtImageProcessor.from_pretrained("Zetatech/pvt-tiny-224")
        model = PvtForImageClassification.from_pretrained("Zetatech/pvt-tiny-224").to(torch_device).eval()

        image = prepare_img()
        encoded_inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = encoded_inputs.pixel_values.to(torch_device)

        with torch.no_grad():
            outputs = model(pixel_values)

        expected_shape = torch.Size((1, model.config.num_labels))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expectations = Expectations(
            {
                (None, None): [-1.4192, -1.9158, -0.9702],
                ("cuda", 8): [-1.4194, -1.9161, -0.9705],
            }
        )
        expected_slice = torch.tensor(expectations.get_expectation()).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=2e-4, atol=2e-4)

    @slow
    def test_inference_model(self):
        model = PvtModel.from_pretrained("Zetatech/pvt-tiny-224").to(torch_device).eval()

        image_processor = PvtImageProcessor.from_pretrained("Zetatech/pvt-tiny-224")
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values)

        # verify the logits
        expected_shape = torch.Size((1, 50, 512))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expectations = Expectations(
            {
                (None, None): [[-0.3086, 1.0402, 1.1816], [-0.2880, 0.5781, 0.6124], [0.1480, 0.6129, -0.0590]],
                ("cuda", 8): [[-0.3084, 1.0402, 1.1816], [-0.2883, 0.5781, 0.6123], [0.1487, 0.6119, -0.0584]],
            }
        )
        expected_slice = torch.tensor(expectations.get_expectation()).to(torch_device)

        torch.testing.assert_close(outputs.last_hidden_state[0, :3, :3], expected_slice, rtol=2e-4, atol=2e-4)

    @slow
    @require_accelerate
    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_fp16(self):
        r"""
        A small test to make sure that inference work in half precision without any problem.
        """
        model = PvtForImageClassification.from_pretrained("Zetatech/pvt-tiny-224", torch_dtype=torch.float16)
        model.to(torch_device)
        image_processor = PvtImageProcessor(size=224)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch_device, dtype=torch.float16)

        # forward pass to make sure inference works in fp16
        with torch.no_grad():
            _ = model(pixel_values)
