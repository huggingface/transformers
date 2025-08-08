# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch AIMv2 model."""

import inspect
import tempfile
import unittest

import numpy as np
import requests
from parameterized import parameterized
from pytest import mark

from transformers import Aimv2Config, Aimv2TextConfig, Aimv2VisionConfig
from transformers.testing_utils import (
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    require_torch_sdpa,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import (
    is_torch_available,
    is_vision_available,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    _config_zero_init,
    _test_eager_matches_sdpa_inference,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        Aimv2Model,
        Aimv2TextModel,
        Aimv2VisionModel,
    )


if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor, AutoProcessor


class Aimv2VisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=False,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return Aimv2VisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
        )

    def create_and_check_model(self, config, pixel_values):
        model = Aimv2VisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class Aimv2ModelTesterMixin(ModelTesterMixin):
    """
    Subclass of ModelTesterMixin with methods specific to testing Aimv2 models.
    The SDPA equivalence test is overridden here because Aimv2 models may have test/vision/text+vision inputs,
    different output logits, and are not supposed to be used or tested with padding_side="left".
    """

    def test_sdpa_can_dispatch_composite_models(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Load the model with SDPA
                model_sdpa = model_class.from_pretrained(tmpdirname)

                # Load model with eager attention
                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

            if hasattr(model_sdpa, "vision_model"):
                self.assertTrue(model_sdpa.vision_model.config._attn_implementation == "sdpa")
                self.assertTrue(model_eager.vision_model.config._attn_implementation == "eager")

            if hasattr(model_sdpa, "text_model"):
                self.assertTrue(model_sdpa.text_model.config._attn_implementation == "sdpa")
                self.assertTrue(model_eager.text_model.config._attn_implementation == "eager")

            self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
            self.assertTrue(model_eager.config._attn_implementation == "eager")


@require_torch
class Aimv2VisionModelTest(Aimv2ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Aimv2 does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (Aimv2VisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Aimv2VisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Aimv2VisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Aimv2 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)


class Aimv2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=25,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return Aimv2TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = Aimv2TextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class Aimv2TextModelTest(Aimv2ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Aimv2TextModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False
    test_resize_embeddings = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Aimv2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Aimv2TextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Aimv2 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass


class Aimv2ModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=False):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = Aimv2TextModelTester(parent, **text_kwargs)
        self.vision_model_tester = Aimv2VisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return Aimv2Config.from_text_vision_configs(
            self.text_model_tester.get_config(), self.vision_model_tester.get_config(), projection_dim=64
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = Aimv2Model(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, pixel_values, attention_mask)
        self.parent.assertEqual(
            result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict


@require_torch
class Aimv2ModelTest(Aimv2ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    additional_model_inputs = ["pixel_values"]
    all_model_classes = (Aimv2Model,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": Aimv2Model, "image-feature-extraction": Aimv2VisionModel}
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_attention_outputs = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Aimv2ModelTester(self)
        common_properties = ["projection_dim", "logit_scale_init_value"]
        self.config_tester = ConfigTester(
            self, config_class=Aimv2Config, has_text_modality=False, common_properties=common_properties
        )

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        print(config_and_inputs)
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Aimv2Model does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("Size mismatch on CUDA")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    # Override as the `logit_scale` parameter initialization is different for Aimv2
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # check if `logit_scale` is initialized as per the original implementation
                    if name == "logit_scale":
                        self.assertAlmostEqual(
                            param.data.item(),
                            np.log(1 / 0.07),
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save Aimv2Config and check if we can load Aimv2VisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Aimv2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save Aimv2Config and check if we can load Aimv2TextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = Aimv2TextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, dtype=torch.bfloat16)
                model.to(torch_device)

                dummy_pixel_values = inputs_dict["pixel_values"].to(torch.bfloat16)
                dummy_input_ids = inputs_dict["input_ids"]

                outputs = model(pixel_values=dummy_pixel_values, input_ids=dummy_input_ids, output_hidden_states=True)
                outputs_fa = model_fa(
                    pixel_values=dummy_pixel_values, input_ids=dummy_input_ids, output_hidden_states=True
                )

                self.assertTrue(
                    torch.allclose(outputs.logits_per_image, outputs_fa.logits_per_image, atol=4e-2, rtol=4e-2),
                    f"Image logits max diff: {torch.max(torch.abs(outputs.logits_per_image - outputs_fa.logits_per_image))}",
                )
                self.assertTrue(
                    torch.allclose(outputs.logits_per_text, outputs_fa.logits_per_text, atol=4e-2, rtol=4e-2),
                    f"Text logits max diff: {torch.max(torch.abs(outputs.logits_per_text - outputs_fa.logits_per_text))}",
                )

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, dtype=torch.bfloat16, attn_implementation="eager")
                model.to(torch_device)

                dummy_pixel_values = inputs_dict["pixel_values"].to(torch.bfloat16)
                dummy_input_ids = inputs_dict["input_ids"]
                dummy_pixel_mask = inputs_dict["attention_mask"]

                # right padding
                dummy_pixel_mask[:] = 1
                dummy_pixel_mask[:, -1:] = 0

                outputs = model(pixel_values=dummy_pixel_values, input_ids=dummy_input_ids, output_hidden_states=True)
                outputs_fa = model_fa(
                    pixel_values=dummy_pixel_values, input_ids=dummy_input_ids, output_hidden_states=True
                )

                logits_per_image_eager = outputs.logits_per_image[:, :-1]
                logits_per_text_eager = outputs.logits_per_text[:, :-1]

                logits_per_image_sdpa = outputs_fa.logits_per_image[:, :-1]
                logits_per_text_sdpa = outputs_fa.logits_per_text[:, :-1]

                self.assertTrue(
                    torch.allclose(logits_per_image_eager, logits_per_image_sdpa, atol=4e-2, rtol=4e-2),
                    f"Image logits max diff: {torch.max(torch.abs(logits_per_image_eager - logits_per_image_sdpa))}",
                )
                self.assertTrue(
                    torch.allclose(logits_per_text_eager, logits_per_text_sdpa, atol=4e-2, rtol=4e-2),
                    f"Text logits max diff: {torch.max(torch.abs(logits_per_text_eager - logits_per_text_sdpa))}",
                )

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @require_torch_sdpa
    def test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
    ):
        "We need to relax a bit the `atols` for fp32 here due to the altup projections"
        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 3e-2,  # this was relaxed
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 3e-2,  # this was relaxed
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 3e-2,  # this was relaxed
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 3e-2,  # this was relaxed
            ("cuda", True, torch.float16): 5e-3,
        }
        _test_eager_matches_sdpa_inference(
            self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels, atols=atols
        )


@require_vision
@require_torch
class Aimv2ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "apple/aimv2-large-patch14-224-lit"
        model = Aimv2Model.from_pretrained(model_name, device_map=torch_device)
        processor = AutoProcessor.from_pretrained(model_name)

        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"], images=image, padding=True, return_tensors="pt"
        ).to(model.device)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Verify the logits
        self.assertEqual(
            outputs.logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        # handle device
        expected_logits = torch.tensor([[33.3550, 26.4255]]).to(model.device)
        torch.testing.assert_close(outputs.logits_per_image, expected_logits, atol=1e-3, rtol=1e-3)


@require_vision
@require_torch
class Aimv2VisionModelIntegrationTests(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "apple/aimv2-large-patch14-224"

        model = Aimv2VisionModel.from_pretrained(model_name, device_map=torch_device)
        processor = AutoImageProcessor.from_pretrained(model_name)

        image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
        inputs = processor(image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model(**inputs)

        # Verify logits shape
        self.assertEqual(output.last_hidden_state.shape, torch.Size([1, 256, 1024]))

        # Verify logits slice
        # fmt: off
        expected_logits = torch.tensor(
        [[ 0.0510,  0.0806, -0.0990, -0.0154],
        [ 2.7850, -2.5143, -0.3320,  2.4196],
        [ 2.8179, -2.4089, -0.2770,  2.3218],
        [ 2.7641, -2.4114, -0.3684,  2.2998],
        [ 2.7972, -2.3180, -0.4490,  2.2302],
        [ 2.8584, -2.5322, -0.2302,  2.4936],
        [-2.7849,  2.4121,  1.3670, -1.5514]]).to(model.device)
        # fmt: on

        output_slice = output.last_hidden_state.squeeze(0)[0:7, 0:4]
        self.assertTrue(torch.allclose(output_slice, expected_logits, atol=1e-3))

    @slow
    def test_inference_for_native_resolution(self):
        model_name = "apple/aimv2-large-patch14-native"

        model = Aimv2VisionModel.from_pretrained(model_name, device_map="auto")
        processor = AutoImageProcessor.from_pretrained(model_name)

        image = image = Image.open(
            requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
        )
        inputs = processor(image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model(**inputs)

        # Verify logits shape
        self.assertEqual(output.last_hidden_state.shape, torch.Size([1, 1530, 1024]))

        # Verify logits slice
        # fmt: off
        expected_logits = torch.tensor(
        [[-1.3342,  0.3720,  0.0963,  0.4159],
        [-1.5328,  0.4677,  0.0936,  0.4321],
        [-0.3775, -0.2758, -0.0803, -0.5367],
        [-1.3877,  0.5561, -1.9064, -1.1766],
        [-0.5148,  0.0108, -0.4515, -0.6402],
        [-0.3400, -0.1711, -0.1855, -0.4219],
        [-1.2877, -0.0585, -0.1646,  0.7420]]).to(model.device)
        # fmt: on

        output_slice = output.last_hidden_state.squeeze(0)[0:7, 0:4]
        self.assertTrue(torch.allclose(output_slice, expected_logits, atol=1e-3))
