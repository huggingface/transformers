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
"""Testing suite for the PyTorch Fgclip2 model."""

import inspect
import tempfile
import unittest

import numpy as np
import requests
from parameterized import parameterized
from pytest import mark

from transformers import Fgclip2Config, Fgclip2TextConfig, Fgclip2VisionConfig
from transformers.testing_utils import (
    is_flaky,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
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
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import Fgclip2Model, Fgclip2TextModel, Fgclip2VisionModel

if is_vision_available():
    from PIL import Image

    from transformers import Fgclip2Processor


class Fgclip2ModelTesterMixin(ModelTesterMixin):
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

            if hasattr(model_sdpa, "vision_model"):
                self.assertTrue(model_sdpa.vision_model.config._attn_implementation == "sdpa")
                self.assertTrue(model_eager.vision_model.config._attn_implementation == "eager")

            if hasattr(model_sdpa, "text_model"):
                self.assertTrue(model_sdpa.text_model.config._attn_implementation == "sdpa")
                self.assertTrue(model_eager.text_model.config._attn_implementation == "eager")

            self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
            self.assertTrue(model_eager.config._attn_implementation == "eager")

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence(self):
        dtype = torch.float16

        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            # Prepare inputs
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            if "pixel_values" in inputs_dict:
                inputs_dict["pixel_values"] = inputs_dict["pixel_values"].to(dtype)

            # Separate masks
            attention_masks = {}
            if "attention_mask" in inputs_dict:
                # attention_masks["attention_mask"] = inputs_dict.pop("attention_mask")
                inputs_dict["attention_mask"] = None
            if "pixel_attention_mask" in inputs_dict:
                attention_masks["pixel_attention_mask"] = inputs_dict.pop("pixel_attention_mask")
                inputs_dict["pixel_attention_mask"] = None

            # Save and load model with flash attention 2 and eager attentions
            with tempfile.TemporaryDirectory() as tmp_dir:
                model = model_class(config)
                model.save_pretrained(tmp_dir)

                model = model_class.from_pretrained(tmp_dir, dtype=dtype)
                model_fa = model_class.from_pretrained(tmp_dir, dtype=dtype, attn_implementation="flash_attention_2")

            model_fa.to(torch_device)
            model.to(torch_device)

            # Run forward pass without attention masks
            with torch.no_grad():
                outputs = model(**inputs_dict, output_hidden_states=True)
                outputs_fa = model_fa(**inputs_dict, output_hidden_states=True)

            # Choose which key to compare
            key = [k for k in ["logits", "logits_per_image", "last_hidden_state"] if k in outputs][0]

            torch.testing.assert_close(outputs[key], outputs_fa[key], atol=4e-2, rtol=4e-2)

            # Run forward pass with attention masks
            inputs_dict.update(attention_masks)
            with torch.no_grad():
                outputs = model(**inputs_dict, output_hidden_states=True)
                outputs_fa = model_fa(**inputs_dict, output_hidden_states=True)

            output_tensor = outputs[key]
            output_tensor_fa = outputs_fa[key]

            # Mask out padded tokens, they are different for SDPA and Flash Attention 2
            if key == "last_hidden_state" and "pixel_attention_mask" in inputs_dict:
                output_tensor = output_tensor * inputs_dict["pixel_attention_mask"][..., None]
                output_tensor_fa = output_tensor_fa * inputs_dict["pixel_attention_mask"][..., None]
            elif key == "last_hidden_state" and inputs_dict.get("attention_mask", None) is not None:
                output_tensor = output_tensor * inputs_dict["attention_mask"][..., None]
                output_tensor_fa = output_tensor_fa * inputs_dict["attention_mask"][..., None]

            torch.testing.assert_close(output_tensor, output_tensor_fa, atol=4e-2, rtol=4e-2)

            # Check with inference + dropout
            model.train()
            _ = model_fa(**inputs_dict, output_hidden_states=True)

    @unittest.skip(reason="Fgclip2 has default right padding (tested in test_flash_attn_2_inference_equivalence)")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(reason="SDPA can't dispatch on flash with not None `attention_mask`")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, *args):
        # adding only flaky decorator here and call the parent test method
        return getattr(ModelTesterMixin, self._testMethodName)(self)

    @is_flaky()
    def test_batching_equivalence(self, *args):
        return getattr(ModelTesterMixin, self._testMethodName)(self)


class Fgclip2VisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        num_patches=16,
        image_num_patches=24,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope
        self.seq_length = image_num_patches

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.seq_length,
                self.num_channels * self.patch_size * self.patch_size,
            ]
        )
        pixel_attention_mask = torch.zeros(self.batch_size, self.seq_length, device=torch_device, dtype=torch.long)

        spatial_shapes = [
            (height, width)
            for height in range(1, self.seq_length)
            for width in range(1, self.seq_length)
            if height * width <= self.seq_length
        ] * self.batch_size
        spatial_shapes = spatial_shapes[: self.batch_size]
        spatial_shapes = torch.tensor(spatial_shapes, device=torch_device, dtype=torch.long)

        for i, (height, width) in enumerate(spatial_shapes):
            pixel_attention_mask[i, : height * width] = 1

        config = self.get_config()

        return config, pixel_values, pixel_attention_mask, spatial_shapes

    def get_config(self):
        return Fgclip2VisionConfig(
            num_patches=self.num_patches,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values, pixel_attention_mask, spatial_shapes):
        model = Fgclip2VisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values, pixel_attention_mask, spatial_shapes)

        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_attention_mask, spatial_shapes = self.prepare_config_and_inputs()
        inputs_dict = {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
        }
        return config, inputs_dict


@require_torch
class Fgclip2VisionModelTest(Fgclip2ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as FG-CLIP2 does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (Fgclip2VisionModel,) if is_torch_available() else ()
    additional_model_inputs = ["pixel_attention_mask", "spatial_shapes"]

    test_resize_embeddings = False
    # MP works but offload doesn't work when the MultiheadAttention is offloaded
    # TODO: One potential solution would be to add to set preload_module_classes = ["Fgclip2MultiheadAttentionPoolingHead"]
    # in the dispatch_model function
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def setUp(self):
        self.model_tester = Fgclip2VisionModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Fgclip2VisionConfig,
            has_text_modality=False,
            hidden_size=37,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="FG-CLIP2 does not use inputs_embeds")
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

    @unittest.skip(reason="Fgclip2VisionModel does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="Fgclip2VisionModel does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Fgclip2VisionModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="Fgclip2VisionModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Fgclip2 uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    # @slow
    def test_model_from_pretrained(self):
        model_name = "qihoo360/fg-clip2-base"
        model = Fgclip2VisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, *args):
        # adding only flaky decorator here and call the parent test method
        return getattr(ModelTesterMixin, self._testMethodName)(self)


class Fgclip2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope

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
        return Fgclip2TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = Fgclip2TextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class Fgclip2TextModelTest(Fgclip2ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Fgclip2TextModel,) if is_torch_available() else ()
    fx_compatible = False
    test_resize_embeddings = False
    test_pruning = False
    test_head_masking = False
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = Fgclip2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Fgclip2TextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Fgclip2TextModel does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="Fgclip2TextModel does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Fgclip2TextModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="Fgclip2TextModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Fgclip2 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Fgclip2 uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    # @slow
    def test_model_from_pretrained(self):
        model_name = "qihoo360/fg-clip2-base"
        model = Fgclip2TextModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class Fgclip2ModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = Fgclip2TextModelTester(parent, **text_kwargs)
        self.vision_model_tester = Fgclip2VisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values, pixel_attention_mask, spatial_shapes = (
            self.vision_model_tester.prepare_config_and_inputs()
        )

        config = self.get_config()

        return (
            config,
            input_ids,
            attention_mask,
            pixel_values,
            pixel_attention_mask,
            spatial_shapes,
        )

    def get_config(self):
        return Fgclip2Config(
            text_config=self.text_model_tester.get_config().to_dict(),
            vision_config=self.vision_model_tester.get_config().to_dict(),
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_attention_mask,
        spatial_shapes,
    ):
        model = Fgclip2Model(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(
                input_ids,
                pixel_values,
                pixel_attention_mask,
                spatial_shapes,
                attention_mask,
            )
        self.parent.assertEqual(
            result.logits_per_image.shape,
            (self.vision_model_tester.batch_size, self.text_model_tester.batch_size),
        )
        self.parent.assertEqual(
            result.logits_per_text.shape,
            (self.text_model_tester.batch_size, self.vision_model_tester.batch_size),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            pixel_values,
            pixel_attention_mask,
            spatial_shapes,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
            "attention_mask": attention_mask,
            "position_ids": None,
            "return_loss": False,
        }
        return config, inputs_dict


@require_torch
class Fgclip2ModelTest(Fgclip2ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Fgclip2Model,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": Fgclip2Model} if is_torch_available() else {}
    additional_model_inputs = [
        "pixel_values",
        "pixel_attention_mask",
        "spatial_shapes",
    ]
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    # MP works but offload doesn't work when the MultiheadAttention is offloaded
    # TODO: One potential solution would be to add to set preload_module_classes = ["Fgclip2MultiheadAttentionPoolingHead"]
    # in the dispatch_model function
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Fgclip2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Fgclip2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Fgclip2Model does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Fgclip2 uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    def test_load_vision_text_config(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Save Fgclip2Config and check if we can load Fgclip2VisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Fgclip2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save Fgclip2Config and check if we can load Fgclip2TextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = Fgclip2TextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    # @slow
    def test_model_from_pretrained(self):
        model_name = "qihoo360/fg-clip2-base"
        model = Fgclip2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest("Fgclip2 does not support right padding")


r"""
Test performance, including:
full-image short text retrieval (Chinese-English bilingual),
full-image long text retrieval (Chinese-English bilingual),
image local region and text retrieval,
"""


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
@require_torch
class Fgclip2ModelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_name = "qihoo360/fg-clip2-base"
        cls.model = Fgclip2Model.from_pretrained(model_name).to(torch_device)
        cls.processor = Fgclip2Processor.from_pretrained(model_name)
        cls.image = prepare_img()

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.processor
        torch.cuda.empty_cache()

    @slow
    def test_en_inference(self):
        short_en_text = [
            "Two cats sprawl on a pink sofa, blissed out.",
            "Bird hops on sunlit sill, pecking for crumbs.",
            "Elderly man jogs, smiling brightly.",
            "Post-rain street glows with rainbow reflections.",
            "Kids build sandcastles, laughter rides the waves.",
            "Students read quietly in the hushed library.",
        ]

        inputs = self.processor(text=short_en_text, images=self.image, return_tensors="pt")
        inputs = inputs.to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image

        # verify shape
        self.assertEqual(
            logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        expected_logits = torch.tensor(
            [[7.1313, -12.3647, -26.1456, -22.0568, -17.3236, -13.9281]],
            device=torch_device,
        )
        torch.testing.assert_close(logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)

        long_en_text = [
            "In this heartwarming photograph, two fluffy cats lounge lazily on a plush pink sofa, one curled sideways like a crescent moon, the other sprawled on its back with all four paws relaxed, seemingly basking in the gentle warmth of afternoon sunlight filtering through sheer curtains; their eyes are peacefully shut, whiskers twitching faintly, tails occasionally swaying—a portrait of pure, unguarded contentment. The softness of the sofa fabric and the pastel hue of pink create a dreamlike, therapeutic ambiance that invites you to reach out and stroke their velvety fur or even collapse beside them to share the tranquility. This posture of total vulnerability is the feline language of trust and safety, a silent testament to the comfort of home. They need no words; their very stillness speaks volumes about joy and fulfillment. In a world racing toward productivity and noise, these cats offer a gentle, furry reminder: sometimes the most radical act of self-care is to pause, curl up in something soft, and let your mind drift—because true peace isn’t found in hustle, but in the quiet surrender to comfort.",
            "At 5:30 AM, the city still slumbers, streets empty save for the rhythmic swish of a street cleaner’s broom and the distant hum of an occasional taxi—a symphony of pre-dawn stillness. I jog along the riverside, breathing crisp, dew-laced air that scrubs my lungs clean like mountain springwater. Above me, indigo night clings stubbornly to the sky, a few defiant stars still twinkling, while the eastern horizon blushes with the first streaks of dawn, heralding rebirth. By the third bridge, a white egret bursts from reeds, wings slicing through mist with balletic grace—and in that suspended moment, I understand: I don’t run to burn calories or post selfies. I run to sync my pulse with the earth’s before chaos erupts, to let thoughts settle like sediment in a shaken jar, to awaken my soul in the cathedral of morning light. This ritual isn’t exercise; it’s communion.",
            "In the hushed corner of a university library, a bespectacled student buries herself in a dog-eared copy of Kant’s Critique, her finger tracing yellowed pages, brow furrowing then smoothing as she scribbles marginalia. Rain taps gently against the windowpane, but inside, only the whisper of turning pages and HVAC’s low drone break the silence. Around her, towers of Nietzsche, Sartre, and Heidegger stand like silent mentors—she’s not studying; she’s conversing across centuries. In an age of TikTok lectures and algorithm-driven distractions, her choice to wrestle with dense, slow-burning texts is an act of rebellion. She seeks not credentials but clarity, not trends but truth. Reading here isn’t escapism—it’s excavation. When she finally closes the book and gazes out at the rain-streaked glass, her eyes hold a new stillness: the quiet certainty of someone who’s mapped her own mind and found coordinates for navigating a noisy world.",
            "In the kitchen at 3 AM, a mother simmers a pot of old-fashioned chicken soup for her child leaving home tomorrow. The stove glows amber, golden droplets of fat shimmer on the broth’s surface, and the aroma—rich with ginger, goji berries, and nostalgia—weaves through the air like a lullaby. She skims foam with a spoon, adjusts the heat with surgeon-like precision, each motion a ritual of love. On the fridge, a crooked childhood drawing of their family smiles beside a calendar circled with departure dates. Chopped scallions wait in neat rows like obedient soldiers. She doesn’t speak; her devotion simmers in the broth, her worries dissolve into the steam. This soup won’t win Michelin stars, but to her child, it’s the taste of sanctuary—a flavor no restaurant can replicate. Here, in the quiet alchemy of bone, water, and time, maternal love transforms into liquid gold: not loud, not lavish, but life-sustaining. The greatest gifts are often served in chipped bowls at ungodly hours.",
            "Artificial intelligence is irrevocably reshaping education: adaptive platforms tweak quiz difficulty based on response time, AI tutors dissect Shakespearean sonnets at midnight, and VR headsets teleport students to the Colosseum’s bloodstained sands. The monolithic ‘lecture hall’ model is crumbling, replaced by personalized learning pathways that cater to individual rhythms. Yet this isn’t utopia—algorithmic bias may entrench inequality, data harvesting threatens privacy, and over-reliance on tech could atrophy critical thinking. The real revolution lies not in tools but philosophy: when AI recites Plato, humans must learn to question; when machines solve equations, students should marvel at mathematical elegance. Technology is the oar, humanity the compass. The classroom of tomorrow must let AI handle repetition so teachers can ignite wonder, empowering students to wield tech without surrendering curiosity, skepticism, or the courage to create what algorithms cannot imagine.",
            "High-altitude lakes gleam like emerald shards dropped by gods between snow-capped peaks and rolling meadows, their crystalline waters mirroring scudding clouds and darting kingfishers. Yaks wade through shallows, bells chiming softly, while prayer flags snap in the wind like whispered sutras above a herder’s smoke-trailing tent. I sit on sun-warmed rocks, alpine flowers brushing my ankles, breathing air scented with pine resin and yak-butter tea. Here, no Wi-Fi bars blink, no calendar alerts chime—time flows with sun arcs and birdcalls. Urbanites chase ‘productivity’ yet starve for meaning; the plateau teaches that ‘being’ is enough. Watch a cloud morph, hear a raindrop kiss the lake, feel sunlight migrate across your skin. When nature recalibrates your soul, you realize: poetry isn’t found in plane tickets to Bali, but in the trembling awe of noticing a single wildflower after months of scrolling screens.",
        ]

        inputs = self.processor(images=self.image, text=long_en_text, max_length=196, return_tensors="pt")
        inputs = inputs.to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs, walk_type="long")

        logits_per_image = outputs.logits_per_image

        # verify shape
        self.assertEqual(
            logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )

        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        expected_logits = torch.tensor(
            [[1.5820, -14.4048, -23.1075, -16.6545, -16.3500, -21.9193]],
            device=torch_device,
        )
        torch.testing.assert_close(logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)

    @slow
    def test_cn_inference(self):
        short_cn_text = [
            "两只猫瘫在粉沙发上，一脸惬意。",
            "小鸟蹦跳，啄食窗台阳光。",
            "老人晨跑，笑容满面。",
            "雨后街道，彩虹倒影美。",
            "孩子堆沙堡，笑声随浪飞。",
            "图书馆里，学生安静读书。",
        ]

        inputs = self.processor(text=short_cn_text, images=self.image, return_tensors="pt")
        inputs = inputs.to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image

        # verify shape
        self.assertEqual(
            logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )

        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        expected_logits = torch.tensor(
            [[6.5309, -16.2254, -18.9228, -16.0596, -15.2771, -13.9292]],
            device=torch_device,
        )
        torch.testing.assert_close(logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)

        long_cn_text = [
            "在这张温馨的照片中，两只毛茸茸的猫咪慵懒地瘫在一张粉红色的绒布沙发上，一只侧卧蜷缩如月牙，另一只仰面朝天四脚舒展，仿佛在享受午后阳光透过窗帘洒下的温柔暖意；它们闭着眼睛，胡须微微颤动，尾巴偶尔轻摆，完全沉浸在无拘无束的放松状态中，沙发的柔软与色彩的甜美共同营造出一种童话般的治愈氛围，让人不禁想伸手抚摸它们蓬松的毛发，或干脆也躺下来共享这份宁静；猫咪的这种毫无防备的姿态，正是对家的安全感与主人信任的最好表达，它们不需要言语，仅凭姿态就能传递幸福与满足，也提醒着忙碌的人类：偶尔停下脚步，蜷在柔软处，放空大脑，才是对抗焦虑最温柔的方式。",
            "清晨五点半，城市尚未苏醒，街道空无一人，只有清洁工扫帚划过地面的沙沙声与远处偶尔驶过的出租车引擎低鸣交织成一首寂静的序曲；我沿着河岸慢跑，呼吸着微凉而湿润的空气，肺部像被山泉洗过般清爽，头顶是尚未褪去的深蓝夜幕，几颗倔强的星星仍在闪烁，东方天际却已悄悄泛起鱼肚白，预示着新一天的诞生；跑过第三座桥时，一只白鹭从芦苇丛中振翅飞起，翅膀划破薄雾，优雅如诗，那一刻我忽然明白，坚持晨跑不是为了减肥或打卡，而是为了在喧嚣来临前，独享这片刻与自然同步的宁静，让心跳与大地脉搏共振，让思绪在奔跑中沉淀，让灵魂在晨光中苏醒。",
            "图书馆的角落，一位戴眼镜的女生正埋首于厚重的哲学典籍，她指尖划过泛黄纸页，时而皱眉沉思，时而快速笔记，窗外雨滴敲打玻璃，室内却只有翻页声与空调低鸣；她身旁堆叠着康德、尼采与萨特，仿佛在与古今智者隔空对话，每一行字都是思想的阶梯，每一次停顿都是灵魂的震颤；在这个信息碎片化、注意力被算法切割的时代，她选择用整块时间沉浸于艰深文本，不是为了炫耀学识，而是为了在喧嚣世界中锚定自我，在他人追逐热点时，她正构建属于自己的认知宇宙；阅读不是逃避现实，而是更深刻地理解现实，当她合上书本望向窗外雨幕，眼神已不再迷茫，而是带着穿透表象的清澈与坚定。",
            "深夜厨房里，母亲正为明天远行的孩子熬制一锅老火鸡汤，灶火微红，汤面浮着金黄油星，香气如丝如缕钻入鼻腔，勾起童年无数个生病或考试前的温暖记忆；她轻轻撇去浮沫，加入枸杞与红枣，动作缓慢却精准，仿佛在进行某种神圣仪式；冰箱贴着孩子小时候画的歪扭全家福，墙上日历圈出出发日期，案板上切好的姜片整齐如士兵列队；她不说话，只是偶尔掀盖尝味，调整火候，把牵挂与不舍熬进汤里，把担忧与祝福炖入骨髓；这锅汤不会出现在任何美食榜单，却是孩子心中无可替代的‘家的味道’，它不昂贵，却无价，不炫技，却倾注了全部的爱——原来最深沉的母爱，往往藏在无声的厨房烟火里，藏在凌晨三点不肯熄灭的灶火中。",
            "人工智能正以不可逆之势重塑教育体系：从自适应学习平台根据学生答题速度调整难度，到AI助教24小时解答语法疑问，再到虚拟现实课堂让学生‘亲临’古罗马战场；传统‘教师讲、学生听’的单向灌输模式正在瓦解，取而代之的是以学习者为中心的个性化路径；但这并非乌托邦——算法偏见可能固化阶层，数据监控引发隐私焦虑，过度依赖技术或削弱批判思维；真正的教育革命不在工具升级，而在理念进化：当AI能背诵《论语》时，人类更需学会提问；当机器能解微积分时，学生更应理解数学之美；技术是桨，人文是舵，缺一不可；未来教室的模样，应是AI处理重复劳动，教师专注启迪心灵，学生在科技赋能下，依然保有好奇、质疑与创造的勇气。",
            "高原湖泊如一块被神明遗落的翡翠，镶嵌在雪山与草甸之间，湖水清澈见底，倒映着流云与飞鸟，牦牛群缓步涉水，脖铃叮当，牧民的帐篷升起袅袅炊烟；我坐在湖畔岩石上，脚下野花摇曳，风里带着雪松与酥油茶的气息，远处经幡在风中猎猎作响，仿佛在诵读古老的祈愿；这里没有Wi-Fi信号，没有会议提醒，时间以日升月落为刻度，生命以呼吸心跳为节奏；都市人追逐‘效率’与‘成就’，却常在深夜被空虚啃噬；而高原教会我的，是‘存在’本身即意义——看一朵云变形，听一滴水落湖，感受阳光在皮肤上移动的轨迹；当心灵被自然重新校准，我们才懂得：所谓诗与远方，不在机票终点，而在放下手机、凝视一朵野花时，内心涌起的久违悸动。",
        ]

        inputs = self.processor(images=self.image, text=long_cn_text, max_length=196, return_tensors="pt")
        inputs = inputs.to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs, walk_type="long")

        logits_per_image = outputs.logits_per_image
        # verify shape
        self.assertEqual(
            logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )

        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        expected_logits = torch.tensor(
            [[-0.3012, -15.8156, -21.3586, -13.2546, -15.2005, -23.6982]],
            device=torch_device,
        )
        torch.testing.assert_close(logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)

    @slow
    def test_region_inference(self):
        text = "remote control"

        inputs = self.processor(images=self.image, text=text, return_tensors="pt")
        inputs = inputs.to(torch_device)

        right_cat_bbox = [345, 27, 640, 372]
        left_remote_bbox = [41, 74, 177, 118]
        image_width, image_height = self.image.size

        bbox_features = self.model.get_image_region_features(
            **inputs,
            image_sizes=[[image_height, image_width]],
            region_infos=[[right_cat_bbox, left_remote_bbox]],
        )

        bbox_embeds = bbox_features[0]
        bbox_embeds = bbox_embeds / bbox_embeds.norm(p=2, dim=-1, keepdim=True)

        text_embeds = self.model.get_text_features(**inputs, walk_type="box")
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_image = (100.0 * text_embeds @ bbox_embeds.t()).softmax(dim=-1)
        print(logits_per_image)

        expected_logits = torch.tensor([[2.0169e-05, 9.9998e-01]], device=torch_device)

        torch.testing.assert_close(logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)
