# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch CLIP model."""

import inspect
import os
import tempfile
import unittest

import numpy as np
import requests
from parameterized import parameterized
from pytest import mark

from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
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
    floats_tensor,
    ids_tensor,
    is_flaky,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        CLIPForImageClassification,
        CLIPModel,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModel,
        CLIPVisionModelWithProjection,
    )

if is_vision_available():
    from PIL import Image

    from transformers import CLIPProcessor


class CLIPVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
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
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return CLIPVisionConfig(
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
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = CLIPVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_with_projection(self, config, pixel_values):
        model = CLIPVisionModelWithProjection(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.image_embeds.shape, (self.batch_size, self.projection_dim))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @require_torch_sdpa
    def test_eager_matches_sdpa_inference(self, *args):
        return getattr(ModelTesterMixin, self._testMethodName)(self)


class CLIPModelTesterMixin(ModelTesterMixin):
    """
    Subclass of ModelTesterMixin with methods specific to testing CLIP models.
    The SDPA equivalence test is overridden here because CLIP models may have test/vision/text+vision inputs,
    different output logits, and are not supposed to be used or tested with padding_side="left".
    """

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Load the model with SDPA (it is the default, but we explicit it for clarity)
                model_sdpa = model_class.from_pretrained(tmpdirname, attn_implementation="sdpa")
                model_sdpa = model_sdpa.eval().to(torch_device)

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
class CLIPVisionModelTest(CLIPModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CLIP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (CLIPVisionModel, CLIPVisionModelWithProjection) if is_torch_available() else ()
    fx_compatible = True
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = CLIPVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLIPVisionConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CLIP does not use inputs_embeds")
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

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_projection(*config_and_inputs)

    @unittest.skip
    def test_training(self):
        pass

    @unittest.skip
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPVisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "visual_projection"))

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @require_torch_sdpa
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, *args):
        # adding only flaky decorator here and call the parent test method
        return getattr(ModelTesterMixin, self._testMethodName)(self)

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()


class CLIPTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
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
        self.projection_dim = projection_dim
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
        return CLIPTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = CLIPTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_with_projection(self, config, input_ids, input_mask):
        model = CLIPTextModelWithProjection(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.text_embeds.shape, (self.batch_size, self.projection_dim))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class CLIPTextModelTest(CLIPModelTesterMixin, unittest.TestCase):
    all_model_classes = (CLIPTextModel, CLIPTextModelWithProjection) if is_torch_available() else ()
    fx_compatible = True
    test_pruning = False
    test_head_masking = False
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = CLIPTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLIPTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_projection(*config_and_inputs)

    @unittest.skip
    def test_training(self):
        pass

    @unittest.skip
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="CLIP does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPTextModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPTextModelWithProjection.from_pretrained(model_name)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "text_projection"))

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @require_torch_sdpa
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, *args):
        # adding only flaky decorator here and call the parent test method
        return getattr(ModelTesterMixin, self._testMethodName)(self)

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()

    @require_torch_sdpa
    def test_sdpa_can_dispatch_on_flash(self):
        self.skipTest(reason="CLIPTextModel has two attention masks: `causal_attention_mask` and `attention_mask`")


class CLIPModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = CLIPTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = CLIPVisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return CLIPConfig(
            text_config=self.text_model_tester.get_config().to_dict(),
            vision_config=self.vision_model_tester.get_config().to_dict(),
            projection_dim=64,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = CLIPModel(config).to(torch_device).eval()
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
            "return_loss": True,
        }
        return config, inputs_dict


@require_torch
class CLIPModelTest(CLIPModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (CLIPModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": CLIPModel, "image-feature-extraction": CLIPVisionModel} if is_torch_available() else {}
    )
    additional_model_inputs = ["pixel_values"]
    fx_compatible = True
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    _is_composite = True

    def setUp(self):
        self.model_tester = CLIPModelTester(self)
        common_properties = ["projection_dim", "logit_scale_init_value"]
        self.config_tester = ConfigTester(
            self, config_class=CLIPConfig, has_text_modality=False, common_properties=common_properties
        )

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="CLIPModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    # override as the `logit_scale` parameter initialization is different for CLIP
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

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()

            try:
                input_ids = inputs_dict["input_ids"]
                pixel_values = inputs_dict["pixel_values"]  # CLIP needs pixel_values
                traced_model = torch.jit.trace(model, (input_ids, pixel_values))
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            non_persistent_buffers = {}
            for key in loaded_model_state_dict:
                if key not in model_state_dict:
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save CLIPConfig and check if we can load CLIPVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = CLIPVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save CLIPConfig and check if we can load CLIPTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = CLIPTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @require_torch_sdpa
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, *args):
        # adding only flaky decorator here and call the parent test method
        return getattr(ModelTesterMixin, self._testMethodName)(self)

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()

    @require_torch_sdpa
    def test_sdpa_can_dispatch_on_flash(self):
        self.skipTest(reason="CLIP text tower has two attention masks: `causal_attention_mask` and `attention_mask`")

    @require_torch_sdpa
    def test_sdpa_can_compile_dynamic(self):
        self.skipTest(reason="CLIP model can't be compiled dynamic, error in clip_loss`")

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

                model = model_class.from_pretrained(
                    tmpdirname, dtype=torch.bfloat16, attn_implementation="eager"
                )
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


class CLIPForImageClassificationModelTester(CLIPModelTester):
    def __init__(self, parent):
        super().__init__(parent)
        self.batch_size = self.vision_model_tester.batch_size
        self.num_hidden_layers = self.vision_model_tester.num_hidden_layers
        self.hidden_size = self.vision_model_tester.hidden_size
        self.seq_length = self.vision_model_tester.seq_length

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class CLIPForImageClassificationModelTest(CLIPModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (CLIPForImageClassification,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-classification": CLIPForImageClassification} if is_torch_available() else {}
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    _is_composite = True

    def setUp(self):
        self.model_tester = CLIPForImageClassificationModelTester(self)

    @unittest.skip(reason="CLIPForImageClassification does not support inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="CLIPForImageClassification does not support inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="CLIPForImageClassification does not support gradient checkpointing yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="CLIPForImageClassification does not support gradient checkpointing yet")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="CLIPForImageClassification does not support gradient checkpointing yet")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="CLIP uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @require_torch_sdpa
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, *args):
        # adding only flaky decorator here and call the parent test method
        return getattr(ModelTesterMixin, self._testMethodName)(self)

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
@require_torch
class CLIPModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name, attn_implementation="sdpa").to(torch_device)
        processor = CLIPProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"], images=image, padding=True, return_tensors="pt"
        ).to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        self.assertEqual(
            outputs.logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        expected_logits = torch.tensor([[24.5701, 19.3049]], device=torch_device)

        torch.testing.assert_close(outputs.logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_interpolate_pos_encoding(self):
        # CLIP models have an `interpolate_pos_encoding` argument in their forward method,
        # allowing to interpolate the pre-trained position embeddings in order to use
        # the model on higher resolutions. The DINO model by Facebook AI leverages this
        # to visualize self-attention on higher resolution images.
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(torch_device)

        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", size={"height": 180, "width": 180}, crop_size={"height": 180, "width": 180}
        )

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = processor(text="what's in the image", images=image, return_tensors="pt").to(torch_device)

        # interpolate_pos_encodiung false should return value error
        with self.assertRaises(ValueError, msg="doesn't match model"):
            with torch.no_grad():
                model(**inputs, interpolate_pos_encoding=False)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)

        # verify the logits
        expected_shape = torch.Size((1, 26, 768))

        self.assertEqual(outputs.vision_model_output.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-0.1538, 0.0322, -0.3235], [0.2893, 0.1135, -0.5708], [0.0461, 0.1540, -0.6018]]
        ).to(torch_device)

        torch.testing.assert_close(
            outputs.vision_model_output.last_hidden_state[0, :3, :3], expected_slice, rtol=6e-3, atol=4e-4
        )
