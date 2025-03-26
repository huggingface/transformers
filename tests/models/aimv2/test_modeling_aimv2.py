# coding=utf-8
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
"""Testing suite for the PyTorch AIMv2 model."""

import inspect
import os
import tempfile
import unittest

import numpy as np
from parameterized import parameterized
from pytest import mark

from transformers import AIMv2Config, AIMv2TextConfig, AIMv2VisionConfig
from transformers.testing_utils import (
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)
from transformers.utils import (
    is_torch_available,
    is_vision_available,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
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
        AIMv2Model,
        AIMv2TextModel,
        AIMv2VisionModel,
    )


if is_vision_available():
    pass


class AIMv2VisionModelTester:
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

        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return AIMv2VisionConfig(
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
        model = AIMv2VisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class AIMv2ModelTesterMixin(ModelTesterMixin):
    """
    Subclass of ModelTesterMixin with methods specific to testing AIMv2 models.
    The SDPA equivalence test is overridden here because AIMv2 models may have test/vision/text+vision inputs,
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
                model_sdpa = model_sdpa.eval().to(torch_device)

                # Load model with eager attention
                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

            # SigLip has one shared cls attr for all models, so we assign both submodels heer
            vision_attn = text_attn = "sdpa" if model._supports_sdpa else "eager"

            # `None` as it is the requested one which will be assigned to each sub-config
            # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
            if hasattr(model_sdpa, "vision_model") and hasattr(model_sdpa, "text_model"):
                self.assertTrue(model_sdpa.vision_model.config._attn_implementation == vision_attn)
                self.assertTrue(model_sdpa.text_model.config._attn_implementation == text_attn)
                self.assertTrue(model_eager.vision_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.text_model.config._attn_implementation == "eager")

            self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
            self.assertTrue(model_eager.config._attn_implementation == "eager")

            for name, submodule in model_eager.named_modules():
                class_name = submodule.__class__.__name__
                if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                    raise ValueError("The eager model should not have SDPA attention layers")

            has_sdpa = False
            for name, submodule in model_sdpa.named_modules():
                class_name = submodule.__class__.__name__
                if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                    has_sdpa = True
                    break
            if not has_sdpa and model_sdpa.config.model_type != "falcon":
                raise ValueError("The SDPA model should have SDPA attention layers")


@require_torch
class AIMv2VisionModelTest(AIMv2ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as AIMv2 does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (AIMv2VisionModel,) if is_torch_available() else ()
    fx_compatible = True
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = AIMv2VisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=AIMv2VisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="AIMv2 does not use inputs_embeds")
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

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        super().test_eager_matches_sdpa_inference(
            torch_dtype=torch_dtype,
            logit_keys=("last_hidden_state", "pooler_output", "image_embeds"),
            use_attention_mask_options=(None,),
        )

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()


class AIMv2TextModelTester:
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
        return AIMv2TextConfig(
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
        model = AIMv2TextModel(config=config)
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
class AIMv2TextModelTest(AIMv2ModelTesterMixin, unittest.TestCase):
    all_model_classes = (AIMv2TextModel,) if is_torch_available() else ()
    fx_compatible = True
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = AIMv2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AIMv2TextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

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

    # @unittest.skip(reason="AIMv2 does not use inputs_embeds")
    # def test_inputs_embeds(self):
    #     pass

    # @unittest.skip(reason="AIMv2TextModel has no base class and is not available in MODEL_MAPPING")
    # def test_save_load_fast_init_from_base(self):
    #     pass

    # @unittest.skip(reason="AIMv2TextModel has no base class and is not available in MODEL_MAPPING")
    # def test_save_load_fast_init_to_base(self):
    #     pass

    # @slow
    # def test_model_from_pretrained(self):
    #     model_name = "openai/AIMv2-vit-base-patch32"
    #     model = AIMv2TextModel.from_pretrained(model_name)
    #     self.assertIsNotNone(model)

    # @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    # @require_torch_sdpa
    # @slow
    # @is_flaky()
    # def test_eager_matches_sdpa_inference(self, torch_dtype: str):
    #     super().test_eager_matches_sdpa_inference(
    #         torch_dtype=torch_dtype,
    #         logit_keys=("last_hidden_state", "pooler_output", "text_embeds"),
    #         use_attention_mask_options=(None, "right"),  # "left" is not supported for text model
    #     )

    # @require_torch_sdpa
    # def test_sdpa_can_dispatch_composite_models(self):
    #     super().test_sdpa_can_dispatch_composite_models()

    # @require_torch_sdpa
    # def test_sdpa_can_dispatch_on_flash(self):
    #     self.skipTest(reason="AIMv2TextModel has two attention masks: `causal_attention_mask` and `attention_mask`")


class AIMv2ModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = AIMv2TextModelTester(parent, **text_kwargs)
        self.vision_model_tester = AIMv2VisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return AIMv2Config.from_text_vision_configs(
            self.text_model_tester.get_config(), self.vision_model_tester.get_config(), projection_dim=64
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = AIMv2Model(config).to(torch_device).eval()
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
class AIMv2ModelTest(AIMv2ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (AIMv2Model,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": AIMv2Model, "image-feature-extraction": AIMv2VisionModel}
        if is_torch_available()
        else {}
    )
    fx_compatible = True
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    _is_composite = True

    def setUp(self):
        self.model_tester = AIMv2ModelTester(self)
        common_properties = ["projection_dim", "logit_scale_init_value"]
        self.config_tester = ConfigTester(
            self, config_class=AIMv2Config, has_text_modality=False, common_properties=common_properties
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

    @unittest.skip(reason="AIMv2Model does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    # override as the `logit_scale` parameter initialization is different for AIMv2
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
                pixel_values = inputs_dict["pixel_values"]  # AIMv2 needs pixel_values
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
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
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

        # Save AIMv2Config and check if we can load AIMv2VisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = AIMv2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save AIMv2Config and check if we can load AIMv2TextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = AIMv2TextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        super().test_eager_matches_sdpa_inference(
            torch_dtype=torch_dtype,
            logit_keys=("logits_per_image", "logits_per_text"),
            use_attention_mask_options=(None, "right"),  # "left" is not supported for text model
        )

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
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
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="eager"
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
