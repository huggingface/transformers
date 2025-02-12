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
"""Testing suite for the PyTorch Siglip2 model."""

import inspect
import tempfile
import unittest
from typing import Tuple

import numpy as np
from parameterized import parameterized
from pytest import mark

from transformers import Siglip2Config, Siglip2TextConfig, Siglip2VisionConfig
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
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
    is_torch_sdpa_available,
    is_vision_available,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    is_flaky,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import Siglip2ForImageClassification, Siglip2Model, Siglip2TextModel, Siglip2VisionModel

if is_torch_sdpa_available():
    from torch.nn.attention import SDPBackend, sdpa_kernel

if is_vision_available():
    from PIL import Image, ImageDraw

    from transformers import Siglip2Processor


class Siglip2ModelTesterMixin(ModelTesterMixin):
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

    def test_eager_matches_sdpa_inference(
        self,
        torch_dtype: str,
        use_attention_mask_options: Tuple[bool, ...] = (True, False),
        logit_keys: Tuple[str, ...] = ("logits_per_image", "logits_per_text", "image_embeds", "text_embeds"),
    ):
        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        if torch_dtype == "float16" and not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        if torch_dtype == "bfloat16" and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(
                f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
            )

        # Convert to torch dtype
        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtypes[torch_dtype]

        atols = {
            torch.float32: 1e-5,
            torch.bfloat16: 3e-2,
            torch.float16: 5e-3,
        }
        rtols = {
            torch.float32: 1e-4,
            torch.bfloat16: 3e-2,
            torch.float16: 5e-3,
        }

        atol = atols[torch_dtype]
        rtol = rtols[torch_dtype]

        def get_mean_reldiff(msg, current_case, x, ref, atol, rtol):
            return f"{msg} {current_case}: mean relative difference: {((x - ref).abs() / (ref.abs() + 1e-12)).mean():.3e}, torch atol = {atol}, torch rtol = {rtol}"

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Load the model with SDPA
                model_sdpa = model_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype)
                model_sdpa = model_sdpa.eval().to(torch_device)

                # Load model with eager attention
                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

            # We use these for loops instead of parameterized.expand just for the interest of avoiding loading/saving the model each time,
            # but it would be nicer to have an efficient way to use parameterized.expand
            cases = [
                (use_mask, output_attentions, sdpa_backend, batch_size)
                for use_mask in use_attention_mask_options
                for output_attentions in [True, False]
                for sdpa_backend in [
                    SDPBackend.MATH,
                    [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH],
                    [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
                    [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
                ]
                for batch_size in [1, 5]
            ]
            fail_cases = []

            for use_mask, output_attentions, sdpa_backend, batch_size in cases:
                processed_inputs = inputs_dict.copy()

                # convert to torch_dtype
                if "pixel_values" in processed_inputs:
                    processed_inputs["pixel_values"] = processed_inputs["pixel_values"].to(torch_dtype)

                # slice for different batch sizes
                for key in processed_inputs.keys():
                    if isinstance(processed_inputs[key], (torch.Tensor, list, tuple)):
                        processed_inputs[key] = processed_inputs[key][:batch_size]

                # set attention mask with left padding
                if not use_mask:
                    processed_inputs.pop("attention_mask", None)
                else:
                    dummy_attention_mask = processed_inputs["attention_mask"]
                    dummy_attention_mask[:] = 1
                    dummy_attention_mask[:, :1] = 0
                    processed_inputs["attention_mask"] = dummy_attention_mask

                processed_inputs["output_attentions"] = output_attentions
                processed_inputs["output_hidden_states"] = True

                current_case = (
                    f"padding_side=left, use_mask={use_mask}, batch_size={batch_size}, sdpa_backend={sdpa_backend}"
                )

                prepared_inputs = self._prepare_for_class(processed_inputs, model_class)

                with torch.no_grad():
                    try:
                        with sdpa_kernel(sdpa_backend):
                            outputs_eager = model_eager(**prepared_inputs)
                            outputs_sdpa = model_sdpa(**prepared_inputs)
                    except Exception as e:
                        fail_cases.append(f"{current_case}: {e}")
                        continue

                for key in logit_keys:
                    eager_logits = outputs_eager[key]
                    sdpa_logits = outputs_sdpa[key]

                    if use_mask:
                        eager_logits = eager_logits[:, 1:]
                        sdpa_logits = sdpa_logits[:, 1:]

                    is_close = torch.allclose(eager_logits, sdpa_logits, atol=atol, rtol=rtol)
                    if not is_close:
                        fail_cases.append(get_mean_reldiff(key, current_case, sdpa_logits, eager_logits, atol, rtol))

            self.assertTrue(len(fail_cases) == 0, "\n".join(fail_cases))

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence(self):
        dtype = torch.bfloat16

        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            # Prepare inputs
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            if "pixel_values" in inputs_dict:
                inputs_dict["pixel_values"] = inputs_dict["pixel_values"].to(dtype)

            # Separate masks
            attention_masks = {}
            if "attention_mask" in inputs_dict:
                attention_masks["attention_mask"] = inputs_dict.pop("attention_mask")
                inputs_dict["attention_mask"] = None
            if "pixel_attention_mask" in inputs_dict:
                attention_masks["pixel_attention_mask"] = inputs_dict.pop("pixel_attention_mask")
                inputs_dict["pixel_attention_mask"] = None

            # Save and load model with flash attention 2 and eager attentions
            with tempfile.TemporaryDirectory() as tmp_dir:
                model = model_class(config)
                model.save_pretrained(tmp_dir)

                model_fa = model_class.from_pretrained(
                    tmp_dir, torch_dtype=dtype, attn_implementation="flash_attention_2"
                )
                model = model_class.from_pretrained(tmp_dir, torch_dtype=dtype, attn_implementation="sdpa")

            model_fa.to(torch_device)
            model.to(torch_device)

            # Run forward pass without attention masks
            with torch.no_grad():
                outputs = model(**inputs_dict, output_hidden_states=True)
                outputs_fa = model_fa(**inputs_dict, output_hidden_states=True)

            # Choose which key to compare
            key = [k for k in ["last_hidden_state", "logits", "logits_per_image"] if k in outputs][0]

            torch.testing.assert_close(outputs[key], outputs_fa[key], atol=4e-2, rtol=4e-2)

            # Run forward pass with attention masks
            inputs_dict.update(attention_masks)
            with torch.no_grad():
                outputs = model(**inputs_dict, output_hidden_states=True)
                outputs_fa = model_fa(**inputs_dict, output_hidden_states=True)

            torch.testing.assert_close(outputs[key], outputs_fa[key], atol=4e-2, rtol=4e-2)

            # Check with inference + dropout
            model.train()
            _ = model_fa(**inputs_dict, output_hidden_states=True)

    @unittest.skip(reason="Siglip2 has default right padding (tested in test_flash_attn_2_inference_equivalence)")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass
    
    @unittest.skip(reason="SDPA can't dispatch on flash with not None `attention_mask`")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

class Siglip2VisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        num_patches=16,
        image_num_patches=24,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
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
            [self.batch_size, self.seq_length, self.num_channels * self.patch_size * self.patch_size]
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
        return Siglip2VisionConfig(
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
        model = Siglip2VisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values, pixel_attention_mask, spatial_shapes)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
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
class Siglip2VisionModelTest(Siglip2ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SIGLIP2 does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (Siglip2VisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    # MP works but offload doesn't work when the MultiheadAttention is offloaded
    # TODO: One potential solution would be to add to set preload_module_classes = ["Siglip2MultiheadAttentionPoolingHead"]
    # in the dispatch_model function
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def setUp(self):
        self.model_tester = Siglip2VisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Siglip2VisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SIGLIP2 does not use inputs_embeds")
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

    @unittest.skip(reason="Siglip2VisionModel does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="Siglip2VisionModel does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Siglip2VisionModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="Siglip2VisionModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Siglip2VisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Siglip2VisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="Siglip2 uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "s0225/siglip2-base-patch-16-naflex-256"
        model = Siglip2VisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        super().test_eager_matches_sdpa_inference(
            torch_dtype=torch_dtype,
            logit_keys=("pooler_output", "last_hidden_state"),
            use_attention_mask_options=(False,),
        )

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()


class Siglip2TextModelTester:
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
        return Siglip2TextConfig(
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
        model = Siglip2TextModel(config=config)
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
class Siglip2TextModelTest(Siglip2ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Siglip2TextModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = Siglip2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Siglip2TextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Siglip2TextModel does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="Siglip2TextModel does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Siglip2TextModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="Siglip2TextModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Siglip2 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Siglip2TextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Siglip2TextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="Siglip2 uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "s0225/siglip2-base-patch-16-naflex-256"
        model = Siglip2TextModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        super().test_eager_matches_sdpa_inference(
            torch_dtype=torch_dtype,
            logit_keys=("pooler_output", "last_hidden_state"),
            use_attention_mask_options=(False, True),
        )

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()


class Siglip2ModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = Siglip2TextModelTester(parent, **text_kwargs)
        self.vision_model_tester = Siglip2VisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values, pixel_attention_mask, spatial_shapes = (
            self.vision_model_tester.prepare_config_and_inputs()
        )

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values, pixel_attention_mask, spatial_shapes

    def get_config(self):
        return Siglip2Config.from_text_vision_configs(
            self.text_model_tester.get_config(),
            self.vision_model_tester.get_config(),
        )

    def create_and_check_model(
        self, config, input_ids, attention_mask, pixel_values, pixel_attention_mask, spatial_shapes
    ):
        model = Siglip2Model(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, pixel_values, pixel_attention_mask, spatial_shapes, attention_mask)
        self.parent.assertEqual(
            result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values, pixel_attention_mask, spatial_shapes = config_and_inputs
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
class Siglip2ModelTest(Siglip2ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Siglip2Model,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": Siglip2Model} if is_torch_available() else {}
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    # MP works but offload doesn't work when the MultiheadAttention is offloaded
    # TODO: One potential solution would be to add to set preload_module_classes = ["Siglip2MultiheadAttentionPoolingHead"]
    # in the dispatch_model function
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Siglip2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Siglip2Config, has_text_modality=False)

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

    @unittest.skip(reason="Siglip2Model does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Siglip2 uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    def test_load_vision_text_config(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Save Siglip2Config and check if we can load Siglip2VisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Siglip2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save Siglip2Config and check if we can load Siglip2TextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = Siglip2TextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "s0225/siglip2-base-patch-16-naflex-256"
        model = Siglip2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest("Siglip2 does not support right padding")

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        super().test_eager_matches_sdpa_inference(
            torch_dtype=torch_dtype,
            logit_keys=("logits_per_image", "logits_per_text", "image_embeds", "text_embeds"),
            use_attention_mask_options=(False, True),
        )

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()


class Siglip2ForImageClassificationModelTester(Siglip2ModelTester):
    def __init__(self, parent):
        super().__init__(parent)
        self.batch_size = self.vision_model_tester.batch_size
        self.num_hidden_layers = self.vision_model_tester.num_hidden_layers
        self.hidden_size = self.vision_model_tester.hidden_size
        self.seq_length = self.vision_model_tester.seq_length

    def prepare_config_and_inputs(self):
        _, pixel_values, pixel_attention_mask, spatial_shapes = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()

        return config, pixel_values, pixel_attention_mask, spatial_shapes

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, pixel_attention_mask, spatial_shapes = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
        }
        return config, inputs_dict


@require_torch
class Siglip2ForImageClassificationModelTest(Siglip2ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Siglip2ForImageClassification,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-classification": Siglip2ForImageClassification} if is_torch_available() else {}
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    # MP works but offload doesn't work when the MultiheadAttention is offloaded
    # TODO: One potential solution would be to add to set preload_module_classes = ["Siglip2MultiheadAttentionPoolingHead"]
    # in the dispatch_model function
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Siglip2ForImageClassificationModelTester(self)

    @unittest.skip(reason="Siglip2ForImageClassification does not support inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Siglip2ForImageClassification does not support inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Siglip2ForImageClassification does not support gradient checkpointing yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Siglip2ForImageClassification does not support gradient checkpointing yet")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="Siglip2ForImageClassification does not support gradient checkpointing yet")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Siglip2 uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        super().test_eager_matches_sdpa_inference(
            torch_dtype=torch_dtype, logit_keys=("logits",), use_attention_mask_options=(False,)
        )

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()


# Draw a circle on an images with different aspect ratios
def prepare_images():
    shapes = [(224, 224), (1024, 1024), (224, 1024)]
    images = []
    for height, width in shapes:
        image = Image.new("RGB", (width, height), color="red")
        draw = ImageDraw.Draw(image)
        center_x = image.width // 2
        center_y = image.height // 2
        radius = min(center_x, center_y) // 8 * 7
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill="blue",
            outline="green",
            width=image.width // 20,
        )
        images.append(image)
    return images


@require_vision
@require_torch
class Siglip2ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "s0225/siglip2-base-patch-16-naflex-256"
        model = Siglip2Model.from_pretrained(model_name).to(torch_device)
        processor = Siglip2Processor.from_pretrained(model_name)

        images = prepare_images()
        text = [
            "circle",
            "ellipsoid",
            "blue circle on red background",
            "blue circle with green border on red background",
            "green circle on red background",
            "a dog",
            "a blue dog with a green border on a red background",
        ]

        inputs = processor(text=text, images=images, return_tensors="pt")
        inputs = inputs.to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        # verify the logits shape
        self.assertEqual(
            logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        # verify the logits values
        # fmt: off
        expected_logits_per_text = torch.tensor(
            [
                [  1.0195,  -0.0280,  -1.4468],
                [ -4.5395,  -6.2269,  -1.5667],
                [  4.1757,   5.0358,   3.5159],
                [  9.4264,  10.1879,   6.3353],
                [  2.4409,   3.1058,   4.5491],
                [-12.3230, -13.7355, -13.4632],
                [  1.1520,   1.1687,  -1.9647],
            ]
        ).to(torch_device)
        # fmt: on

        torch.testing.assert_close(outputs.logits_per_text, expected_logits_per_text, rtol=1e-3, atol=1e-3)
