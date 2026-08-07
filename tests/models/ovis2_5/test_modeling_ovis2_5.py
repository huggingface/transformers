# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Ovis2.5 model."""

import copy
import tempfile
import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_bf16_available_on_device, is_torch_fp16_available_on_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    sdpa_kernel,
)
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Ovis2_5Config,
        Ovis2_5ForConditionalGeneration,
        Ovis2_5Model,
        Ovis2_5VisionConfig,
        Ovis2_5VisionModel,
    )
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


if is_torch_available():

    class Ovis2_5VisionModelTester:
        def __init__(self, parent):
            self.parent = parent
            self.batch_size = 2
            self.image_size = 4
            self.patch_size = 2
            self.num_channels = 3
            self.hidden_size = 16
            self.intermediate_size = 32
            self.num_hidden_layers = 1
            self.expected_num_hidden_layers = self.num_hidden_layers
            self.num_attention_heads = 4
            self.seq_length = (self.image_size // self.patch_size) ** 2
            self.is_training = True

        def get_config(self):
            return Ovis2_5VisionConfig(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                num_channels=self.num_channels,
                image_size=self.image_size,
                patch_size=self.patch_size,
                hidden_stride=2,
                window_size=self.image_size,
                attention_dropout=0.0,
                vocab_size=12,
                num_visual_indicator_tokens=4,
                preserve_original_pe=True,
                fullatt_block_indexes=None,
            )

        def prepare_config_and_inputs_for_common(self):
            config = self.get_config()
            pixel_values = torch.randn(
                self.batch_size * self.seq_length,
                self.num_channels * self.patch_size**2,
                device=torch_device,
            )
            grid_size = self.image_size // self.patch_size
            grid_thw = torch.tensor(
                [[1, grid_size, grid_size]] * self.batch_size,
                dtype=torch.long,
                device=torch_device,
            )
            return config, {"pixel_values": pixel_values, "grid_thw": grid_thw}

    class Ovis2_5VisionText2TextModelTester(VLMModelTester):
        base_model_class = Ovis2_5Model
        config_class = Ovis2_5Config
        text_config_class = Qwen3Config
        vision_config_class = Ovis2_5VisionConfig
        conditional_generation_class = Ovis2_5ForConditionalGeneration

        def __init__(self, parent, **kwargs):
            kwargs.setdefault("batch_size", 2)
            kwargs.setdefault("seq_length", 8)
            kwargs.setdefault("vocab_size", 32)
            kwargs.setdefault("hidden_size", 16)
            kwargs.setdefault("intermediate_size", 32)
            kwargs.setdefault("num_hidden_layers", 1)
            kwargs.setdefault("num_attention_heads", 4)
            kwargs.setdefault("num_key_value_heads", 2)
            kwargs.setdefault("head_dim", 4)
            kwargs.setdefault("max_position_embeddings", 32)
            kwargs.setdefault("attention_dropout", 0.0)
            kwargs.setdefault("hidden_act", "silu")
            kwargs.setdefault("bos_token_id", 1)
            kwargs.setdefault("eos_token_id", 2)
            kwargs.setdefault("pad_token_id", 0)
            kwargs.setdefault("tie_word_embeddings", False)
            kwargs.setdefault("num_channels", 3)
            kwargs.setdefault("image_size", 4)
            kwargs.setdefault("patch_size", 2)
            kwargs.setdefault("num_image_tokens", 1)
            kwargs.setdefault("image_token_id", 4)
            kwargs.setdefault("visual_atom_token_id", 4)
            kwargs.setdefault("image_start_token_id", 5)
            kwargs.setdefault("image_end_token_id", 6)
            kwargs.setdefault("video_start_token_id", 7)
            kwargs.setdefault("video_end_token_id", 8)
            kwargs.setdefault("visual_vocab_size", 12)
            super().__init__(parent, **kwargs)
            self.num_image_patches = (self.image_size // self.patch_size) ** 2

        @property
        def _special_token_ids(self):
            return super()._special_token_ids | {
                self.visual_atom_token_id,
                self.image_start_token_id,
                self.image_end_token_id,
                self.video_start_token_id,
                self.video_end_token_id,
            }

        def get_text_config(self):
            return Qwen3Config(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                attention_dropout=self.attention_dropout,
                hidden_act=self.hidden_act,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                tie_word_embeddings=self.tie_word_embeddings,
            )

        def get_vision_config(self):
            return Ovis2_5VisionConfig(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                num_channels=self.num_channels,
                image_size=self.image_size,
                patch_size=self.patch_size,
                hidden_stride=2,
                window_size=self.image_size,
                attention_dropout=self.attention_dropout,
                vocab_size=self.visual_vocab_size,
                num_visual_indicator_tokens=4,
                preserve_original_pe=True,
                use_rope=True,
                fullatt_block_indexes=None,
            )

        def get_config(self):
            return Ovis2_5Config(
                text_config=self.get_text_config(),
                vision_config=self.get_vision_config(),
                visual_vocab_size=self.visual_vocab_size,
                visual_atom_token_id=self.visual_atom_token_id,
                image_start_token_id=self.image_start_token_id,
                image_end_token_id=self.image_end_token_id,
                video_start_token_id=self.video_start_token_id,
                video_end_token_id=self.video_end_token_id,
            )

        def create_attention_mask(self, input_ids):
            return torch.ones_like(input_ids, device=torch_device)

        def create_pixel_values(self):
            return torch.randn(
                self.batch_size * self.num_image_patches,
                self.num_channels * self.patch_size**2,
                device=torch_device,
            )

        def place_image_tokens(self, input_ids, config):
            input_ids = input_ids.clone()
            for token_id in self._special_token_ids:
                input_ids[input_ids == token_id] = self._safe_token_id()
            input_ids[:, 0] = config.image_start_token_id
            input_ids[:, 1] = config.visual_atom_token_id
            input_ids[:, 2] = config.image_end_token_id
            return input_ids

        def get_additional_inputs(self, config, input_ids, modality_inputs):
            grid_size = self.image_size // self.patch_size
            return {
                "image_grid_thw": torch.tensor(
                    [[1, grid_size, grid_size]] * self.batch_size,
                    dtype=torch.long,
                    device=torch_device,
                )
            }

        def prepare_video_inputs(self):
            num_frames = 2
            input_ids = torch.tensor(
                [
                    [
                        self.bos_token_id,
                        self.video_start_token_id,
                        self.visual_atom_token_id,
                        self.visual_atom_token_id,
                        self.video_end_token_id,
                        self._safe_token_id(),
                        self.eos_token_id,
                    ]
                ]
                * self.batch_size,
                dtype=torch.long,
                device=torch_device,
            )
            grid_size = self.image_size // self.patch_size
            patches_per_video = num_frames * grid_size**2
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "pixel_values_videos": torch.randn(
                    self.batch_size * patches_per_video,
                    self.num_channels * self.patch_size**2,
                    device=torch_device,
                ),
                "video_grid_thw": torch.tensor(
                    [[num_frames, grid_size, grid_size]] * self.batch_size,
                    dtype=torch.long,
                    device=torch_device,
                ),
            }


@require_torch
class Ovis2_5VisionModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Ovis2_5VisionModel,) if is_torch_available() else ()
    additional_model_inputs = ["grid_thw"]
    test_resize_embeddings = False
    # Packed grid metadata uses data-dependent Python control flow.
    test_torch_exportable = False

    def setUp(self):
        self.model_tester = Ovis2_5VisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Ovis2_5VisionConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = Ovis2_5VisionModel(config)
        self.assertIsInstance(model.get_input_embeddings(), torch.nn.Module)
        self.assertIsNone(model.get_output_embeddings())

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
    ):
        if use_attention_mask:
            self.skipTest("Ovis2.5's packed vision tower does not use an attention mask.")

        dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
        if dtype == torch.float16 and not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 is not supported on {torch_device}")
        if dtype == torch.bfloat16 and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(f"bfloat16 is not supported on {torch_device}")

        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        patches_per_item = int(inputs["grid_thw"][0].prod())
        inputs["pixel_values"] = inputs["pixel_values"][:patches_per_item].repeat(7, 1).to(dtype)
        inputs["grid_thw"] = inputs["grid_thw"][:1].repeat(7, 1)
        inputs["output_hidden_states"] = True
        inputs["output_attentions"] = output_attentions

        torch.manual_seed(42)
        model_eager = Ovis2_5VisionModel._from_config(copy.deepcopy(config), attn_implementation="eager")
        model_sdpa = Ovis2_5VisionModel._from_config(copy.deepcopy(config), attn_implementation="sdpa")
        model_sdpa.load_state_dict(model_eager.state_dict())
        model_eager = model_eager.to(torch_device, dtype=dtype).eval()
        model_sdpa = model_sdpa.to(torch_device, dtype=dtype).eval()
        self.assertEqual(model_eager.config._attn_implementation, "eager")
        self.assertEqual(model_sdpa.config._attn_implementation, "sdpa")

        with (
            torch.no_grad(),
            sdpa_kernel(
                enable_flash=enable_kernels,
                enable_math=True,
                enable_mem_efficient=enable_kernels,
            ),
        ):
            outputs_eager = model_eager(**inputs)
            outputs_sdpa = model_sdpa(**inputs)

        atol = {torch.float32: 1e-6, torch.float16: 5e-3, torch.bfloat16: 1e-2}[dtype]
        rtol = {torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 1e-2}[dtype]
        torch.testing.assert_close(
            outputs_eager.last_hidden_state, outputs_sdpa.last_hidden_state, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(outputs_eager.pooler_output, outputs_sdpa.pooler_output, atol=atol, rtol=rtol)

    def flash_attn_inference_equivalence(
        self,
        attn_implementation: str,
        padding_side: str,
        atol: float = 4e-2,
        rtol: float = 4e-2,
    ) -> None:
        """Compare packed vision inputs without separating their patch tensor from `grid_thw`."""
        torch.manual_seed(42)
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config = self._prepare_config_headdim(config, 16)
        model = Ovis2_5VisionModel(config)

        compatible_implementations = model._compatible_flash_implementations
        if compatible_implementations is not None and attn_implementation not in compatible_implementations:
            self.skipTest(f"Ovis2_5VisionModel does not support {attn_implementation}.")

        grid_thw = inputs["grid_thw"][:1]
        num_patches = int(grid_thw.prod())
        packed_inputs = {
            "pixel_values": inputs["pixel_values"][:num_patches].to(torch.bfloat16),
            "grid_thw": grid_thw,
            "output_hidden_states": True,
        }

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            with torch.no_grad():
                model_eager = Ovis2_5VisionModel.from_pretrained(
                    tmpdirname,
                    dtype=torch.bfloat16,
                    attn_implementation="eager",
                    device_map=torch_device,
                )
                self.assertEqual(model_eager.config._attn_implementation, "eager")
                hidden_states_eager = model_eager(**packed_inputs).hidden_states[-1]

                model_flash = Ovis2_5VisionModel.from_pretrained(
                    tmpdirname,
                    dtype=torch.bfloat16,
                    attn_implementation=attn_implementation,
                    device_map=torch_device,
                )
                self.assertEqual(model_flash.config._attn_implementation, attn_implementation)
                hidden_states_flash = model_flash(**packed_inputs).hidden_states[-1]

        torch.testing.assert_close(hidden_states_eager, hidden_states_flash, atol=atol, rtol=rtol)

    def test_attention_outputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        def check_attention_outputs(model, inputs):
            with torch.no_grad():
                outputs = model(**inputs)
            self.assertEqual(len(outputs.attentions), self.model_tester.num_hidden_layers)
            for layer_attentions in outputs.attentions:
                self.assertEqual(len(layer_attentions), inputs["grid_thw"].shape[0])
                for attention, grid in zip(layer_attentions, inputs["grid_thw"]):
                    sequence_length = int(grid.prod())
                    self.assertListEqual(
                        list(attention.shape[-3:]),
                        [self.model_tester.num_attention_heads, sequence_length, sequence_length],
                    )

        model = Ovis2_5VisionModel._from_config(config, attn_implementation="eager").to(torch_device).eval()
        check_attention_outputs(model, {**inputs, "output_attentions": True})

        config.output_attentions = True
        model = Ovis2_5VisionModel._from_config(config, attn_implementation="eager").to(torch_device).eval()
        check_attention_outputs(model, inputs)

    def test_hidden_states_output(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        expected_shape = [int(inputs["grid_thw"].prod(dim=1).sum()), self.model_tester.hidden_size]

        def check_hidden_states(model, inputs):
            with torch.no_grad():
                outputs = model(**inputs)
            self.assertEqual(len(outputs.hidden_states), self.model_tester.expected_num_hidden_layers)
            for hidden_state in outputs.hidden_states:
                self.assertListEqual(list(hidden_state.shape), expected_shape)

        model = Ovis2_5VisionModel(config).to(torch_device).eval()
        check_hidden_states(model, {**inputs, "output_hidden_states": True})

        config.output_hidden_states = True
        model = Ovis2_5VisionModel(config).to(torch_device).eval()
        check_hidden_states(model, inputs)

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = Ovis2_5VisionModel._from_config(config, attn_implementation="eager").to(torch_device)
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        hidden_state = outputs.hidden_states[0]
        attention = outputs.attentions[0][0]
        hidden_state.retain_grad()
        attention.retain_grad()

        outputs.last_hidden_state.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_state.grad)
        self.assertIsNotNone(attention.grad)


@require_torch
class Ovis2_5ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Ovis2_5VisionText2TextModelTester if is_torch_available() else None
    # The visual-tokenizer head consumes the encoder state before post_layernorm.
    test_all_params_have_gradient = False
    # Packed patch counts and grid metadata use data-dependent Python control flow.
    test_torch_exportable = False

    def _image_features_get_expected_num_hidden_states(self, model_tester=None):
        model_tester = self.model_tester if model_tester is None else model_tester
        return model_tester.num_hidden_layers

    def _video_features_get_expected_num_hidden_states(self, model_tester=None):
        model_tester = self.model_tester if model_tester is None else model_tester
        return model_tester.num_hidden_layers

    def test_reverse_loading_mapping(self):
        # The official-key mapping targets the conditional model's `model.*` subtree.
        super().test_reverse_loading_mapping(skip_base_model=True)

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        filtered_inputs = {}
        for key, value in inputs_dict.items():
            if key == "pixel_values":
                filtered_inputs[key] = value[: batch_size * self.model_tester.num_image_patches]
            elif key == "image_grid_thw":
                filtered_inputs[key] = value[:batch_size]
            elif isinstance(value, torch.Tensor):
                filtered_inputs[key] = value[:batch_size]
            else:
                filtered_inputs[key] = value

        text_config = config.get_text_config(decoder=True)
        text_config.eos_token_id = None
        text_config.forced_eos_token_id = None
        return config, filtered_inputs

    def test_mismatching_num_image_tokens(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        patches_per_image = self.model_tester.num_image_patches

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            model(**inputs)

            one_image_inputs = {
                "input_ids": inputs["input_ids"][:1],
                "attention_mask": inputs["attention_mask"][:1],
                "pixel_values": inputs["pixel_values"][:patches_per_image],
                "image_grid_thw": inputs["image_grid_thw"][:1],
            }
            two_prompt_inputs = {
                "input_ids": one_image_inputs["input_ids"].repeat(2, 1),
                "attention_mask": one_image_inputs["attention_mask"].repeat(2, 1),
                "pixel_values": one_image_inputs["pixel_values"],
                "image_grid_thw": one_image_inputs["image_grid_thw"],
            }
            with self.assertRaises(ValueError):
                model(**two_prompt_inputs)

            two_prompt_inputs["pixel_values"] = one_image_inputs["pixel_values"].repeat(2, 1)
            two_prompt_inputs["image_grid_thw"] = one_image_inputs["image_grid_thw"].repeat(2, 1)
            model(**two_prompt_inputs)

    def test_legacy_subconfig_names(self):
        text_config = self.model_tester.get_text_config().to_dict()
        vision_config = self.model_tester.get_vision_config().to_dict()
        config = Ovis2_5Config(
            llm_config=text_config,
            vit_config=vision_config,
            visual_vocab_size=self.model_tester.visual_vocab_size,
            visual_atom_token_id=self.model_tester.visual_atom_token_id,
            image_start_token_id=self.model_tester.image_start_token_id,
            image_end_token_id=self.model_tester.image_end_token_id,
            video_start_token_id=self.model_tester.video_start_token_id,
            video_end_token_id=self.model_tester.video_end_token_id,
        )

        self.assertIs(config.llm_config, config.text_config)
        self.assertIs(config.vit_config, config.vision_config)
        self.assertIsInstance(config.text_config, Qwen3Config)
        self.assertIsInstance(config.vision_config, Ovis2_5VisionConfig)
        serialized_config = config.to_dict()
        self.assertIn("text_config", serialized_config)
        self.assertIn("vision_config", serialized_config)
        self.assertNotIn("llm_config", serialized_config)
        self.assertNotIn("vit_config", serialized_config)

    def test_visual_tokenizer_distribution(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = Ovis2_5ForConditionalGeneration(config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model.model.vision_tower(
                pixel_values=inputs["pixel_values"],
                grid_thw=inputs["image_grid_thw"],
            )

        num_indicators = config.vision_config.num_visual_indicator_tokens
        torch.testing.assert_close(
            outputs.pooler_output[:, -num_indicators:],
            torch.zeros_like(outputs.pooler_output[:, -num_indicators:]),
        )
        torch.testing.assert_close(
            outputs.pooler_output[:, :-num_indicators].sum(dim=-1),
            torch.ones(outputs.pooler_output.shape[0], device=torch_device),
        )

    def test_visual_feature_helpers_split_per_input(self):
        config = self.model_tester.get_config()
        model = Ovis2_5ForConditionalGeneration(config).to(torch_device).eval()
        grid_thw = torch.tensor([[1, 2, 2], [1, 4, 2]], dtype=torch.long, device=torch_device)
        pixel_values = torch.randn(
            int(grid_thw.prod(dim=1).sum()),
            self.model_tester.num_channels * self.model_tester.patch_size**2,
            device=torch_device,
        )
        expected_shapes = [(1, self.model_tester.hidden_size), (2, self.model_tester.hidden_size)]

        feature_calls = (
            (model.get_image_features, {"pixel_values": pixel_values, "image_grid_thw": grid_thw}),
            (
                model.get_video_features,
                {"pixel_values_videos": pixel_values, "video_grid_thw": grid_thw},
            ),
        )
        for get_features, kwargs in feature_calls:
            with self.subTest(get_features=get_features.__name__):
                with torch.no_grad():
                    outputs = get_features(**kwargs)

                self.assertIsInstance(outputs.pooler_output, tuple)
                self.assertEqual([tuple(features.shape) for features in outputs.pooler_output], expected_shapes)

    def test_precomputed_vision_kwargs(self):
        config, image_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = Ovis2_5ForConditionalGeneration(config).to(torch_device).eval()
        video_inputs = self.model_tester.prepare_video_inputs()

        cases = (
            (image_inputs, "image_grid_thw", "image_position_ids"),
            (video_inputs, "video_grid_thw", "video_position_ids"),
        )
        for inputs, grid_key, position_ids_key in cases:
            position_ids = torch.full(
                (int(inputs[grid_key].prod(dim=1).sum()), 2),
                7,
                dtype=torch.long,
                device=torch_device,
            )
            captured_position_ids = []

            def capture_position_ids(module, args):
                captured_position_ids.append(args[0])

            with self.subTest(position_ids_key=position_ids_key):
                hook = model.model.vision_tower.transformer.rotary_pos_emb.register_forward_pre_hook(
                    capture_position_ids
                )
                try:
                    with torch.no_grad():
                        model(**inputs, **{position_ids_key: position_ids})
                finally:
                    hook.remove()

                self.assertEqual(len(captured_position_ids), 1)
                torch.testing.assert_close(captured_position_ids[0], position_ids)

    def test_video_forward(self):
        config = self.model_tester.get_config()
        model = Ovis2_5ForConditionalGeneration(config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(**self.model_tester.prepare_video_inputs())

        self.assertIsNotNone(outputs.video_hidden_states)
        self.assertIsNone(outputs.image_hidden_states)

    def test_vision_hidden_states_preserve_patch_order(self):
        config = self.model_tester.get_vision_config()
        config.window_size = 8
        model = Ovis2_5VisionModel(config).to(torch_device).eval()
        grid_thw = torch.tensor([[1, 4, 8]], dtype=torch.long, device=torch_device)
        pixel_values = torch.randn(
            int(grid_thw.prod()),
            config.num_channels * config.patch_size**2,
            device=torch_device,
        )

        with torch.no_grad():
            baseline = model(
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                output_hidden_states=False,
                return_dict=True,
            )
            outputs = model(
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                output_hidden_states=True,
                return_dict=True,
            )

        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers)
        torch.testing.assert_close(outputs.last_hidden_state, baseline.last_hidden_state)
        torch.testing.assert_close(
            model.transformer.post_layernorm(outputs.hidden_states[-1]),
            outputs.last_hidden_state,
        )

    def test_linear_patch_embedding(self):
        config = self.model_tester.get_vision_config()
        config.num_patches = 16
        model = Ovis2_5VisionModel(config).to(torch_device).eval()
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=torch_device)
        pixel_values = torch.randn(
            int(grid_thw.prod()),
            config.num_channels * config.patch_size**2,
            device=torch_device,
        )

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, grid_thw=grid_thw)

        self.assertIsInstance(model.get_input_embeddings(), torch.nn.Linear)
        self.assertEqual(model.transformer.embeddings.position_embedding.num_embeddings, config.num_patches)
        self.assertEqual(outputs.pooler_output.shape[0], 4)

        config.preserve_original_pe = False
        model = Ovis2_5VisionModel(config)
        self.assertNotIn("position_embedding.weight", model.transformer.embeddings.state_dict())


if __name__ == "__main__":
    unittest.main()
