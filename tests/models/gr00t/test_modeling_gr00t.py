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
"""Testing suite for the PyTorch GR00T model."""

import unittest

from parameterized import parameterized

from transformers import Gr00tConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch

    from transformers import Gr00tForConditionalGeneration


class Gr00tModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        text_seq_length=12,
        num_image_tokens=4,
        image_token_id=3,
        vocab_size=99,
        action_horizon=4,
        max_state_dim=6,
        max_action_dim=5,
        max_num_embodiments=3,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.text_seq_length = text_seq_length
        # Attributes checked by the common tests on the recorded DiT hidden states and attentions
        self.seq_length = 1 + action_horizon
        self.key_length = text_seq_length
        self.hidden_size = 24
        self.num_hidden_layers = 4
        self.num_attention_heads = 4
        self.num_image_tokens = num_image_tokens
        self.image_token_id = image_token_id
        self.vocab_size = vocab_size
        self.action_horizon = action_horizon
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_num_embodiments = max_num_embodiments
        self.is_training = is_training
        # Image of 2x2 patches of 4x4 pixels, 2 frames, no spatial merging: 4 image tokens
        self.patch_size = 4
        self.temporal_patch_size = 2
        self.image_grid_thw = [1, 2, 2]

    def get_config(self):
        return Gr00tConfig(
            vlm_config={
                "model_type": "qwen3_vl",
                "image_token_id": self.image_token_id,
                "video_token_id": 4,
                "vision_start_token_id": 5,
                "vision_end_token_id": 6,
                "text_config": {
                    "model_type": "qwen3_vl_text",
                    "vocab_size": self.vocab_size,
                    "hidden_size": 32,
                    "intermediate_size": 37,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 8,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 10000.0,
                        "mrope_section": [2, 1, 1],
                        "mrope_interleaved": True,
                    },
                },
                "vision_config": {
                    "depth": 2,
                    "hidden_size": 32,
                    "intermediate_size": 37,
                    "num_heads": 4,
                    "out_hidden_size": 32,
                    "patch_size": self.patch_size,
                    "spatial_merge_size": 1,
                    "temporal_patch_size": self.temporal_patch_size,
                    "num_position_embeddings": 16,
                    "deepstack_visual_indexes": [0],
                },
            },
            action_horizon=self.action_horizon,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            max_num_embodiments=self.max_num_embodiments,
            projector_hidden_size=16,
            action_head_config={
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "refiner_num_hidden_layers": 2,
                "refiner_num_attention_heads": 4,
                "timestep_embedding_dim": 8,
            },
            max_position_embeddings=8,
        )

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()

        # Text tokens never collide with the special token ids 3-6
        input_ids = ids_tensor([self.batch_size, self.text_seq_length], self.vocab_size - 7) + 7
        input_ids[:, 1] = 5
        input_ids[:, 2 : 2 + self.num_image_tokens] = self.image_token_id
        input_ids[:, 2 + self.num_image_tokens] = 6
        attention_mask = torch.ones_like(input_ids)
        # Left-pad the last sample
        attention_mask[-1, :1] = 0

        patch_dim = 3 * self.temporal_patch_size * self.patch_size**2
        pixel_values = floats_tensor([self.batch_size * self.num_image_tokens, patch_dim])
        image_grid_thw = torch.tensor([self.image_grid_thw] * self.batch_size, device=torch_device)
        mm_token_type_ids = (input_ids == self.image_token_id).int()

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
            "state": floats_tensor([self.batch_size, self.max_state_dim]),
            "embodiment_id": ids_tensor([self.batch_size], self.max_num_embodiments),
            "actions": floats_tensor([self.batch_size, self.action_horizon, self.max_action_dim]),
            "noise": floats_tensor([self.batch_size, self.action_horizon, self.max_action_dim]),
            "timestep": torch.rand(self.batch_size, device=torch_device),
        }
        return config, inputs_dict


@require_torch
class Gr00tForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Gr00tForConditionalGeneration,) if is_torch_available() else ()
    test_resize_embeddings = False
    _is_composite = True
    additional_model_inputs = [
        "input_ids",
        "attention_mask",
        "image_grid_thw",
        "mm_token_type_ids",
        "embodiment_id",
        "actions",
        "noise",
        "timestep",
    ]

    def setUp(self):
        self.model_tester = Gr00tModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gr00tConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_loss_per_sample(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.loss_reduction = "none"
        model = Gr00tForConditionalGeneration(config).eval().to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs_dict)
        self.assertEqual(
            outputs.loss.shape, (self.model_tester.batch_size, config.action_horizon, config.max_action_dim)
        )

    def test_action_mask_excludes_padded_dims(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Gr00tForConditionalGeneration(config).eval().to(torch_device)
        action_mask = torch.ones_like(inputs_dict["actions"])
        action_mask[..., -2:] = 0
        with torch.no_grad():
            loss = model(**inputs_dict, action_mask=action_mask).loss
            # Changing the targets of masked-out dimensions must not change the loss
            inputs_dict["actions"][..., -2:] += 10.0
            loss_perturbed = model(**inputs_dict, action_mask=action_mask).loss
        torch.testing.assert_close(loss, loss_perturbed)

    def test_vlm_features_are_last_layer_before_final_norm(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Gr00tForConditionalGeneration(config).eval().to(torch_device)
        vlm_inputs = {
            key: inputs_dict[key]
            for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
        }
        captured = {}
        hook = model.model.vlm.language_model.layers[-1].register_forward_hook(
            lambda module, args, output: captured.update(features=output)
        )
        with torch.no_grad():
            features = model.model.get_vlm_features(**vlm_inputs)
        hook.remove()
        with torch.no_grad():
            expected = model.model.vl_refiner(model.model.vl_layer_norm(captured["features"]))
        torch.testing.assert_close(features, expected)

    def test_precomputed_vlm_features_match(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Gr00tForConditionalGeneration(config).eval().to(torch_device)
        vlm_inputs = {
            key: inputs_dict[key]
            for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
        }
        with torch.no_grad():
            expected = model(**inputs_dict).logits
            vlm_features = model.model.get_vlm_features(**vlm_inputs)
            inputs_dict.pop("pixel_values")
            actual = model(**inputs_dict, vlm_features=vlm_features).logits
        torch.testing.assert_close(actual, expected)

    def test_embodiment_selects_projection_slot(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Gr00tForConditionalGeneration(config).eval().to(torch_device)
        inputs_dict["embodiment_id"] = torch.zeros_like(inputs_dict["embodiment_id"])
        with torch.no_grad():
            expected = model(**inputs_dict).logits
            # Weights of an unused embodiment slot must not affect the output
            model.action_decoder.fc2.weight[1].normal_()
            model.state_encoder.fc1.weight[1].normal_()
            actual = model(**inputs_dict).logits
        torch.testing.assert_close(actual, expected)

    def test_sample_actions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Gr00tForConditionalGeneration(config).eval().to(torch_device)
        for key in ("actions", "timestep"):
            inputs_dict.pop(key)
        with torch.no_grad():
            actions = model.sample_actions(**inputs_dict)
            actions_again = model.sample_actions(**inputs_dict)
            # One Euler step from t=0 is `noise + velocity(noise, t=0)`
            single_step = model.sample_actions(**inputs_dict, num_steps=1)
            velocity = model(
                **{k: v for k, v in inputs_dict.items() if k != "noise"},
                noise=inputs_dict["noise"],
                timestep=torch.zeros(self.model_tester.batch_size, device=torch_device),
            ).logits
        self.assertEqual(actions.shape, (self.model_tester.batch_size, config.action_horizon, config.max_action_dim))
        self.assertTrue(torch.isfinite(actions).all())
        torch.testing.assert_close(actions, actions_again)
        torch.testing.assert_close(single_step, inputs_dict["noise"] + velocity)

    def test_half_precision_model_accepts_float32_inputs(self):
        # The processor returns a float32 `state`: the model runs it, the noise and the actions in its own dtype
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Gr00tForConditionalGeneration(config).to(torch_device, torch.bfloat16)
        inputs_dict["pixel_values"] = inputs_dict["pixel_values"].to(torch.bfloat16)
        self.assertEqual(inputs_dict["state"].dtype, torch.float32)
        loss = model(**inputs_dict).loss
        self.assertEqual(loss.dtype, torch.bfloat16)
        loss.backward()

        model.eval()
        inputs_dict.pop("actions"), inputs_dict.pop("timestep")
        with torch.no_grad():
            actions = model.sample_actions(**inputs_dict)
        self.assertEqual(actions.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(actions).all())

    def test_eager_matches_sdpa_sample_actions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for key in ("actions", "timestep"):
            inputs_dict.pop(key)
        model_sdpa = Gr00tForConditionalGeneration._from_config(config, attn_implementation="sdpa")
        model_sdpa = model_sdpa.eval().to(torch_device)
        model_eager = Gr00tForConditionalGeneration._from_config(config, attn_implementation="eager")
        model_eager.load_state_dict(model_sdpa.state_dict())
        model_eager = model_eager.eval().to(torch_device)
        with torch.no_grad():
            torch.testing.assert_close(
                model_eager.sample_actions(**inputs_dict),
                model_sdpa.sample_actions(**inputs_dict),
                atol=1e-5,
                rtol=1e-5,
            )

    def test_training_reduces_loss(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.action_head_config.dropout = 0.0
        config.state_dropout_prob = 0.0
        torch.manual_seed(0)
        model = Gr00tForConditionalGeneration(config).to(torch_device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Fixed noise and timesteps, so that only the parameters change between steps
        losses = []
        for _ in range(150):
            loss = model(**inputs_dict).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # The model can fit the fixed batch: the loss drops well below the variance of the target velocity
        self.assertLess(losses[-1], 0.5 * losses[0], losses)

    def test_training_compiled_action_head_step(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Gr00tForConditionalGeneration(config).to(torch_device).eval()
        vlm_inputs = {
            key: inputs_dict[key]
            for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
        }
        with torch.no_grad():
            step_inputs = {
                key: value
                for key, value in inputs_dict.items()
                if key not in vlm_inputs or key in ("input_ids", "attention_mask")
            }
            step_inputs["vlm_features"] = model.model.get_vlm_features(**vlm_inputs)
            expected = model(**step_inputs).logits
            # One denoising step from precomputed backbone features compiles without graph breaks
            actual = torch.compile(lambda **kwargs: model(**kwargs).logits, fullgraph=True)(**step_inputs)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_training_gradients_reach_used_parameters(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Gr00tForConditionalGeneration(config).to(torch_device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        model(**inputs_dict).loss.backward()
        # The final norm of the backbone is unused: the action head reads the last decoder layer before it
        unused = {"model.vlm.language_model.norm.weight"}
        for name, param in model.named_parameters():
            if name in unused:
                self.assertIsNone(param.grad, name)
            else:
                self.assertIsNotNone(param.grad, f"{name} has no gradient")

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip(
        "The common test builds MAE-style `noise` from `image_size`; covered by `test_eager_matches_sdpa_sample_actions`"
    )
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        pass

    @unittest.skip(
        "The DiT always cross-attends with text-only or image-only masks, so SDPA can't use its flash backend, as in PI0"
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(
        "The backbone's final norm is unused by design and gets no gradient; "
        "covered by `test_training_gradients_reach_used_parameters`"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        "The backbone's final norm is unused by design and gets no gradient; "
        "covered by `test_training_gradients_reach_used_parameters`"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        "The backbone's final norm is unused by design and gets no gradient; "
        "covered by `test_training_gradients_reach_used_parameters`"
    )
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass


@slow
@require_torch_accelerator
class Gr00tModelIntegrationTest(unittest.TestCase):
    checkpoint = "nvidia/GR00T-N1.7-3B"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @staticmethod
    def get_inputs():
        # Tokenized inputs built by hand, as the tokenizer of the backbone repository is gated: two 256x256 images
        # (64 image tokens each) between text tokens
        generator = torch.Generator().manual_seed(0)
        image_tokens = [151652] + [151655] * 64 + [151653]
        input_ids = torch.tensor(
            [
                torch.randint(1000, 30000, (4,), generator=generator).tolist()
                + image_tokens
                + image_tokens
                + torch.randint(1000, 30000, (12,), generator=generator).tolist()
            ]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": (input_ids == 151655).int(),
            "pixel_values": torch.randn(2 * 256, 1536, generator=generator),
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 16, 16]]),
            "state": torch.rand(1, 132, generator=generator) * 2 - 1,
            "embodiment_id": torch.tensor([24]),
            "noise": torch.randn(1, 40, 132, generator=generator),
        }

    def test_original_checkpoint_loads_all_weights(self):
        _, loading_info = Gr00tForConditionalGeneration.from_pretrained(self.checkpoint, output_loading_info=True)
        for key, values in loading_info.items():
            self.assertEqual(len(values), 0, f"{key}: {values}")

    def _sample_actions(self, dtype):
        model = Gr00tForConditionalGeneration.from_pretrained(self.checkpoint, dtype=dtype, device_map=torch_device)
        inputs = {key: value.to(torch_device) for key, value in self.get_inputs().items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.no_grad():
            actions = model.sample_actions(**inputs)
        self.assertEqual(tuple(actions.shape), (1, 40, 132))
        self.assertEqual(actions.dtype, dtype)
        # The first three and the last action steps, first six dims
        return torch.cat([actions[0, :3, :6], actions[0, -1:, :6]]).float().cpu()

    def test_sample_actions(self):
        # Matches the original GR00T N1.7 implementation (Isaac-GR00T `n1.7-release`) within 5e-5
        expected = Expectations(
            {
                (None, None): [
                    [-0.1292, -0.0473, 0.1391, 0.9806, -0.1003, -0.0310],
                    [-0.0541, 0.1053, 0.0915, 1.0008, 0.0806, 0.0461],
                    [-0.0534, 0.1125, 0.0912, 1.0006, 0.0559, 0.0444],
                    [0.0054, -0.0755, -0.1167, 0.9673, 0.1616, -0.1279],
                ],
            }
        ).get_expectation()  # fmt: skip
        actions = self._sample_actions(torch.float32)
        torch.testing.assert_close(actions, torch.tensor(expected), atol=1e-3, rtol=1e-3)

    def test_sample_actions_bf16(self):
        expected = Expectations(
            {
                ("cuda", (8, 6)): [
                    [-0.1318, -0.0483, 0.1367, 0.9805, -0.0996, -0.0293],
                    [-0.0552, 0.1045, 0.0889, 1.0000, 0.0801, 0.0488],
                    [-0.0537, 0.1123, 0.0898, 1.0000, 0.0547, 0.0449],
                    [0.0000, -0.0762, -0.1133, 0.9688, 0.1582, -0.1279],
                ],
            }
        ).get_expectation()  # fmt: skip
        actions = self._sample_actions(torch.bfloat16)
        torch.testing.assert_close(actions, torch.tensor(expected), atol=2e-2, rtol=2e-2)
