# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch PI0 model."""

import tempfile
import unittest

from datasets import Dataset, load_dataset
from parameterized import parameterized

from transformers import PI0Config, PI0Processor, Trainer, TrainingArguments, is_torch_available
from transformers.image_utils import load_image
from transformers.testing_utils import (
    require_torch,
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
from ...trainer.trainer_test_utils import StoreLossCallback


if is_torch_available():
    import torch

    from transformers import PI0ForConditionalGeneration


class PI0ModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.is_training = True
        self.batch_size = 4
        self.num_cameras = 2
        self.image_size = 8
        self.patch_size = 4
        self.num_channels = 3
        self.vocab_size = 128
        self.hidden_size = 16
        self.num_hidden_layers = 2
        self.num_attention_heads = 2
        self.chunk_size = 4
        self.max_state_dim = 8
        self.max_action_dim = 8
        self.num_inference_steps = 3
        self.image_token_index = 127
        self.pad_token_id = 0
        self.num_image_tokens = (self.image_size // self.patch_size) ** 2 * self.num_cameras
        self.encoder_seq_length = 5
        self.seq_length = self.encoder_seq_length + self.num_image_tokens
        self.key_length = self.encoder_seq_length + self.seq_length

        self.vision_config = {
            "model_type": "siglip_vision_model",
            "hidden_size": self.hidden_size,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": self.patch_size,
            "image_size": self.image_size,
            "vision_use_head": False,
            "num_channels": self.num_channels,
        }
        self.text_config = {
            "model_type": "gemma",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "pad_token_id": 0,
        }
        self.dit_config = {
            "model_type": "gemma",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "pad_token_id": 0,
        }

    def get_config(self):
        return PI0Config(
            dit_config=self.dit_config,
            vlm_config={
                "vision_config": self.vision_config,
                "text_config": self.text_config,
                "image_token_index": self.image_token_index,
                "projection_dim": self.hidden_size,
            },
            chunk_size=self.chunk_size,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            num_inference_steps=self.num_inference_steps,
        )

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        pixel_values = floats_tensor(
            [self.batch_size, self.num_cameras, self.num_channels, self.image_size, self.image_size]
        )
        pixel_attention_mask = torch.tensor(
            [[True, True], [True, True], [True, False], [True, False]], dtype=torch.bool, device=torch_device
        )

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1)
        attention_mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.long, device=torch_device)
        input_ids[input_ids == config.vlm_config.image_token_id] = self.pad_token_id
        # Pixel attention mask is not completely-unmasked, so we create different input ids
        input_ids[:2, : self.num_image_tokens] = config.vlm_config.image_token_id
        input_ids[2:4, : self.num_image_tokens // 2] = config.vlm_config.image_token_id

        state = floats_tensor([self.batch_size, self.max_state_dim])
        actions = floats_tensor([self.batch_size, self.chunk_size, self.max_action_dim])
        noise = floats_tensor([self.batch_size, self.chunk_size, self.max_action_dim])
        timestep = torch.tensor([0.3, 0.5, 0.8, 0.9], dtype=torch.float32, device=torch_device)

        inputs_dict = {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "state": state,
            "actions": actions,
            "noise": noise,
            "timestep": timestep,
        }
        return config, inputs_dict


@require_torch
class PI0ForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (PI0ForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False
    test_resize_embeddings = False
    test_torch_exportable = False
    test_all_params_have_gradient = False
    has_attentions = True
    _is_composite = True
    additional_model_inputs = ["input_ids", "attention_mask", "state", "actions", "timestep"]

    def setUp(self):
        self.model_tester = PI0ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PI0Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = PI0ForConditionalGeneration(config).eval().to(device=torch_device)
        with torch.no_grad():
            outputs = model(**inputs_dict)
        self.assertEqual(outputs.loss.shape, (self.model_tester.batch_size, config.chunk_size, config.max_action_dim))

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip("Model architecture is special and requires much higher `tols`")
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        pass

    @unittest.skip(
        "Skip until the official weights add `embed_tokens`. Currently weights have only `lm_head` saved but"
        " PI0 doesn't create any lm-head. So we added it in conversion mapping"
    )
    def test_reverse_loading_mapping(self):
        pass

    @unittest.skip("Prefix tuning doesn't work with GC and the model uses prefix tuning to fuse VLM outputs")
    def test_enable_input_require_grads_with_gradient_checkpointing(self):
        pass

    @unittest.skip("Prefix tuning doesn't work with GC and the model uses prefix tuning to fuse VLM outputs")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip("Prefix tuning doesn't work with GC and the model uses prefix tuning to fuse VLM outputs")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip("Prefix tuning doesn't work with GC and the model uses prefix tuning to fuse VLM outputs")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass


@require_torch
class PI0ModelSmokeTest(unittest.TestCase):
    def test_full_run_smoke(self):
        torch.manual_seed(0)
        tester = PI0ModelTester(self)
        config, inputs_dict = tester.prepare_config_and_inputs_for_common()
        model = PI0ForConditionalGeneration(config).to(device=torch_device).eval()

        with torch.no_grad():
            outputs = model(**inputs_dict)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.loss.ndim, 3)

        sample_inputs = {k: v for k, v in inputs_dict.items() if k not in ["actions", "timestep"]}
        with torch.no_grad():
            sampled_actions = model.sample_actions(**sample_inputs, num_steps=2)
        self.assertEqual(sampled_actions.shape, (tester.batch_size, config.chunk_size, config.max_action_dim))
        self.assertTrue(torch.isfinite(sampled_actions).all())


@require_torch
@slow
class PI0ModelIntegrationTest(unittest.TestCase):
    def test_pi0_base_reference_values(self):
        model = PI0ForConditionalGeneration.from_pretrained("lerobot/pi0_base", torch_dtype=torch.float32).eval()
        processor = PI0Processor.from_pretrained("google/paligemma-3b-pt-224")

        inputs = processor(
            text=["Pick up the object"],
            images=load_image(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/vla_pi0.jpg"
            ),
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=304,
            return_tensors="pt",
        )

        # Generate random state and actions for prediction
        torch.manual_seed(42)
        state = torch.randn(1, 32)
        actions = torch.randn(1, 50, 32)
        noise = torch.randn(1, 50, 32)
        timestep = torch.tensor([0.5], dtype=torch.float32)

        with torch.no_grad():
            suffix_embs = model.embed_action_time(state, noise, timestep)
        self.assertEqual(suffix_embs.shape, (1, 51, 1024))
        self.assertAlmostEqual(suffix_embs.mean().item(), -0.10177, delta=0.002)
        torch.testing.assert_close(
            suffix_embs[0, 0, :4], torch.tensor([-0.0460, 0.8858, 0.7172, -0.7538]), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            suffix_embs[0, -1, :4], torch.tensor([0.7107, -1.3107, -4.8396, -6.9446]), atol=1e-3, rtol=1e-3
        )

        with torch.no_grad():
            prefix_embs = model.model.embed_prefix(**inputs)

        self.assertEqual(prefix_embs.shape, (1, 304, 2048))
        self.assertAlmostEqual(prefix_embs.mean().item(), 0.0224, places=3)
        torch.testing.assert_close(
            prefix_embs[0, 0, :4], torch.tensor([1.1781, 0.1176, -0.2231, -0.3662]), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            prefix_embs[0, -1, :4], torch.tensor([23.8649, -1.4916, 4.2868, 4.3973]), atol=1e-3, rtol=1e-3
        )

        with torch.no_grad():
            outputs = model(
                **inputs,
                state=state,
                actions=actions,
                noise=noise,
                timestep=timestep,
            )
        self.assertEqual(outputs.loss.shape, (1, 50, 32))
        self.assertAlmostEqual(outputs.loss.mean().item(), 3.950, places=3)

        torch.manual_seed(99)  # different seed to sample random noise
        model.model.dit.config._attn_implementation = "eager"
        with torch.no_grad():
            sampled = model.sample_actions(**inputs, state=state, num_steps=3)
        self.assertEqual(sampled.shape, (1, 50, 32))
        self.assertAlmostEqual(sampled.mean().item(), -0.0764, places=3)
        self.assertAlmostEqual(sampled.std().item(), 0.2300, places=3)
        torch.testing.assert_close(
            sampled[0, 0, :4], torch.tensor([0.0602, -0.1177, -0.5010, -0.0028]), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            sampled[0, -1, :4], torch.tensor([0.0615, 0.0161, -0.3112, -0.9186]), atol=1e-3, rtol=1e-3
        )

    def test_pi0_base_libero(self):
        model = PI0ForConditionalGeneration.from_pretrained("lerobot/pi0_base", torch_dtype=torch.float32).eval()
        processor = PI0Processor.from_pretrained("google/paligemma-3b-pt-224")

        small_data = load_dataset("RaushanTurganbay/libero-small-testing", split="train")
        first_sample = small_data[0]
        timestep = torch.tensor([first_sample["timestamp"]])

        inputs = processor(
            images=[first_sample["observation.images.image"], first_sample["observation.images.wrist_image"]],
            text="put the white mug on the left plate and put the yellow and white mug on the right plate",
            actions=small_data["action"][:50],  # chunk size is 50
            state=first_sample["observation.state"],
            padding=True,
            padding_side="right",
            truncation=True,
            return_tensors="pt",
        )

        # Generate random noise
        torch.manual_seed(63)
        noise = torch.randn(1, 50, 32)

        with torch.no_grad():
            outputs = model(**inputs, noise=noise, timestep=timestep)
        self.assertEqual(outputs.loss.shape, (1, 50, 32))
        self.assertAlmostEqual(outputs.loss.mean().item(), 2.5087, places=3)

        with torch.no_grad():
            sampled = model.sample_actions(**inputs, num_steps=5)
        self.assertEqual(sampled.shape, (1, 50, 32))
        self.assertAlmostEqual(sampled.mean().item(), -0.0192, places=3)
        self.assertAlmostEqual(sampled.std().item(), 0.1267, places=3)
        torch.testing.assert_close(
            sampled[0, 0, :4], torch.tensor([-0.2456, -0.1260, -0.2977, 0.2654]), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            sampled[0, -1, :4], torch.tensor([-0.2541, -0.1213, -0.2637, 0.2935]), atol=1e-3, rtol=1e-3
        )

    def test_train_pi0_base_libero(self):
        model = PI0ForConditionalGeneration.from_pretrained("lerobot/pi0_base", torch_dtype=torch.float32).eval()
        processor = PI0Processor.from_pretrained("google/paligemma-3b-pt-224")

        small_data = load_dataset("RaushanTurganbay/libero-small-testing", split="train")
        train_actions = [small_data["action"][i : i + 50] for i in range(len(small_data) - 50)]

        def preprocess(example):
            # format images as nested list
            example["images"] = [[im] for im in example["images"]]
            encodings = processor(**example, return_tensors="pt")
            encodings["timestep"] = example["timestep"]
            return encodings

        train_data = Dataset.from_dict(
            {
                "actions": train_actions[:50],
                "text": ["put the white mug on the left plate"] * 50,
                "state": small_data["observation.state"][:50],
                "timestep": small_data["timestamp"][:50],
                "images": small_data["observation.images.image"][:50],
            }
        )
        train_data = train_data.map(preprocess, batched=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                max_steps=5,
                learning_rate=1e-4,
                logging_steps=1,
                disable_tqdm=True,
            )
            loss_callback = StoreLossCallback()
            trainer = Trainer(
                model,
                args,
                train_dataset=train_data,
                callbacks=[loss_callback],
                processing_class=processor,
            )
            trainer.train()

        # Loss is steadily decreasing
        self.assertTrue(sorted(loss_callback.losses, reverse=True) == loss_callback.losses)
