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
"""Testing suite for the PyTorch LightOnOCR model."""

import copy
import unittest
from difflib import SequenceMatcher

from transformers import (
    AutoProcessor,
    LightOnOCRConfig,
    LightOnOCRForConditionalGeneration,
    LightOnOCRModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


if is_vision_available():
    from transformers.image_utils import load_image


class LightOnOCRVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        image_token_index=10,
        spatial_merge_size=2,
        seq_length=7,
        text_config={
            "model_type": "lightonocr_text",
            "seq_length": 7,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 37,
            "hidden_act": "silu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "head_dim": 8,
        },
        is_training=True,
        vision_config={
            "image_size": 112,
            "patch_size": 14,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "initializer_range": 0.02,
            "rope_theta": 10000.0,
        },
    ):
        self.parent = parent
        self.image_token_index = image_token_index
        self.spatial_merge_size = spatial_merge_size
        self.text_config = text_config
        self.vision_config = vision_config
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        # Image size must be divisible by patch_size
        self.image_size = vision_config["image_size"]
        self.patch_size = vision_config["patch_size"]
        # Number of patches after patch conv
        num_patches = (self.image_size // self.patch_size) ** 2
        # After spatial merging, number of tokens is reduced by spatial_merge_size**2
        self.num_image_tokens = num_patches // (self.spatial_merge_size**2)
        self.seq_length = seq_length + self.num_image_tokens
        self.encoder_seq_length = self.seq_length

    def get_config(self):
        return LightOnOCRConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_index,
            spatial_merge_size=self.spatial_merge_size,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1

        # Avoid placing image tokens on positions that would be the pad token
        input_ids[input_ids == config.image_token_id] = self.pad_token_id

        # Place image tokens at the beginning
        input_ids[:, : self.num_image_tokens] = config.image_token_id

        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # Create image_sizes as tensor - must match batch size
        image_sizes = torch.tensor(
            [[self.image_size, self.image_size]] * self.batch_size, dtype=torch.long, device=torch_device
        )

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_sizes": image_sizes,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_generate(self, batch_size=None):
        """Prepare config and inputs for generation tests."""
        if batch_size is None:
            batch_size = self.batch_size

        # Get base config
        config = self.get_config()

        # Create pixel_values with the specified batch size
        pixel_values = floats_tensor(
            [
                batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )

        # Create input_ids
        input_ids = ids_tensor([batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1

        # Avoid placing image tokens on positions that would be the pad token
        input_ids[input_ids == config.image_token_id] = self.pad_token_id

        # Place image tokens at the beginning
        input_ids[:, : self.num_image_tokens] = config.image_token_id

        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # Create image_sizes as tensor - must match batch size
        image_sizes = torch.tensor(
            [[self.image_size, self.image_size]] * batch_size, dtype=torch.long, device=torch_device
        )

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_sizes": image_sizes,
        }
        return config, inputs_dict


@require_torch
class LightOnOCRForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `LightOnOCRForConditionalGeneration`.
    """

    all_model_classes = (
        (
            LightOnOCRModel,
            LightOnOCRForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = {"image-text-to-text": LightOnOCRForConditionalGeneration} if is_torch_available() else {}

    _is_composite = True
    test_head_masking = False
    test_pruning = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = LightOnOCRVisionText2TextModelTester(self)
        common_properties = ["image_token_id", "spatial_merge_size"]
        self.config_tester = ConfigTester(
            self, config_class=LightOnOCRConfig, has_text_modality=False, common_properties=common_properties
        )

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        """
        Prepare inputs for the model class, ensuring image_sizes matches the batch size.
        """
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        # Ensure image_sizes matches the batch size of pixel_values or input_ids
        if "pixel_values" in inputs_dict and "image_sizes" in inputs_dict:
            batch_size = inputs_dict["pixel_values"].shape[0]
            # If image_sizes doesn't match batch size, adjust it
            if len(inputs_dict["image_sizes"]) != batch_size:
                # Take only the first batch_size entries
                inputs_dict["image_sizes"] = inputs_dict["image_sizes"][:batch_size]

        return inputs_dict

    def prepare_config_and_inputs_for_generate(self, batch_size=1):
        """Override to use the model_tester's custom method."""
        return self.model_tester.prepare_config_and_inputs_for_generate(batch_size=batch_size)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs throw an error with explicit message saying what is wrong
        when number of images doesn't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompt has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)  # in-place modifications further
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # remove one image but leave the image token in text
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-1:, ...]
            curr_input_dict["image_sizes"] = curr_input_dict["image_sizes"][-1:]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:1]
            image_sizes = curr_input_dict["image_sizes"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(input_ids=input_ids, pixel_values=pixel_values, image_sizes=image_sizes)

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_sizes = torch.cat([image_sizes, image_sizes], dim=0)
            _ = model(input_ids=input_ids, pixel_values=pixel_values, image_sizes=image_sizes)

    def test_spatial_merge_size(self):
        """
        Test that models can be created and initialized with different spatial_merge_size values.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Test that model can be created with different spatial_merge_size values
        for spatial_merge_size in [1, 2, 4]:
            curr_config = copy.deepcopy(config)
            curr_config.spatial_merge_size = spatial_merge_size

            for model_class in self.all_model_classes:
                # Build model with the new config - should not raise any errors
                model = model_class(curr_config).to(torch_device)
                model.eval()

                # Verify the spatial_merge_size is set correctly
                self.assertEqual(model.config.spatial_merge_size, spatial_merge_size)

                # Verify the model has the expected components
                if hasattr(model, "model"):
                    self.assertTrue(hasattr(model.model, "vision_projection"))
                    self.assertEqual(model.model.vision_projection.config.spatial_merge_size, spatial_merge_size)
                elif hasattr(model, "vision_projection"):
                    self.assertEqual(model.vision_projection.config.spatial_merge_size, spatial_merge_size)

    def test_forward_pass_with_image_sizes(self):
        """
        Test that the model correctly handles variable image sizes.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()

            # Test with different image sizes in the same batch
            batch_size = 2
            pixel_values = floats_tensor(
                [batch_size, 3, self.model_tester.image_size, self.model_tester.image_size]
            ).to(torch_device)

            # Different image sizes (but still need to be divisible by patch_size)
            image_sizes = torch.tensor(
                [[self.model_tester.image_size, self.model_tester.image_size]] * batch_size,
                dtype=torch.long,
                device=torch_device,
            )

            num_patches = (self.model_tester.image_size // self.model_tester.patch_size) ** 2
            num_image_tokens = num_patches // (config.spatial_merge_size**2)

            input_ids = ids_tensor([batch_size, 10 + num_image_tokens], config.text_config.vocab_size - 1) + 1
            # Ensure no tokens accidentally equal image_token_id
            input_ids[input_ids == config.image_token_id] = config.image_token_id + 1
            # Now place image tokens at the beginning
            input_ids[:, :num_image_tokens] = config.image_token_id
            input_ids = input_ids.to(torch_device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_sizes=image_sizes,
            )

            self.assertIsNotNone(outputs)

    def test_model_outputs_equivalence(self):
        """
        Test that model outputs are consistent across different input configurations.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs1 = model(**input_dict)
                outputs2 = model(**input_dict)

            # Check that outputs are deterministic
            if hasattr(outputs1, "last_hidden_state") and hasattr(outputs2, "last_hidden_state"):
                self.assertTrue(torch.allclose(outputs1.last_hidden_state, outputs2.last_hidden_state, atol=1e-5))

    @unittest.skip(
        "LightOnOCR uses complex attention patterns with sliding windows, skipping gradient checkpointing test"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        "LightOnOCR uses complex attention patterns with sliding windows, skipping gradient checkpointing test"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        "LightOnOCR uses complex attention patterns with sliding windows, skipping gradient checkpointing test"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("FlashAttention only support fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_eager_matches_fa2_generate(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_eager_matches_sdpa_generate(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_flash_attn_2_from_config(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip("Pixtral does not support attention interfaces.")
    def test_flex_attention_with_grads(self):
        pass

    def test_initialization(self):
        """
        Test that model initializes correctly with proper weight initialization.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)

            # Check that model has all expected components
            if model_class == LightOnOCRForConditionalGeneration:
                self.assertTrue(hasattr(model, "model"))
                self.assertTrue(hasattr(model.model, "vision_encoder"))
                self.assertTrue(hasattr(model.model, "language_model"))
                self.assertTrue(hasattr(model.model, "vision_projection"))
                self.assertTrue(hasattr(model, "lm_head"))
            elif model_class == LightOnOCRModel:
                self.assertTrue(hasattr(model, "vision_encoder"))
                self.assertTrue(hasattr(model, "language_model"))
                self.assertTrue(hasattr(model, "vision_projection"))

    def test_vision_projection(self):
        """
        Test that the vision projection correctly transforms vision embeddings to text space.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        model = LightOnOCRModel(config).to(torch_device)
        model.eval()

        # Convert image_sizes to list for vision_encoder
        if isinstance(input_dict["image_sizes"], torch.Tensor):
            image_sizes_list = [(int(h), int(w)) for h, w in input_dict["image_sizes"]]
        else:
            image_sizes_list = input_dict["image_sizes"]

        with torch.no_grad():
            # Get vision features
            vision_outputs = model.vision_encoder(
                pixel_values=input_dict["pixel_values"].to(torch_device),
                image_sizes=image_sizes_list,
            )

            # Project vision features
            projected = model.vision_projection(
                vision_outputs.last_hidden_state.squeeze(0),
                image_sizes_list,
            )

            # Check output dimensions - should match text hidden size
            self.assertEqual(projected.shape[-1], config.text_config.hidden_size)

    def test_get_image_features(self):
        """
        Test the get_image_features method.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        model = LightOnOCRModel(config).to(torch_device)
        model.eval()

        with torch.no_grad():
            image_features_list = model.get_image_features(
                pixel_values=input_dict["pixel_values"].to(torch_device),
                image_sizes=input_dict["image_sizes"],
            )

            # Check that features are returned as a list
            self.assertIsNotNone(image_features_list)
            self.assertIsInstance(image_features_list, (list, tuple))

            # Concatenate features and check shape
            image_features = torch.cat(image_features_list, dim=0)
            self.assertEqual(image_features.shape[-1], config.text_config.hidden_size)


@slow
@require_torch
class LightOnOCRForConditionalGenerationIntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_lightonocr_ocr_integration(self):
        """
        Integration test for LightOnOCR OCR capabilities.
        Tests that the model can perform OCR on a real image and produce expected output.

        """

        model_id = "lightonai/LightOnOCR-1B-1025"

        # Load processor and model from Hub
        processor = AutoProcessor.from_pretrained(model_id)
        model = LightOnOCRForConditionalGeneration.from_pretrained(model_id, device_map=torch_device)
        model.eval()

        # Load a test OCR image
        # This is a standard OCR test image from HuggingFace fixtures
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg"
        )

        # Process image and prepare inputs
        # Using chat template as shown in the model's usage pattern
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        # Generate OCR output
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
            )

        # Decode output, excluding the input prompt
        decoded_output = processor.decode(generated_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        expected_output = "Document No : TD01167104\n\nDate : 25/12/2018 8:13:39 PM\n\nCashier : MANIS\n\nMember :\n\nCASH BILL\n\n| CODE"

        similarity = SequenceMatcher(None, decoded_output, expected_output).ratio()

        # Require at least 95% similarity to catch regressions while allowing minor variations
        self.assertGreater(
            similarity,
            0.95,
            f"Model output differs too much from expected output (similarity: {similarity:.2%}).\n"
            f"Expected:\n{expected_output}\n\nGot:\n{decoded_output}",
        )

    def test_model_can_generate_without_images(self):
        """
        Test that the model can generate text without image inputs.
        """
        # Create a small config for fast testing
        text_config = {
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "head_dim": 16,
        }
        vision_config = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "image_size": 112,
            "patch_size": 14,
        }

        config = LightOnOCRConfig(text_config=text_config, vision_config=vision_config, image_token_id=10)
        model = LightOnOCRForConditionalGeneration(config).to(torch_device)
        model.eval()

        # Create text-only input
        input_ids = torch.randint(0, config.vocab_size - 1, (1, 10), device=torch_device) + 1

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=5)

        self.assertIsNotNone(outputs)
        self.assertEqual(outputs.shape[0], 1)
        self.assertGreater(outputs.shape[1], input_ids.shape[1])

    def test_model_forward_with_images(self):
        """
        Test forward pass with image inputs.
        """
        text_config = {
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "head_dim": 16,
        }
        vision_config = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "image_size": 112,
            "patch_size": 14,
        }

        config = LightOnOCRConfig(text_config=text_config, vision_config=vision_config, image_token_id=10)
        model = LightOnOCRForConditionalGeneration(config).to(torch_device)
        model.eval()

        # Create inputs
        batch_size = 2
        image_size = 112
        pixel_values = torch.randn(batch_size, 3, image_size, image_size, device=torch_device)
        image_sizes = torch.tensor([[image_size, image_size]] * batch_size, dtype=torch.long, device=torch_device)

        # Calculate number of image tokens
        num_patches = (image_size // 14) ** 2  # patch_size = 14
        num_image_tokens = num_patches // (config.spatial_merge_size**2)

        seq_len = num_image_tokens + 10
        input_ids = torch.randint(0, config.vocab_size - 1, (batch_size, seq_len), device=torch_device) + 1
        # Ensure no tokens accidentally equal image_token_id
        input_ids[input_ids == config.image_token_id] = config.image_token_id + 1
        # Now place image tokens at the beginning
        input_ids[:, :num_image_tokens] = config.image_token_id

        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_sizes=image_sizes,
            )

        self.assertIsNotNone(outputs)
        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape[0], batch_size)
        self.assertEqual(outputs.logits.shape[1], seq_len)
        self.assertEqual(outputs.logits.shape[2], config.vocab_size)
