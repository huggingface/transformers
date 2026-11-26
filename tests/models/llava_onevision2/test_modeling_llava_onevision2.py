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

import tempfile
import unittest

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    LlavaOnevision2Config,
    LlavaOnevision2ForConditionalGeneration,
    LlavaOnevision2Model,
    LlavaOnevision2Processor,
    LlavaOnevision2VisionConfig,
    LlavaOnevision2VisionModel,
    Qwen3Config,
    is_torch_available,
)
from transformers.testing_utils import require_torch, require_torch_accelerator, require_vision, slow, torch_device
from transformers.vision_utils import get_vision_position_ids

from ...test_image_processing_common import load_test_image
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class LlavaOnevision2ModelTester(VLMModelTester):
    base_model_class = LlavaOnevision2Model
    conditional_generation_class = LlavaOnevision2ForConditionalGeneration
    config_class = LlavaOnevision2Config
    text_config_class = Qwen3Config
    vision_config_class = LlavaOnevision2VisionConfig

    small_config_overrides = {
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "num_image_tokens": 4,
        "out_hidden_size": 64,
    }

    def create_pixel_values(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        grid_size = self.image_size // self.patch_size
        return floats_tensor([batch_size * grid_size**2, self.num_channels * self.patch_size**2])

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        grid_size = self.image_size // self.patch_size
        return {"image_grid_thw": torch.tensor([[1, grid_size, grid_size]] * batch_size, device=torch_device)}


@require_torch
class LlavaOnevision2ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = LlavaOnevision2ModelTester
    # The first dimension contains spatially merged patches across the batch.
    skip_test_image_features_output_shape = True

    @unittest.skip("The packed vision attention returns hidden states without attention weights.")
    def test_get_image_features_attentions(self):
        pass

    def test_generate_encodes_visuals_once(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = LlavaOnevision2ForConditionalGeneration(config).to(torch_device).eval()
        model.generation_config.eos_token_id = None
        for beams in (1, 3):
            calls = []
            handle = model.model.visual.register_forward_hook(lambda *args: calls.append(1))
            try:
                result = model.generate(**inputs, max_new_tokens=3, do_sample=False, num_beams=beams)
            finally:
                handle.remove()
            batch_size, seq_length = inputs["input_ids"].shape
            self.assertEqual(result.shape, (batch_size, seq_length + 3))
            self.assertEqual(len(calls), 1)

    def test_precomputed_visual_features(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = LlavaOnevision2ForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            text_inputs = {key: inputs[key] for key in ("input_ids", "attention_mask")}
            features = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            logits = model(**text_inputs, mm_encoder_outputs={"image": features}).logits
            torch.testing.assert_close(logits, model(**inputs).logits)
            with self.assertRaises(ValueError):
                model(**inputs, mm_encoder_outputs={"image": features})


@require_torch
class LlavaOnevision2VisionModelTest(unittest.TestCase):
    all_model_classes = (LlavaOnevision2VisionModel,)

    def setUp(self):
        torch.manual_seed(0)
        self.config = LlavaOnevision2ModelTester(self).get_config().vision_config
        self.model = LlavaOnevision2VisionModel(self.config).eval()

    def test_positions_and_hidden_states(self):
        grid = torch.tensor([[1, 4, 4], [1, 2, 4]])
        pixels = torch.randn(24, 12)
        positions = get_vision_position_ids(grid, 2, include_temporal=True)
        with torch.no_grad():
            implicit = self.model(pixels, grid, output_hidden_states=True)
            explicit = self.model(pixels, grid, patch_positions=positions)
        self.assertEqual(implicit.last_hidden_state.shape, (6, 64))
        self.assertEqual(len(implicit.hidden_states), self.config.num_hidden_layers + 1)
        torch.testing.assert_close(implicit.last_hidden_state, explicit.last_hidden_state)

    def test_temporal_window_isolation(self):
        grid = torch.tensor([[8, 2, 2]])
        pixels = torch.randn(32, 12)
        changed = pixels.clone()
        changed[16:] += torch.randn_like(changed[16:])
        with torch.no_grad():
            before = self.model(pixels, grid).last_hidden_state
            after = self.model(changed, grid).last_hidden_state
        torch.testing.assert_close(before[:4], after[:4], rtol=0, atol=0)
        self.assertFalse(torch.equal(before[4:], after[4:]))

    def test_geometry_validation(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(12, 12), torch.tensor([[1, 4, 4]]))
        with self.assertRaises(ValueError):
            self.model(torch.randn(16, 12), torch.tensor([[1, 4, 4]]), patch_positions=torch.zeros(8, 3))

    def test_config_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            config = AutoConfig.from_pretrained(directory)
            model = AutoModel.from_pretrained(directory)
        self.assertIsInstance(config, LlavaOnevision2VisionConfig)
        self.assertIsInstance(model, LlavaOnevision2VisionModel)


@slow
@require_torch
@require_torch_accelerator
@require_vision
class LlavaOnevision2IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"

    def test_image_generation_and_vision_loading(self):
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=False)
        self.assertIsInstance(processor, LlavaOnevision2Processor)
        model, loading_info = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype="auto", device_map=torch_device, output_loading_info=True, trust_remote_code=False
        )
        self.assertFalse(any(loading_info.values()), loading_info)

        image = load_test_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
        )
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What animal is in the image?"}]}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        result = processor.decode(generated[0, inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        self.assertRegex(result.lower(), r"cat|lynx")

        vision_model, loading_info = LlavaOnevision2VisionModel.from_pretrained(
            self.model_id, dtype="auto", device_map=torch_device, output_loading_info=True
        )
        unexpected_keys = loading_info.pop("unexpected_keys")
        self.assertTrue(all(key.startswith(("model.language_model.", "lm_head.")) for key in unexpected_keys))
        self.assertFalse(any(loading_info.values()), loading_info)
        vision_inputs = {name: inputs[name] for name in ("pixel_values", "image_grid_thw", "patch_positions")}
        expected = model.get_image_features(**vision_inputs).last_hidden_state
        result = vision_model(**vision_inputs).last_hidden_state
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
