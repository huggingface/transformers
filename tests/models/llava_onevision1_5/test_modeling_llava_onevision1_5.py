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
"""Testing suite for the PyTorch Llava-OneVision-1.5 model."""

import copy
import json
import tempfile
import unittest

from transformers import (
    AutoProcessor,
    LlavaOnevision15Config,
    LlavaOnevision15ForConditionalGeneration,
    LlavaOnevision15Model,
    LlavaOnevision15VisionConfig,
    LlavaOnevision15VisionModel,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    require_torch_greater_or_equal,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class LlavaOnevision15ModelTester(VLMModelTester):
    base_model_class = LlavaOnevision15Model
    config_class = LlavaOnevision15Config
    conditional_generation_class = LlavaOnevision15ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 5)
        kwargs.setdefault("video_token_id", 6)
        kwargs.setdefault("vision_start_token_id", 3)
        kwargs.setdefault("vision_end_token_id", 4)
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("intermediate_size", 37)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("vision_hidden_size", 32)
        kwargs.setdefault("vision_intermediate_size", 37)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("num_heads", 2)
        kwargs.setdefault("out_hidden_size", 32)
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("temporal_patch_size", 1)
        kwargs.setdefault("spatial_merge_size", 2)
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("vision_hidden_act", "gelu")
        super().__init__(parent, **kwargs)

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.video_token_id, self.vision_start_token_id, self.vision_end_token_id}

    def create_pixel_values(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return floats_tensor(
            [batch_size * (self.image_size // self.patch_size) ** 2, self.num_channels * self.patch_size**2]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[:, 0] = self.vision_start_token_id
        input_ids[:, 1 : 1 + self.num_image_tokens] = self.image_token_id
        input_ids[:, 1 + self.num_image_tokens] = self.vision_end_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        side = self.image_size // self.patch_size
        return {"image_grid_thw": torch.tensor([[1, side, side]] * batch_size, device=torch_device)}


@require_torch
class LlavaOnevision15ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = LlavaOnevision15ModelTester

    def test_tied_checkpoint_roundtrip(self):
        """Root-level tying must survive saving a checkpoint with a shared output head."""
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        values = config.to_dict()
        values.pop("transformers_version", None)
        values["tie_word_embeddings"] = True
        values["text_config"]["tie_word_embeddings"] = False
        config = LlavaOnevision15Config.from_dict(values)
        self.assertTrue(config.tie_word_embeddings)
        model = LlavaOnevision15ForConditionalGeneration(config).to(torch_device).eval()
        self.assertIs(model.lm_head.weight, model.get_input_embeddings().weight)
        with torch.no_grad():
            expected = model(**inputs).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded, info = LlavaOnevision15ForConditionalGeneration.from_pretrained(
                directory, output_loading_info=True
            )
            self.assertFalse(any(info.values()), info)
            self.assertIs(reloaded.lm_head.weight, reloaded.get_input_embeddings().weight)
            reloaded = reloaded.to(torch_device).eval()
            with torch.no_grad():
                torch.testing.assert_close(reloaded(**inputs).logits, expected)
        values["transformers_version"] = "4.58.0"
        self.assertFalse(LlavaOnevision15Config.from_dict(values).tie_word_embeddings)

    def test_vision_loading_from_full_checkpoint(self):
        """Extract the vision tower from a full checkpoint using the published legacy config type."""
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = LlavaOnevision15ForConditionalGeneration(config).to(torch_device).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            config_path = f"{directory}/config.json"
            with open(config_path, encoding="utf-8") as f:
                values = json.load(f)
            values["vision_config"]["model_type"] = "rice_vit"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(values, f)
            vision, info = LlavaOnevision15VisionModel.from_pretrained(directory, output_loading_info=True)
            self.assertEqual(vision.config.model_type, "llava_onevision1_5_vision")
            self.assertFalse(info["missing_keys"], info)
            self.assertFalse(info["mismatched_keys"], info)
            self.assertFalse(info["error_msgs"], info)
            for name, parameter in model.model.visual.state_dict().items():
                torch.testing.assert_close(vision.state_dict()[name].to(torch_device), parameter)
            vision = vision.to(torch_device).eval()
            with torch.no_grad():
                expected = model.model.visual(inputs["pixel_values"], inputs["image_grid_thw"]).pooler_output
                actual = vision(inputs["pixel_values"], inputs["image_grid_thw"]).pooler_output
            torch.testing.assert_close(actual, expected)

    def test_generate_multiple_sequences_preserves_image_features(self):
        """Beam and sample expansion must preserve the image features associated with each prompt."""
        config = self.model_tester.get_config()
        config.vision_config.spatial_merge_size = 2
        model = LlavaOnevision15ForConditionalGeneration(config).to(torch_device).eval()
        input_ids = torch.tensor(
            [[10, config.vision_start_token_id, *([config.image_token_id] * 4), config.vision_end_token_id, 11]],
            device=torch_device,
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": floats_tensor([16, 3 * config.vision_config.patch_size**2]),
            "image_grid_thw": torch.tensor([[1, 4, 4]], device=torch_device),
        }
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1]
            for generation_kwargs, expansion_size in (
                ({"do_sample": False, "num_beams": 2}, 2),
                ({"do_sample": True, "temperature": 1e-6, "num_return_sequences": 3}, 3),
            ):
                with self.subTest(**generation_kwargs):
                    output = model.generate(
                        **inputs,
                        **generation_kwargs,
                        max_new_tokens=2,
                        output_logits=True,
                        return_dict_in_generate=True,
                    )
                    torch.testing.assert_close(
                        output.logits[0], logits.expand(expansion_size, -1), atol=1e-5, rtol=1e-4
                    )

    @slow
    @require_torch_greater_or_equal("2.11")
    def test_export_preserves_class_tokens_and_frame_isolation(self):
        from transformers.exporters import DynamoConfig, DynamoExporter
        from transformers.exporters.utils import prepare_for_export

        config = LlavaOnevision15VisionConfig(
            hidden_size=32, out_hidden_size=32, intermediate_size=48, depth=2, num_heads=4
        )
        model = LlavaOnevision15VisionModel(config).to(torch_device).eval()
        # One image and two video frames, with 16 patches per frame.
        inputs = {
            "pixel_values": floats_tensor([48, 3 * 14 * 14]),
            "grid_thw": torch.tensor([[1, 4, 4], [2, 4, 4]], device=torch_device),
        }
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                sample = copy.deepcopy(inputs)
                exported = DynamoExporter().export(model, sample, config=DynamoConfig(dynamic=dynamic))
                # Replay with new pixels and aspect ratio, retaining 16 patches per frame.
                sample["pixel_values"] = floats_tensor([48, 3 * 14 * 14])
                sample["grid_thw"] = torch.tensor([[1, 2, 8], [2, 2, 8]], device=torch_device)
                model, sample, _ = prepare_for_export(model, sample)
                with torch.no_grad():
                    actual = exported.module()(**sample)
                    expected = model(sample["pixel_values"], sample["grid_thw"])
                    separate = torch.cat(
                        [
                            model(frame, sample["grid_thw"][:1]).pooler_output
                            for frame in sample["pixel_values"].split(16)
                        ]
                    )
                torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)
                torch.testing.assert_close(actual.pooler_output, expected.pooler_output)
                torch.testing.assert_close(actual.pooler_output, separate)

    def test_standalone_save_load_and_frame_isolation(self):
        config = LlavaOnevision15VisionConfig(
            hidden_size=32, out_hidden_size=32, intermediate_size=48, depth=2, num_heads=4
        )
        model = LlavaOnevision15VisionModel(config).to(torch_device).eval()
        pixels = floats_tensor([40, 3 * 14 * 14])
        # One image and two video frames must attend independently, each with its own class token.
        grid = torch.tensor([[1, 4, 4], [2, 2, 6]], device=torch_device)
        frame_grids = grid[[0, 1, 1]].clone()
        frame_grids[:, 0] = 1
        with torch.no_grad():
            batched = model(pixels, grid).pooler_output
            separate = torch.cat(
                [
                    model(part, frame_grid.unsqueeze(0)).pooler_output
                    for part, frame_grid in zip(pixels.split([16, 12, 12]), frame_grids)
                ]
            )
        torch.testing.assert_close(batched, separate)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = LlavaOnevision15VisionModel.from_pretrained(directory).to(torch_device).eval()
            with torch.no_grad():
                torch.testing.assert_close(reloaded(pixels, grid).pooler_output, batched)

    def test_class_token_gradients_with_checkpointing(self):
        self._check_class_token_gradients_with_checkpointing(None, torch.float32)

    @require_flash_attn
    @require_torch_gpu
    def test_class_token_gradients_with_flash_attention(self):
        self._check_class_token_gradients_with_checkpointing("flash_attention_2", torch.bfloat16)

    def _check_class_token_gradients_with_checkpointing(self, attn_implementation, dtype):
        config = LlavaOnevision15VisionConfig(
            hidden_size=32, out_hidden_size=32, intermediate_size=48, depth=2, num_heads=4
        )
        model = (
            LlavaOnevision15VisionModel._from_config(config, attn_implementation=attn_implementation, dtype=dtype)
            .to(torch_device)
            .train()
        )
        pixels = floats_tensor([16, 3 * 14 * 14])
        grid = torch.tensor([[1, 4, 4]], device=torch_device)
        model(pixels, grid).pooler_output.square().sum().backward()
        expected = {name: p.grad.clone() for name, p in model.named_parameters()}
        for reentrant in (True, False):
            model.zero_grad(set_to_none=True)
            model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": reentrant})
            model(pixels, grid).pooler_output.square().sum().backward()
            for name, p in model.named_parameters():
                torch.testing.assert_close(p.grad, expected[name])
            self.assertGreater(model.class_embedding.grad.abs().sum().item(), 0)
            self.assertGreater(model.class_pos_emb.grad.abs().sum().item(), 0)


@require_torch
class LlavaOnevision15IntegrationTest(unittest.TestCase):
    model_id = "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, padding_side="left")
        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
                        ),
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _check_image_generation(self, attn_implementation):
        model = LlavaOnevision15ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device, attn_implementation=attn_implementation
        )
        inputs = self.processor.apply_chat_template(
            self.messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = self.processor.batch_decode(output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        expected = Expectations(
            {
                (None, None): [
                    "The image depicts a snow leopard walking on a snowy surface. The snow leopard is characterized by its thick"
                ]
            }
        ).get_expectation()
        self.assertEqual(text, expected)

    @slow
    @require_torch_gpu
    def test_image_generation(self):
        self._check_image_generation("sdpa")

    @slow
    @require_torch_gpu
    @require_flash_attn
    def test_image_generation_flash_attention_2(self):
        self._check_image_generation("flash_attention_2")

    @slow
    @require_torch_gpu
    def test_batched_generation(self):
        model = LlavaOnevision15ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        )
        text_only = [{"role": "user", "content": [{"type": "text", "text": "What is one plus one?"}]}]
        conversations = [self.messages, self.messages, text_only]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            processor_kwargs={"padding": True},
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = self.processor.batch_decode(output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        expected = Expectations(
            {
                (None, None): [
                    "The image depicts a snow leopard walking on a snowy surface. The snow leopard is characterized by its thick",
                    "The image depicts a snow leopard walking on a snowy surface. The snow leopard is characterized by its thick",
                    "One plus one equals two.",
                ]
            }
        ).get_expectation()
        self.assertEqual(text, expected)

    @slow
    @require_torch_gpu
    def test_video_generation(self):
        model = LlavaOnevision15ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tiny_video.mp4"
                        ),
                    },
                    {"type": "text", "text": "Describe the video."},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            processor_kwargs={"num_frames": 2},
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = self.processor.batch_decode(output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        expected = Expectations(
            {
                (None, None): [
                    "The image is a black and white photograph of a person lying on their back on a mat, with"
                ]
            }
        ).get_expectation()
        self.assertEqual(text, expected)

    @slow
    @require_torch_gpu
    def test_base_checkpoint_loading(self):
        model, info = LlavaOnevision15Model.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device, output_loading_info=True
        )
        self.assertFalse(info["missing_keys"])
        self.assertFalse(info["mismatched_keys"])
        self.assertFalse(info["error_msgs"])
        self.assertLessEqual(set(info["unexpected_keys"]), {"lm_head.weight"})
        inputs = self.processor.apply_chat_template(
            self.messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            output = model(**inputs)
        self.assertEqual(
            output.last_hidden_state.shape, (*inputs.input_ids.shape, model.config.text_config.hidden_size)
        )
