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
import unittest

import requests
import torch

from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor, is_torch_available
from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
from transformers.testing_utils import require_torch, require_vision, slow, torch_device


if is_torch_available():
    from transformers import LocateAnythingForConditionalGeneration


@require_torch
@require_vision
class LocateAnythingModelTest(unittest.TestCase):
    all_model_classes = (LocateAnythingForConditionalGeneration,) if is_torch_available() else ()

    model_id = "nvidia/LocateAnything-3B"

    def get_tiny_config(self):
        from transformers import LocateAnythingConfig

        return LocateAnythingConfig(
            vision_config={
                "model_type": "moonvit",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "patch_size": 14,
                "init_pos_emb_height": 4,
                "init_pos_emb_width": 4,
            },
            text_config={
                "architectures": ["Qwen2ForCausalLM"],
                "vocab_size": 200,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
            },
            image_token_index=199,
        )

    def test_config(self):
        from transformers import LocateAnythingConfig

        config = LocateAnythingConfig()
        self.assertEqual(config.model_type, "locateanything")
        self.assertEqual(config.vision_config.model_type, "moonvit")
        self.assertEqual(config.text_config.architectures[0], "Qwen2ForCausalLM")
        self.assertEqual(LocateAnythingConfig.from_dict(config.to_dict()).to_dict(), config.to_dict())

    def test_model_can_instantiate_from_tiny_config(self):
        from transformers import LocateAnythingForConditionalGeneration

        model = LocateAnythingForConditionalGeneration(self.get_tiny_config())
        self.assertEqual(model.config.model_type, "locateanything")

    def test_auto_model_for_image_text_to_text_from_config(self):
        from transformers import LocateAnythingForConditionalGeneration

        model = AutoModelForImageTextToText.from_config(self.get_tiny_config())

        self.assertIsInstance(model, LocateAnythingForConditionalGeneration)

    def test_auto_processor_and_image_processor_mappings(self):
        from transformers import LocateAnythingConfig, LocateAnythingImageProcessor, LocateAnythingProcessor

        self.assertIs(PROCESSOR_MAPPING[LocateAnythingConfig], LocateAnythingProcessor)
        self.assertEqual(IMAGE_PROCESSOR_MAPPING[LocateAnythingConfig]["torchvision"], LocateAnythingImageProcessor)

    def test_get_image_features(self):
        from transformers import LocateAnythingForConditionalGeneration

        config = self.get_tiny_config()
        model = LocateAnythingForConditionalGeneration(config)
        model.extract_feature = lambda pixel_values, image_grid_hws: [
            torch.ones(1, config.vision_config.hidden_size * 4, device=pixel_values.device)
        ]

        image_features = model.get_image_features(torch.ones(1, 3, 14, 14))

        self.assertEqual(image_features.shape, (1, config.text_config.hidden_size))

    def test_get_placeholder_mask(self):
        from transformers import LocateAnythingForConditionalGeneration

        config = self.get_tiny_config()
        model = LocateAnythingForConditionalGeneration(config)
        input_ids = torch.tensor([[1, config.image_token_index, 2]])
        inputs_embeds = model.get_input_embeddings()(input_ids)
        image_features = torch.ones(1, config.text_config.hidden_size)

        image_mask = model.get_placeholder_mask(input_ids, inputs_embeds, image_features)

        self.assertEqual(image_mask.shape, inputs_embeds.shape)
        self.assertEqual(image_mask[..., 0].sum(), 1)

        with self.assertRaises(ValueError):
            model.get_placeholder_mask(input_ids, inputs_embeds, torch.ones(2, config.text_config.hidden_size))

    @slow
    def test_real_model_matches_remote_code_generation(self):
        from PIL import Image

        image = Image.open(
            requests.get(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                stream=True,
                timeout=30,
            ).raw
        ).convert("RGB")
        prompt = "Locate a single instance that matches the following description: animal."
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

        remote_processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        local_processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=False)

        remote_text = remote_processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        local_text = local_processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(remote_text, local_text)

        remote_images, remote_videos = remote_processor.process_vision_info(messages)
        local_images, local_videos = local_processor.process_vision_info(messages)
        remote_inputs = remote_processor(
            text=[remote_text], images=remote_images, videos=remote_videos, return_tensors="pt"
        )
        local_inputs = local_processor(
            text=[local_text], images=local_images, videos=local_videos, return_tensors="pt"
        )

        for key in remote_inputs:
            if hasattr(remote_inputs[key], "shape"):
                self.assertEqual(remote_inputs[key].shape, local_inputs[key].shape)

        dtype = torch.bfloat16
        remote_model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=dtype).to(
            torch_device
        )
        local_model = AutoModel.from_pretrained(self.model_id, trust_remote_code=False, torch_dtype=dtype).to(
            torch_device
        )
        remote_model.eval()
        local_model.eval()

        remote_inputs = remote_inputs.to(torch_device)
        local_inputs = local_inputs.to(torch_device)

        remote_output = remote_model.generate(
            pixel_values=remote_inputs["pixel_values"].to(dtype),
            input_ids=remote_inputs["input_ids"],
            attention_mask=remote_inputs["attention_mask"],
            image_grid_hws=remote_inputs.get("image_grid_hws", None),
            tokenizer=remote_processor.tokenizer,
            max_new_tokens=64,
            use_cache=True,
            generation_mode="slow",
            do_sample=False,
            verbose=False,
        )
        local_output = local_model.generate(
            pixel_values=local_inputs["pixel_values"].to(dtype),
            input_ids=local_inputs["input_ids"],
            attention_mask=local_inputs["attention_mask"],
            image_grid_hws=local_inputs.get("image_grid_hws", None),
            tokenizer=local_processor.tokenizer,
            max_new_tokens=64,
            use_cache=True,
            generation_mode="slow",
            do_sample=False,
            verbose=False,
        )

        self.assertEqual(remote_output[0], local_output[0])
