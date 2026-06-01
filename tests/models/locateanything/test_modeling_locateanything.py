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
        model.model.extract_feature = lambda pixel_values, image_grid_hws: [
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

        image_mask = model.model.get_placeholder_mask(input_ids, inputs_embeds, image_features)

        self.assertEqual(image_mask.shape, inputs_embeds.shape)
        self.assertEqual(image_mask[..., 0].sum(), 1)

        with self.assertRaises(ValueError):
            model.model.get_placeholder_mask(input_ids, inputs_embeds, torch.ones(2, config.text_config.hidden_size))

    @slow
    def test_real_model_slow_generation(self):
        # Expected strings were captured from the original `trust_remote_code` model
        # (transformers==4.57.1) on the same inputs. The original remote code does not import on
        # current Transformers, so instead of comparing at runtime we guard numerical equivalence of
        # the ported model in pure auto-regressive ("slow") decoding against those captured outputs.
        from PIL import Image

        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=False)
        dtype = torch.bfloat16
        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=False, torch_dtype=dtype).to(torch_device)
        model.eval()

        cases = [
            (
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                "Locate a single instance that matches the following description: animal.",
                "<ref>animal</ref><box><501><459><547><541></box><|im_end|>",
            ),
            (
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
                "Locate all instances that match the following description: cat.",
                "<ref>cat</ref><box><0><112><494><988></box><|im_end|>",
            ),
        ]
        for url, prompt, expected in cases:
            image = Image.open(requests.get(url, stream=True, timeout=30).raw).convert("RGB")
            messages = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
            ]
            text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = processor.process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(torch_device)

            output = model.generate(
                pixel_values=inputs["pixel_values"].to(dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=processor.tokenizer,
                max_new_tokens=64,
                use_cache=True,
                generation_mode="slow",
                do_sample=False,
            )
            self.assertEqual(output[0], expected)
