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
"""Testing suite for the PyTorch LocateAnything model."""

import unittest

import requests

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LocateAnythingConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
from transformers.testing_utils import cleanup, require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import LocateAnythingForConditionalGeneration, LocateAnythingModel

if is_vision_available():
    from PIL import Image


class LocateAnythingVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        # Keep the placeholder id low so vocab-resize tests (which shrink the vocab) don't clamp it away.
        image_token_index=1,
        is_training=True,
        text_config={
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "vocab_size": 99,
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 512,
            "rope_theta": 10000,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "pad_token_id": 4,
        },
        vision_config={
            "model_type": "moonvit",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "patch_size": 14,
            "init_pos_emb_height": 4,
            "init_pos_emb_width": 4,
            "merge_kernel_size": [2, 2],
        },
    ):
        self.parent = parent
        self.image_token_index = image_token_index
        self.text_config = text_config
        self.vision_config = vision_config
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training

        # Each image is a 2x2 patch grid; the [2, 2] merge kernel folds it into a single visual token.
        self.image_grid_hw = (2, 2)
        self.num_patches_per_image = self.image_grid_hw[0] * self.image_grid_hw[1]
        self.num_image_tokens_per_image = 1
        self.seq_length = seq_length + self.num_image_tokens_per_image

        self.pad_token_id = text_config["pad_token_id"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]

    def get_config(self):
        return LocateAnythingConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_index=self.image_token_index,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = self.vision_config["patch_size"]
        # MoonViT consumes packed patches: (num_images * grid_h * grid_w, channels, patch, patch).
        pixel_values = floats_tensor(
            [self.batch_size * self.num_patches_per_image, self.num_channels, patch_size, patch_size]
        )
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[input_ids == self.image_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens_per_image] = self.image_token_index

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_hws": torch.tensor([list(self.image_grid_hw)] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class LocateAnythingModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `LocateAnythingModel` and `LocateAnythingForConditionalGeneration`.

    `GenerationTesterMixin` is not applicable: `generate()` is overridden with the model's Parallel
    Box Decoding loop, which has its own (tokenizer-driven) API instead of the standard one.
    """

    all_model_classes = (LocateAnythingModel, LocateAnythingForConditionalGeneration) if is_torch_available() else ()
    # `generate()` is fully overridden by the Parallel Box Decoding loop (tokenizer-driven API);
    # the standard `GenerationMixin` test surface does not apply.
    all_generative_model_classes = ()
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = LocateAnythingVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LocateAnythingConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_auto_model_for_image_text_to_text_from_config(self):
        config = self.model_tester.get_config()
        model = AutoModelForImageTextToText.from_config(config)
        self.assertIsInstance(model, LocateAnythingForConditionalGeneration)

    @require_vision
    def test_auto_processor_and_image_processor_mappings(self):
        from transformers import LocateAnythingImageProcessor, LocateAnythingProcessor

        self.assertIs(PROCESSOR_MAPPING[LocateAnythingConfig], LocateAnythingProcessor)
        self.assertEqual(IMAGE_PROCESSOR_MAPPING[LocateAnythingConfig]["torchvision"], LocateAnythingImageProcessor)

    def test_reverse_loading_mapping(self):
        # As with the Llava-family mappings, the `model.` re-prefixing renames are only visible on
        # the model with head; the base model round-trips via `base_model_prefix`.
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_get_placeholder_mask(self):
        config = self.model_tester.get_config()
        model = LocateAnythingForConditionalGeneration(config)
        input_ids = torch.tensor([[5, config.image_token_index, 2]])
        inputs_embeds = model.get_input_embeddings()(input_ids)
        image_features = torch.ones(1, config.text_config.hidden_size)

        image_mask = model.model.get_placeholder_mask(input_ids, inputs_embeds, image_features)

        # The inherited Llava `get_placeholder_mask` returns a `(batch, seq_len, 1)` mask that broadcasts
        # over the hidden dimension in `masked_scatter`.
        self.assertEqual(image_mask.shape, (*input_ids.shape, 1))
        self.assertEqual(image_mask[..., 0].sum(), 1)

        with self.assertRaises(ValueError):
            model.model.get_placeholder_mask(input_ids, inputs_embeds, torch.ones(2, config.text_config.hidden_size))


@require_torch
class LocateAnythingIntegrationTest(unittest.TestCase):
    model_id = "nvidia/LocateAnything-3B"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_vision
    def test_real_model_slow_generation(self):
        # Expected strings were captured from the original `trust_remote_code` model
        # (transformers==4.57.1) on the same inputs. The original remote code does not import on
        # current Transformers, so instead of comparing at runtime we guard numerical equivalence of
        # the ported model in pure auto-regressive ("slow") decoding against those captured outputs.
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=False)
        dtype = torch.bfloat16
        model = LocateAnythingForConditionalGeneration.from_pretrained(
            self.model_id, trust_remote_code=False, torch_dtype=dtype
        ).to(torch_device)
        model.eval()

        # Note: MoonViT packs all image patches into a single sequence. Without flash-attention the
        # sdpa fallback attends per image chunk, so a moderately sized image keeps memory bounded.
        cases = [
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
