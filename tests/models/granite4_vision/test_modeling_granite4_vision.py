# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Granite4Vision model."""

import unittest

from transformers import (
    AutoProcessor,
    CLIPVisionConfig,
    Granite4VisionConfig,
    Granite4VisionForConditionalGeneration,
    Granite4VisionModel,
    GraniteConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class Granite4VisionModelTester(VLMModelTester):
    base_model_class = Granite4VisionModel
    config_class = Granite4VisionConfig
    conditional_generation_class = Granite4VisionForConditionalGeneration
    text_config_class = GraniteConfig
    vision_config_class = CLIPVisionConfig

    def __init__(self, parent, **kwargs):
        # Vision hidden_size must be divisible by 64 (QFormer num_attention_heads = hidden_size // 64)
        kwargs.setdefault("hidden_size", 64)
        kwargs.setdefault("intermediate_size", 64)
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("num_hidden_layers", 2)
        # Image/patch sizes: image_side = image_size // patch_size must be divisible by window_side
        kwargs.setdefault("image_size", 8)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("projection_dim", 64)
        kwargs.setdefault("num_patches_per_image", 2)
        # Granite4Vision-specific
        kwargs.setdefault("downsample_rate", "1/2")
        kwargs.setdefault("deepstack_layer_map", [[1, 0]])
        kwargs.setdefault("projector_dropout", 0.0)
        kwargs.setdefault("image_token_index", kwargs.get("image_token_id", 3))

        # Compute num_image_tokens after downsampling:
        # image_side = image_size/patch_size = 4, ds 1/2 -> patches_h = patches_w = 2
        # pinpoints [[8,8]] -> scale 1x1 -> current_h = current_w = 2
        # unpadded = 2*2 = 4, newline = 2, base = 2*2 = 4 -> total = 10
        kwargs.setdefault("num_image_tokens", 10)

        super().__init__(parent, **kwargs)

    def create_pixel_values(self):
        """Granite4Vision expects 5D pixel_values: (batch_size, num_patches, channels, height, width)"""
        return floats_tensor(
            [
                self.batch_size,
                self.num_patches_per_image,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )

    def get_additional_inputs(self, config, input_ids, pixel_values):
        """Granite4Vision requires image_sizes tensor"""
        return {
            "image_sizes": torch.tensor([[self.image_size, self.image_size]] * self.batch_size),
        }

    def get_config(self):
        config = super().get_config()
        config.image_grid_pinpoints = [[self.image_size, self.image_size]]
        config.downsample_rate = self.downsample_rate
        config.deepstack_layer_map = self.deepstack_layer_map
        config.projector_dropout = self.projector_dropout
        config.qformer_config.intermediate_size = 64
        return config


@require_torch
class Granite4VisionModelTest(VLMModelTest, unittest.TestCase):
    """
    Model tester for `Granite4VisionForConditionalGeneration`.
    """

    model_tester_class = Granite4VisionModelTester
    skip_test_image_features_output_shape = True
    test_torch_exportable = False
    # Custom layer-by-layer forward doesn't support output_attentions
    # (GraniteDecoderLayer discards attention weights internally)
    test_attention_outputs = False
    has_attentions = False
    test_all_params_have_gradient = False

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Custom layer-by-layer forward has graph breaks incompatible with fullgraph compile")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip("Blip2QFormerModel in WindowQFormerDownsampler does not support SDPA dispatch")
    def test_can_set_attention_dynamically_composite_model(self):
        pass


@require_torch
class Granite4VisionIntegrationTest(unittest.TestCase):
    model_id = "ibm-granite/granite-vision-4.1-4b"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = Image.open(url_to_local_path(url))

    def make_prompt(self, question):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_integration_test(self):
        model = Granite4VisionForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(
            torch_device
        )

        prompt = self.make_prompt("Describe this image briefly.")
        inputs = self.processor(text=prompt, images=self.image, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]

        EXPECTED_RESPONSE = "The image depicts two cats resting on a pink couch. They are lying in a relaxed, sprawled position, with one cat appearing to be in a"  # fmt: skip
        self.assertEqual(self.processor.decode(new_tokens[0], skip_special_tokens=True), EXPECTED_RESPONSE)

    @slow
    def test_small_model_integration_test_batch(self):
        model = Granite4VisionForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(
            torch_device
        )

        url2 = "http://images.cocodataset.org/val2017/000000001000.jpg"
        image2 = Image.open(url_to_local_path(url2))

        prompt = self.make_prompt("What do you see in this image?")
        inputs = self.processor(
            text=[prompt, prompt],
            images=[self.image, image2],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        responses = self.processor.batch_decode(new_tokens, skip_special_tokens=True)

        self.assertIn("cat", responses[0].lower())
        self.assertIn("tennis", responses[1].lower())

    @slow
    def test_small_model_integration_test_batch_matches_single(self):
        model = Granite4VisionForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(
            torch_device
        )

        prompt = self.make_prompt("What do you see in this image?")

        # Single inference
        inputs_single = self.processor(text=prompt, images=self.image, return_tensors="pt").to(model.device)
        output_single = model.generate(**inputs_single, max_new_tokens=30, do_sample=False)
        decoded_single = self.processor.decode(
            output_single[0, inputs_single["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Batch inference (same image as first in batch)
        url2 = "http://images.cocodataset.org/val2017/000000001000.jpg"
        image2 = Image.open(url_to_local_path(url2))
        inputs_batch = self.processor(
            text=[prompt, prompt],
            images=[self.image, image2],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        output_batch = model.generate(**inputs_batch, max_new_tokens=30, do_sample=False)
        decoded_batch = self.processor.decode(
            output_batch[0, inputs_batch["input_ids"].shape[1] :], skip_special_tokens=True
        )

        self.assertEqual(decoded_single, decoded_batch)
