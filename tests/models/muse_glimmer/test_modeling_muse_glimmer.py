# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch MuseGlimmer model."""

import copy
import os
import unittest

from transformers import MuseGlimmerConfig, is_torch_available
from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig, MuseGlimmerVisionConfig
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        MuseGlimmerForConditionalGeneration,
        MuseGlimmerModel,
    )


MUSE_GLIMMER_CHECKPOINT_DIR = os.environ.get("MUSE_GLIMMER_CHECKPOINT_DIR", "/raid/pablo/muse_glimmer_early/muse_glimmer-hf")


class MuseGlimmerVision2TextModelTester(VLMModelTester):
    if is_torch_available():
        base_model_class = MuseGlimmerModel
        config_class = MuseGlimmerConfig
        text_config_class = MuseGlimmerTextConfig
        vision_config_class = MuseGlimmerVisionConfig
        conditional_generation_class = MuseGlimmerForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("num_image_tokens", 1)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("patch_temporal", 2)
        kwargs.setdefault("merge_size", 1)
        kwargs.setdefault("layer_types", ["full_attention", "sliding_attention"])
        kwargs.setdefault("pos_emb_height", 4)
        kwargs.setdefault("pos_emb_width", 4)
        kwargs.setdefault("intermediate_size", 37)
        kwargs.setdefault("projector_hidden_size", 32)
        super().__init__(parent, **kwargs)
        self.image_grid_thw = (1, 1, 1)
        self.out_hidden_size = self.hidden_size * self.merge_size**2

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.video_token_id}

    def get_vision_config(self):
        # `layer_types` is shared with the text config by name, but the vision tower uses
        # "window_attention" instead of "sliding_attention".
        config = super().get_vision_config()
        config.layer_types = ["window_attention"] * (config.num_hidden_layers - 1) + ["full_attention"]
        return config

    def create_pixel_values(self):
        grid_t, grid_h, grid_w = self.image_grid_thw
        num_patches = self.batch_size * grid_t * grid_h * grid_w
        return floats_tensor([num_patches, self.patch_temporal * self.num_channels * self.patch_size**2])

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        return {"image_grid_thw": torch.tensor([list(self.image_grid_thw)] * self.batch_size, device=torch_device)}


@require_torch
class MuseGlimmerVision2TextModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MuseGlimmerVision2TextModelTester

    def test_reverse_loading_mapping(self):
        # The vendor checkpoint layout is defined relative to the `model.` prefix, which the base
        # MuseGlimmerModel serializes without.
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_mismatching_num_image_tokens(self):
        # Overwritten -- MuseGlimmer packs patches along the first `pixel_values` dim, so removing an image
        # means dropping its patch rows and its `image_grid_thw` row together.
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        patches_per_image = input_dict["pixel_values"].shape[0] // input_dict["image_grid_thw"].shape[0]
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)

            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][:-patches_per_image]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][:-1]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)


@slow
@require_torch_accelerator
class MuseGlimmerIntegrationTest(unittest.TestCase):
    EXPECTED_TEXT_PREFIX = " to find your gift. The purpose of life is to give it away."
    EXPECTED_IMAGE_PREFIX = " two cats lying on a pink blanket. "

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def get_model_and_processor(cls):
        model = MuseGlimmerForConditionalGeneration.from_pretrained(
            MUSE_GLIMMER_CHECKPOINT_DIR, dtype=torch.bfloat16, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(MUSE_GLIMMER_CHECKPOINT_DIR)
        return model, processor

    def test_text_generation_matches_reference(self):
        # The reference implementation tokenizes raw completions as [bos] + encode(prompt).
        model, processor = self.get_model_and_processor()
        tokenizer = processor.tokenizer

        prompt = "The meaning of life is"
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        input_ids = torch.tensor([[tokenizer.bos_token_id] + prompt_ids], device=torch_device)

        output = model.generate(input_ids=input_ids, max_new_tokens=30, do_sample=False)
        completion = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
        self.assertEqual(completion[: len(self.EXPECTED_TEXT_PREFIX)], self.EXPECTED_TEXT_PREFIX)

    def test_image_generation_matches_reference(self):
        # The reference implementation tokenizes image completions as
        # [bos] + [patch] * num_vision_tokens + encode(prompt), with no image start/end wrappers.
        from PIL import Image

        model, processor = self.get_model_and_processor()
        tokenizer = processor.tokenizer

        image = Image.open(os.path.join(os.path.dirname(MUSE_GLIMMER_CHECKPOINT_DIR), "test_images", "cats.jpg")).convert(
            "RGB"
        )
        image_inputs = processor.image_processor(images=[image], return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        self.assertEqual(image_grid_thw.tolist(), [[1, 34, 46]])

        num_vision_tokens = int(image_grid_thw.prod(dim=-1).sum() // processor.image_processor.merge_size**2)
        self.assertEqual(num_vision_tokens, 391)

        prompt_ids = tokenizer("In this photo we can see", add_special_tokens=False).input_ids
        input_ids = torch.tensor(
            [[tokenizer.bos_token_id] + [model.config.image_token_id] * num_vision_tokens + prompt_ids],
            device=torch_device,
        )

        output = model.generate(
            input_ids=input_ids,
            pixel_values=image_inputs["pixel_values"].to(torch_device, torch.bfloat16),
            image_grid_thw=image_grid_thw.to(torch_device),
            max_new_tokens=60,
            do_sample=False,
        )
        completion = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
        self.assertEqual(completion[: len(self.EXPECTED_IMAGE_PREFIX)], self.EXPECTED_IMAGE_PREFIX)
