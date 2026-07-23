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
"""Testing suite for the LocateAnything model."""

import copy
import gc
import unittest

import requests
from PIL import Image

from transformers import (
    AutoProcessor,
    LocateAnythingConfig,
    LocateAnythingForConditionalGeneration,
    LocateAnythingModel,
    LocateAnythingVisionConfig,
    LocateAnythingVisionModel,
    is_torch_available,
)
from transformers.testing_utils import backend_empty_cache, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
TEST_PAD_ID = 0
TEST_BOS_ID = 1
TEST_EOS_ID = 2


class LocateAnythingVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=13,
        image_size=28,
        is_training=True,
        text_config={
            "model_type": "qwen2",
            "vocab_size": 99,
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
            "pad_token_id": TEST_PAD_ID,
            "bos_token_id": TEST_BOS_ID,
            "eos_token_id": TEST_EOS_ID,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "patch_size": 14,
            "init_pos_emb_height": 8,
            "init_pos_emb_width": 8,
            "spatial_merge_size": 2,
        },
        image_token_id=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_training = is_training
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.hidden_size = text_config["hidden_size"]
        self.vocab_size = text_config["vocab_size"]
        self.patch_size = vision_config["patch_size"]
        self.merge = vision_config["spatial_merge_size"]
        self.grid = self.image_size // self.patch_size
        self.num_image_tokens = (self.grid // self.merge) ** 2
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return LocateAnythingConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor([self.batch_size * self.grid * self.grid, 3, self.patch_size, self.patch_size])
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1) + 1
        input_ids[input_ids == self.image_token_id] = self.image_token_id - 1
        input_ids[:, : self.num_image_tokens] = self.image_token_id
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, self.grid, self.grid]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class LocateAnythingVisionModelTest(unittest.TestCase):
    all_model_classes = (LocateAnythingVisionModel,) if is_torch_available() else ()

    def test_model(self):
        config = LocateAnythingVisionConfig(
            hidden_size=32,
            intermediate_size=37,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=14,
            init_pos_emb_height=8,
            init_pos_emb_width=8,
            spatial_merge_size=2,
        )
        model = LocateAnythingVisionModel(config).to(torch_device).eval()
        pixel_values = floats_tensor([4, 3, 14, 14])
        image_grid_thw = torch.tensor([[1, 2, 2]], device=torch_device)
        with torch.no_grad():
            output = model(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        self.assertEqual(output.last_hidden_state.shape, (4, config.hidden_size))


@require_torch
class LocateAnythingModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (LocateAnythingModel, LocateAnythingForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = {"image-text-to-text": LocateAnythingForConditionalGeneration}
    _is_composite = True

    def setUp(self):
        self.model_tester = LocateAnythingVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LocateAnythingConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_reverse_loading_mapping(self):
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_mismatching_num_image_tokens(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            curr = copy.deepcopy(input_dict)
            _ = model(**curr)
            one_img = self.model_tester.grid * self.model_tester.grid
            curr["pixel_values"] = curr["pixel_values"][-one_img:, ...]
            curr["image_grid_thw"] = curr["image_grid_thw"][-1:, ...]
            with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
                _ = model(**curr)

    # pixel_values are packed as (num_patches, channels, patch, patch), so generation can't slice them by batch
    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        ignore = ["decoder_input_ids", "decoder_attention_mask", "use_cache", "labels"]
        per_image = self.model_tester.grid * self.model_tester.grid
        filtered = {
            k: v[:batch_size, ...] if isinstance(v, torch.Tensor) else v
            for k, v in inputs_dict.items()
            if k not in ignore
        }
        filtered["pixel_values"] = inputs_dict["pixel_values"][: batch_size * per_image]
        text_gen_config = config.get_text_config(decoder=True)
        if text_gen_config.eos_token_id is not None and text_gen_config.pad_token_id is None:
            text_gen_config.pad_token_id = text_gen_config.eos_token_id
        text_gen_config.eos_token_id = None
        text_gen_config.forced_eos_token_id = None
        return config, filtered


@slow
@require_torch
class LocateAnythingIntegrationTest(unittest.TestCase):
    model_id = "nvidia/LocateAnything-3B"
    revision = MODEL_REVISION
    cases = (
        (
            "coco-cat",
            "https://images.cocodataset.org/val2017/000000039769.jpg",
            "Locate all the instances that matches the following description: cat.",
            "<ref>cat</ref><box><0><115><494><988></box><|im_end|>",
        ),
        (
            "warehouse-industrial-assets",
            "https://live.staticflickr.com/6061/6097691785_925e4687b4_o.jpg",
            "Locate all the instances that matches the following description: "
            "forklift</c>pallet</c>stacked supply boxes</c>warehouse aisle.",
            "<ref>forklift</ref><box><586><126><979><956></box><ref>pallet</ref>"
            "<box><388><113><597><951></box><box><588><191><645><831></box>"
            "<ref>stacked supply boxes</ref><box><322><264><396><826></box>"
            "<box><388><113><597><951></box><box><588><191><645><831></box>"
            "<ref>warehouse aisle</ref><box><0><0><1000><1000></box><|im_end|>",
        ),
    )

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_detection_generation(self):
        model = LocateAnythingForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, dtype=torch.float32
        )
        model = model.to(torch_device).eval()
        for case_name, image_url, question, expected_output in self.cases:
            with self.subTest(case_name=case_name):
                image = Image.open(requests.get(image_url, stream=True, timeout=30).raw).convert("RGB")
                prompt = (
                    "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
                    f"<|im_start|>user\n<image-1>{question}<|im_end|>\n<|im_start|>assistant\n"
                )
                inputs = self.processor(images=image, text=[prompt], return_tensors="pt").to(torch_device)
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=128)
                decoded = self.processor.decode(
                    generated[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=False
                )
                self.assertEqual(decoded, expected_output)
