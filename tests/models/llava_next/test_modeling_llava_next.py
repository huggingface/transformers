# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Llava-NeXT model."""

import unittest

import pytest
import requests
from huggingface_hub import hf_hub_download

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    CLIPVisionConfig,
    LlamaConfig,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_bitsandbytes,
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import check_torch_load_is_safe

from ...test_modeling_common import (
    floats_tensor,
    ids_tensor,
)
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches


if is_vision_available():
    from PIL import Image


class LlavaNextVisionText2TextModelTester(VLMModelTester):
    base_model_class = LlavaNextModel
    config_class = LlavaNextConfig
    conditional_generation_class = LlavaNextForConditionalGeneration
    text_config_class = LlamaConfig
    vision_config_class = CLIPVisionConfig
    all_model_classes = (LlavaNextModel, LlavaNextForConditionalGeneration)

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Use a distinct image_token_index that won't conflict with pad_token_id (0) or random tokens
        self.image_token_index = self.vocab_size - 1

        # LlavaNext needs specific grid pinpoints that match our test image size
        # With image_grid_pinpoints=[[image_size, image_size]] and vision_config.image_size=image_size:
        # num_patches = (image_size/image_size)*(image_size/image_size) + 1 = 2 patches per image
        self.num_patches_per_image = 2  # 1 grid patch + 1 base patch

        # Calculate num_image_tokens based on LlavaNext's pack_image_features logic:
        # - base_image_feature: (image_size/patch_size)^2 tokens
        # - grid patches (1 patch with grid_shape 1x1):
        #   height = width = image_size/patch_size
        #   After unpad + newline: height * (width + 1) tokens
        # Total per image: base_tokens + grid_tokens
        tokens_per_patch = (self.image_size // self.patch_size) ** 2
        height = width = self.image_size // self.patch_size
        grid_tokens = height * (width + 1)  # includes newline token
        self.num_image_tokens = tokens_per_patch + grid_tokens

        # Ensure seq_length is large enough to hold image tokens plus some text
        self.seq_length = self.num_image_tokens + 7

    def get_config(self):
        config = super().get_config()
        # Set grid pinpoints compatible with our small test image size
        config.image_grid_pinpoints = [[self.image_size, self.image_size]]
        return config

    def prepare_config_and_inputs(self):
        # LlavaNext expects pixel_values with shape (batch_size, num_patches, channels, height, width)
        # unlike most other VLMs so we have to override
        config = self.get_config()

        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_patches_per_image,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        return config, input_ids, None, input_mask, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, token_type_ids, input_mask, pixel_values = self.prepare_config_and_inputs()

        input_ids[input_ids == config.image_token_index] = self.bos_token_id
        input_ids[:, : self.num_image_tokens] = config.image_token_index

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "pixel_values": pixel_values,
            "image_sizes": torch.tensor([[self.image_size, self.image_size]] * self.batch_size),
        }
        return config, inputs_dict


@require_torch
class LlavaNextForConditionalGenerationModelTest(VLMModelTest, unittest.TestCase):
    """
    Model tester for `LlavaNextForConditionalGeneration`.
    """

    model_tester_class = LlavaNextVisionText2TextModelTester
    test_resize_embeddings = False

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing(self):
        super().test_training_gradient_checkpointing()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        super().test_training_gradient_checkpointing_use_reentrant_false()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        super().test_training_gradient_checkpointing_use_reentrant_true()

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass


@require_torch
class LlavaNextForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        self.image = Image.open(requests.get(url, stream=True).raw)

        self.prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )

        inputs = self.processor(images=self.image, text=self.prompt, return_tensors="pt").to(torch_device)

        # verify inputs against original implementation
        filepath = hf_hub_download(
            repo_id="nielsr/test-image",
            filename="llava_1_6_input_ids.pt",
            repo_type="dataset",
        )
        check_torch_load_is_safe()
        original_input_ids = torch.load(filepath, map_location="cpu", weights_only=True)
        # replace -200 by image_token_index (since we use token ID = 32000 for the image token)
        # remove image token indices because HF impl expands image tokens `image_seq_length` times
        original_input_ids = original_input_ids[original_input_ids != -200]
        observed_input_ids = inputs.input_ids[inputs.input_ids != model.config.image_token_index]
        assert original_input_ids[0].tolist() == observed_input_ids[0].tolist()

        filepath = hf_hub_download(
            repo_id="nielsr/test-image",
            filename="llava_1_6_pixel_values.pt",
            repo_type="dataset",
        )
        check_torch_load_is_safe()
        original_pixel_values = torch.load(filepath, map_location="cpu", weights_only=True)
        assert torch.allclose(
            original_pixel_values, inputs.pixel_values.to(device="cpu", dtype=original_pixel_values.dtype)
        )

        # verify generation
        output = model.generate(**inputs, max_new_tokens=100)
        EXPECTED_DECODED_TEXT = '[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot that displays values for multiple quantitative variables represented on axes starting from the same point. This particular radar chart is showing the performance of various models or systems across different metrics or datasets.\n\nThe chart is divided into several sections, each representing a different model or dataset. The axes represent different metrics or datasets, such as "MMM-Vet," "MMM-Bench," "L'

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch(self):
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        cats_image = Image.open(requests.get(url, stream=True).raw)

        inputs = self.processor(
            images=[self.image, cats_image],
            text=[self.prompt, self.prompt],
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ['[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot that displays', '[INST]  \nWhat is shown in this image? [/INST] The image shows two cats lying on a pink surface, which appears to be a couch or a cush']  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_unk_token(self):
        # related to (#29835)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )

        prompt_with_unk = "[INST] <image>\nWhat is shown in this <unk> image? [/INST]"
        inputs = self.processor(images=self.image, text=prompt_with_unk, return_tensors="pt")

        # verify single forward pass
        inputs = inputs.to(torch_device)
        with torch.no_grad():
            output = model(**inputs)

        # verify generation
        output = model.generate(**inputs, max_new_tokens=40)
        EXPECTED_DECODED_TEXT = '[INST]  \nWhat is shown in this   image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot that displays values for multiple quantitative variables represented on axes starting from the same point. This particular radar chart'  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch_different_resolutions(self):
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        lowres_url = "https://4.img-dpreview.com/files/p/TS560x560~forums/56876524/03975b28741443319e9a94615e35667e"
        cats_image = Image.open(requests.get(url, stream=True).raw)
        lowres_img = Image.open(requests.get(lowres_url, stream=True).raw)

        inputs = self.processor(
            images=[lowres_img, cats_image], text=[self.prompt, self.prompt], return_tensors="pt", padding=True
        ).to(torch_device)
        pixel_values = inputs["pixel_values"]

        # verify pixel values are padded correctly with 0 when one image has more num_patches than the other
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=model.config.image_grid_pinpoints,
                patch_size=model.config.vision_config.image_size,
            )
            for imsize in inputs["image_sizes"]
        ]
        for pix_val, num_patch in zip(pixel_values, image_num_patches):
            self.assertTrue(torch.all(pix_val[num_patch:] == 0))  # pad on the right
            for i in range(num_patch):
                self.assertFalse(torch.all(pix_val[i : i + 1] == 0))  # no padding expected in any of patches

        # verify generation
        output = model.generate(**inputs, max_new_tokens=50)
        EXPECTED_DECODED_TEXT = "[INST]  \nWhat is shown in this image? [/INST] The image shows two deer, likely fawns, in a grassy area with trees in the background. The setting appears to be a forest or woodland, and the photo is taken during what seems to be either dawn or dusk, given"
        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch_matches_single(self):
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        lowres_url = "https://4.img-dpreview.com/files/p/TS560x560~forums/56876524/03975b28741443319e9a94615e35667e"
        cats_image = Image.open(requests.get(url, stream=True).raw)
        lowres_img = Image.open(requests.get(lowres_url, stream=True).raw)

        inputs_batched = self.processor(
            images=[lowres_img, cats_image], text=[self.prompt, self.prompt], return_tensors="pt", padding=True
        ).to(torch_device)

        inputs_single = self.processor(images=lowres_img, text=self.prompt, return_tensors="pt", padding=True).to(
            torch_device
        )

        # verify generation
        output_batched = model.generate(**inputs_batched, max_new_tokens=50)
        output_single = model.generate(**inputs_single, max_new_tokens=50)
        self.assertEqual(
            self.processor.decode(output_batched[0], skip_special_tokens=True),
            self.processor.decode(output_single[0], skip_special_tokens=True),
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_full_vision_state_selection(self):
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )
        # test that changing `strategy` won't error out
        model.vision_feature_select_strategy = "full"

        inputs = self.processor(text=self.prompt, images=self.image, return_tensors="pt").to(model.device)

        # verify generation
        output = model.generate(**inputs, max_new_tokens=30)
        EXPECTED_DECODED_TEXT = '[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot that displays values for multiple quantitative variables represented on axes'  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_granite_vision(self):
        """
        Check the expected output of a granite vision model, which leverages
        multiple vision feature layers and a visual encoder with no CLS (siglip).
        """
        granite_model_path = "ibm-granite/granite-vision-3.1-2b-preview"
        model = LlavaNextForConditionalGeneration.from_pretrained(granite_model_path)
        self.processor = AutoProcessor.from_pretrained(granite_model_path)
        prompt = "<|user|>\n<image>\nWhat is shown in this image?\n<|assistant|>\n"
        inputs = self.processor(text=prompt, images=self.image, return_tensors="pt").to(model.device)

        # verify generation
        output = model.generate(**inputs, max_new_tokens=30)
        EXPECTED_DECODED_TEXT = "<|user|>\n\nWhat is shown in this image?\n<|assistant|>\nThe image displays a radar chart comparing the performance of various machine learning models."  # fmt: skip
        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
