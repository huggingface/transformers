# Copyright (C) 2026 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HunYuanVL model."""

import copy
import unittest

import requests
from huggingface_hub import hf_hub_download

from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    HunYuanVLConfig,
    HunYuanVLForConditionalGeneration,
    HunYuanVLModel,
    HunYuanVLTextConfig,
    HunYuanVLVisionConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_vision,
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


class HunYuanVLVisionText2TextModelTester(VLMModelTester):
    """Build a tiny HunYuanVL config plus matching multimodal inputs for unit tests."""

    base_model_class = HunYuanVLModel
    config_class = HunYuanVLConfig
    text_config_class = HunYuanVLTextConfig
    vision_config_class = HunYuanVLVisionConfig
    conditional_generation_class = HunYuanVLForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("seq_length", 32)
        kwargs.setdefault("vocab_size", 256)
        kwargs.setdefault("hidden_size", 64)
        kwargs.setdefault("intermediate_size", 128)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 4)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 128)
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", 2)
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("rope_theta", 10000.0)
        kwargs.setdefault(
            "rope_parameters", {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 2, 2, 2]}
        )
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("temporal_patch_size", 1)
        kwargs.setdefault("spatial_merge_size", 1)
        kwargs.setdefault("image_size", 64)
        kwargs.setdefault("image_token_id", 5)
        kwargs.setdefault("out_hidden_size", kwargs["hidden_size"])
        kwargs.setdefault("text_hidden_size", kwargs["hidden_size"])
        kwargs.setdefault("max_image_size", kwargs["image_size"])
        kwargs.setdefault("min_image_size", kwargs["image_size"])
        kwargs.setdefault("anyres_vit_max_image_size", kwargs["image_size"])
        grid_hw = kwargs["image_size"] // kwargs["patch_size"]
        # HunYuanVL inserts an extra column per row (newline) and 2 begin/end tokens.
        kwargs.setdefault("num_image_tokens", grid_hw * (grid_hw + 1) + 2)
        kwargs.setdefault("max_vit_seq_len", grid_hw**2)
        super().__init__(parent, **kwargs)
        self.device = torch_device
        self.grid_hw = self.image_size // self.patch_size
        self.num_image_patches = self.grid_hw**2
        self.num_image_placeholder_tokens = self.num_image_tokens

    def get_config(self):
        return HunYuanVLConfig(
            attn_implementation="eager",
            text_config=self.get_text_config().to_dict(),
            vision_config=self.get_vision_config().to_dict(),
            image_token_id=self.image_token_id,
        )

    def create_attention_mask(self, input_ids):
        return torch.ones_like(input_ids, device=torch_device)

    def create_pixel_values(self):
        return floats_tensor(
            [self.batch_size * self.num_image_patches, self.num_channels * self.patch_size * self.patch_size]
        ).to(torch_device)

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[input_ids == self.image_token_id] = config.text_config.pad_token_id
        input_ids[:, : self.num_image_placeholder_tokens] = self.image_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        mm_token_type_ids = torch.zeros_like(input_ids, device=torch_device)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        return {
            "image_grid_thw": torch.tensor([[1, self.grid_hw, self.grid_hw]] * self.batch_size, device=torch_device),
            "mm_token_type_ids": mm_token_type_ids,
        }

    def prepare_config_and_inputs(self):
        config, inputs_dict = self.prepare_config_and_inputs_for_common()
        config.text_config.rope_parameters["mrope_section"] = [2, 2, 2, 2]
        # HunYuanVL uses 4 multimodal RoPE axes: position, width, height, and temporal.
        inputs_dict["position_ids"] = (
            torch.arange(self.seq_length, device=torch_device).view(1, 1, -1).expand(4, self.batch_size, -1)
        )
        return config, inputs_dict


@require_torch
class HunYuanVLModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = HunYuanVLVisionText2TextModelTester
    test_all_params_have_gradient = False
    test_torch_exportable = False
    # HunYuanVL packs all images into one flat patch stream; pixel_values.shape[0] is total patches, not batch size.
    skip_test_image_features_output_shape = True

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        filtered_inputs_dict = {}
        for key, value in inputs_dict.items():
            if key == "pixel_values":
                filtered_inputs_dict[key] = value[: batch_size * self.model_tester.num_image_patches]
            elif key == "image_grid_thw":
                filtered_inputs_dict[key] = value[:batch_size]
            elif key == "position_ids":
                continue
            elif isinstance(value, torch.Tensor):
                filtered_inputs_dict[key] = value[:batch_size, ...]
            else:
                filtered_inputs_dict[key] = value

        text_gen_config = config.get_text_config(decoder=True)
        if text_gen_config.eos_token_id is not None and text_gen_config.pad_token_id is None:
            text_gen_config.pad_token_id = (
                text_gen_config.eos_token_id
                if isinstance(text_gen_config.eos_token_id, int)
                else text_gen_config.eos_token_id[0]
            )
        text_gen_config.eos_token_id = None
        text_gen_config.forced_eos_token_id = None

        return config, filtered_inputs_dict

    def test_auto_model_uses_base_model(self):
        config = self.model_tester.get_config()
        model = AutoModel.from_config(config).to(self.model_tester.device)
        self.assertIsInstance(model, HunYuanVLModel)
        self.assertFalse(hasattr(model, "lm_head"))

    def test_mrope_embeddings_are_built_once_per_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        inputs_dict.pop("position_ids")
        config.text_config.rope_parameters["mrope_section"] = [2, 2, 2, 2]
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)
        model.eval()

        embedding_call_count = 0
        rotary_forward = model.model.language_model.rotary_emb.forward

        def wrapped_rotary_forward(*args, **kwargs):
            nonlocal embedding_call_count
            embedding_call_count += 1
            return rotary_forward(*args, **kwargs)

        model.model.language_model.rotary_emb.forward = wrapped_rotary_forward
        with torch.no_grad():
            model(**inputs_dict)

        self.assertEqual(embedding_call_count, 1)

    def test_model_builds_mrope_position_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)

        position_ids, rope_deltas = model.model.get_rope_index(
            inputs_dict["input_ids"],
            mm_token_type_ids=inputs_dict["mm_token_type_ids"],
            image_grid_thw=inputs_dict["image_grid_thw"],
            attention_mask=inputs_dict["attention_mask"],
        )

        grid_tokens = self.model_tester.grid_hw * (self.model_tester.grid_hw + 1)
        self.assertEqual(position_ids.shape, (4, self.model_tester.batch_size, self.model_tester.seq_length))
        self.assertEqual(rope_deltas.shape, (self.model_tester.batch_size, 1))
        self.assertTrue(
            torch.equal(
                position_ids[1, 0, 1 : 1 + grid_tokens],
                torch.arange(self.model_tester.grid_hw + 1, device=position_ids.device).repeat(
                    self.model_tester.grid_hw
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                position_ids[2, 0, 1 : 1 + grid_tokens],
                torch.arange(self.model_tester.grid_hw, device=position_ids.device).repeat_interleave(
                    self.model_tester.grid_hw + 1
                ),
            )
        )

    def test_legacy_xdrope_section_normalizes_to_mrope_section(self):
        text_config = HunYuanVLTextConfig(
            hidden_size=64,
            num_attention_heads=4,
            head_dim=16,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "xdrope_section": [2.0, 2, 2, 2]},
        )

        self.assertEqual(text_config.rope_parameters["mrope_section"], [2, 2, 2, 2])
        self.assertNotIn("xdrope_section", text_config.rope_parameters)

    def test_mismatching_num_image_tokens(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**input_dict)

            curr_input_dict = copy.deepcopy(input_dict)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][: -self.model_tester.num_image_patches]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][:-1]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            input_ids = input_dict["input_ids"][:1]
            attention_mask = input_dict["attention_mask"][:1]
            pixel_values = input_dict["pixel_values"][: self.model_tester.num_image_patches]
            image_grid_thw = input_dict["image_grid_thw"][:1]
            mm_token_type_ids = input_dict["mm_token_type_ids"][:1]

            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=torch.cat([input_ids, input_ids], dim=0),
                    attention_mask=torch.cat([attention_mask, attention_mask], dim=0),
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=torch.cat([mm_token_type_ids, mm_token_type_ids], dim=0),
                )

            _ = model(
                input_ids=torch.cat([input_ids, input_ids], dim=0),
                attention_mask=torch.cat([attention_mask, attention_mask], dim=0),
                pixel_values=torch.cat([pixel_values, pixel_values], dim=0),
                image_grid_thw=torch.cat([image_grid_thw, image_grid_thw], dim=0),
                mm_token_type_ids=torch.cat([mm_token_type_ids, mm_token_type_ids], dim=0),
            )

    def test_prepare_inputs_for_generation_drops_pixel_values_after_prefill(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = HunYuanVLForConditionalGeneration(config).to(self.model_tester.device)
        model.eval()

        prefill_inputs = model.prepare_inputs_for_generation(
            inputs_dict["input_ids"],
            attention_mask=inputs_dict["attention_mask"],
            position_ids=inputs_dict["position_ids"],
            pixel_values=inputs_dict["pixel_values"],
            image_grid_thw=inputs_dict["image_grid_thw"],
            use_cache=True,
            is_first_iteration=True,
        )
        self.assertIs(prefill_inputs["pixel_values"], inputs_dict["pixel_values"])
        self.assertIs(prefill_inputs["image_grid_thw"], inputs_dict["image_grid_thw"])
        self.assertEqual(prefill_inputs["position_ids"].shape, inputs_dict["position_ids"].shape)

        decode_inputs = model.prepare_inputs_for_generation(
            inputs_dict["input_ids"],
            attention_mask=inputs_dict["attention_mask"],
            position_ids=inputs_dict["position_ids"],
            pixel_values=inputs_dict["pixel_values"],
            image_grid_thw=inputs_dict["image_grid_thw"],
            use_cache=True,
            is_first_iteration=False,
            next_sequence_length=1,
        )
        self.assertIsNone(decode_inputs["pixel_values"])
        self.assertIs(decode_inputs["image_grid_thw"], inputs_dict["image_grid_thw"])
        self.assertEqual(decode_inputs["position_ids"].shape, (4, self.model_tester.batch_size, 1))

    def test_batching_equivalence(self, atol=2e-5, rtol=1e-4):
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    # FIXME raushan, no idea why yet
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip("HunYuanVL currently validates the vision path with eager attention.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip("Model doesn't return attentions for vision tower")
    def test_get_image_features_attentions(self):
        pass

    def test_reverse_loading_mapping(self, check_keys_were_modified=True, skip_base_model=True):
        self.skipTest("HunYuanVL keeps multiple legacy vision-tower source prefixes for checkpoint compatibility.")


@require_torch
@require_vision
@slow
class HunYuanVLForConditionalGenerationIntegrationTest(unittest.TestCase):
    model_id = "tencent/HunyuanOCR"
    candy_image_url = url_to_local_path(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
    )
    lowres_image_url = url_to_local_path(
        "https://4.img-dpreview.com/files/p/TS560x560~forums/56876524/03975b28741443319e9a94615e35667e"
    )
    max_new_tokens = 64

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, backend="pil")
        self.processor.tokenizer.padding_side = "left"

        # TODO: use `url-to-local-file`
        image_file = hf_hub_download(
            repo_id="raushan-testing-hf/images_test", filename="llava_v1_5_radar.jpg", repo_type="dataset"
        )
        with Image.open(image_file) as image:
            self.image = image.convert("RGB")
        self.candy_image = Image.open(requests.get(self.candy_image_url, stream=True).raw).convert("RGB")
        self.lowres_image = Image.open(requests.get(self.lowres_image_url, stream=True).raw).convert("RGB")

        self.radar_prompt = "What is shown in this image?"
        self.ocr_prompt = "Extract the text from the image."
        self.candy_prompt = "What animal is on the candy?"
        self.compare_prompt = "What is shown in the first image, and what animal is on the candy in the second image?"
        self.text_prompt = "Briefly explain what OCR is used for."

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @property
    def dtype(self):
        return torch.float32 if torch_device == "cpu" else torch.bfloat16

    def _load_model(self):
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            attn_implementation="sdpa",
            dtype=self.dtype,
            device_map=torch_device,
        )
        model.eval()
        return model

    @staticmethod
    def _conversation(images, prompt):
        content = [
            {"type": "image", **image} if isinstance(image, dict) else {"type": "image", "image": image}
            for image in images
        ]
        content.append({"type": "text", "text": prompt})
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": content},
        ]

    def _prepare_inputs(self, conversations):
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        return inputs.to(torch_device, dtype=self.dtype)

    def _generate_trimmed_text(self, model, inputs, max_new_tokens=16):
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids_trimmed = generated_ids[:, prompt_length:]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def test_small_model_integration_test(self):
        model = self._load_model()
        inputs = self._prepare_inputs(self._conversation([self.image], self.radar_prompt))

        self.assertIn("pixel_values", inputs)
        self.assertIn("image_grid_thw", inputs)
        self.assertEqual(inputs.image_grid_thw.shape[0], 1)
        self.assertGreater(inputs.input_ids.shape[1], 0)

        expected_texts = Expectations(
            {
                ("cuda", None): "The image is a radar chart that compares the performance of four different models or methods across various benchmarks. The chart is labeled with the names of the benchmarks on the axes, and each model is represented by a different colored line. The models are labeled as BLIP-2, InstructBLIP, Qwen-VL",
            }
        )  # fmt: skip
        decoded_text = self._generate_trimmed_text(model, inputs, max_new_tokens=self.max_new_tokens)[0]
        self.assertEqual(decoded_text, expected_texts.get_expectation())

    def test_small_model_integration_test_batch(self):
        model = self._load_model()
        conversations = [
            self._conversation([self.image], self.radar_prompt),
            self._conversation([self.candy_image], self.candy_prompt),
        ]
        inputs = self._prepare_inputs(conversations)
        self.assertEqual(inputs.image_grid_thw.shape[0], 2)

        expected_texts = Expectations(
            {
                ("cuda", None): [
                    "The image is a radar chart that compares the performance of four different models or methods across various benchmarks. The chart is labeled with the names of the benchmarks on the axes, and each model is represented by a different colored line. The models are labeled as BLIP-2, InstructBLIP, Qwen-VL",
                    "To determine the animal on the candy, observe the image: there are two candies—one teal and one orange. The teal candy has a black silhouette of a bird (a type of bird in the family **passerina**). The orange candy also has a black silhouette of a bird, but",
                ]
            }
        )  # fmt: skip
        decoded_texts = self._generate_trimmed_text(model, inputs, max_new_tokens=self.max_new_tokens)
        self.assertListEqual(decoded_texts, expected_texts.get_expectation())

    def test_small_model_integration_test_multi_image(self):
        model = self._load_model()
        inputs = self._prepare_inputs(self._conversation([self.image, self.candy_image], self.compare_prompt))
        self.assertEqual(inputs.image_grid_thw.shape[0], 2)

        expected_texts = Expectations(
            {
                ("cuda", None): "To determine the answer, we analyze the radar chart:  \n\n1. **First image**: The first image shows a hand with multiple colored candy beads. The top - most bead is teal, and the second bead from the top is green. The third bead from the top is orange. The fourth bead from the",
            }
        )  # fmt: skip
        decoded_text = self._generate_trimmed_text(model, inputs, max_new_tokens=self.max_new_tokens)[0]
        self.assertEqual(decoded_text, expected_texts.get_expectation())

    def test_small_model_integration_test_multi_image_nested(self):
        model = self._load_model()
        conversations = [
            self._conversation([], self.text_prompt),
            self._conversation([self.image, self.candy_image], self.compare_prompt),
            self._conversation([self.image], self.radar_prompt),
        ]
        inputs = self._prepare_inputs(conversations)

        self.assertEqual(inputs.image_grid_thw.shape[0], 3)
        expected_texts = Expectations(
            {
                ("cuda", None): [
                    "OCR (Optical Character Recognition) is a computer technology that uses **Optical Character Recognition (OCR)** to extract text from images or documents. It is a powerful tool for automating tasks like text extraction, image analysis, and document processing.\n\n### Brief Explanation:\n1. **Purpose**: OCR is used to recognize and extract",
                    "To determine the answer, we analyze the radar chart:  \n\n1. **First image**: The first image shows a hand with multiple colored candy beads. The top - most bead is teal, and the second bead from the top is green. The third bead from the top is orange. The fourth bead from the",
                    "The image is a radar chart that compares the performance of four different models or methods across various benchmarks. The chart is labeled with the names of the benchmarks on the axes, and each model is represented by a different colored line. The models are labeled as BLIP-2, InstructBLIP, Qwen-VL",
                ]
            }
        )  # fmt: skip
        decoded_texts = self._generate_trimmed_text(model, inputs, max_new_tokens=self.max_new_tokens)
        self.assertListEqual(decoded_texts, expected_texts.get_expectation())

    def test_small_model_integration_test_batch_different_resolutions(self):
        model = self._load_model()
        conversations = [
            self._conversation([self.lowres_image], self.ocr_prompt),
            self._conversation([self.candy_image], self.candy_prompt),
        ]
        inputs = self._prepare_inputs(conversations)
        self.assertEqual(inputs.image_grid_thw.shape[0], 2)
        self.assertFalse(torch.equal(inputs.image_grid_thw[0], inputs.image_grid_thw[1]))

        expected_texts = Expectations(
            {
                ("cuda", None): [
                    "STEALTH CAM 07:59 AM 09/01/15 69 F FRONT CBN",
                    "To determine the animal on the candy, observe the image: there are two candies—one teal and one orange. The teal candy has a black silhouette of a bird (a type of bird in the family **passerina**). The orange candy also has a black silhouette of a bird, but",
                ]
            }
        )  # fmt: skip
        decoded_texts = self._generate_trimmed_text(model, inputs, max_new_tokens=self.max_new_tokens)
        self.assertListEqual(decoded_texts, expected_texts.get_expectation())

    def test_small_model_integration_test_batch_matches_single(self):
        model = self._load_model()
        conversations = [
            self._conversation([self.lowres_image], self.ocr_prompt),
            self._conversation([self.candy_image], self.candy_prompt),
        ]
        inputs_batched = self._prepare_inputs(conversations)
        inputs_single = self._prepare_inputs(self._conversation([self.lowres_image], self.ocr_prompt))

        expected_texts_batch = Expectations(
            {
                ("cuda", None): [
                    "STEALTH CAM 07:59 AM 09/01/15 69 F FRONT CBN",
                    "To determine the animal on the candy, observe the image: there are two candies—one teal and one orange. The teal candy has a black silhouette of a bird (a type of bird in the family **passerina**). The orange candy also has a black silhouette of a bird, but",
                ]
            }
        )  # fmt: skip
        expected_texts_single = Expectations(
            {
                ("cuda", None): "STEALTH CAM 07:59 AM 09/01/15 69 F FRONT CBN",
            }
        )  # fmt: skip

        decoded_batched = self._generate_trimmed_text(model, inputs_batched, max_new_tokens=self.max_new_tokens)
        decoded_single = self._generate_trimmed_text(model, inputs_single, max_new_tokens=self.max_new_tokens)

        self.assertListEqual(decoded_batched, expected_texts_batch.get_expectation())
        self.assertEqual(decoded_single[0], expected_texts_single.get_expectation())
        self.assertEqual(decoded_batched[0], decoded_single[0])
