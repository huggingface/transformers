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
"""Testing suite for the PyTorch UnlimitedOcr model."""

import unittest

from transformers import (
    AutoProcessor,
    UnlimitedOcrConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        DynamicCache,
        StaticCache,
        UnlimitedOcrForConditionalGeneration,
        UnlimitedOcrModel,
    )
    from transformers.models.unlimited_ocr.configuration_unlimited_ocr import (
        UnlimitedOcrTextConfig,
        UnlimitedOcrVisionConfig,
    )

if is_vision_available():
    from transformers.image_utils import load_image


class UnlimitedOcrVisionText2TextModelTester(VLMModelTester):
    base_model_class = UnlimitedOcrModel
    config_class = UnlimitedOcrConfig
    conditional_generation_class = UnlimitedOcrForConditionalGeneration
    text_config_class = UnlimitedOcrTextConfig
    vision_config_class = UnlimitedOcrVisionConfig

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("num_image_tokens", 7)
        kwargs.setdefault("image_token_id", 1)
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("hidden_size", 128)
        kwargs.setdefault("intermediate_size", 256)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 4)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 512)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("bos_token_id", 2)
        kwargs.setdefault("eos_token_id", 3)
        kwargs.setdefault("pad_token_id", 4)
        kwargs.setdefault("n_routed_experts", 8)
        kwargs.setdefault("n_shared_experts", 1)
        kwargs.setdefault("mlp_layer_types", ["dense", "sparse"])
        kwargs.setdefault("moe_intermediate_size", 64)
        kwargs.setdefault("num_experts_per_tok", 2)
        kwargs.setdefault("sliding_window", 4)
        kwargs.setdefault("use_sliding_window", True)
        super().__init__(parent, **kwargs)

        self.sam_config = {
            "hidden_size": 32,
            "output_channels": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 16,
            "patch_size": 2,
            "hidden_act": "gelu",
            "mlp_ratio": 4.0,
            "mlp_dim": 128,
            "window_size": 4,
            "global_attn_indexes": [1],
            "downsample_channels": [32, 64],
        }
        self.encoder_config = {
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "rms_norm_eps": 1.0,
        }

    def get_vision_config(self):
        return UnlimitedOcrVisionConfig(
            sam_config=self.sam_config,
            encoder_config=self.encoder_config,
        )

    def get_config(self):
        return self.config_class(
            vision_config=self.get_vision_config(),
            text_config=self.get_text_config(),
            image_token_id=self.image_token_id,
        )


@require_torch
class UnlimitedOcrModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = UnlimitedOcrVisionText2TextModelTester
    test_all_params_have_gradient = False

    @unittest.skip(
        reason="UnlimitedOcrVisionModel builds a hybrid bidirectional+causal mask internally, so SDPA is always called with a non-null `attn_mask`."
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def _image_features_prepare_config_and_inputs(self):
        # `test_get_image_features_output` requires `vision_config.hidden` size to be set.
        # This is not the case by default as the vision model is a combination of two submodels (SAM + CLIP vision encoder).
        config, inputs_dict = super()._image_features_prepare_config_and_inputs()
        config.vision_config.hidden_size = (
            config.vision_config.sam_config.downsample_channels[-1] + config.vision_config.encoder_config.hidden_size
        )
        return config, inputs_dict

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        # max cache length can be smaller than sequence length
        max_length = past_key_values.get_max_length()
        seq_length = min(seq_length, max_length) if max_length >= 0 else seq_length
        super()._check_past_key_values_for_generate(batch_size, past_key_values, seq_length, config)

    def _check_generate_cache_sliding_window_too_small(self, cache_implementation: str, prefill_max_new_tokens: int):
        """Test that reference sliding window cache works correctly when decoding more than sliding_window tokens at once."""
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            self.assertEqual(config.text_config.sliding_window, 4)

            model = model_class(config).to(torch_device).eval()

            # Resume from cache doesn't work with random attention mask.
            inputs_dict["attention_mask"] = torch.ones_like(inputs_dict["attention_mask"])

            out_reference = model.generate(**inputs_dict, max_new_tokens=15, do_sample=False)

            # Prefill the cache
            out_prefill = model.generate(
                **inputs_dict,
                max_new_tokens=prefill_max_new_tokens,
                max_cache_len=100,
                do_sample=False,
                return_dict_in_generate=True,
                use_cache=True,
                cache_implementation=cache_implementation,
            )

            # Decode from cache and pass more than sliding_window unseen input ids.
            input_ids_reference = out_reference[:, :-3]
            out = model.generate(
                input_ids=input_ids_reference,
                # Pass attention mask explicitly as input_ids_reference sometimes contains randomly generated
                # pad tokens which trips up the automatic attention mask generation.
                attention_mask=torch.ones_like(input_ids_reference),
                past_key_values=out_prefill.past_key_values,
                max_new_tokens=3,
                do_sample=False,
                return_dict_in_generate=True,
                use_cache=True,
            )
            self.assertEqual(out.sequences.tolist(), out_reference.tolist())

            prompt_length = inputs_dict["input_ids"].shape[1]
            reference_layers = [
                layer for layer in out.past_key_values.layers if layer._layer_type == "reference_sliding_attention"
            ]
            self.assertGreater(len(reference_layers), 0)
            for layer in reference_layers:
                self.assertEqual(layer.prefill_length, prompt_length)

    def test_generate_dynamic_cache_sliding_window_too_small(self):
        self._check_generate_cache_sliding_window_too_small(cache_implementation="dynamic", prefill_max_new_tokens=1)

    def test_generate_static_cache_sliding_window_too_small(self):
        self._check_generate_cache_sliding_window_too_small(cache_implementation="static", prefill_max_new_tokens=1)

    def test_generate_dynamic_cache_sliding_window_too_small_cache_full(self):
        """Continue from full cache"""
        self._check_generate_cache_sliding_window_too_small(cache_implementation="dynamic", prefill_max_new_tokens=6)

    def test_generate_static_cache_sliding_window_too_small_cache_full(self):
        """Continue from full cache"""
        self._check_generate_cache_sliding_window_too_small(cache_implementation="static", prefill_max_new_tokens=6)

    def _check_manual_forward_cache(self, cache_implementation: str):
        """Test that the states of the first forward pass are the prefill states if set_prefill_length is not explicitly called."""
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            sliding_window = config.text_config.sliding_window
            num_decode_steps = 2 * sliding_window

            model = model_class(config).to(torch_device).eval()
            input_ids = inputs_dict["input_ids"]
            prompt_length = input_ids.shape[1]
            past_key_values = (
                StaticCache(config=config.get_text_config(), max_cache_len=prompt_length + num_decode_steps)
                if cache_implementation == "static"
                else DynamicCache(config=config.get_text_config())
            )

            next_tokens = input_ids
            with torch.no_grad():
                for _ in range(num_decode_steps + 1):
                    outputs = model(input_ids=next_tokens, past_key_values=past_key_values, use_cache=True)
                    next_tokens = outputs.logits[:, -1:].argmax(dim=-1)

            reference_layers = [
                layer for layer in past_key_values.layers if layer._layer_type == "reference_sliding_attention"
            ]
            self.assertGreater(len(reference_layers), 0)
            for layer in reference_layers:
                self.assertEqual(layer.prefill_length, prompt_length)

    def test_manual_forward_dynamic_cache(self):
        self._check_manual_forward_cache(cache_implementation="dynamic")

    def test_manual_forward_loop_static_cache(self):
        self._check_manual_forward_cache(cache_implementation="static")

    def test_generate_cache_chunked_prefill(self):
        """Test that a chunked prefill keeps all prompt tokens as prefill, not only the first chunk."""
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).to(torch_device).eval()

            input_ids = inputs_dict["input_ids"]
            prompt_length = input_ids.shape[1]
            out = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=3,
                min_new_tokens=3,
                do_sample=False,
                prefill_chunk_size=prompt_length // 2,
                return_dict_in_generate=True,
            )

            reference_layers = [
                layer for layer in out.past_key_values.layers if layer._layer_type == "reference_sliding_attention"
            ]
            self.assertGreater(len(reference_layers), 0)
            for layer in reference_layers:
                self.assertEqual(layer.prefill_length, prompt_length)

    def test_generate_without_sliding_window(self):
        """With `use_sliding_window=False` every layer is a full attention layer."""
        model_tester = self.model_tester_class(self, use_sliding_window=False)
        config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()
        self.assertEqual(config.text_config.layer_types, ["full_attention"] * config.text_config.num_hidden_layers)

        for model_class in self.all_generative_model_classes:
            model = model_class(config).to(torch_device).eval()
            out = model.generate(**inputs_dict, max_new_tokens=3, do_sample=False, return_dict_in_generate=True)
            self.assertTrue(all(not layer.is_sliding for layer in out.past_key_values.layers))


@require_torch
class UnlimitedOcrIntegrationTest(unittest.TestCase):
    model_id = "baidu/Unlimited-OCR"
    # TODO: remove revision before merge
    revision = "refs/pr/13"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def get_messages(self, images, text):
        if not isinstance(images, list):
            images = [images]
        return [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": text}],
            }
        ]

    @slow
    @require_torch_accelerator
    def test_small_model_integration_test_document_parsing(self):
        model = UnlimitedOcrForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, device_map=torch_device
        ).eval()
        image = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
            )
        )
        messages = self.get_messages(image, "document parsing.")
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.autocast(device_type=torch_device, dtype=torch.bfloat16):
            generate_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=20,
            )
        decoded = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", 9): "image [383, 88, 497, 175]\ntitle [333",
                ("cpu", None): "image [383, 87, 497, 171]\ntitle [333",
            }
        ).get_expectation()  # fmt: skip
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_test_document_parsing_no_skip_special_tokens(self):
        model = UnlimitedOcrForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, device_map=torch_device
        ).eval()
        image = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
            )
        )
        messages = self.get_messages(image, "document parsing.")
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.autocast(device_type=torch_device, dtype=torch.bfloat16):
            generate_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=20,
            )
        decoded, detections = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False, return_detections=True
        )
        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", 9): "<|det|>image [383, 88, 497, 175]<|/det|>\n<|det|>title [333",
                ("cpu", None): "<|det|>image [383, 87, 497, 171]<|/det|>\n<|det|>title [333",
            }
        ).get_expectation()  # fmt: skip
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)

        EXPECTED_DETECTIONS = Expectations(
            {
                ("cuda", 9): [{"region_type": "image", "box": [383, 88, 497, 175], "text": "\n"}],
                ("cpu", None): [{"region_type": "image", "box": [383, 87, 497, 171], "text": "\n"}],
            }
        ).get_expectation()  # fmt: skip
        self.assertEqual(detections, EXPECTED_DETECTIONS)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_test_document_parsing_batched(self):
        model = UnlimitedOcrForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, device_map=torch_device
        ).eval()
        image1 = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
            )
        )
        image2 = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
            )
        )
        messages = [self.get_messages(image, "document parsing.") for image in [image1, image2]]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        with torch.autocast(device_type=torch_device, dtype=torch.bfloat16):
            generate_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=20,
            )
        decoded = self.processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", 9): [
                    "image [383, 88, 497, 174]\ntitle [333",
                    "header [53, 23, 365, 41]Advanced Template and Styl",
                ],
                ("cpu", None): [
                    "image [383, 88, 497, 171]\ntitle [333",
                    "header [53, 23, 365, 41]Advanced Template and Styl",
                ],
            }
        ).get_expectation()
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_test_multi_page_document_parsing(self):
        model = UnlimitedOcrForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, device_map=torch_device
        ).eval()
        image1 = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
            )
        )
        image2 = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
            )
        )
        messages = self.get_messages([image1, image2], "Multi page parsing.")
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"crop_to_patches": False},
        ).to(model.device)

        with torch.autocast(device_type=torch_device, dtype=torch.bfloat16):
            generate_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=20,
            )
        decoded = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", 9): "<PAGE>image [382, 87, 489, 180]\n",
                ("cpu", None): "<PAGE>image [382, 87, 489, 174]\n",
            }
        ).get_expectation()
        self.assertEqual(decoded, EXPECTED_DECODED_TEXT)
