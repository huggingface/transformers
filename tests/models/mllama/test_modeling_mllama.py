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
"""Testing suite for the PyTorch Mllama model."""

import unittest

import pytest
import requests

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaConfig,
    MllamaForCausalLM,
    MllamaForConditionalGeneration,
    MllamaModel,
    is_torch_available,
    is_vision_available,
)
from transformers.cache_utils import Cache
from transformers.models.mllama.configuration_mllama import MllamaTextConfig
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_bitsandbytes,
    require_optimum_quanto,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class MllamaText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        seq_length=7,
        is_training=True,
        text_config={
            "model_type": "mllama",
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "max_position_embeddings": 512,
            "initializer_range": 0.02,
            "rope_scaling": {"rope_type": "default"},
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.text_config = text_config
        self.seq_length = seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training
        self.pad_token_id = self.text_config["pad_token_id"]
        self.batch_size = 3

    def get_config(self):
        return MllamaTextConfig(**self.text_config)

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict

    def create_and_check_mllama_model_fp16_forward(self, config, input_ids, attention_mask):
        model = MllamaForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class MllamaForCausalLMModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `MllamaForConditionalGeneration`.
    """

    all_model_classes = (MllamaForCausalLM,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = MllamaText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MllamaTextConfig, has_text_modality=True)


class MllamaVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=4,
        seq_length=7,
        is_training=True,
        text_config={
            "model_type": "mllama",
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "max_position_embeddings": 512,
            "initializer_range": 0.02,
            "rope_scaling": {"rope_type": "default"},
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "cross_attention_layers": [1],
        },
        vision_config={
            "image_size": 30,
            "patch_size": 2,
            "num_channels": 3,
            "hidden_size": 16,
            "intermediate_layers_indices": [0],
            "vision_output_dim": 32,
            "projection_dim": 32,
            "num_hidden_layers": 6,
            "num_global_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "initializer_range": 0.02,
            "supported_aspect_ratios": [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]],
        },
    ):
        self.parent = parent
        self.is_training = is_training
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.pad_token_id = self.text_config["pad_token_id"]

        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 224
        self.max_num_images = 1
        self.max_image_tiles = 4
        self.image_length = 904

    def get_config(self):
        return MllamaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_index=self.image_token_index,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.max_num_images,
                self.max_image_tiles,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        aspect_ratio_ids = torch.tensor([[6] * self.batch_size], device=torch_device).transpose(0, 1)
        aspect_ratio_mask = torch.ones(self.batch_size, self.max_num_images, self.max_image_tiles)
        config = self.get_config()

        return config, pixel_values, aspect_ratio_ids, aspect_ratio_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, aspect_ratio_ids, aspect_ratio_mask = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)
        aspect_ratio_mask = aspect_ratio_mask.to(torch_device)
        cross_attention_mask = torch.ones(
            (self.batch_size, self.seq_length, self.max_num_images, self.max_image_tiles), device=torch_device
        )

        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[:, 1] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "aspect_ratio_ids": aspect_ratio_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "aspect_ratio_mask": aspect_ratio_mask,
            "cross_attention_mask": cross_attention_mask,
            "use_cache": True,
        }
        return config, inputs_dict

    def create_and_check_mllama_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = MllamaForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())

    @unittest.skip("Mllama applies key/query norm which doesn't work with packing")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Mllama applies key/query norm which doesn't work with packing")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass


@require_torch
class MllamaForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `MllamaForConditionalGeneration`.
    """

    all_model_classes = (
        (
            MllamaModel,
            MllamaForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = {"image-text-to-text": MllamaForConditionalGeneration} if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False
    _is_composite = True

    def setUp(self):
        self.model_tester = MllamaVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=MllamaConfig, has_text_modality=False, common_properties=["image_token_index"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_resize_embeddings_results_in_successful_loss(self):
        # resizing embeddings should result in successful loss computation
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        model = MllamaForConditionalGeneration(config).to(torch_device)
        model_vocab_size = config.get_text_config().vocab_size
        inputs = self._prepare_for_class(inputs, MllamaForConditionalGeneration, return_labels=True)
        # Resize embeddings and call forward
        model.resize_token_embeddings(model_vocab_size + 10)
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            return_dict=True,
        )
        self.assertTrue("loss" in output)

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_past_key_values
    ):
        # Mllama has cross attention layers and those have a different shape than normal attention layers
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (output_length - prompt_length))

        cross_attention_layers = self.model_tester.text_config["cross_attention_layers"]
        use_cache = decoder_past_key_values is not None

        for generated_length, iter_attentions in enumerate(attentions):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length
            query_length = prompt_length + generated_length

            expected_shape = (
                batch_size,
                config.num_attention_heads,
                model_input_length,
                query_length,
            )

            expected_shape_cross = (
                batch_size,
                config.num_attention_heads,
                model_input_length,
                self.model_tester.image_length,
            )

            expected_shapes = [
                expected_shape if layer_idx not in cross_attention_layers else expected_shape_cross
                for layer_idx in range(len(iter_attentions))
            ]

            self.assertListEqual([layer_attention.shape for layer_attention in iter_attentions], expected_shapes)

    @require_optimum_quanto
    @pytest.mark.generate
    @unittest.skip("Mllama is actually an encoder decoder cache and thus can't supports quant cache")
    def test_generate_with_quant_cache(self):
        pass

    @unittest.skip("For some unknown reasons the tests fails in CrossAttention layer when doing torch.sdpa(). ")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="AssertionError: Items in the second set but not the first: might be a setting issue")
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="Mllama can't assisted decoding due to cache format and `Cache.crop()`")
    def test_assisted_decoding_with_num_logits_to_keep(self):
        pass

    @unittest.skip(reason="Mllama uses self.weights directly causing device mismatch when offloading`")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Mllama uses self.weights directly causing device mismatch when offloading`")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Mllama uses self.weights directly causing device mismatch when offloading`")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip("Mllama applies key/query norm which doesn't work with packing")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Mllama applies key/query norm which doesn't work with packing")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids_and_fa_kwargs(self):
        pass

    @unittest.skip("Mllama applies key/query norm which doesn't work with packing")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Mllama applies key/query norm which doesn't work with packing")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @pytest.mark.generate
    # overridden because mllama is not an encoder-decoder model, but has encoder-decoder-like cache
    def test_past_key_values_format(self):
        # Test that the KV cache is formatted correctly. Exceptions need to explicitly overwrite this test. Having a
        # standard KV cache format is important for a consistent API (and for advanced generation methods).
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            model = model_class(config).to(torch_device)
            if "use_cache" not in inputs:
                inputs["use_cache"] = True
            outputs = model(**inputs)

            text_config = config.get_text_config()
            num_hidden_layers = (
                getattr(text_config, "decoder_layers", None)
                or getattr(text_config, "num_decoder_layers", None)
                or text_config.num_hidden_layers
            )
            num_attention_heads = getattr(text_config, "decoder_attention_heads", text_config.num_attention_heads)
            embed_dim = getattr(text_config, "d_model", text_config.hidden_size)
            per_head_embed_dim = embed_dim // num_attention_heads

            # some models have different num-head for query vs key/value so we need to assign correct value
            # BUT only after `per_head_embed_dim` is set
            num_attention_heads = (
                text_config.num_key_value_heads
                if getattr(text_config, "num_key_value_heads", None) is not None
                else num_attention_heads
            )

            past_kv = outputs["past_key_values"]
            self.assertEqual(len(past_kv), num_hidden_layers)
            batch_size, seq_length = inputs["input_ids"].shape
            for i in range(num_hidden_layers):
                self.assertEqual(len(past_kv[0]), 2)  # K V for the decoder = 2
                if i in self.model_tester.text_config["cross_attention_layers"]:
                    self.assertEqual(
                        past_kv[i][0].shape,
                        (batch_size, num_attention_heads, self.model_tester.image_length, per_head_embed_dim),
                    )
                    self.assertEqual(
                        past_kv[i][1].shape,
                        (batch_size, num_attention_heads, self.model_tester.image_length, per_head_embed_dim),
                    )
                else:
                    self.assertEqual(
                        past_kv[i][0].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    self.assertEqual(
                        past_kv[i][1].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )

    # overridden because mllama has special cache for self and cross attentions
    def _check_past_key_values_for_generate(self, batch_size, decoder_past_key_values, cache_length, config):
        self.assertIsInstance(decoder_past_key_values, Cache)
        self.assertListEqual(
            [isinstance(iter_past_key_values, tuple) for iter_past_key_values in decoder_past_key_values],
            [True] * len(decoder_past_key_values),
        )

        for layer_idx, layer_past_key_values in enumerate(decoder_past_key_values):
            if layer_idx in self.model_tester.text_config["cross_attention_layers"]:
                expected_shape = (
                    batch_size,
                    config.num_key_value_heads
                    if hasattr(config, "num_key_value_heads")
                    else config.num_attention_heads,
                    self.model_tester.image_length,
                    config.hidden_size // config.num_attention_heads,
                )
            else:
                # (batch, head, cache_length, head_features)
                expected_shape = (
                    batch_size,
                    config.num_key_value_heads
                    if hasattr(config, "num_key_value_heads")
                    else config.num_attention_heads,
                    cache_length,
                    config.hidden_size // config.num_attention_heads,
                )
            # check shape key, value
            self.assertListEqual([layer_past_key_values[0].shape], [expected_shape])
            self.assertListEqual([layer_past_key_values[1].shape], [expected_shape])

    def test_generate_text_only_with_cache(self):
        """
        Tests that our cached generation with text-only inputs works. When mllama was introduced, this feature
        required cache modifications (because layers are skipped in practice). This test should prevent regressions.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]

            model.generate(input_ids, use_cache=True)


@require_torch
class MllamaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.base_model_checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        self.instruct_model_checkpoint = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_accelerator
    @require_bitsandbytes
    @require_read_token
    def test_11b_model_integration_generate(self):
        # Prepare inputs
        processor = AutoProcessor.from_pretrained(self.base_model_checkpoint)

        prompt = "<|image|>If I had to write a haiku for this one"
        url = "https://llava-vl.github.io/static/images/view.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch_device)

        input_ids = inputs["input_ids"]

        # Check inputs ids
        expected_input_ids_all = Expectations(
            {
                ("xpu", 3): torch.tensor([[128000, 128256, 128000, 2746, 358, 1047, 311, 3350, 264, 6520, 39342, 369, 420, 832]], device=torch_device),
                ("cuda", 7): torch.tensor([[128000, 128256, 128000, 2746, 358, 1047, 311, 3350, 264, 6520, 39342, 369, 420, 832]], device=torch_device),
                ("cuda", 8): torch.tensor([[128000, 128256, 128000, 2746, 358, 1047, 311, 3350, 264, 6520, 39342, 369, 420, 832]], device=torch_device),
            }
        )  # fmt: skip
        expected_input_ids = expected_input_ids_all.get_expectation()
        self.assertTrue(torch.equal(input_ids, expected_input_ids))

        # Load model in 4 bit
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = MllamaForConditionalGeneration.from_pretrained(
            self.base_model_checkpoint, quantization_config=quantization_config
        )

        # Generate
        output = model.generate(**inputs, do_sample=False, max_new_tokens=25)

        decoded_output = processor.decode(output[0], skip_special_tokens=True)
        expected_outputs = Expectations(
                {
                    ("xpu", 3): "If I had to write a haiku for this one, it would be:.\\nA dock on a lake.\\nA mountain in the distance.\\nA long exposure.",
                    ("cuda", 7): "If I had to write a haiku for this one, it would be:.\\nA dock in the lake.\\nA mountain in the distance.\\nA long exposure.",
                    ("cuda", 8): 'If I had to write a haiku for this one, it would be:.\\nA dock in the lake.\\nA mountain in the distance.\\nA long exposure.',
                }
            )  # fmt: skip
        expected_output = expected_outputs.get_expectation()

        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )

    @slow
    @require_torch_accelerator
    @require_bitsandbytes
    @require_read_token
    def test_11b_model_integration_generate_text_only(self):
        # Prepare inputs
        processor = AutoProcessor.from_pretrained(self.base_model_checkpoint)
        prompt = "If I had to write a haiku"
        inputs = processor(text=prompt, return_tensors="pt").to(torch_device)
        input_ids = inputs["input_ids"].cpu().squeeze().tolist()

        # Check inputs ids
        expected_input_ids_all = Expectations(
            {
                ("xpu", 3): [128000, 128000, 2746, 358, 1047, 311, 3350, 264, 6520, 39342],
                ("cuda", 7): [128000, 128000, 2746, 358, 1047, 311, 3350, 264, 6520, 39342],
                ("cuda", 8): [128000, 128000, 2746, 358, 1047, 311, 3350, 264, 6520, 39342],
            }
        )
        expected_input_ids = expected_input_ids_all.get_expectation()
        self.assertEqual(input_ids, expected_input_ids)

        # Load model in 4 bit
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = MllamaForConditionalGeneration.from_pretrained(
            self.base_model_checkpoint, quantization_config=quantization_config
        )

        # Generate
        output = model.generate(**inputs, do_sample=False, max_new_tokens=25)

        decoded_output = processor.decode(output[0], skip_special_tokens=True)
        expected_outputs = Expectations(
                {
                    ("xpu", 3): "If I had to write a haiku about my life, I would write:\nLife is a messy tapestry\n Threads of joy and sorrow\nWeft of memories",
                    ("cuda", 7): "If I had to write a haiku about my life, I would write:\nLife is a messy stream\nRipples of joy and pain\nFlowing, ever",
                    ("cuda", 8): "If I had to write a haiku about my life, I would write:\nLife is a messy stream\nRipples of joy and pain\nFlowing, ever",
                }
            )  # fmt: skip
        expected_output = expected_outputs.get_expectation()
        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )

    @slow
    @require_torch_accelerator
    @require_bitsandbytes
    @require_read_token
    def test_11b_model_integration_forward(self):
        # Prepare inputs
        processor = AutoProcessor.from_pretrained(self.base_model_checkpoint)

        prompt = "<|image|>If I had to write a haiku for this one"
        url = "https://llava-vl.github.io/static/images/view.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch_device)

        # Load model in 4 bit
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = MllamaForConditionalGeneration.from_pretrained(
            self.base_model_checkpoint, quantization_config=quantization_config
        )

        # Forward
        with torch.inference_mode():
            output = model(**inputs)

        actual_logits = output.logits[0, -1, :5].cpu()
        expected_logits_all = Expectations(
            {
                ("xpu", 3): torch.tensor([9.1562, 8.9141, 5.0664, 1.6855, 3.2324], dtype=actual_logits.dtype),
                ("cuda", 7): torch.tensor([9.0781, 8.8750, 5.0781, 1.6221, 3.2207], dtype=actual_logits.dtype),
                ("cuda", 8): torch.tensor([9.0703, 8.8750, 5.0781, 1.6279, 3.2207], dtype=actual_logits.dtype),
            }
        )

        expected_logits = expected_logits_all.get_expectation()
        self.assertTrue(
            torch.allclose(actual_logits, expected_logits, atol=0.1),
            f"Actual logits: {actual_logits}"
            f"\nExpected logits: {expected_logits}"
            f"\nDifference: {torch.abs(actual_logits - expected_logits)}",
        )

    @slow
    @require_torch_accelerator
    @require_bitsandbytes
    @require_read_token
    def test_11b_model_integration_batched_generate(self):
        processor = AutoProcessor.from_pretrained(self.base_model_checkpoint)

        # Prepare inputs
        prompt = [
            "<|image|>If I had to write a haiku for this one",
            "<|image|>This image shows",
        ]
        image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
        image2 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg",
                stream=True,
            ).raw
        )

        inputs = processor(text=prompt, images=[[image1], [image2]], padding=True, return_tensors="pt").to(
            torch_device
        )

        # Load model in 4 bit
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = MllamaForConditionalGeneration.from_pretrained(
            self.base_model_checkpoint, quantization_config=quantization_config
        )

        output = model.generate(**inputs, do_sample=False, max_new_tokens=25)

        # Check first output
        decoded_output = processor.decode(output[0], skip_special_tokens=True)
        expected_outputs = Expectations(
                {
                    ("xpu", 3): "If I had to write a haiku for this one, it would be:.\\nA dock on a lake.\\nA mountain in the distance.\\nA long exposure.",
                    ("cuda", 7): "If I had to write a haiku for this one, it would be:.\\nA dock on a lake.\\nA mountain in the distance.\\nA long exposure.",
                    ("cuda", 8): 'If I had to write a haiku for this one, it would be:.\\nA dock in the lake.\\nA mountain in the distance.\\nA long exposure.',
                 }
            )  # fmt: skip
        expected_output = expected_outputs.get_expectation()

        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )

        # Check second output
        decoded_output = processor.decode(output[1], skip_special_tokens=True)
        expected_outputs = Expectations(
                {
                    ("xpu", 3): "This image shows\nI'm not able to provide information on the person in this image. I can give you an idea of what's happening",
                    ("cuda", 7): "This image shows\nI'm not able to provide information on the person in this image. I can give you an idea of what's happening",
                    ("cuda", 8): "This image shows\nI'm not able to provide information on the person in this image. I can give you an idea of what's happening",
                }
            )  # fmt: skip
        expected_output = expected_outputs.get_expectation()

        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )

    @slow
    @require_torch_accelerator
    @require_bitsandbytes
    @require_read_token
    def test_11b_model_integration_multi_image_generate(self):
        processor = AutoProcessor.from_pretrained(self.instruct_model_checkpoint)

        # Prepare inputs
        image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
        image2 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg",
                stream=True,
            ).raw
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What’s shown in this image?"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "This image shows a long wooden dock extending out into a lake."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What about this one, what do you see here? Can you describe in detail?"},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[[image1, image2]], return_tensors="pt").to(torch_device)
        prompt_len = inputs["input_ids"].shape[-1]

        # Load model in 4 bit
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = MllamaForConditionalGeneration.from_pretrained(
            self.instruct_model_checkpoint, quantization_config=quantization_config
        )

        output = model.generate(**inputs, do_sample=False, max_new_tokens=25)

        # Check first output
        generated_output = output[0][prompt_len:]
        decoded_output = processor.decode(generated_output, skip_special_tokens=False)

        # model should response about "stop sign", however it responses about "dock"
        # this happens only in quantized version, bfloat16 works fine
        expected_output = "This image shows a long wooden dock extending out into a lake. The dock is made of wooden planks and has a railing"

        self.assertEqual(
            decoded_output,
            expected_output,
            f"Decoded output: {decoded_output}\nExpected output: {expected_output}",
        )
