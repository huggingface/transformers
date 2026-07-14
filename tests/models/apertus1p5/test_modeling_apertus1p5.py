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
"""Testing suite for the PyTorch Apertus 1.5 model."""

import unittest

import pytest
import requests

from transformers import (
    Apertus1p5Config,
    Apertus1p5TextConfig,
    BitsAndBytesConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
    require_bitsandbytes,
    require_torch,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

    from transformers import (
        Apertus1p5ForCausalLM,
        Apertus1p5ForConditionalGeneration,
        Apertus1p5Model,
        Apertus1p5Processor,
    )


class Apertus1p5Text2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        max_position_embeddings=512,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        config = self.get_config()

        return config, input_ids, attention_mask

    def get_config(self):
        return Apertus1p5TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class Apertus1p5Text2TextModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Apertus1p5ForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "text-generation": Apertus1p5ForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = Apertus1p5Text2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Apertus1p5TextConfig, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip("Doesn't work, tensors are not almost same")  # TODO raushan fixme
    def test_custom_4d_attention_mask(self):
        pass


class Apertus1p5Vision2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        max_position_embeddings=512,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=3,
        image_size=15,
        codebook_size=20,
        base_channels=32,
        vq_channel_multiplier=[1, 2, 1],
        vq_num_res_blocks=2,
        image_seq_length=12,
        vq_img_token_start_id=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.image_size = image_size
        self.codebook_size = codebook_size
        self.vq_channel_multiplier = vq_channel_multiplier
        self.vq_num_res_blocks = vq_num_res_blocks
        self.vq_img_token_start_id = vq_img_token_start_id
        self.base_channels = base_channels
        self.seq_length = seq_length + image_seq_length
        self.image_seq_length = image_seq_length

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size)
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        pixel_values = floats_tensor(
            [
                self.batch_size,
                3,
                self.image_size,
                self.image_size,
            ]
        )
        image_sizes = [[self.image_size, self.image_size]] * self.batch_size
        image_sizes = torch.tensor(image_sizes, device=torch_device, dtype=torch.int64)

        return config, input_ids, attention_mask, pixel_values, image_sizes

    def get_config(self):
        # create dummy vocab map for image2bpe mapping if it needs remapping
        # we assume that vocab size is big enough to account for `codebook_size` amount of
        # image tokens somewhere at the beginning of total vocab size

        vocab_map = {i: chr(i) for i in range(self.vocab_size)}
        start = self.vq_img_token_start_id
        end = self.vq_img_token_start_id + self.codebook_size
        for i in range(start, end):
            # dummy str for each token, anything that fits pattern "<|visual token XXXXXX|>"
            vocab_map[i] = f"<|visual token{i:06d}|>"

        # add tokens that have to be in the vocab, we'll retrieve their ids later in modeling code
        vocab_map[self.image_token_id] = "<image>"
        vocab_map[self.image_token_id + 1] = "<|extra_200|>"
        vocab_map = {v: k for k, v in vocab_map.items()}

        text_config = Apertus1p5TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

        vq_config = {
            "codebook_size": self.codebook_size,
            "base_channels": self.base_channels,
            "channel_multiplier": self.vq_channel_multiplier,
            "num_res_blocks": self.vq_num_res_blocks,
            "embed_dim": 16,
            "latent_channels": 16,
            "attn_resolutions": [],
            "resolution": self.image_size,
        }
        return Apertus1p5Config(text_config=text_config, vq_config=vq_config, vocabulary_map=vocab_map)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            pixel_values,
            image_sizes,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        }
        return config, inputs_dict


@require_torch
class Apertus1p5Vision2TextModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Apertus1p5Model,
            Apertus1p5ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"any-to-any": Apertus1p5ForConditionalGeneration, "image-text-to-text": Apertus1p5ForConditionalGeneration}
        if is_torch_available()
        else {}
    )
    skip_test_image_features_output_shape = True  # Apertus1p5 uses index -3 for hidden_size instead of -1

    test_torch_exportable = False  # data-dependent control flow in vision/segmentation head

    def setUp(self):
        self.model_tester = Apertus1p5Vision2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Apertus1p5Config, has_text_modality=False, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    @pytest.mark.generate
    @unittest.skip("Apertus1p5 has dynamic control flow in vision backbone")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip(
        "The vision tokenizer encodes each image individually (its global attention makes batch padding perturb "
        "codes), so per-block output recording is not supported"
    )
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip(
        "The vision tokenizer encodes each image individually (its global attention makes batch padding perturb "
        "codes), so per-block output recording is not supported"
    )
    def test_get_image_features_hidden_states(self):
        pass

    def test_pixel_values_influence_logits(self):
        """Regression test: the generation wrapper must pass image tensors down to the base model
        (the inherited Emu3 forward silently dropped them)."""
        config, input_ids, attention_mask, pixel_values, image_sizes = self.model_tester.prepare_config_and_inputs()
        model = Apertus1p5ForConditionalGeneration(config).to(torch_device).eval()
        torch.manual_seed(0)
        other_pixel_values = pixel_values * 3.0 + 2.0
        with torch.no_grad():
            logits_a = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            ).logits
            logits_b = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=other_pixel_values,
                image_sizes=image_sizes,
            ).logits
        self.assertFalse(torch.allclose(logits_a, logits_b), "different images must yield different logits")

    def test_images_encoded_one_by_one(self):
        """Images must be encoded individually (batch size 1) even when a batched tensor is given: the vision
        tokenizer contains global attention, so padded-batch encoding would perturb the codes."""
        config, *_ = self.model_tester.prepare_config_and_inputs()
        model = Apertus1p5Model(config).to(torch_device).eval()
        factor = model.vqmodel.vision_spatial_factor

        sizes = [(12, 12), (8, 16), (16, 8)]
        pixel_values = torch.zeros(3, 3, 16, 16, device=torch_device)
        for i, (height, width) in enumerate(sizes):
            pixel_values[i, :, :height, :width] = floats_tensor([3, height, width]).to(torch_device)
        image_sizes = torch.tensor(sizes, device=torch_device, dtype=torch.int64)

        encode_shapes = []
        original_encode = model.vqmodel.encode

        def spy_encode(pixel_values, *args, **kwargs):
            encode_shapes.append(tuple(pixel_values.shape))
            return original_encode(pixel_values, *args, **kwargs)

        model.vqmodel.encode = spy_encode
        with torch.no_grad():
            bpe_tokens = model.get_image_tokens(pixel_values, image_sizes)
        model.vqmodel.encode = original_encode

        self.assertTrue(all(shape[0] == 1 for shape in encode_shapes), f"expected batch-1 calls, got {encode_shapes}")
        self.assertEqual([shape[2:] for shape in encode_shapes], list(sizes))

        # per-image results equal encoding each image alone, and counts match the split formula
        expected = []
        with torch.no_grad():
            for i, (height, width) in enumerate(sizes):
                codes = model.vqmodel.encode(pixel_values[i : i + 1, :, :height, :width])[0]
                expected.append(model.vocabulary_mapping.convert_img2bpe(codes).flatten())
        self.assertTrue(torch.equal(bpe_tokens, torch.cat(expected)))
        expected_counts = [(h // factor) * (w // factor + 1) for h, w in sizes]
        self.assertEqual([len(t) for t in expected], expected_counts)

    def test_mismatched_image_sizes_raise(self):
        config, *_ = self.model_tester.prepare_config_and_inputs()
        model = Apertus1p5Model(config).to(torch_device).eval()
        pixel_values = torch.randn(2, 3, 16, 16, device=torch_device)
        error_message = "The number of images in `pixel_values` must match the number of entries in `image_sizes`"

        for image_sizes in (
            torch.tensor([[16, 16]], device=torch_device),
            torch.tensor([[16, 16], [16, 16], [16, 16]], device=torch_device),
        ):
            with self.subTest(num_image_sizes=image_sizes.shape[0]):
                with self.assertRaisesRegex(ValueError, error_message):
                    model.get_image_tokens(pixel_values, image_sizes)
                with self.assertRaisesRegex(ValueError, error_message):
                    model.get_image_features(pixel_values, image_sizes)

    def test_image_sizes_are_required(self):
        config, *_ = self.model_tester.prepare_config_and_inputs()
        model = Apertus1p5Model(config).to(torch_device).eval()
        pixel_values = torch.randn(1, 3, 16, 16, device=torch_device)

        with self.assertRaisesRegex(ValueError, "`image_sizes` must be provided"):
            model.get_image_tokens(pixel_values, None)
        with self.assertRaisesRegex(ValueError, "`image_sizes` must be provided"):
            model.get_image_features(pixel_values, None)


@require_torch
class Apertus1p5VQVAETest(unittest.TestCase):
    """Unit tests for the encode-only IBQ vision tokenizer, with the real 16x geometry at tiny width."""

    def get_model(self):
        from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5VQVAE, Apertus1p5VQVAEConfig

        config = Apertus1p5VQVAEConfig(
            codebook_size=64,
            embed_dim=16,
            latent_channels=16,
            base_channels=32,
            channel_multiplier=(1, 1, 1, 1, 1),  # 5 stages -> 16x downsample, all widths 32
            num_res_blocks=1,
            attn_resolutions=[2],  # attention at the last stage, like the real config (resolution // 16)
            resolution=32,
        )
        torch.manual_seed(0)
        return Apertus1p5VQVAE(config).to(torch_device).eval()

    def test_code_grid_geometry(self):
        model = self.get_model()
        self.assertEqual(model.config.spatial_scale_factor, 16)
        with torch.no_grad():
            codes = model.encode(torch.randn(1, 3, 32, 48, device=torch_device))
        self.assertEqual(codes.shape, (1, 2, 3))  # (32/16, 48/16)
        self.assertEqual(codes.dtype, torch.int64)
        self.assertGreaterEqual(codes.min().item(), 0)
        self.assertLess(codes.max().item(), model.config.codebook_size)

    def test_encode_deterministic(self):
        model = self.get_model()
        pixel_values = torch.randn(1, 3, 32, 32, device=torch_device)
        with torch.no_grad():
            codes_1 = model.encode(pixel_values)
            codes_2 = model.encode(pixel_values)
        self.assertTrue(torch.equal(codes_1, codes_2))

    def test_per_image_encode_matches_same_size_batch(self):
        model = self.get_model()
        pixel_values = torch.randn(2, 3, 32, 32, device=torch_device)
        with torch.no_grad():
            batched = model.encode(pixel_values)
            singles = torch.cat([model.encode(pixel_values[i : i + 1]) for i in range(2)], dim=0)
        self.assertTrue(torch.equal(batched, singles))

    def test_codes_not_degenerate(self):
        model = self.get_model()
        with torch.no_grad():
            codes = model.encode(torch.randn(1, 3, 64, 64, device=torch_device))
        self.assertGreater(codes.unique().numel(), 1)

    def test_no_decoder(self):
        model = self.get_model()
        self.assertFalse(hasattr(model, "decoder"))
        self.assertFalse(hasattr(model, "post_quant_conv"))

    def test_kept_in_fp32_when_loaded_in_half_precision(self):
        """Codes are an argmax over codebook logits: the tokenizer must stay fp32 even in a bf16 model."""
        import tempfile

        model = self.get_model()
        pixel_values = torch.randn(1, 3, 32, 32, device=torch_device)
        with torch.no_grad():
            codes_fp32 = model.encode(pixel_values)

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5VQVAE

            reloaded = Apertus1p5VQVAE.from_pretrained(tmp, dtype=torch.bfloat16).to(torch_device).eval()

        self.assertEqual(reloaded.encoder.conv_in.weight.dtype, torch.float32)
        self.assertEqual(reloaded.quantize.embedding.weight.dtype, torch.float32)
        # fp32 inputs (the processor's output dtype) must yield exactly the fp32 model's codes
        with torch.no_grad():
            codes_reloaded = reloaded.encode(pixel_values)
        self.assertTrue(torch.equal(codes_fp32, codes_reloaded))
        # half-precision inputs are cast internally and must not crash
        with torch.no_grad():
            reloaded.encode(pixel_values.to(torch.bfloat16))


@require_torch
class Apertus1p5IntegrationTest(unittest.TestCase):
    @slow
    @require_bitsandbytes
    def test_model_generation(self):
        model = Apertus1p5ForConditionalGeneration.from_pretrained(
            "BAAI/Apertus1p5-Chat-hf", quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )
        processor = Apertus1p5Processor.from_pretrained("BAAI/Apertus1p5-Chat-hf")

        image = Image.open(requests.get("https://picsum.photos/id/237/200/200", stream=True).raw)
        prompt = "USER: <image>Describe what do you see here and tell me about the history behind it? ASSISTANT:"

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETION = ['USER: 64*64Describe what do you see here and tell me about the history behind it? ASSISTANT: The image captures a moment of tranquility with a black Labrador Retriever resting on a wooden floor. The dog, with its glossy black coat, is lying down with its front legs stretched out in']  # fmt: skip
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_bitsandbytes
    @require_torch_large_accelerator
    def test_model_generation_batched(self):
        model = Apertus1p5ForConditionalGeneration.from_pretrained(
            "BAAI/Apertus1p5-Chat-hf", quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )
        processor = Apertus1p5Processor.from_pretrained("BAAI/Apertus1p5-Chat-hf")
        processor.tokenizer.padding_side = "left"

        image = Image.open(requests.get("https://picsum.photos/id/237/50/50", stream=True).raw)
        image_2 = Image.open(requests.get("https://picsum.photos/id/247/50/50", stream=True).raw)
        prompts = [
            "USER: <image>Describe what do you see here? ASSISTANT:",
            "USER: <image>What can you say about the image? ASSISTANT:",
        ]

        inputs = processor(images=[image, image_2], text=prompts, padding=True, return_tensors="pt").to(
            model.device, torch.float16
        )

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETIONS = Expectations(
            {
                ("xpu", 3): [
                    "USER: 64*64Describe what do you see here? ASSISTANT: The image depicts a black panther in a crouched position. The panther's body is elongated and its head is lowered, suggesting a state of alertness or readiness. The animal's",
                    "USER: 64*64What can you say about the image? ASSISTANT: The image depicts a serene natural landscape. The foreground consists of a grassy area with some patches of bare earth. The middle ground shows a gently sloping hill with a reddish-brown hue,",
                ],
                (None, None): [
                    "USER: 64*64Describe what do you see here? ASSISTANT: The image depicts a black panther in a crouched position. The panther's body is elongated and curved, with its head lowered and ears pointed forward, suggesting alertness or focus.",
                    "USER: 64*64What can you say about the image? ASSISTANT: The image depicts a serene natural landscape. The foreground consists of a grassy area with some patches of bare earth. The middle ground shows a steep, reddish-brown cliff, which could be a",
                ],
                # We switch to A10 on 2025/06/29, and A10 gives strange values
                ("cuda", 8): [
                    'USER: 64*64Describe what do you see here? ASSISTANT: 1.Filed with 1.Computing theComputing.Computing.',
                    'USER: 64*64What can you say about the image? ASSISTANT: 1.Filed with theComputing theComputing.Computing.',
                ],
            }
        )  # fmt: skip
        EXPECTED_TEXT_COMPLETION = EXPECTED_TEXT_COMPLETIONS.get_expectation()

        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_bitsandbytes
    @require_torch_large_accelerator
    def test_model_generation_multi_image(self):
        model = Apertus1p5ForConditionalGeneration.from_pretrained(
            "BAAI/Apertus1p5-Chat-hf", quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )
        processor = Apertus1p5Processor.from_pretrained("BAAI/Apertus1p5-Chat-hf")

        image = Image.open(requests.get("https://picsum.photos/id/237/50/50", stream=True).raw)
        image_2 = Image.open(requests.get("https://picsum.photos/id/247/50/50", stream=True).raw)
        prompt = "USER: <image><image>What do these two images have in common? ASSISTANT:"

        inputs = processor(images=[image, image_2], text=prompt, return_tensors="pt").to(model.device, torch.float16)

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETIONS = Expectations(
                {
                    ("xpu", 3): ['USER: 64*6464*64What do these two images have in common? ASSISTANT: The two images both depict a rhinoceros, yet they are significantly different in terms of focus and clarity. The rhinoceros in the upper image is in sharp focus, showing detailed textures'],
                    (None, None): ["USER: 64*6464*64What do these two images have in common? ASSISTANT: Both images feature a black animal, but they are not the same animal. The top image shows a close-up of a black cow's head, while the bottom image depicts a black cow in a natural"],
                    # We switch to A10 on 2025/06/29, and A10 gives strange values
                    ("cuda", 8): ['USER: 64*6464*64What do these two images have in common? ASSISTANT:Computing.Filed.Filed.11.Computing theComputing.Computing.'],
                }
            )  # fmt: skip
        EXPECTED_TEXT_COMPLETION = EXPECTED_TEXT_COMPLETIONS.get_expectation()
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
