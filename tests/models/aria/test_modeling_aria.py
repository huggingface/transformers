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
"""Testing suite for the PyTorch Aria model."""

import unittest

import pytest

from transformers import (
    AriaConfig,
    AriaForConditionalGeneration,
    AriaModel,
    AriaTextConfig,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    is_torch_available,
)
from transformers.models.idefics3 import Idefics3VisionConfig
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_bitsandbytes,
    require_torch,
    require_torch_large_accelerator,
    require_vision,
    slow,
    torch_device,
)

from ...test_image_processing_common import load_coco_image, load_test_image
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


# Synthetic stand-in for the view image the aria-vl/llava-vl demos used.
IMAGE_OF_VIEW_URL = "https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/llava_view.jpg"


class AriaVisionText2TextModelTester(VLMModelTester):
    base_model_class = AriaModel
    config_class = AriaConfig
    text_config_class = AriaTextConfig
    vision_config_class = Idefics3VisionConfig
    conditional_generation_class = AriaForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 9)
        kwargs.setdefault("pad_token_id", 1)
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 8)
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("intermediate_size", 16)
        kwargs.setdefault("head_dim", 4)
        kwargs.setdefault("max_position_embeddings", 60)
        kwargs.setdefault("moe_num_experts", 3)
        kwargs.setdefault("moe_topk", 2)
        super().__init__(parent, **kwargs)
        self.projector_patch_to_query_dict = {(self.image_size // self.patch_size) ** 2: self.num_image_tokens}


@require_torch
class AriaForConditionalGenerationModelTest(VLMModelTest, unittest.TestCase):
    """
    Model tester for `AriaForConditionalGeneration`.
    """

    model_tester_class = AriaVisionText2TextModelTester

    @pytest.mark.xfail(
        reason="This architecture seems to not compute gradients for the last vision-layernorm because the model uses hidden states pre-norm"
    )
    def test_training_gradient_checkpointing(self):
        super().test_training_gradient_checkpointing()

    @pytest.mark.xfail(
        reason="This architecture seems to not compute gradients for the last vision-layernorm because the model uses hidden states pre-norm"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        super().test_training_gradient_checkpointing_use_reentrant_false()

    @pytest.mark.xfail(
        reason="This architecture seems to not compute gradients for the last vision-layernorm because the model uses hidden states pre-norm"
    )
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        super().test_training_gradient_checkpointing_use_reentrant_true()


SKIP = False
torch_accelerator_module = getattr(torch, torch_device)
memory = 48  # skip on devices that cannot fit Aria's non-quantized MoE expert weights
if hasattr(torch_accelerator_module, "get_device_properties"):
    if torch_accelerator_module.get_device_properties(0).total_memory / 1024**3 < memory:
        SKIP = True


@unittest.skipIf(SKIP, reason="Not enough accelerator memory for Aria integration tests")
@require_torch
@slow
class AriaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("rhymes-ai/Aria")
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_torch_large_accelerator
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        # Let's make sure we test the preprocessing to replace what is used
        model = AriaForConditionalGeneration.from_pretrained(
            "rhymes-ai/Aria",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=["multihead_attn"]),
        )

        prompt = "<|img|>\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT:"
        raw_image = load_test_image(IMAGE_OF_VIEW_URL)
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device, model.dtype)

        non_img_tokens = [
            109, 3905, 2000, 93415, 4551, 1162, 901, 3894, 970, 2478, 1017, 19312, 2388, 1596, 1809, 970, 5449, 1235,
            3333, 93483, 109, 61081, 11984, 14800, 93415
        ]  # fmt: skip
        EXPECTED_INPUT_IDS = torch.tensor([[9] * 256 + non_img_tokens]).to(inputs["input_ids"].device)
        self.assertTrue(torch.equal(inputs["input_ids"], EXPECTED_INPUT_IDS))

        output = model.generate(**inputs, max_new_tokens=20)
        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)

        expected_output = Expectations(
            {
                (
                    "cuda",
                    None,
                ): "\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT: When visiting this place, there are a few things one should be cautious about. Firstly,",
                (
                    "rocm",
                    (9, 5),
                ): "\n USER: What are the things I should be cautious about when I visit this place?\n ASSISTANT: When you visit this place, you should be cautious about the following things:\n\n- The",
            }
        ).get_expectation()
        self.assertEqual(decoded_output, expected_output)

    @require_torch_large_accelerator
    @require_bitsandbytes
    def test_small_model_integration_test_llama_single(self):
        # Let's make sure we test the preprocessing to replace what is used
        model_id = "rhymes-ai/Aria"

        model = AriaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=["multihead_attn"]),
        )
        processor = AutoProcessor.from_pretrained(model_id)

        prompt = "USER: <|img|>\nWhat are the things I should be cautious about when I visit this place? ASSISTANT:"
        raw_image = load_test_image(IMAGE_OF_VIEW_URL)
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device, model.dtype)

        output = model.generate(**inputs, max_new_tokens=90, do_sample=False)
        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", (8, 0)): "USER: \n What are the things I should be cautious about when I visit this place? ASSISTANT: When visiting this beautiful location, it's important to be mindful of a few things to ensure both your safety and the preservation of the environment. Firstly, always be cautious when walking on the wooden pier, as it can be slippery, especially during or after rain. Secondly, be aware of the local wildlife and do not feed or disturb them. Lastly, respect the natural surroundings by not littering and sticking to",
                ("rocm", (9, 5)): "USER: \n What are the things I should be cautious about when I visit this place? ASSISTANT: \n\nWhen visiting this place, you should be cautious about the following:\n\n1. **Weather Conditions**: The weather can be unpredictable, so it's important to check the forecast and dress in layers. Sudden changes in weather can occur, so be prepared for rain or cold temperatures.\n\n2. **Safety on the Dock**: The dock may be slippery, especially when",
            }
        ).get_expectation()  # fmt: off

        decoded_output = processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertEqual(
            decoded_output,
            EXPECTED_DECODED_TEXT,
            f"Expected: {repr(EXPECTED_DECODED_TEXT)}\nActual: {repr(decoded_output)}",
        )

    @require_torch_large_accelerator
    @require_bitsandbytes
    def test_small_model_integration_test_llama_batched(self):
        # Let's make sure we test the preprocessing to replace what is used
        model_id = "rhymes-ai/Aria"

        model = AriaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=["multihead_attn"]),
        )
        processor = AutoProcessor.from_pretrained(model_id)

        prompts = [
            "USER: <|img|>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me? ASSISTANT:",
            "USER: <|img|>\nWhat is this? ASSISTANT:",
        ]
        image1 = load_test_image(IMAGE_OF_VIEW_URL)
        image2 = load_coco_image("000000039769.jpg")

        inputs = processor(images=[image1, image2], text=prompts, return_tensors="pt", padding=True).to(
            model.device, model.dtype
        )

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = Expectations(
            {
                ("cuda", None): [
                    "USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring with me? ASSISTANT: When visiting this place, which is a pier or dock extending over a body of water, you",
                    "USER:  \nWhat is this? ASSISTANT: The image features two cats lying down on a pink couch. One cat is located on",
                ],
                ("rocm", (9, 5)): [
                    "USER: \n What are the things I should be cautious about when I visit this place? What should I bring with me? ASSISTANT: \n\nWhen visiting this place, you should be cautious about the weather conditions, as it",
                    "USER: \n What is this? ASSISTANT: This is a picture of two cats sleeping on a couch. USER: What is the color of",
                ],
            }
        ).get_expectation()

        decoded_output = processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(decoded_output, EXPECTED_DECODED_TEXT)

    @require_torch_large_accelerator
    @require_bitsandbytes
    def test_small_model_integration_test_batch(self):
        # Let's make sure we test the preprocessing to replace what is used
        model = AriaForConditionalGeneration.from_pretrained(
            "rhymes-ai/Aria",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=["multihead_attn"]),
        )
        # The first batch is longer in terms of text, but only has 1 image. The second batch will be padded in text, but the first will be padded because images take more space!.
        prompts = [
            "USER: <|img|>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
            "USER: <|img|>\nWhat is this?\nASSISTANT:",
        ]
        image1 = load_test_image(IMAGE_OF_VIEW_URL)
        image2 = load_coco_image("000000039769.jpg")

        inputs = self.processor(images=[image1, image2], text=prompts, return_tensors="pt", padding=True).to(
            model.device, model.dtype
        )

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = Expectations({
            ("cuda", None): [
                'USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT: When visiting this place, there are a few things to be cautious about and items to bring.',
                'USER:  \nWhat is this?\nASSISTANT: Cats',
            ],
            ("rocm", (9, 5)): [
                'USER: \n What are the things I should be cautious about when I visit this place? What should I bring with me?\n ASSISTANT: \n\nWhen visiting this place, you should be cautious about the following:\n\n-',
                'USER: \n What is this?\n ASSISTANT: This is a picture of two cats sleeping on a couch. The couch is red, and the cats',
            ],
        }).get_expectation()  # fmt: skip

        decoded_output = self.processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(decoded_output, EXPECTED_DECODED_TEXT)

    @require_torch_large_accelerator
    @require_bitsandbytes
    def test_small_model_integration_test_llama_batched_regression(self):
        # Let's make sure we test the preprocessing to replace what is used
        model_id = "rhymes-ai/Aria"

        # Multi-image & multi-prompt (e.g. 3 images and 2 prompts now fails with SDPA, this tests if "eager" works as before)
        model = AriaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=["multihead_attn"]),
        )
        processor = AutoProcessor.from_pretrained(model_id, pad_token="<pad>")

        prompts = [
            "USER: <|img|>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
            "USER: <|img|>\nWhat is this?\nASSISTANT: Two cats lying on a bed!\nUSER: <|img|>\nAnd this?\nASSISTANT:",
        ]
        image1 = load_test_image(IMAGE_OF_VIEW_URL)
        image2 = load_coco_image("000000039769.jpg")

        inputs = processor(images=[image1, image2, image1], text=prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device, model.dtype)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = Expectations({
            ("cuda", None): ['USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT: When visiting this place, which appears to be a dock or pier extending over a body of water', 'USER:  \nWhat is this?\nASSISTANT: Two cats lying on a bed!\nUSER:  \nAnd this?\nASSISTANT: A cat sleeping on a bed.'],
            ("rocm", (9, 5)): ['USER: \n What are the things I should be cautious about when I visit this place? What should I bring with me?\n ASSISTANT: \n\nWhen visiting this place, you should be cautious about the weather conditions, as it', 'USER: \n What is this?\n ASSISTANT: Two cats lying on a bed!\n USER: \n And this?\n ASSISTANT: A serene lake scene with a wooden dock extending into the water.\n USER: \n']
        }).get_expectation()  # fmt: skip

        decoded_output = processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(decoded_output, EXPECTED_DECODED_TEXT)

    @require_torch_large_accelerator
    @require_vision
    @require_bitsandbytes
    def test_batched_generation(self):
        # Skip multihead_attn for 4bit because MHA will read the original weight without dequantize.
        # See https://github.com/huggingface/transformers/pull/37444#discussion_r2045852538.
        model = AriaForConditionalGeneration.from_pretrained(
            "rhymes-ai/Aria",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=["multihead_attn"]),
        )
        processor = AutoProcessor.from_pretrained("rhymes-ai/Aria")

        prompt1 = "<image>\n<image>\nUSER: What's the difference of two images?\nASSISTANT:"
        prompt2 = "<image>\nUSER: Describe the image.\nASSISTANT:"
        prompt3 = "<image>\nUSER: Describe the image.\nASSISTANT:"
        url1 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/unsplash_1552053831-71594a27632d.jpg"
        url2 = "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/unsplash_1617258683320-61900b281ced.jpg"
        image1 = load_test_image(url1)
        image2 = load_test_image(url2)

        # Create inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt1},
                    {"type": "image"},
                    {"type": "text", "text": prompt2},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt3},
                ],
            },
        ]

        prompts = [processor.apply_chat_template([message], add_generation_prompt=True) for message in messages]
        images = [[image1, image2], [image2]]
        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(
            device=model.device, dtype=model.dtype
        )

        EXPECTED_OUTPUTS = Expectations(
            {
                ("cpu", None): [
                    "<|im_start|>user\n<fim_prefix><fim_suffix> <image>\n <image>\n USER: What's the difference of two images?\n ASSISTANT:<fim_prefix><fim_suffix> <image>\n USER: Describe the image.\n ASSISTANT:<|im_end|>\n <|im_start|>assistant\n The first image features a cute, light-colored puppy sitting on a paved surface with",
                    "<|im_start|>user\n<fim_prefix><fim_suffix> <image>\n USER: Describe the image.\n ASSISTANT:<|im_end|>\n <|im_start|>assistant\n The image shows a young alpaca standing on a grassy hill. The alpaca has",
                ],
                ("cuda", None): [
                    "<|im_start|>user\n<fim_prefix><fim_suffix> <image>\n <image>\n USER: What's the difference of two images?\n ASSISTANT:<fim_prefix><fim_suffix> <image>\n USER: Describe the image.\n ASSISTANT:<|im_end|>\n <|im_start|>assistant\n The first image features a cute, light-colored puppy sitting on a paved surface with",
                    "<|im_start|>user\n<fim_prefix><fim_suffix> <image>\n USER: Describe the image.\n ASSISTANT:<|im_end|>\n <|im_start|>assistant\n The image shows a young alpaca standing on a patch of ground with some dry grass. The",
                ],
                ("xpu", 3): [
                    "<|im_start|>user\n<fim_prefix><fim_suffix> <image>\n <image>\n USER: What's the difference of two images?\n ASSISTANT:<fim_prefix><fim_suffix> <image>\n USER: Describe the image.\n ASSISTANT:<|im_end|>\n <|im_start|>assistant\n The first image features a cute, light-colored puppy sitting on a paved surface with",
                    "<|im_start|>user\n<fim_prefix><fim_suffix> <image>\n USER: Describe the image.\n ASSISTANT:<|im_end|>\n <|im_start|>assistant\n The image shows a young alpaca standing on a patch of ground with some dry grass. The",
                ],
                ("rocm", (9, 5)): [
                    "<|im_start|>user\n<fim_prefix><fim_suffix> <image>\n <image>\n USER: What's the difference of two images?\n ASSISTANT:<fim_prefix><fim_suffix> <image>\n USER: Describe the image.\n ASSISTANT:<|im_end|>\n <|im_start|>assistant\n The first image shows a cute golden retriever puppy sitting on a paved surface with a stick",
                    '<|im_start|>user\n<fim_prefix><fim_suffix> <image>\n USER: Describe the image.\n ASSISTANT:<|im_end|>\n <|im_start|>assistant\n The image shows a young llama standing on a patch of ground with some dry grass and dirt. The'
                ],
            }
        )  # fmt: skip
        EXPECTED_OUTPUT = EXPECTED_OUTPUTS.get_expectation()
        generate_ids = model.generate(**inputs, max_new_tokens=20)
        outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertListEqual(outputs, EXPECTED_OUTPUT)

    def test_tokenizer_integration(self):
        model_id = "rhymes-ai/Aria"
        slow_tokenizer = AutoTokenizer.from_pretrained(
            model_id, bos_token="<|startoftext|>", eos_token="<|endoftext|>", use_fast=False
        )
        slow_tokenizer.add_tokens("<image>", True)

        fast_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            from_slow=True,
            legacy=False,
        )
        fast_tokenizer.add_tokens("<image>", True)

        prompt = "<|startoftext|><|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>"
        EXPECTED_OUTPUT = ['<|startoftext|>', '<', '|', 'im', '_', 'start', '|', '>', 'system', '\n', 'Answer', '▁the', '▁questions', '.<', '|', 'im', '_', 'end', '|', '><', '|', 'im', '_', 'start', '|', '>', 'user', '\n', '<image>', '\n', 'What', '▁is', '▁shown', '▁in', '▁this', '▁image', '?', '<', '|', 'im', '_', 'end', '|', '>']  # fmt: skip
        self.assertEqual(slow_tokenizer.tokenize(prompt), EXPECTED_OUTPUT)
        self.assertEqual(fast_tokenizer.tokenize(prompt), EXPECTED_OUTPUT)

    @require_torch_large_accelerator
    @require_bitsandbytes
    def test_generation_no_images(self):
        model_id = "rhymes-ai/Aria"
        model = AriaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=["multihead_attn"]),
        )
        processor = AutoProcessor.from_pretrained(model_id)
        # Prepare inputs with no images
        inputs = processor(text="Hello, I am", return_tensors="pt").to(torch_device)

        # Make sure that `generate` works
        _ = model.generate(**inputs, max_new_tokens=20)
