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
import requests

from transformers import (
    AriaConfig,
    AriaForConditionalGeneration,
    AriaModel,
    AriaTextConfig,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    is_torch_available,
    is_vision_available,
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

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image

# Used to be https://aria-vl.github.io/static/images/view.jpg but it was removed, llava-vl has the same image
IMAGE_OF_VIEW_URL = "https://llava-vl.github.io/static/images/view.jpg"


class AriaVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=9,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        text_config=AriaTextConfig(
            seq_length=7,
            is_training=True,
            use_input_mask=True,
            use_token_type_ids=False,
            use_labels=True,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            type_vocab_size=16,
            type_sequence_label_size=2,
            initializer_range=0.02,
            num_labels=3,
            num_choices=4,
            pad_token_id=1,
            hidden_size=32,
            intermediate_size=64,
            max_position_embeddings=60,
            model_type="aria_moe_lm",
            moe_intermediate_size=4,
            moe_num_experts=4,
            moe_topk=2,
            num_attention_heads=8,
            num_experts_per_tok=3,
            num_hidden_layers=2,
            num_key_value_heads=8,
            rope_theta=5000000,
            vocab_size=99,
            eos_token_id=2,
            head_dim=4,
        ),
        is_training=True,
        vision_config=Idefics3VisionConfig(
            image_size=358,
            patch_size=10,
            num_channels=3,
            is_training=True,
            hidden_size=32,
            projection_dim=20,
            num_hidden_layers=2,
            num_attention_heads=16,
            intermediate_size=10,
            dropout=0.1,
            attention_dropout=0.1,
            initializer_range=0.02,
        ),
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.pad_token_id = text_config.pad_token_id
        self.eos_token_id = text_config.eos_token_id
        self.num_hidden_layers = text_config.num_hidden_layers
        self.vocab_size = text_config.vocab_size
        self.hidden_size = text_config.hidden_size
        self.num_attention_heads = text_config.num_attention_heads
        self.is_training = is_training

        self.batch_size = 10
        self.num_channels = 3
        self.image_size = 358
        self.num_image_tokens = 128
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return AriaConfig(
            text_config=self.text_config.to_dict(),
            vision_config=self.vision_config.to_dict(),
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
            eos_token_id=self.eos_token_id,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config.num_channels,
                self.vision_config.image_size,
                self.vision_config.image_size,
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)
        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@slow
@require_torch
class AriaForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `AriaForConditionalGeneration`.
    """

    all_model_classes = (AriaModel, AriaForConditionalGeneration) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False
    _is_composite = True

    def setUp(self):
        self.model_tester = AriaVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AriaConfig, has_text_modality=False)

    @unittest.skip(
        reason="This architecture seems to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seems to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in LLava models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in LLava models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Feedforward chunking is not yet supported")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="Unstable test")
    def test_initialization(self):
        pass

    @unittest.skip(reason="Dynamic control flow due to MoE")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip(reason="Dynamic control flow due to MoE")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip(reason="Aria uses nn.MHA which is not compatible with offloading")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Aria uses nn.MHA which is not compatible with offloading")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Aria uses nn.MHA which is not compatible with offloading")
    def test_disk_offload_safetensors(self):
        pass


SKIP = False
torch_accelerator_module = getattr(torch, torch_device)
memory = 23  # skip on T4 / A10
if hasattr(torch_accelerator_module, "get_device_properties"):
    if torch_accelerator_module.get_device_properties(0).total_memory / 1024**3 < memory:
        SKIP = True


@unittest.skipIf(SKIP, reason="A10 doesn't have enough GPU memory for this tests")
@require_torch
class AriaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("rhymes-ai/Aria")
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_large_accelerator
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        # Let's make sure we test the preprocessing to replace what is used
        model = AriaForConditionalGeneration.from_pretrained(
            "rhymes-ai/Aria",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=["multihead_attn"]),
        )

        prompt = "<|img|>\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT:"
        raw_image = Image.open(requests.get(IMAGE_OF_VIEW_URL, stream=True).raw)
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

    @slow
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
        raw_image = Image.open(requests.get(IMAGE_OF_VIEW_URL, stream=True).raw)
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

    @slow
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
        image1 = Image.open(requests.get(IMAGE_OF_VIEW_URL, stream=True).raw)
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

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

    @slow
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
        image1 = Image.open(requests.get(IMAGE_OF_VIEW_URL, stream=True).raw)
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

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

    @slow
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
        image1 = Image.open(requests.get(IMAGE_OF_VIEW_URL, stream=True).raw)
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(images=[image1, image2, image1], text=prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device, model.dtype)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = Expectations({
            ("cuda", None): ['USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT: When visiting this place, which appears to be a dock or pier extending over a body of water', 'USER:  \nWhat is this?\nASSISTANT: Two cats lying on a bed!\nUSER:  \nAnd this?\nASSISTANT: A cat sleeping on a bed.'],
            ("rocm", (9, 5)): ['USER: \n What are the things I should be cautious about when I visit this place? What should I bring with me?\n ASSISTANT: \n\nWhen visiting this place, you should be cautious about the weather conditions, as it', 'USER: \n What is this?\n ASSISTANT: Two cats lying on a bed!\n USER: \n And this?\n ASSISTANT: A serene lake scene with a wooden dock extending into the water.\n USER: \n']
        }).get_expectation()  # fmt: skip

        decoded_output = processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(decoded_output, EXPECTED_DECODED_TEXT)

    @slow
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
        url1 = "https://images.unsplash.com/photo-1552053831-71594a27632d?q=80&w=3062&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        url2 = "https://images.unsplash.com/photo-1617258683320-61900b281ced?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        image1 = Image.open(requests.get(url1, stream=True).raw)
        image2 = Image.open(requests.get(url2, stream=True).raw)

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

    @slow
    @require_torch_large_accelerator
    @require_bitsandbytes
    def test_generation_no_images(self):
        model_id = "rhymes-ai/Aria"
        model = AriaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, llm_int8_skip_modules=["multihead_attn"]),
        )
        processor = AutoProcessor.from_pretrained(model_id)
        assert model.device.type == "cuda", "This test is only supported on CUDA"  # TODO: remove this
        # Prepare inputs with no images
        inputs = processor(text="Hello, I am", return_tensors="pt").to(torch_device)

        # Make sure that `generate` works
        _ = model.generate(**inputs, max_new_tokens=20)
