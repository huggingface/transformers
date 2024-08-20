# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Llava model."""

import gc
import unittest

import requests
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_bitsandbytes,
    require_torch,
    require_torch_gpu,
    require_vision,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class LlavaVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        text_config={
            "model_type": "llama",
            "seq_length": 7,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
        },
        is_training=True,
        vision_config={
            "image_size": 30,
            "patch_size": 2,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 336
        self.encoder_seq_length = 231

    def get_config(self):
        return LlavaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)
        # set to random non-image token to prevent flakiness
        input_ids[input_ids == config.image_token_index] = 1
        # we are giving 3 images let's make sure we pass in 3 image tokens
        input_ids[:, 1] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_llava_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = LlavaForConditionalGeneration(config=config)
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


@require_torch
class LlavaForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `LlavaForConditionalGeneration`.
    """

    all_model_classes = (LlavaForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (LlavaForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-to-text": LlavaForConditionalGeneration} if is_torch_available() else {}
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = LlavaVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlavaConfig, has_text_modality=False)

    @parameterized.expand([(True,), (False,)])
    def test_greedy_generation(self, use_cache: bool):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            out = model.generate(
                **inputs_dict,
                min_new_tokens=20,
                max_new_tokens=20,
                use_cache=use_cache,
                bad_words_ids=[[config.image_token_index]],
            )
            self.assertTrue(out.shape[1] == inputs_dict["input_ids"].shape[1] + 20)

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in LLava models")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in LLava models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@require_torch
class LlavaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("llava-hf/bakLlava-v1-hf")

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/bakLlava-v1-hf", load_in_4bit=True)

        prompt = "<image>\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT:"
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(prompt, raw_image, return_tensors="pt")

        EXPECTED_INPUT_IDS = torch.tensor([[1, 32000, 28705, 13, 11123, 28747, 1824, 460, 272, 1722,315, 1023, 347, 13831, 925, 684, 739, 315, 3251, 456,1633, 28804, 13, 4816, 8048, 12738, 28747]])  # fmt: skip
        self.assertTrue(torch.equal(inputs["input_ids"], EXPECTED_INPUT_IDS))

        output = model.generate(**inputs, max_new_tokens=20)
        EXPECTED_DECODED_TEXT = "\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT: When visiting this place, there are a few things one should be cautious about. Firstly,"  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama_single(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "llava-hf/llava-1.5-7b-hf"

        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(model_id)

        prompt = "USER: <image>\nWhat are the things I should be cautious about when I visit this place? ASSISTANT:"
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = processor(prompt, raw_image, return_tensors="pt").to(torch_device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=900, do_sample=False)
        EXPECTED_DECODED_TEXT = "USER:  \nWhat are the things I should be cautious about when I visit this place? ASSISTANT: When visiting this place, which is a pier or dock extending over a body of water, there are a few things to be cautious about. First, be aware of the weather conditions, as sudden changes in weather can make the pier unsafe to walk on. Second, be mindful of the water depth and any potential hazards, such as submerged rocks or debris, that could cause accidents or injuries. Additionally, be cautious of the tides and currents, as they can change rapidly and pose a risk to swimmers or those who venture too close to the edge of the pier. Lastly, be respectful of the environment and other visitors, as the pier is a shared space where people can enjoy the view, relax, or engage in recreational activities."  # fmt: skip

        self.assertEqual(
            processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama_batched(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "llava-hf/llava-1.5-7b-hf"

        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(model_id)

        prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me? ASSISTANT:",
            "USER: <image>\nWhat is this? ASSISTANT:",
        ]
        image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(prompts, images=[image1, image2], return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ['USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring with me? ASSISTANT: When visiting this place, which is a pier or dock extending over a body of water, you', 'USER:  \nWhat is this? ASSISTANT: The image features two cats lying down on a pink couch. One cat is located on']  # fmt: skip

        self.assertEqual(
            processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/bakLlava-v1-hf", load_in_4bit=True)
        # The first batch is longer in terms of text, but only has 1 image. The second batch will be padded in text, but the first will be padded because images take more space!.
        prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
            "USER: <image>\nWhat is this?\nASSISTANT:",
        ]
        image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = self.processor(prompts, images=[image1, image2], return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ['USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT: When visiting this place, there are a few things to be cautious about and items to bring along', 'USER:  \nWhat is this?\nASSISTANT: Cats']  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama_batched_regression(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "llava-hf/llava-1.5-7b-hf"

        # Multi-image & multi-prompt (e.g. 3 images and 2 prompts now fails with SDPA, this tests if "eager" works as before)
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", load_in_4bit=True, attn_implementation="eager"
        )
        processor = AutoProcessor.from_pretrained(model_id, pad_token="<pad>")

        prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
            "USER: <image>\nWhat is this?\nASSISTANT: Two cats lying on a bed!\nUSER: <image>\nAnd this?\nASSISTANT:",
        ]
        image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(prompts, images=[image1, image2, image1], return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ['USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT: When visiting this place, which appears to be a dock or pier extending over a body of water', 'USER:  \nWhat is this?\nASSISTANT: Two cats lying on a bed!\nUSER:  \nAnd this?\nASSISTANT: A cat sleeping on a bed.']  # fmt: skip

        self.assertEqual(
            processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_torch
    @require_vision
    def test_batched_generation(self):
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf").to(torch_device)

        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        prompt1 = "<image>\n<image>\nUSER: What's the the difference of two images?\nASSISTANT:"
        prompt2 = "<image>\nUSER: Describe the image.\nASSISTANT:"
        prompt3 = "<image>\nUSER: Describe the image.\nASSISTANT:"
        url1 = "https://images.unsplash.com/photo-1552053831-71594a27632d?q=80&w=3062&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        url2 = "https://images.unsplash.com/photo-1617258683320-61900b281ced?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        image1 = Image.open(requests.get(url1, stream=True).raw)
        image2 = Image.open(requests.get(url2, stream=True).raw)

        inputs = processor(
            text=[prompt1, prompt2, prompt3],
            images=[image1, image2, image1, image2],
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        model = model.eval()

        EXPECTED_OUTPUT = [
            "\n \nUSER: What's the the difference of two images?\nASSISTANT: In the two images, the primary difference is the presence of a small dog in one and a ll",
            "\nUSER: Describe the image.\nASSISTANT: The image features a small, fluffy dog sitting on a sidewalk. The dog is holding",
            "\nUSER: Describe the image.\nASSISTANT: The image features a lone, adult llama standing on a grassy hill. The llama",
        ]

        generate_ids = model.generate(**inputs, max_new_tokens=20)
        outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertEqual(outputs, EXPECTED_OUTPUT)

    @slow
    @require_bitsandbytes
    def test_llava_index_error_bug(self):
        # This is a reproducer of https://github.com/huggingface/transformers/pull/28032 and makes sure it does not happen anymore
        # Please refer to that PR, or specifically https://github.com/huggingface/transformers/pull/28032#issuecomment-1860650043 for
        # more details
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(model_id, load_in_4bit=True)

        processor = AutoProcessor.from_pretrained(model_id)

        # Simulate a super long prompt
        user_prompt = "Describe the image:?\n" * 200
        prompt = f"USER: <image>\n{user_prompt}ASSISTANT:"
        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = processor(prompt, raw_image, return_tensors="pt").to(torch_device, torch.float16)

        # Make sure that `generate` works
        _ = model.generate(**inputs, max_new_tokens=20)

    @slow
    @require_torch_gpu
    def test_llava_merge_inputs_error_bug(self):
        # This is a reproducer of https://github.com/huggingface/transformers/pull/28333 and makes sure it does not happen anymore
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(torch_device)

        # Simulate some user inputs
        pixel_values = torch.randn(
            (2, 3, 336, 336),
            dtype=torch.float,
            device=torch_device,
        )
        input_ids = torch.tensor(
            [
                [32001, 32001, 1, 15043, 7084, 32000, 29871, 13, 7900],
                [1, 15043, 7084, 29901, 29871, 32000, 29871, 13, 7900],
            ],
            dtype=torch.long,
            device=torch_device,
        )
        attention_mask = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.long,
            device=torch_device,
        )

        # Make sure that the loss is properly computed
        loss = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        ).loss
        loss.backward()

    def test_tokenizer_integration(self):
        slow_tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.6-34b", use_fast=False)
        slow_tokenizer.add_tokens("<image>", True)

        fast_tokenizer = AutoTokenizer.from_pretrained(
            "liuhaotian/llava-v1.6-34b",
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            from_slow=True,
            legacy=False,
        )
        fast_tokenizer.add_tokens("<image>", True)

        prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"
        EXPECTED_OUTPUT = ['<|im_start|>', 'system', '\n', 'Answer', '▁the', '▁questions', '.', '<|im_end|>', '<|im_start|>', 'user', '\n', '<image>', '\n', 'What', '▁is', '▁shown', '▁in', '▁this', '▁image', '?', '<|im_end|>', '<|im_start|>', 'ass', 'istant', '\n']  # fmt: skip
        self.assertEqual(slow_tokenizer.tokenize(prompt), EXPECTED_OUTPUT)
        self.assertEqual(fast_tokenizer.tokenize(prompt), EXPECTED_OUTPUT)

    @slow
    @require_bitsandbytes
    def test_generation_no_images(self):
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(model_id, load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(model_id)

        # Prepare inputs with no images
        inputs = processor("Hello, I am", return_tensors="pt").to(torch_device)

        # Make sure that `generate` works
        _ = model.generate(**inputs, max_new_tokens=20)
