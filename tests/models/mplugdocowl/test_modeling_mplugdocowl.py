# coding=utf-8
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
"""Testing suite for the PyTorch MPLUGDocOwl model."""

import gc
import unittest

from transformers import (
    AutoProcessor,
    MPLUGDocOwlConfig,
    MPLUGDocOwlForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor, ids_tensor


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class MPLUGDocOwlVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=32000,
        hreducer_hidden_size=1024,
        hreducer_initializer_range=0.02,
        hreducer_layer_norm=1e-6,
        hreducer_conv_shape="1x4",
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="full",
        vision_feature_layer=-2,
        text_config={
            "model_type": "llama",
            # "seq_length": 7,
            # "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "hidden_act": "silu",
            # "hidden_dropout_prob": 0.1,
            # "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            # "type_vocab_size": 16,
            # "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            # "num_labels": 3,
            # "num_choices": 4,
            "pad_token_id": 0,
        },
        is_training=True,
        vision_config={
            "image_size": 448,
            "patch_size": 14,
            "num_channels": 3,
            # "is_training": True,
            "hidden_size": 1024,
            "projection_dim": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            # "intermediate_size": 37,
            # "dropout": 0.1,
            "attention_dropout": 0.0,
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
        return MPLUGDocOwlConfig(
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
        # we are giving 3 images let's make sure we pass in 3 image tokens
        input_ids[:, 1] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_mplugdocowl_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = MPLUGDocOwlForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.float16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


'''
@require_torch
class MPLUGDocOwlForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `MPLUGDocOwlForConditionalGeneration`.
    """

    all_model_classes = (MPLUGDocOwlForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = MPLUGDocOwlVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MPLUGDocOwlConfig, has_text_modality=False)

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

'''


@require_torch
class MPLUGDocOwlForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("/raid/dana/mplug_model_hf")

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    # @require_bitsandbytes
    def test_small_model_integration_test(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = MPLUGDocOwlForConditionalGeneration.from_pretrained("/raid/dana/mplug_model_hf", load_in_4bit=False)

        prompt = "<image>What's the value of the Very well bar in the 65+ age group? Answer the question with detailed explanation."
        image_file = "/raid/dana/test_image.png"
        # raw_image = Image.open(requests.get(image_file, stream=True).raw)
        raw_image = Image.open(image_file)
        inputs = self.processor(prompt, raw_image, return_tensors="pt")
        print(inputs["input_ids"])
        EXPECTED_INPUT_IDS = torch.tensor([[ 1,  3148,  1001, 29901,   529, 10945, 29918,  2492, 29958,  32000,
           529, 29883,  1336, 29918,  2492, 29918,   798, 29900, 29918,  1054,
         29900, 29958,  32000,   529, 29883,  1336, 29918,  2492, 29918,   798,
         29900, 29918,  1054, 29896, 29958,  32000,   529, 29883,  1336, 29918,
          2492, 29918,   798, 29896, 29918,  1054, 29900, 29958,  32000,   529,
         29883,  1336, 29918,  2492, 29918,   798, 29896, 29918,  1054, 29896,
         29958,  32000,   529, 29883,  1336, 29918,  2492, 29918,   798, 29906,
         29918,  1054, 29900, 29958,  32000,   529, 29883,  1336, 29918,  2492,
         29918,   798, 29906, 29918,  1054, 29896, 29958,  32000,  1724, 29915,
         29879,   278,   995,   310,   278, 18064,  1532,  2594,   297,   278,
         29871, 29953, 29945, 29974,  5046,  2318, 29973,   673,   278,  1139,
           411, 13173,  8252, 29889,   319,  1799,  9047, 13566, 29901]])  # fmt: skip

        self.assertTrue(torch.equal(inputs["input_ids"], EXPECTED_INPUT_IDS))

        output = model.generate(**inputs, max_new_tokens=500)
        EXPECTED_DECODED_TEXT = "USER: <global_img> <crop_img_row0_col0> <crop_img_row0_col1> <crop_img_row1_col0> <crop_img_row1_col1> <crop_img_row2_col0> <crop_img_row2_col1> What's the value of the Very well bar in the 65+ age group? Answer the question with detailed explanation. ASSISTANT: 68%\nIn the image, which appears to be a chart from a Pew Research Center report, the bar representing the percentage of people aged 65 and older who believe that Trump fights for their beliefs 'very well' is at 68%."  # fmt: skip
        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    # @require_bitsandbytes
    def test_small_model_integration_test_llama_single(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "/raid/dana/mplug_model_hf"

        model = MPLUGDocOwlForConditionalGeneration.from_pretrained("/raid/dana/mplug_model_hf", load_in_4bit=False)
        processor = AutoProcessor.from_pretrained(model_id)

        prompt = "<image>Recognize text in the image."
        image_file = "/raid/dana/test_image.tif"
        raw_image = Image.open(image_file)
        inputs = processor(prompt, raw_image, return_tensors="pt")  # .to(torch_device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=500, do_sample=False)

        EXPECTED_DECODED_TEXT = "USER: <global_img> <crop_img_row0_col0> <crop_img_row0_col1> <crop_img_row1_col0> <crop_img_row1_col1> <crop_img_row2_col0> <crop_img_row2_col1> Recognize text in the image. ASSISTANT: PHILIP MORRIS MANAGEMENT CORP."
        self.assertEqual(
            processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )


"""
    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama_batched(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "mplugdocowl-hf/mplugdocowl-1.5-7b-hf"

        model = MPLUGDocOwlForConditionalGeneration.from_pretrained("/raid/dana/mplug_model_hf", load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(model_id)

        prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me? ASSISTANT:",
            "USER: <image>\nWhat is this? ASSISTANT:",
        ]
        image1 = Image.open(requests.get("https://mplugdocowl-vl.github.io/static/images/view.jpg", stream=True).raw)
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
        model = MPLUGDocOwlForConditionalGeneration.from_pretrained("mplugdocowl-hf/bakMPLUGDocOwl-v1-hf", load_in_4bit=False)
        # The first batch is longer in terms of text, but only has 1 image. The second batch will be padded in text, but the first will be padded because images take more space!.
        prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
            "USER: <image>\nWhat is this?\nASSISTANT:",
        ]
        image1 = Image.open(requests.get("https://mplugdocowl-vl.github.io/static/images/view.jpg", stream=True).raw)
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
        model_id = "/raid/dana/mplug_model_hf"

        # Multi-image & multi-prompt (e.g. 3 images and 2 prompts now fails with SDPA, this tests if "eager" works as before)
        model = MPLUGDocOwlForConditionalGeneration.from_pretrained(
            "/raid/dana/mplug_model_hf", load_in_4bit=True, attn_implementation="eager"
        )
        processor = AutoProcessor.from_pretrained(model_id, pad_token="<pad>")

        prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
            "USER: <image>\nWhat is this?\nASSISTANT: Two cats lying on a bed!\nUSER: <image>\nAnd this?\nASSISTANT:",
        ]
        image1 = Image.open(requests.get("https://mplugdocowl-vl.github.io/static/images/view.jpg", stream=True).raw)
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
        model = MPLUGDocOwlForConditionalGeneration.from_pretrained("mplugdocowl-hf/mplugdocowl-1.5-7b-hf").to(torch_device)

        processor = AutoProcessor.from_pretrained("/raid/dana/mplug_model_hf")

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
    def test_mplugdocowl_index_error_bug(self):
        # This is a reproducer of https://github.com/huggingface/transformers/pull/28032 and makes sure it does not happen anymore
        # Please refer to that PR, or specifically https://github.com/huggingface/transformers/pull/28032#issuecomment-1860650043 for
        # more details
        model_id = "mplugdocowl-hf/mplugdocowl-1.5-7b-hf"
        model = MPLUGDocOwlForConditionalGeneration.from_pretrained(model_id, load_in_4bit=True)

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
    def test_mplugdocowl_merge_inputs_error_bug(self):
        # This is a reproducer of https://github.com/huggingface/transformers/pull/28333 and makes sure it does not happen anymore
        model_id = "mplugdocowl-hf/mplugdocowl-1.5-7b-hf"
        model = MPLUGDocOwlForConditionalGeneration.from_pretrained(
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
        slow_tokenizer = AutoTokenizer.from_pretrained("/raid/dana/mplug_model_hf", use_fast=False)
        slow_tokenizer.add_tokens("<image>", True)

        fast_tokenizer = AutoTokenizer.from_pretrained(
            "liuhaotian/mplugdocowl-v1.6-34b",
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
"""
