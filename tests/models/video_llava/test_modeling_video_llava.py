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
"""Testing suite for the PyTorch VideoLlava model."""

import gc
import unittest

import numpy as np
import requests
from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers import (
    VideoLlavaConfig,
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import require_bitsandbytes, require_torch, require_torch_gpu, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class VideoLlavaVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        video_token_index=1,
        projector_hidden_act="gelu",
        seq_length=13,
        num_frames=8,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        text_config={
            "model_type": "llama",
            "seq_length": 13,
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
            "max_position_embeddings": 2048,  # we need it high because videos are 8 frames
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
        },
        is_training=True,
        vision_config={
            "model_type": "clip_vision_model",
            "batch_size": 12,
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
        self.video_token_index = video_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.num_frames = num_frames

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 5
        self.num_channels = 3
        self.image_size = 224
        self.encoder_seq_length = 2044

    def get_config(self):
        return VideoLlavaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            video_token_index=self.video_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
        )

    def prepare_config_and_inputs(self):
        pixel_values_videos = floats_tensor(
            [
                self.batch_size,
                self.num_frames,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )

        pixel_values_images = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values_images, pixel_values_videos

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values_images, pixel_values_videos = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)
        # set to random non-image token to prevent flakiness
        input_ids[input_ids == config.image_token_index] = 2
        input_ids[input_ids == config.video_token_index] = 2

        # we are giving 3 videos and 3 images. Need to pass in image and video tokens
        input_ids[:, 0] = config.video_token_index
        input_ids[:, 1:2] = config.image_token_index
        inputs_dict = {
            "pixel_values_videos": pixel_values_videos,
            "pixel_values_images": pixel_values_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_batched_test(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, _, pixel_values_videos = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)

        # make sure no other special tokens are set
        input_ids[(input_ids == 0) | (input_ids == 1)] = 3
        input_ids[:, 0] = config.video_token_index
        inputs_dict = {
            "pixel_values_videos": pixel_values_videos,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class VideoLlavaForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `VideoLlavaForConditionalGeneration`.
    """

    all_model_classes = (VideoLlavaForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (VideoLlavaForConditionalGeneration,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False

    def setUp(self):
        self.model_tester = VideoLlavaVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VideoLlavaConfig, has_text_modality=False)

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
                bad_words_ids=[[config.image_token_index], [config.video_token_index]],
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

    @unittest.skip(reason="Pass because video-LLava requires `attention_mask is not None`")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Pass because video-LLava requires `attention_mask is not None`")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_mixed_input(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            # test that the forward does not fail
            with torch.no_grad():
                _ = model(**inputs)

            # if we remove some images from inputs leaving only one
            # image number mismatch error should raise
            inputs["pixel_values_images"] = inputs["pixel_values_images"][:1]
            with self.assertRaises(ValueError):
                _ = model(**inputs)

    def test_video_only_input(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            # replace video_token with dummy id which is not video token id
            # error that video-tokens and num-of-video-inputs mismatch will be raised
            inputs["input_ids"][:, 1:2] = 2
            with self.assertRaises(ValueError):
                _ = model(**inputs)

            inputs["pixel_values_images"] = None
            _ = model(**inputs)

    def test_image_only_input(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            # set dummy id, which is not image token id, same as above
            inputs["input_ids"][:, :1] = 2
            with self.assertRaises(ValueError):
                _ = model(**inputs)

            inputs["pixel_values_videos"] = None
            _ = model(**inputs)

    def test_batching_equivalence(self):
        def recursive_check(batched_object, single_row_object, model_name, key):
            if isinstance(batched_object, (list, tuple)):
                for batched_object_value, single_row_object_value in zip(batched_object, single_row_object):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            # do not compare returned loss (0-dim tensor) / codebook ids (int) / caching objects
            elif batched_object is None or not isinstance(batched_object, torch.Tensor):
                return
            elif batched_object.dim() == 0:
                return
            else:
                batched_row = batched_object[:1]
                self.assertFalse(
                    torch.isnan(batched_row).any(), f"Batched output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(batched_row).any(), f"Batched output has `inf` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isnan(single_row_object).any(), f"Single row output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(single_row_object).any(), f"Single row output has `inf` in {model_name} for key={key}"
                )
                self.assertTrue(
                    (torch.max(torch.abs(batched_row - single_row_object))) <= 1e-03,
                    msg=(
                        f"Batched and Single row outputs are not equal in {model_name} for key={key}. "
                        f"Difference={torch.max(torch.abs(batched_row - single_row_object))}."
                    ),
                )

        config, batched_input = self.model_tester.prepare_config_and_inputs_for_batched_test()

        for model_class in self.all_model_classes:
            config.output_hidden_states = True

            model_name = model_class.__name__
            batched_input_prepared = self._prepare_for_class(batched_input, model_class)
            model = model_class(config).to(torch_device).eval()

            single_row_input = {}
            for key, value in batched_input_prepared.items():
                single_row_input[key] = value[:1]

            with torch.no_grad():
                model_batched_output = model(**batched_input_prepared)
                model_row_output = model(**single_row_input)

            for key in model_batched_output:
                recursive_check(model_batched_output[key], model_row_output[key], model_name, key)


@require_torch
class VideoLlavaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", load_in_4bit=True)

        prompt = "USER: <video>Why is this video funny? ASSISTANT:"
        video_file = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
        )
        video_file = np.load(video_file)
        inputs = self.processor(prompt, videos=video_file, return_tensors="pt")

        EXPECTED_INPUT_IDS = torch.tensor([[1,  3148, 1001, 29901, 29871, 32001, 3750, 338, 445, 4863, 2090, 1460, 29973, 319, 1799, 9047, 13566, 29901]])  # fmt: skip
        self.assertTrue(torch.equal(inputs["input_ids"], EXPECTED_INPUT_IDS))

        output = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        EXPECTED_DECODED_TEXT = "USER:  Why is this video funny? ASSISTANT: The video is funny because the baby is playing with a Wii remote while sitting on a bed"  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_mixed_inputs(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", load_in_4bit=True)

        prompts = [
            "USER: <image>What are the cats in the image doing? ASSISTANT:",
            "USER: <video>Why is this video funny? ASSISTANT:",
        ]
        video_file = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
        )
        video_file = np.load(video_file)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = self.processor(prompts, images=[image], videos=[video_file], padding=True, return_tensors="pt")
        output = model.generate(**inputs, do_sample=False, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = [
            'USER:  What are the cats in the image doing? ASSISTANT: The cats in the image are lying down on a red couch, possibly sleeping or rest',
            'USER:  Why is this video funny? ASSISTANT: The video is funny because the baby is playing with a Wii remote while sitting on a bed'
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama(self):
        # Let' s make sure we test the preprocessing to replace what is used

        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", load_in_4bit=True)
        processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

        prompt = "USER: <video>Describe the video in details. ASSISTANT:"
        video_file = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
        )
        video_file = np.load(video_file)
        inputs = self.processor(prompt, videos=video_file, return_tensors="pt").to(torch_device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=900, do_sample=False)
        EXPECTED_DECODED_TEXT = "USER:  Describe the video in details. ASSISTANT: The video features a young child sitting on a bed, holding a book and reading it. " \
            "The child appears to be enjoying the book, as they are fully engaged in the reading process. The bed is located in a bedroom, and there is a chair nearby. " \
            "The child is wearing a light blue shirt and pink pants, and they have glasses on. The room is well-lit, and there is a clock on the wall. The child seems " \
            "to be in a comfortable and relaxed environment, which is conducive to reading and learning. Overall, the video captures a heartwarming moment of a child " \
            "engaging in a simple yet essential activity, which is reading."  # fmt: skip

        self.assertEqual(
            processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama_batched(self):
        # Let' s make sure we test the preprocessing to replace what is used

        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", load_in_4bit=True)
        processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
        processor.tokenizer.padding_side = "left"

        prompts = [
            "USER: <video>What is the baby doing? ASSISTANT:",
            "USER: <video>Who is sitting next to the woman? ASSISTANT:",
        ]
        video_1 = np.load(
            hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset")
        )
        video_2 = np.load(
            hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="video_demo_2.npy", repo_type="dataset")
        )

        inputs = processor(prompts, videos=[video_1, video_2], return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = [
            'USER:  What is the baby doing? ASSISTANT: The baby is sitting on a bed and reading a book.Ъ',
            'USER:  Who is sitting next to the woman? ASSISTANT: A small dog is sitting next to the woman.Ъ'
        ]  # fmt: skip

        self.assertEqual(processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama_batched_regression(self):
        # Let' s make sure we test the preprocessing to replace what is used

        # Multi-image & multi-prompt (e.g. 3 images and 2 prompts now fails with SDPA, this tests if "eager" works as before)
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            "LanguageBind/Video-LLaVA-7B-hf", load_in_4bit=True, attn_implementation="eager"
        )
        processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", pad_token="<pad>")
        processor.tokenizer.padding_side = "left"

        prompts = [
            "USER: <video>What is the baby doing? ASSISTANT:",
            "USER: <video>Who is sitting next to the woman? ASSISTANT: A small dog is sitting next to the woman. USER: <video>What about this video? ASSITANT:",
        ]
        video_1 = np.load(
            hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset")
        )
        video_2 = np.load(
            hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="video_demo_2.npy", repo_type="dataset")
        )

        inputs = processor(prompts, videos=[video_1, video_2, video_1], return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=20)

        # fmt: off
        EXPECTED_DECODED_TEXT = [
            'USER:  What is the baby doing? ASSISTANT: The baby is sitting on a bed and reading a book.Ъ',
            'USER:  Who is sitting next to the woman? ASSISTANT: A small dog is sitting next to the woman. USER:  What about this video? ASSITANT: The video shows a baby sitting on a bed, reading a book. The baby is wearing glass'
        ]
        # fmt: on

        self.assertEqual(processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    @slow
    @require_bitsandbytes
    def test_video_llava_index_error_bug(self):
        # This is a reproducer of https://github.com/huggingface/transformers/pull/28032 and makes sure it does not happen anymore
        # Please refer to that PR, or specifically https://github.com/huggingface/transformers/pull/28032#issuecomment-1860650043 for
        # more details
        model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", load_in_4bit=True)

        # Simulate a super long prompt
        user_prompt = "Describe the video:?\n" * 200
        prompt = f"USER: <video>{user_prompt}ASSISTANT:"
        video_file = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
        )
        video_file = np.load(video_file)

        # let's expand it for 16 frames, to check model can handle any number of frames
        video_file = video_file.repeat(2, 0)
        inputs = self.processor(prompt, videos=video_file, return_tensors="pt").to(torch_device, torch.float16)

        # Make sure that `generate` works
        _ = model.generate(**inputs, max_new_tokens=20)

    @slow
    @require_torch_gpu
    def test_video_llava_merge_inputs_error_bug(self):
        # This is a reproducer of https://github.com/huggingface/transformers/pull/28333 and makes sure it does not happen anymore
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            "LanguageBind/Video-LLaVA-7B-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(torch_device)

        # Simulate some user inputs
        pixel_values_videos = torch.randn(
            (2, 8, 3, 224, 224),
            dtype=torch.float,
            device=torch_device,
        )
        # fmt: off
        input_ids = torch.tensor(
            [
                [
                    32001, 32001, 1, 15043, 7084, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 29871, 13, 7900
                ],
                [
                    1, 15043, 7084, 29901, 29871, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 29871, 13, 7900
                ],
            ],
            dtype=torch.long,
            device=torch_device,
        )
        # fmt: on
        attention_mask = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.long,
            device=torch_device,
        )

        # Make sure that the loss is properly computed
        loss = model(
            pixel_values_videos=pixel_values_videos,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        ).loss
        loss.backward()
