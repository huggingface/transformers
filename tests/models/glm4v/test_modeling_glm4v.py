# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GLM-4.1V model."""

import copy
import gc
import unittest

from transformers import (
    AutoProcessor,
    Glm4vConfig,
    Glm4vForConditionalGeneration,
    Glm4vModel,
    is_torch_available,
)
from transformers.testing_utils import (
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch


class Glm4vVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=112,
        video_start_token_id=3,
        video_end_token_id=4,
        image_start_token_id=5,
        image_end_token_id=6,
        image_token_id=7,
        video_token_id=8,
        is_training=True,
        text_config={
            "vocab_size": 99,
            "hidden_size": 16,
            "intermediate_size": 22,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "output_channels": 64,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "rope_scaling": {"type": "default", "mrope_section": [2, 1, 1]},
            "rope_theta": 10000,
            "tie_word_embeddings": True,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 2,
            "hidden_act": "silu",
            "hidden_size": 48,
            "out_hidden_size": 16,
            "intermediate_size": 22,
            "patch_size": 14,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = text_config["bos_token_id"]
        self.eos_token_id = text_config["eos_token_id"]
        self.pad_token_id = text_config["pad_token_id"]
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.text_config = text_config
        self.vision_config = vision_config
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.hidden_size = text_config["hidden_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.vocab_size = text_config["vocab_size"]
        self.num_image_tokens = 64
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return Glm4vConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            video_start_token_id=self.video_start_token_id,
            video_end_token_id=self.video_end_token_id,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.video_start_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_start_token_id] = self.pad_token_id
        input_ids[input_ids == self.video_end_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_end_token_id] = self.pad_token_id

        input_ids[:, 0] = self.image_start_token_id
        input_ids[:, 1 : 1 + self.num_image_tokens] = self.image_token_id
        input_ids[:, 1 + self.num_image_tokens] = self.image_end_token_id
        patch_size = config.vision_config.patch_size
        patches_per_side = self.image_size // patch_size

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, patches_per_side, patches_per_side]] * self.batch_size),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Glm4vModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Glm4vModel, Glm4vForConditionalGeneration) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False
    model_split_percents = [0.7, 0.9]  # model too big to split at 0.5
    _is_composite = True

    def setUp(self):
        self.model_tester = Glm4vVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Glm4vConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    # GLM4V has images shaped as (bs*patch_len, dim) so we can't slice to batches in generate
    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # We don't want a few model inputs in our model input dictionary for generation tests
        input_keys_to_ignore = [
            # we don't want to mask attention heads
            "head_mask",
            "decoder_head_mask",
            "cross_attn_head_mask",
            # we don't want encoder-decoder models to start from filled decoder ids
            "decoder_input_ids",
            "decoder_attention_mask",
            # we'll set cache use in each test differently
            "use_cache",
            # Ignore labels if it is in the input dict
            "labels",
            # model-specific exceptions should overload/overwrite this function
        ]

        # The diff from the general `prepare_config_and_inputs_for_generate` lies here
        patch_size = config.vision_config.patch_size
        filtered_image_length = batch_size * (self.model_tester.image_size**2) // (patch_size**2)
        filtered_inputs_dict = {
            k: v[:batch_size, ...] if isinstance(v, torch.Tensor) else v
            for k, v in inputs_dict.items()
            if k not in input_keys_to_ignore
        }
        filtered_inputs_dict["pixel_values"] = inputs_dict["pixel_values"][:filtered_image_length]

        # It is important set `eos_token_id` to `None` to avoid early stopping (would break for length-based checks)
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

    @unittest.skip(reason="No available kernels - not supported")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Size mismatch")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("Error with compilation")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]
            del inputs["image_grid_thw"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)
            with torch.no_grad():
                model(**inputs)[0]

    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]
            del inputs["image_grid_thw"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            torch.testing.assert_close(out_embeds, out_ids)


@require_torch
class Glm4vIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
        self.message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        self.message2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
                    },
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    def test_small_model_integration_test(self):
        model = Glm4vForConditionalGeneration.from_pretrained(
            "THUDM/GLM-4.1V-9B-Thinking", dtype="auto", device_map="auto"
        )

        inputs = self.processor.apply_chat_template(
            self.message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        expected_input_ids = [151331, 151333, 151336, 198, 151339, 151343, 151343, 151343, 151343, 151343, 151343, 151343, 151343, 151343, 151343, 151343, 151343]  # fmt: skip
        assert expected_input_ids == inputs.input_ids[0].tolist()[:17]

        expected_pixel_slice = torch.tensor(
            [
                [-0.0988, -0.0842, -0.0842],
                [-0.5660, -0.5514, -0.4200],
                [-0.0259, -0.0259, -0.0259],
                [-0.1280, -0.0988, -0.2010],
                [-0.4638, -0.5806, -0.6974],
                [-1.2083, -1.2229, -1.2083],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        assert torch.allclose(expected_pixel_slice, inputs.pixel_values[:6, :3], atol=3e-3)

        # verify generation
        inputs = inputs.to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30)
        EXPECTED_DECODED_TEXT = "\nWhat kind of dog is this?\n<think>Got it, let's look at the image. The animal in the picture is not a dog; it's a cat. Specifically, it looks"
        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch(self):
        model = Glm4vForConditionalGeneration.from_pretrained(
            "THUDM/GLM-4.1V-9B-Thinking", dtype="auto", device_map="auto"
        )
        batch_messages = [self.message] * 2
        inputs = self.processor.apply_chat_template(
            batch_messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "\nWhat kind of dog is this?\n<think>Got it, let's look at the image. The animal in the picture is not a dog; it's a cat. Specifically, it looks",
            "\nWhat kind of dog is this?\n<think>Got it, let's look at the image. The animal in the picture is not a dog; it's a cat. Specifically, it looks"
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_with_video(self):
        processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking", max_image_size={"longest_edge": 50176})
        model = Glm4vForConditionalGeneration.from_pretrained(
            "THUDM/GLM-4.1V-9B-Thinking", dtype=torch.float16, device_map="auto"
        )
        questions = ["Describe this video."] * 2
        video_urls = [
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"
        ] * 2
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_url,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            for question, video_url in zip(questions, video_urls)
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True
        ).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=30)
        EXPECTED_DECODED_TEXT = [
            "\n012345Describe this video.\n<think>Got it, let's analyze the video. First, the scene is a room with a wooden floor, maybe a traditional Japanese room with tatami",
            "\n012345Describe this video.\n<think>Got it, let's analyze the video. First, the scene is a room with a wooden floor, maybe a traditional Japanese room with tatami"
        ]  # fmt: skip
        self.assertEqual(
            processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_expand(self):
        model = Glm4vForConditionalGeneration.from_pretrained(
            "THUDM/GLM-4.1V-9B-Thinking", dtype="auto", device_map="auto"
        )
        inputs = self.processor.apply_chat_template(
            self.message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, num_beams=2, num_return_sequences=2)

        EXPECTED_DECODED_TEXT = [
            "\nWhat kind of dog is this?\n<think>Got it, let's look at the image. The animal in the picture doesn't look like a dog; it's actually a cat. Specifically",
            "\nWhat kind of dog is this?\n<think>Got it, let's look at the image. The animal in the picture doesn't look like a dog; it's actually a cat, specifically"
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch_wo_image(self):
        model = Glm4vForConditionalGeneration.from_pretrained(
            "THUDM/GLM-4.1V-9B-Thinking", dtype="auto", device_map="auto"
        )
        message_wo_image = [
            {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
        ]
        batched_messages = [self.message, message_wo_image]
        inputs = self.processor.apply_chat_template(
            batched_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "\nWhat kind of dog is this?\n<think>Got it, let's look at the image. The animal in the picture is not a dog; it's a cat. Specifically, it looks",
            '\nWho are you?\n<think>Got it, the user is asking "Who are you?" I need to respond appropriately. First, I should clarify that I\'m an AI assistant'
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch_different_resolutions(self):
        model = Glm4vForConditionalGeneration.from_pretrained(
            "THUDM/GLM-4.1V-9B-Thinking", dtype="auto", device_map="auto"
        )
        batched_messages = [self.message, self.message2]
        inputs = self.processor.apply_chat_template(
            batched_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "\nWhat kind of dog is this?\n<think>Got it, let's look at the image. The animal in the picture is not a dog; it's a cat. Specifically, it looks",
            "\nWhat kind of dog is this?\n<think>Got it, let's look at the image. Wait, the animals here are cats, not dogs. The question is about a dog, but"
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_flash_attn
    @require_torch_gpu
    def test_small_model_integration_test_batch_flashatt2(self):
        model = Glm4vForConditionalGeneration.from_pretrained(
            "THUDM/GLM-4.1V-9B-Thinking",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        batched_messages = [self.message, self.message2]
        inputs = self.processor.apply_chat_template(
            batched_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "\nWhat kind of dog is this?\n<think>Got it, let's look at the image. The animal in the picture has a stocky build, thick fur, and a face that's",
            "\nWhat kind of dog is this?\n<think>Got it, let's look at the image. Wait, the animals here are cats, not dogs. The question is about a dog, but"
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_flash_attn
    @require_torch_gpu
    def test_small_model_integration_test_batch_wo_image_flashatt2(self):
        model = Glm4vForConditionalGeneration.from_pretrained(
            "THUDM/GLM-4.1V-9B-Thinking",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        message_wo_image = [
            {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
        ]
        batched_messages = [self.message, message_wo_image]
        inputs = self.processor.apply_chat_template(
            batched_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "\nWhat kind of dog is this?\n<think>Got it, let's look at the image. The animal in the picture is not a dog; it's a cat. Specifically, it looks",
            '\nWho are you?\n<think>Got it, let\'s look at the question. The user is asking "Who are you?" which is a common question when someone meets an AI'
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
