# Copyright 2025 Baidu and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Ernie 4.5 VL model."""

import unittest

from transformers import (
    AutoProcessor,
    Ernie4_5_VLConfig,
    Ernie4_5_VLForConditionalGeneration,
    Ernie4_5_VLModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    require_torch_large_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_cv2_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_cv2_available():
    pass

if is_torch_available():
    import torch

if is_vision_available():
    pass


class Ernie4_5_VLVisionText2TextModelTester:
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
        text_config=None,
        vision_config=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.ignore_index = ignore_index
        self.image_size = image_size
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.is_training = is_training

        self.text_config = text_config
        if text_config is None:
            self.text_config = {
                "vocab_size": 99,
                "hidden_size": 16,
                "intermediate_size": 22,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_act": "silu",
                "max_position_embeddings": 512,
                "tie_word_embeddings": True,
                "rope_parameters": {"type": "ernie_3d", "freq_allocation": 2},
                "moe_intermediate_size": [22, 22],
                "moe_k": 2,
                "moe_num_experts": 8,
                "moe_num_shared_experts": 2,
                "moe_layer_start_index": 0,
                "moe_layer_interval": 1,
                "moe_norm_min": 1e-12,
            }

        self.vision_config = vision_config
        if vision_config is None:
            self.vision_config = {
                "depth": 2,
                "hidden_size": 32,
                "hidden_act": "silu",
                "intermediate_size": 22,
                "num_heads": 2,
                "text_hidden_size": 16,
                "spatial_merge_size": 1,
            }

        self.hidden_size = self.text_config["hidden_size"]
        self.num_hidden_layers = self.text_config["num_hidden_layers"]
        self.num_attention_heads = self.text_config["num_attention_heads"]
        self.vocab_size = self.text_config["vocab_size"]

        self.num_image_tokens = 64
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return Ernie4_5_VLConfig(
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
        pixel_values = floats_tensor(
            [self.batch_size * (self.image_size**2) // (patch_size**2), self.num_channels * (patch_size**2)]
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
            "image_grid_thw": torch.tensor(
                [[1, patches_per_side, patches_per_side]] * self.batch_size, device=torch_device
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Ernie4_5_VLModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Ernie4_5_VLModel,
            Ernie4_5_VLForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    model_split_percents = [0.7, 0.9]  # model too big to split at 0.5
    test_all_params_have_gradient = False  # e score correction bias + moe
    _is_composite = True

    def setUp(self):
        self.model_tester = Ernie4_5_VLVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Ernie4_5_VLConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        """
        Same as in GLM4V, see `tests/models/glm4v/test_modeling_glm4v.py` for reference
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # We don't want a few model inputs in our model input dictionary for generation tests
        input_keys_to_ignore = [
            # we don't want encoder-decoder models to start from filled decoder ids
            "decoder_input_ids",
            "decoder_attention_mask",
            # we'll set cache use in each test differently
            "use_cache",
            # ignore labels if it is in the input dict
            "labels",
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


@slow
@require_torch_multi_accelerator
@require_torch_large_accelerator
@require_torch
class Ernie4_5_VLIntegrationTest(unittest.TestCase):
    # TODO: update path
    model_id = "/raid/anton/code/forks/transformers/AntonV/ErnieVL"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What kind of dog is this?"},
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                ],
            }
        ]
        self.message2 = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What kind of dog is this?"},
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
                    },
                ],
            }
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_small_model_integration_test(self):
        model = Ernie4_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype="auto", device_map="auto"
        )

        inputs = self.processor.apply_chat_template(
            self.message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        expected_input_ids = [100273, 2969, 93963, 1912, 3836, 315, 9159, 357, 501, 94009, 39082, 93919, 4, 93963, 101304, 100295, 100295]  # fmt: skip
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

        # This model on the hub has `do_sample=True`.
        torch.manual_seed(42)

        output = model.generate(**inputs, max_new_tokens=30)
        EXPECTED_DECODED_TEXT = "User: What kind of dog is this?Picture 1:\nAssistant: \n\n\n\nThe animal in the image is a lynx. It's a medium-sized wild cat characterized by its distinctive facial ruff, short tail"
        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch(self):
        model = Ernie4_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype="auto", device_map="auto"
        )
        batch_messages = [self.message] * 2
        inputs = self.processor.apply_chat_template(
            batch_messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        # This model on the hub has `do_sample=True`.
        torch.manual_seed(42)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "User: What kind of dog is this?Picture 1:\nAssistant: \n\n\n\nThe animal in the image is a lynx. It's a medium-sized wild cat characterized by its distinctive facial ruff, short tail",
            "User: What kind of dog is this?Picture 1:\nAssistant: \n\n\n\nThe animal in the image is a lynx. It's a medium-sized wild cat characterized by its distinctive facial ruff, short tail"
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_with_video(self):
        processor = AutoProcessor.from_pretrained(self.model_id, max_image_size={"longest_edge": 50176})
        model = Ernie4_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.float16, device_map="auto"
        )
        questions = ["Only use English during your responses and describe the following video."]
        video_urls = ["https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/tiny_video.mp4"]
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "video",
                            "video": video_url,
                        },
                    ],
                }
            ]
            for question, video_url in zip(questions, video_urls)
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True
        ).to(torch_device)

        # This model on the hub has `do_sample=True`.
        torch.manual_seed(42)

        output = model.generate(**inputs, max_new_tokens=30)
        EXPECTED_DECODED_TEXT = ['User: Only use English during your responses and describe the following video.Video 1:\nAssistant: \n\n\n\nA black-and-white image shows a person lying on their back on a mat in a dojo. They are dressed in a white judo gi']  # fmt: skip

        self.assertEqual(
            processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_expand(self):
        model = Ernie4_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype="auto", device_map="auto"
        )
        inputs = self.processor.apply_chat_template(
            self.message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        # This model on the hub has `do_sample=True`.
        torch.manual_seed(42)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, num_beams=2, num_return_sequences=2)

        EXPECTED_DECODED_TEXT = [
            'User: What kind of dog is this?Picture 1:\nAssistant: \n\n\n\nThe animal in the image is a lynx, not a dog. It has the distinctive features of a lynx, such as tuft',
            'User: What kind of dog is this?Picture 1:\nAssistant: \n\n\n\nThe animal in the image is a lynx, not a dog. It has the distinctive features of a lynx, such as its short'
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch_wo_image(self):
        model = Ernie4_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype="auto", device_map="auto"
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

        # This model on the hub has `do_sample=True`.
        torch.manual_seed(42)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "User: What kind of dog is this?Picture 1:\nAssistant: \n\n\n\nThe animal in the image is a lynx. It's a medium-sized wild cat characterized by its distinctive facial ruff, short tail",
            "User: Who are you?\nAssistant: \n\n\n\nI'm an AI assistant created to help answer questions and provide information on a wide range of topics! I don't have personal experiences or a"
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch_different_resolutions(self):
        model = Ernie4_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype="auto", device_map="auto"
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

        # This model on the hub has `do_sample=True`.
        torch.manual_seed(42)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "User: What kind of dog is this?Picture 1:\nAssistant: \n\n\n\nThe animal in the image is a lynx. It's a medium-sized wild cat characterized by its distinctive facial ruff, short tail",
            'User: What kind of dog is this?Picture 1:\nAssistant: \n\n\n\nthere are no dogs here, there are 2 cats\nI am an AI assistant. User will you give you a task.',
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @require_flash_attn
    @require_torch_gpu
    def test_small_model_integration_test_batch_flashatt2(self):
        model = Ernie4_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
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

        # This model on the hub has `do_sample=True`.
        torch.manual_seed(42)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "User: What kind of dog is this?Picture 1:\nAssistant: \n\n\n\nThe animal in the image is a lynx. It's a medium-sized wild cat characterized by its distinctive facial ruff, short tail",
            'User: What kind of dog is this?Picture 1:\nAssistant: \n\n\n\nthere are no dogs here, there are 2 cats\nI am an AI visual assistant. Here is my analysis of the image'
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @require_flash_attn
    @require_torch_gpu
    def test_small_model_integration_test_batch_wo_image_flashatt2(self):
        model = Ernie4_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
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

        # This model on the hub has `do_sample=True`.
        torch.manual_seed(42)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "User: What kind of dog is this?Picture 1:\nAssistant: \n\n\n\nThe animal in the image is a lynx. It's a medium-sized wild cat characterized by its distinctive facial ruff, short tail",
            "User: Who are you?\nAssistant: \n\n\n\nI am an AI assistant designed to help answer questions, provide information, and assist with tasks. I don't have personal experiences or a physical form"
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
