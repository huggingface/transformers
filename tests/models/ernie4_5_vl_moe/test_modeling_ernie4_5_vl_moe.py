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

from parameterized import parameterized

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Ernie4_5_VLMoeConfig,
    Ernie4_5_VLMoeForConditionalGeneration,
    Ernie4_5_VLMoeModel,
    is_torch_available,
)
from transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe import (
    Ernie4_5_VLMoeTextConfig,
    Ernie4_5_VLMoeVisionConfig,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_deterministic_for_xpu,
    require_torch,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class Ernie4_5_VLMoeVisionText2TextModelTester(VLMModelTester):
    base_model_class = Ernie4_5_VLMoeModel
    config_class = Ernie4_5_VLMoeConfig
    text_config_class = Ernie4_5_VLMoeTextConfig
    vision_config_class = Ernie4_5_VLMoeVisionConfig
    conditional_generation_class = Ernie4_5_VLMoeForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("video_start_token_id", 3)
        kwargs.setdefault("video_end_token_id", 4)
        kwargs.setdefault("image_start_token_id", 5)
        kwargs.setdefault("image_end_token_id", 6)
        kwargs.setdefault("image_token_id", 7)
        kwargs.setdefault("video_token_id", 8)
        kwargs.setdefault("image_size", 112)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("num_image_tokens", 64)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("num_key_value_heads", 1)
        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("rope_parameters", {"type": "default", "rope_theta": 500_000.0, "mrope_section": [3, 3, 2]})
        kwargs.setdefault("mlp_layer_types", ["dense", "sparse"])
        kwargs.setdefault("moe_intermediate_size", [32, 32])
        kwargs.setdefault("moe_norm_min", 1e-12)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("num_heads", 2)
        kwargs.setdefault("spatial_merge_size", 1)
        super().__init__(parent, **kwargs)

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {
            self.video_token_id,
            self.video_start_token_id,
            self.video_end_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
        }

    def create_attention_mask(self, input_ids):
        # The M-RoPE index only counts unmasked tokens, so the default mask would pad out image placeholders
        return torch.ones_like(input_ids)

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor(
            [
                batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels * (self.patch_size**2),
            ]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[:, 0] = self.image_start_token_id
        input_ids[:, 1 : 1 + self.num_image_tokens] = self.image_token_id
        input_ids[:, 1 + self.num_image_tokens] = self.image_end_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        patches_per_side = self.image_size // self.patch_size
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        return {
            "image_grid_thw": torch.tensor(
                [[1, patches_per_side, patches_per_side]] * batch_size, device=torch_device
            ),
            "mm_token_type_ids": mm_token_type_ids,
        }


@require_torch
class Ernie4_5_VLMoeModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Ernie4_5_VLMoeVisionText2TextModelTester
    model_split_percents = [0.7, 0.9]  # model too big to split at 0.5
    test_all_params_have_gradient = False  # e score correction bias + moe

    @unittest.skip(
        reason="Its TP and EP plans don't reach the text and vision experts yet; they come with their own PR, "
        "together with the loading fix that splits the checkpoint's expert stack."
    )
    def test_moe_parallel_plans_shard_experts(self):
        pass

    def _video_features_prepare_config_and_inputs(self):
        """
        Helper method to extract only video-related inputs from the full set of inputs, for testing `get_video_features`.

        The superclass method simply calls the model_tester.prepare_config_and_inputs_for_common(),
        but that method only prepared image inputs, i.e. where the temporal dimension in grid_thw is 1.
        This override prepares proper video inputs with 12 frames.
        """
        config = self.model_tester.get_config()
        patch_size = config.vision_config.patch_size
        batch_size = self.model_tester.batch_size
        image_size = self.model_tester.image_size
        num_channels = self.model_tester.num_channels
        num_frames = 12
        pixel_values_videos = floats_tensor(
            [num_frames * batch_size * (image_size**2) // (patch_size**2), num_channels * (patch_size**2)]
        )

        patches_per_side = image_size // patch_size
        video_grid_thw = torch.tensor(
            [[num_frames, patches_per_side, patches_per_side]] * batch_size, device=torch_device
        )
        inputs_dict = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }
        return config, inputs_dict

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("Model cannot scale due to pre-rotations when computing freqs")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    @unittest.skip("Model cannot scale due to pre-rotations when computing freqs")
    def test_model_rope_scaling_frequencies(self):
        pass


@slow
@require_torch_large_accelerator(memory=64)  # Tested on A100 / torch 2.9.0
@require_torch
class Ernie4_5_VLMoeIntegrationTest(unittest.TestCase):
    model = None
    model_id = "baidu/ERNIE-4.5-VL-28B-A3B-PT"

    # TODO: remove revision when PR on the hub is merged
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

        self.processor = AutoProcessor.from_pretrained(self.model_id, revision="refs/pr/11")
        self.message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What kind of dog is this?"},
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
                        ),
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
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/coco_sample.png"
                        ),
                    },
                ],
            }
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def load_model(self, dtype, attn_implementation="sdpa"):
        return AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            device_map="auto",
            dtype=dtype,
            attn_implementation=attn_implementation,
            experts_implementation="eager",
            revision="refs/pr/11",
        )

    def test_small_model_integration_test(self):
        model = self.load_model("auto")
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
        EXPECTED_DECODED_TEXT = "The animal in the image is a lynx, not a dog. It's a wild cat species known for its distinctive ear tufts and"
        self.assertEqual(
            self.processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch(self):
        model = self.load_model("auto")
        batch_messages = [self.message] * 2
        inputs = self.processor.apply_chat_template(
            batch_messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        # This model on the hub has `do_sample=True`.
        torch.manual_seed(42)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "The animal in the image is a lynx, not a dog. It's a wild cat species known for its distinctive ear tufts and",
            "The animal in the image is a lynx, not a dog. It's a wild cat species characterized by its distinctive ear tufts,"
        ]  # fmt: skip

        self.assertEqual(
            [
                self.processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
                self.processor.decode(output[1][len(inputs["input_ids"][1]) :], skip_special_tokens=True),
            ],
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_with_video(self):
        processor = AutoProcessor.from_pretrained(
            self.model_id, max_image_size={"longest_edge": 50176}, revision="refs/pr/11"
        )
        model = self.load_model(dtype=torch.float16)
        questions = ["Only use English during your responses. Describe the following video."]
        video_urls = [
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tiny_video.mp4"
        ]
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
        EXPECTED_DECODED_TEXT = 'A black-and-white image shows a person lying on their back on a mat in a dojo. They are dressed in a white judo gi'  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_expand(self):
        model = self.load_model("auto")
        inputs = self.processor.apply_chat_template(
            self.message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        # This model on the hub has `do_sample=True`.
        torch.manual_seed(42)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, num_beams=2, num_return_sequences=2)

        EXPECTED_DECODED_TEXT = [
            'The animal in the image is a lynx, not a dog. It has the distinctive features of a lynx, such as tuft',
            'The animal in the image is a lynx, not a dog. It has the distinctive features of a lynx, including a short tail'
        ]  # fmt: skip

        self.assertEqual(
            [
                self.processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
                self.processor.decode(output[1][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
            ],
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch_wo_image(self):
        model = self.load_model("auto")
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
            "The animal in the image is a lynx. It's a medium-sized wild cat characterized by its distinctive facial ruff, short tail",
            "I am an AI assistant designed to help answer questions, provide information, and assist with tasks. I don't have personal experiences or a physical form"
        ]  # fmt: skip

        self.assertEqual(
            [
                self.processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
                self.processor.decode(output[1][len(inputs["input_ids"][1]) :], skip_special_tokens=True),
            ],
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch_different_resolutions(self):
        model = self.load_model("auto")
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
            'The animal in the image is a lynx, not a dog. It has the distinctive features of a lynx, such as tuft',
            'there are no dogs here, there are 2 cats',
        ]  # fmt: skip

        self.assertEqual(
            [
                self.processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
                self.processor.decode(output[1][len(inputs["input_ids"][1]) :], skip_special_tokens=True),
            ],
            EXPECTED_DECODED_TEXT,
        )


# Garbage output expected as it is a dummy model to be run on the CI
@slow
@require_torch
class Ernie4_5_VLMoeSmallIntegrationTest(unittest.TestCase):
    model = None
    model_id = "hf-internal-testing/Ernie-VL-Moe-Small"

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
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
                        ),
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
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/coco_sample.png"
                        ),
                    },
                ],
            }
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def load_model(self, dtype, attn_implementation="sdpa"):
        return AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            device_map="auto",
            dtype=dtype,
            attn_implementation=attn_implementation,
            experts_implementation="eager",
        )

    def test_small_model_integration_test(self):
        model = self.load_model("auto")
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
        EXPECTED_DECODED_TEXT = "知道了知道了attaatta不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如"
        self.assertEqual(
            self.processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch(self):
        model = self.load_model("auto")
        batch_messages = [self.message] * 2
        inputs = self.processor.apply_chat_template(
            batch_messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        # This model on the hub has `do_sample=True`.
        torch.manual_seed(42)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        # fmt: off
        expectations = Expectations(
            {
                ("xpu", None): [
                    '知道了知道了attaatta不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如',
                    '填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空',
                ],
                (None, None): [
                    '知道了知道了attaatta不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如',
                    '不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊',
                ],
            }
        )
        EXPECTED_DECODED_TEXT = expectations.get_expectation()
        # fmt: on

        self.assertEqual(
            [
                self.processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
                self.processor.decode(output[1][len(inputs["input_ids"][1]) :], skip_special_tokens=True),
            ],
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_with_video(self):
        processor = AutoProcessor.from_pretrained(self.model_id, max_image_size={"longest_edge": 50176})
        model = self.load_model(dtype=torch.float16)
        questions = ["Only use English during your responses. Describe the following video."]
        video_urls = [
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tiny_video.mp4"
        ]
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
        EXPECTED_DECODED_TEXT = 'uschuschusch载载载载载载载载载载载载载载载载载载载载载载载载载载载'  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @require_deterministic_for_xpu
    def test_small_model_integration_test_expand(self):
        model = self.load_model("auto")
        inputs = self.processor.apply_chat_template(
            self.message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        # This model on the hub has `do_sample=True`.
        torch.manual_seed(42)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, num_beams=2, num_return_sequences=2)

        EXPECTED_DECODED_TEXT = [
            '不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊错的错的错的错的错的错的错的错的错的错的错的错的错的',
            '不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊不是啊错的错的错的错的错的错的错的错的错的错的错的错的就是这样',
        ]  # fmt: skip

        self.assertEqual(
            [
                self.processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
                self.processor.decode(output[1][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
            ],
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch_wo_image(self):
        model = self.load_model("auto")
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
            '知道了知道了attaatta不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如',
            '用具柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄柄',
        ]  # fmt: skip

        self.assertEqual(
            [
                self.processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
                self.processor.decode(output[1][len(inputs["input_ids"][1]) :], skip_special_tokens=True),
            ],
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch_different_resolutions(self):
        model = self.load_model("auto")
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
            '知道了知道了attaatta不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如不如',
            '填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空填空',
        ]  # fmt: skip

        self.assertEqual(
            [
                self.processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True),
                self.processor.decode(output[1][len(inputs["input_ids"][1]) :], skip_special_tokens=True),
            ],
            EXPECTED_DECODED_TEXT,
        )
