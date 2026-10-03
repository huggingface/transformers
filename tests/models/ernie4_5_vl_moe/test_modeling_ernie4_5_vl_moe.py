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

import copy
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
        return torch.ones_like(input_ids)

    def create_pixel_values(self):
        return floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels * (self.patch_size**2),
            ]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[:, 0] = self.image_start_token_id
        input_ids[:, 1 : 1 + self.num_image_tokens] = self.image_token_id
        input_ids[:, 1 + self.num_image_tokens] = self.image_end_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        patches_per_side = self.image_size // self.patch_size
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        return {
            "image_grid_thw": torch.tensor(
                [[1, patches_per_side, patches_per_side]] * self.batch_size, device=torch_device
            ),
            "mm_token_type_ids": mm_token_type_ids,
        }


@require_torch
class Ernie4_5_VLMoeModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Ernie4_5_VLMoeVisionText2TextModelTester
    model_split_percents = [0.7, 0.9]  # model too big to split at 0.5
    test_all_params_have_gradient = False  # e score correction bias + moe

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

    def test_mismatching_num_image_tokens(self):
        # Override the base test because we need to slice image_grid_thw too
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**input_dict)  # successful forward with no modifications
            curr_input_dict = copy.deepcopy(input_dict)

            # remove one image but leave the image token in text
            patch_size = config.vision_config.patch_size
            one_img_length = (self.model_tester.image_size**2) // (patch_size**2)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-one_img_length:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
                _ = model(**curr_input_dict)

            model.base_model.rope_deltas = None
            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            mm_token_type_ids = curr_input_dict["mm_token_type_ids"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=torch.cat([mm_token_type_ids, mm_token_type_ids], dim=0),
                )

            model.base_model.rope_deltas = None
            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            mm_token_type_ids = torch.cat(
                [curr_input_dict["mm_token_type_ids"][:1], curr_input_dict["mm_token_type_ids"][:1]], dim=0
            )
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )

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
