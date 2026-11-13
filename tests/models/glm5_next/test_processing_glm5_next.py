# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import copy
import unittest

from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import Glm5NextProcessor


def _image(index):
    # Each image has its own size and color, so pairing an image with the wrong placeholder changes `pixel_values`
    return Image.new("RGB", (28 * (index + 1), 28), color=(40 * index, 255 - 40 * index, 0))


def _image_block(index):
    # Replaced by the image `index` in `_with_images`, so that test cases can be built without vision deps
    return {"type": "image", "image": index}


def _with_images(value):
    if isinstance(value, dict):
        if value.get("type") == "image" and isinstance(value.get("image"), int):
            return {**value, "image": _image(value["image"])}
        return {key: _with_images(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_with_images(item) for item in value]
    return value


def _assistant_calls(*call_ids):
    tool_calls = [
        {"id": call_id, "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        for call_id in call_ids
    ]
    return {"role": "assistant", "content": "", "tool_calls": tool_calls}


def _tool_result(call_id, content, key="tool_call_id"):
    return {"role": "tool", "content": content, **({key: call_id} if call_id is not None else {})}


USER_TURN = {"role": "user", "content": [_image_block(0), {"type": "text", "text": "What is the weather?"}]}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather of a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]


@require_vision
@require_torch
class Glm5NextProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Glm5NextProcessor
    # Tiny processor created with make_tiny_processor.py from "THUDM/GLM-4.1V-9B-Thinking"
    tiny_model_id = "hf-internal-testing/tiny-processor-glm4v"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        # Glm5Next uses the same token for images and video, so a video's tokens only count as
        # video inside the begin/end span
        cls.video_token = f"<|begin_of_video|>{processor.video_token}<|end_of_video|>"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        return super()._setup_from_pretrained(
            model_id,
            do_sample_frames=False,
            patch_size=4,
            size={"shortest_edge": 12 * 12, "longest_edge": 18 * 18},
            **kwargs,
        )

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 4},
            {"num_frames": None, "fps": 16, "expected_dim": 0, "output_length": 12},
            {"do_sample_frames": False, "fps": 2, "expected_dim": 0, "output_length": 24},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 24},
        ]

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image", "video"])
        image_input = self.prepare_images_inputs()
        video_inputs = self.prepare_videos_inputs()
        inputs_dict = {"text": text, "images": image_input, "videos": video_inputs}
        inputs = processor(**inputs_dict, return_tensors="pt", do_sample_frames=False)

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    @parameterized.expand([(1, "pt")])
    @unittest.skip("Mode requires metadata to be always passed by users")
    def test_apply_chat_template_decoded_video(self, batch_size: int, return_tensors: str):
        pass

    def get_glm5_processor(self):
        processor = self.get_processor()
        processor.chat_template = open(
            hf_hub_download(
                "zai-org/GLM-5.3-Flash", "chat_template.jinja", revision="46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
            ),
            encoding="utf-8",
        ).read()
        return processor

    def assert_images_in_order(self, processor, conversations, image_order):
        # Pairing is right when the output equals processing the rendered prompt with the images in the given order
        conversations = _with_images(conversations)
        out = processor.apply_chat_template(
            conversations,
            tools=TOOLS,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        prompts = processor.apply_chat_template(conversations, tools=TOOLS, tokenize=False)
        if isinstance(conversations[0], list):
            images = [[_image(index) for index in order] for order in image_order]
        else:
            images = [_image(index) for index in image_order]
        expected = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        for key in ("input_ids", "attention_mask", "image_grid_thw", "pixel_values"):
            torch.testing.assert_close(out[key], expected[key], msg=key)

    @parameterized.expand(
        [
            (
                "results_in_reverse_order",
                [
                    _assistant_calls("c1", "c2"),
                    _tool_result("c2", [_image_block(1), {"type": "text", "text": "R2"}]),
                    _tool_result("c1", [_image_block(2), {"type": "text", "text": "R1"}]),
                ],
                [0, 2, 1],
            ),
            (
                "string_result_and_integer_ids",
                [
                    _assistant_calls(1, 2, 3),
                    _tool_result(3, "sunny", key="id"),
                    _tool_result(2, [{"type": "text", "text": "R2"}, _image_block(1)], key="id"),
                    _tool_result(1, [_image_block(2)], key="id"),
                ],
                [0, 2, 1],
            ),
            (
                "aggregated_results",
                [
                    _assistant_calls("a", "b"),
                    {
                        "role": "tool",
                        "content": [
                            {"id": "b", "output": [{"type": "text", "text": "RB"}, _image_block(1)]},
                            {"id": "a", "output": [{"type": "text", "text": "RA"}, _image_block(2)]},
                        ],
                    },
                ],
                [0, 2, 1],
            ),
            (
                "aggregated_and_plain_results",
                [
                    _assistant_calls("a", "b", "c"),
                    _tool_result("c", [_image_block(1)]),
                    {
                        "role": "tool",
                        "content": [
                            {"tool_call_id": "b", "output": [_image_block(2)]},
                            {"tool_call_id": "a", "output": [_image_block(3)]},
                        ],
                    },
                ],
                [0, 3, 2, 1],
            ),
            (
                "aggregated_outputs_without_media",
                [
                    _assistant_calls("a", "b", "c", "d", "e"),
                    {
                        "role": "tool",
                        "content": [
                            {"id": "e", "output": "plain text"},
                            {"id": "d", "output": [{"type": "tool_reference", "name": "get_weather"}]},
                            {"id": "c", "output": None},
                            {"id": "b"},
                            {"id": "a", "output": [_image_block(1)]},
                        ],
                    },
                ],
                [0, 1],
            ),
            (
                "aggregated_results_without_id",
                [
                    _assistant_calls("a", "b"),
                    {
                        "role": "tool",
                        "content": [{"id": "b", "output": [_image_block(1)]}, {"output": [_image_block(2)]}],
                    },
                ],
                [0, 1, 2],
            ),
            (
                "result_without_id",
                [
                    _assistant_calls("c1", "c2"),
                    _tool_result("c2", [_image_block(1)]),
                    _tool_result(None, [_image_block(2)]),
                ],
                [0, 1, 2],
            ),
            (
                "duplicate_result_id",
                [
                    _assistant_calls("c1", "c2"),
                    _tool_result("c2", [_image_block(1)]),
                    _tool_result("c2", [_image_block(2)]),
                ],
                [0, 1, 2],
            ),
            (
                "unknown_result_id",
                [
                    _assistant_calls("c1", "c2"),
                    _tool_result("c2", [_image_block(1)]),
                    _tool_result("c3", [_image_block(2)]),
                ],
                [0, 1, 2],
            ),
            (
                "duplicate_call_id",
                [
                    _assistant_calls("c1", "c1"),
                    _tool_result("c1", [_image_block(1)]),
                    _tool_result("c1", [_image_block(2)]),
                ],
                [0, 1, 2],
            ),
            (
                "no_tool_calls",
                [
                    {"role": "assistant", "content": "Let me check."},
                    _tool_result("c2", [_image_block(1)]),
                    _tool_result("c1", [_image_block(2)]),
                ],
                [0, 1, 2],
            ),
            (
                "two_tool_blocks",
                [
                    _assistant_calls("a", "b"),
                    _tool_result("b", [_image_block(1)]),
                    _tool_result("a", [_image_block(2)]),
                    {"role": "user", "content": "And tomorrow?"},
                    _assistant_calls("c", "d"),
                    {
                        "role": "tool",
                        "content": [
                            {"id": "d", "output": [_image_block(3)]},
                            {"id": "c", "output": [_image_block(4)]},
                        ],
                    },
                ],
                [0, 2, 1, 4, 3],
            ),
        ]
    )
    def test_apply_chat_template_tool_result_images(self, name, turns, image_order):
        # The chat template renders a block of tool results in the order of the calls that requested them, and the
        # images nested in aggregated `{"id": ..., "output": [...]}` results
        processor = self.get_glm5_processor()
        self.assert_images_in_order(processor, [USER_TURN, *turns], image_order)

    def test_apply_chat_template_tool_result_images_batched(self):
        processor = self.get_glm5_processor()
        reversed_results = [
            USER_TURN,
            _assistant_calls("c1", "c2"),
            _tool_result("c2", [_image_block(1)]),
            _tool_result("c1", [_image_block(2)]),
        ]
        aggregated = [
            USER_TURN,
            _assistant_calls("a", "b"),
            {
                "role": "tool",
                "content": [{"id": "b", "output": [_image_block(3)]}, {"id": "a", "output": [_image_block(4)]}],
            },
        ]
        self.assert_images_in_order(processor, [reversed_results, aggregated], [[0, 2, 1], [0, 4, 3]])

    def test_apply_chat_template_tool_results_not_modified(self):
        processor = self.get_glm5_processor()
        conversation = [
            USER_TURN,
            _assistant_calls("a", "b"),
            {
                "role": "tool",
                "content": [{"id": "b", "output": [_image_block(1)]}, {"id": "a", "output": [_image_block(2)]}],
            },
        ]
        conversation = _with_images(conversation)
        original = copy.deepcopy(conversation)
        processor.apply_chat_template(conversation, tools=TOOLS, tokenize=True, return_dict=True)
        self.assertEqual(conversation, original)
