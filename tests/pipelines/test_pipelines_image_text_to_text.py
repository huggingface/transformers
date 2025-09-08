# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import base64
import unittest

from transformers import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING, is_vision_available
from transformers.pipelines import ImageTextToTextPipeline, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    require_torch,
    require_vision,
    slow,
)

from .test_pipelines_common import ANY


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@is_pipeline_test
@require_vision
class ImageTextToTextPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING

    def get_test_pipeline(self, model, tokenizer, processor, image_processor, dtype="float32"):
        pipe = ImageTextToTextPipeline(model=model, processor=processor, dtype=dtype, max_new_tokens=10)
        image_token = getattr(processor.tokenizer, "image_token", "")
        examples = [
            {
                "images": Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                "text": f"{image_token}This is a ",
            },
            {
                "images": "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "text": f"{image_token}Here I see a ",
            },
        ]
        return pipe, examples

    def run_pipeline_test(self, pipe, examples):
        outputs = pipe(examples[0].get("images"), text=examples[0].get("text"))
        self.assertEqual(
            outputs,
            [
                {"input_text": ANY(str), "generated_text": ANY(str)},
            ],
        )

    @require_torch
    def test_small_model_pt_token_text_only(self):
        pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
        text = "What is the capital of France? Assistant:"

        outputs = pipe(text=text)
        self.assertEqual(
            outputs,
            [
                {
                    "input_text": "What is the capital of France? Assistant:",
                    "generated_text": "What is the capital of France? Assistant: The capital of France is Paris.",
                }
            ],
        )

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Write a poem on Hugging Face, the company"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the capital of France?"},
                    ],
                },
            ],
        ]
        outputs = pipe(text=messages)
        self.assertEqual(
            outputs,
            [
                [
                    {
                        "input_text": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"}],
                            }
                        ],
                        "generated_text": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"}],
                            },
                            {
                                "role": "assistant",
                                "content": "Hugging Face, a company of minds\nWith tools and services that make our lives easier\nFrom natural language processing\nTo machine learning and more, they've got it all\n\nThey've made it possible for us to be more\nInformed and efficient, with their tools and services\nFrom image and speech recognition\nTo text and language translation, they've got it all\n\nThey've made it possible for us to be more\nInformed and efficient, with their tools and services\nFrom image and speech recognition\nTo text and language translation, they've got it all\n\nThey've made it possible for us to be more\nInformed and efficient, with their tools and services\nFrom image and speech recognition\nTo text and language translation, they've got it all\n\nThey've made it possible for us to be more\nInformed and efficient, with their tools and services\nFrom image and speech recognition\nTo text and language translation, they've got it all\n\nThey've made it possible for us to be more\nInformed and efficient, with their tools and services\nFrom image and speech recognition\nTo text and language translation, they've got it all\n\nThey've made it possible for us to be more\nInformed and efficient, with their tools and",
                            },
                        ],
                    }
                ],
                [
                    {
                        "input_text": [
                            {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}
                        ],
                        "generated_text": [
                            {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
                            {"role": "assistant", "content": "Paris"},
                        ],
                    }
                ],
            ],
        )

    @require_torch
    def test_small_model_pt_token(self):
        pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        text = "<image> What this is? Assistant: This is"

        outputs = pipe(image, text=text)
        self.assertEqual(
            outputs,
            [
                {
                    "input_text": "<image> What this is? Assistant: This is",
                    "generated_text": "<image> What this is? Assistant: This is a photo of two cats lying on a pink blanket. The cats are sleeping and appear to be comfortable. The photo captures a moment of tranquility and companionship between the two feline friends.",
                }
            ],
        )

        outputs = pipe([image, image], text=[text, text])
        self.assertEqual(
            outputs,
            [
                {
                    "input_text": "<image> What this is? Assistant: This is",
                    "generated_text": "<image> What this is? Assistant: This is a photo of two cats lying on a pink blanket. The cats are facing the camera, and they appear to be sleeping or resting. The blanket is placed on a couch, and the cats are positioned in such a way that they are facing the camera. The image captures a peaceful moment between the two cats, and it's a great way to showcase their cuteness and relaxed demeanor.",
                },
                {
                    "input_text": "<image> What this is? Assistant: This is",
                    "generated_text": "<image> What this is? Assistant: This is a photo of two cats lying on a pink blanket. The cats are facing the camera, and they appear to be sleeping or resting. The blanket is placed on a couch, and the cats are positioned in such a way that they are facing the camera. The image captures a peaceful moment between the two cats, and it's a great way to showcase their cuteness and relaxed demeanor.",
                },
            ],
        )

    @require_torch
    def test_consistent_batching_behaviour(self):
        pipe = pipeline("image-text-to-text", model="microsoft/kosmos-2-patch14-224")
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        prompt = "a photo of"

        outputs = pipe([image, image], text=[prompt, prompt], max_new_tokens=10)
        outputs_batched = pipe([image, image], text=[prompt, prompt], batch_size=2, max_new_tokens=10)
        self.assertEqual(outputs, outputs_batched)

    @slow
    @require_torch
    def test_model_pt_chat_template(self):
        pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
        image_ny = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        image_chicago = "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s the difference between these two images?"},
                    {"type": "image"},
                    {"type": "image"},
                ],
            }
        ]
        outputs = pipe([image_ny, image_chicago], text=messages, return_full_text=True, max_new_tokens=10)
        self.assertEqual(
            outputs,
            [
                {
                    "input_text": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What’s the difference between these two images?"},
                                {
                                    "type": "image",
                                    "image": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                                },
                                {
                                    "type": "image",
                                    "image": "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg",
                                },
                            ],
                        }
                    ],
                    "generated_text": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What’s the difference between these two images?"},
                                {
                                    "type": "image",
                                    "image": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                                },
                                {
                                    "type": "image",
                                    "image": "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg",
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": "The first image shows a statue of Liberty in the",
                        },
                    ],
                }
            ],
        )

    @slow
    @require_torch
    def test_model_pt_chat_template_continue_final_message(self):
        pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "There is a dog and"},
                ],
            },
        ]
        outputs = pipe(text=messages, max_new_tokens=10)
        self.assertEqual(
            outputs,
            [
                {
                    "input_text": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                                },
                                {"type": "text", "text": "Describe this image."},
                            ],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": "There is a dog and"}]},
                    ],
                    "generated_text": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                                },
                                {"type": "text", "text": "Describe this image."},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "There is a dog and a person in the image. The dog is sitting",
                                }
                            ],
                        },
                    ],
                }
            ],
        )

    @slow
    @require_torch
    def test_model_pt_chat_template_new_text(self):
        pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        outputs = pipe(text=messages, return_full_text=False, max_new_tokens=10)
        self.assertEqual(
            outputs,
            [
                {
                    "input_text": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                                },
                                {"type": "text", "text": "Describe this image."},
                            ],
                        }
                    ],
                    "generated_text": "In the image, a woman is sitting on the",
                }
            ],
        )

    @slow
    @require_torch
    def test_model_pt_chat_template_image_url(self):
        pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                        },
                    },
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        outputs = pipe(text=messages, return_full_text=False, max_new_tokens=10)[0]["generated_text"]
        self.assertEqual(outputs, "A statue of liberty in the foreground of a city")

    @slow
    @require_torch
    def test_model_pt_chat_template_image_url_base64(self):
        with open("./tests/fixtures/tests_samples/COCO/000000039769.png", "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        pipe = pipeline("image-text-to-text", model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        outputs = pipe(text=messages, return_full_text=False, max_new_tokens=10)[0]["generated_text"]
        self.assertEqual(outputs, "Two cats are sleeping on a pink blanket, with")
