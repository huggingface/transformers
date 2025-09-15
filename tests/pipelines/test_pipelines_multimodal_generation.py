# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import sys
import unittest

import numpy as np

from transformers import MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES, is_vision_available
from transformers.pipelines import MultimodalGenerationPipeline, pipeline
from transformers.testing_utils import (
    is_pipeline_test,
    require_librosa,
    require_torch,
    require_vision,
    slow,
)

from .test_pipelines_common import ANY


sys.path.append(".")
from utils.fetch_hub_objects_for_ci import url_to_local_path


if is_vision_available():
    import PIL


@is_pipeline_test
@require_vision
@require_librosa
@require_torch
class MultimodalGenerationPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES

    def get_test_pipeline(self, model, tokenizer, processor, dtype="float32"):
        _is_images_supported = hasattr(processor, "image_processor")
        _is_videos_supported = hasattr(processor, "video_processor")
        _is_audios_supported = hasattr(processor, "feature_extractor")

        image_token = getattr(processor.tokenizer, "image_token", "")
        video_token = getattr(processor.tokenizer, "video_token", "")
        audio_token = getattr(processor.tokenizer, "audio_token", "")

        images_examples = [
            {
                "images": "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "text": f"{image_token}This is a ",
            },
            {
                "images": "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "text": f"{image_token}Here I see a ",
            },
        ]

        videos_examples = [
            {
                "videos": url_to_local_path(
                    "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/Big_Buck_Bunny_720_10s_10MB.mp4"
                ),
                "text": f"{video_token}This video shows a ",
            },
            {
                "video": url_to_local_path(
                    "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"
                ),
                "text": f"{video_token}In the video I see a ",
            },
        ]
        self.video_path = url_to_local_path(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"
        )

        audio_examples = [
            {
                "audio": url_to_local_path(
                    "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/glass-breaking-151256.mp3"
                ),
                "text": f"{audio_token}This is sound of a ",
            },
            {
                "audio": url_to_local_path(
                    "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav"
                ),
                "text": f"{audio_token}Here I hear a ",
            },
        ]
        self.audio_path = url_to_local_path(
            "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav"
        )

        examples = []
        if _is_images_supported:
            examples.extend(images_examples)
        if _is_videos_supported:
            examples.extend(videos_examples)
        if _is_audios_supported:
            examples.extend(audio_examples)

        pipe = MultimodalGenerationPipeline(model=model, processor=processor, dtype=dtype, max_new_tokens=10)

        return pipe, examples

    def run_pipeline_test_single(self, pipe, examples):
        outputs = pipe(examples[0])
        self.assertEqual(
            outputs,
            [
                {"input_text": ANY(str), "generated_text": ANY(str)},
            ],
        )

    def run_pipeline_test_batched(self, pipe, examples):
        outputs = pipe(examples)
        self.assertEqual(
            outputs,
            [
                {"input_text": ANY(str), "generated_text": ANY(str)},
            ],
        )

    def run_pipeline_test_generation_mode(self, pipe, examples):
        with self.assertRaises(ValueError):
            pipe(examples, generation_mode="video")

        with self.assertRaises(ValueError):
            pipe(examples, generation_mode="audio", return_full_text=True)

        with self.assertRaises(ValueError):
            pipe(examples, generation_mode="image", return_type=1)

    def run_pipeline_test_chat_template(self, pipe, examples):
        if getattr(pipeline.processor, "chat_template", None) is None:
            self.skipTest("The current model has no chat template defined in its processor.")

        pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")

        messages = []
        for example in examples:
            example.pop("text")
            modality_type, modality_data = list(example.items())[0]
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is a "},
                    {"type": modality_type, "path": modality_data},
                ],
            }
            messages.append(message)
        outputs = pipe(messages, return_full_text=True, max_new_tokens=10)

        self.assertEqual(len(outputs), len(messages))
        self.assertIsInstance(outputs[0], dict)
        for out in outputs:
            self.assertTrue("input_text" in out)
            self.assertTrue("generated_text" in out)

    @slow
    def test_small_model_pt_token_text_only(self):
        pipe = pipeline("multimodal-generation", model="Qwen/Qwen2.5-Omni-3B")
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

    @slow
    def test_small_model_pt_token_omni(self):
        pipe = pipeline("multimodal-generation", model="Qwen/Qwen2.5-Omni-3B")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this video."},
                        {"type": "video", "video": self.video_path},
                    ],
                },
            ],
        ]
        outputs = pipe(text=messages, fps=2, max_new_tokens=20)
        self.assertEqual(
            outputs,
            [
                {
                    "input_text": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Describe this video."}],
                        }
                    ],
                    "generated_text": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Describe this video."}],
                        },
                        {
                            "role": "assistant",
                            "content": "The video shows a ",
                        },
                    ],
                }
            ],
        )

        outputs = pipe(text=messages, generation_mode="audio", fps=2, max_new_tokens=20)

        self.assertEqual(len(outputs), len(messages))
        self.assertIsInstance(outputs[0], dict)
        for out in outputs:
            self.assertTrue("input_text" in out)
            self.assertTrue("generated_audio" in out)
            self.assertIsInstance(out["generated_audio"], np.array)

    @slow
    def test_small_model_pt_image_gen(self):
        pipe = pipeline("multimodal-generation", model="deepseek-community/Janus-Pro-1B")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "A dog running under the rain."},
                    ],
                },
            ],
        ]
        outputs = pipe(text=messages, generation_mode="image")

        self.assertEqual(len(outputs), len(messages))
        self.assertIsInstance(outputs[0], dict)
        for out in outputs:
            self.assertTrue("input_text" in out)
            self.assertTrue("generated_audio" in out)
            self.assertIsInstance(out["generated_audio"], PIL.Image.Image)
