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

from transformers import MODEL_FOR_MULTIMODAL_LM_MAPPING, is_vision_available
from transformers.pipelines import AnyToAnyPipeline, pipeline
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
class AnyToAnyPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_MULTIMODAL_LM_MAPPING

    # We only need `processor` but the Mixin will pass all possible preprocessing classes for a model.
    # So we add them all in signature
    def get_test_pipeline(
        self, model, tokenizer, processor, image_processor=None, feature_extractor=None, dtype="float32"
    ):
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

        examples = []
        if _is_images_supported:
            examples.extend(images_examples)
        if _is_videos_supported:
            examples.extend(videos_examples)
        if _is_audios_supported:
            examples.extend(audio_examples)

        pipe = AnyToAnyPipeline(model=model, processor=processor, dtype=dtype, max_new_tokens=10)

        return pipe, examples

    def run_pipeline_test(self, pipe, examples):
        # Single
        outputs = pipe(examples[0])
        self.assertEqual(
            outputs,
            [
                {"input_text": ANY(str), "generated_text": ANY(str)},
            ],
        )

        # Batched but limit to last 2 examples
        outputs = pipe(examples[:2])
        self.assertEqual(
            outputs,
            [
                [
                    {"input_text": ANY(str), "generated_text": ANY(str)},
                ],
                [
                    {"input_text": ANY(str), "generated_text": ANY(str)},
                ],
            ],
        )

        # `generation_mode` raises errors when dosn't match with other params
        with self.assertRaises(ValueError):
            pipe(examples, generation_mode="video")

        with self.assertRaises(ValueError):
            pipe(examples, generation_mode="audio", return_full_text=True)

        with self.assertRaises(ValueError):
            pipe(examples, generation_mode="image", return_type=1)

        # Chat template
        if getattr(pipe.processor, "chat_template", None) is not None:
            messages = []
            for example in examples[:2]:
                example.pop("text")
                modality_type, modality_data = list(example.items())[0]
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This is a "},
                        {"type": modality_type, "path": modality_data},
                    ],
                }
                messages.append([message])
            outputs = pipe(messages, return_full_text=True, max_new_tokens=10)

            self.assertEqual(
                outputs,
                [
                    [
                        {"input_text": ANY(str), "generated_text": ANY(str)},
                    ],
                    [
                        {"input_text": ANY(str), "generated_text": ANY(str)},
                    ],
                ],
            )

    @slow
    def test_small_model_pt_token_text_only(self):
        pipe = pipeline("any-to-any", model="google/gemma-3n-E4B-it")
        text = "What is the capital of France? Assistant:"

        outputs = pipe(text=text, generate_kwargs={"do_sample": False})
        self.assertEqual(
            outputs,
            [
                {
                    "input_text": "What is the capital of France? Assistant:",
                    "generated_None": "What is the capital of France? Assistant: The capital of France is Paris.\n",
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
        outputs = pipe(text=messages, generate_kwargs={"do_sample": False})
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
                        "generated_None": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"}],
                            },
                            {
                                "role": "assistant",
                                "content": "A digital embrace, a friendly face,\nHugging Face rises, setting the pace.\nFor AI's heart, a vibrant core,\nOpen source models, and so much more.\n\nFrom transformers deep, a powerful might,\nNLP's future, shining so bright.\nDatasets curated, a treasure trove found,\nFor researchers and builders, on fertile ground.\n\nA community thriving, a collaborative art,\nSharing knowledge, playing a vital part.\nSpaces to showcase, creations unfold,\nStories in code, bravely told.\n\nWith libraries sleek, and tools so refined,\nDemocratizing AI, for all humankind.\nFrom sentiment analysis to text generation's grace,\nHugging Face empowers, at a rapid pace.\n\nA platform of learning, a place to explore,\nUnlocking potential, and asking for more.\nSo let's give a cheer, for this innovative team,\nHugging Face's vision, a beautiful dream. \n",
                            },
                        ],
                    }
                ],
                [
                    {
                        "input_text": [
                            {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}
                        ],
                        "generated_None": [
                            {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
                            {"role": "assistant", "content": "The capital of France is **Paris**. \n"},
                        ],
                    }
                ],
            ],
        )

    @slow
    def test_small_model_pt_token_audio_input(self):
        pipe = pipeline("any-to-any", model="google/gemma-3n-E4B-it")

        audio_path = url_to_local_path(
            "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you hear in this audio?"},
                    {"type": "audio", "url": audio_path},
                ],
            },
        ]
        outputs = pipe(text=messages, return_type=1, generate_kwargs={"do_sample": False})  # return new text
        self.assertEqual(
            outputs,
            [
                {
                    "input_text": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What do you hear in this audio?"},
                                {
                                    "type": "audio",
                                    "url": "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav",
                                },
                            ],
                        }
                    ],
                    "generated_None": "user\nWhat do you hear in this audio?\n\n\n\n\nmodel\nThe audio contains the repeated sound of someone **coughing**. It's a fairly consistent, forceful cough throughout the duration.",
                }
            ],
        )

    @slow
    def test_small_model_pt_token_audio_gen(self):
        pipe = pipeline("any-to-any", model="Qwen/Qwen2.5-Omni-3B", dtype="bfloat16")

        video_path = url_to_local_path(
            "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/Cooking_cake.mp4"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    {"type": "video", "video": video_path},
                ],
            },
        ]
        outputs = pipe(
            text=messages,
            num_frames=16,
            max_new_tokens=50,
            load_audio_from_video=True,
            generate_kwargs={"use_audio_in_video": True, "talker_do_sample": False, "do_sample": False},
        )
        self.assertEqual(
            outputs,
            [
                {
                    "input_text": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this video."},
                                {
                                    "type": "video",
                                    "video": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/Cooking_cake.mp4",
                                },
                            ],
                        }
                    ],
                    "generated_None": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this video."},
                                {
                                    "type": "video",
                                    "video": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/Cooking_cake.mp4",
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": "system\nYou are a helpful assistant.\nuser\nDescribe this video.\nassistant\nThe video begins with a man standing in a kitchen, wearing a black shirt. He is holding a large glass bowl filled with flour and a spoon. The man starts to mix the flour in the bowl, creating a dough. As he mixes, he continues to talk to the camera, explaining the process. The kitchen has wooden cabinets and a white refrigerator in the background. The man's movements are deliberate and focused as he works with the dough. The video ends with the man still mixing the dough in the bowl. Overall, the video provides a clear and detailed demonstration of how to make dough using flour and a spoon.",
                        },
                    ],
                }
            ],
        )

        outputs = pipe(text=messages, generation_mode="audio", num_frames=16, max_new_tokens=20)

        self.assertEqual(len(outputs), len(messages))
        self.assertIsInstance(outputs[0], dict)
        for out in outputs:
            self.assertTrue("input_text" in out)
            self.assertTrue("generated_audio" in out)
            self.assertIsInstance(out["generated_audio"], np.ndarray)

    @slow
    def test_small_model_pt_image_gen(self):
        pipe = pipeline("any-to-any", model="deepseek-community/Janus-Pro-1B")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A dog running under the rain."},
                ],
            },
        ]
        outputs = pipe(text=messages, generation_mode="image")

        self.assertEqual(len(outputs), len(messages))
        self.assertIsInstance(outputs[0], dict)
        for out in outputs:
            self.assertTrue("input_text" in out)
            self.assertTrue("generated_image" in out)
            self.assertIsInstance(out["generated_image"], PIL.Image.Image)
