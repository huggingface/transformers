# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import unittest

import numpy as np

from transformers import (
    MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
    AutoProcessor,
    TextToAudioPipeline,
    pipeline,
)
from transformers.testing_utils import (
    is_pipeline_test,
    require_torch,
    require_torch_gpu,
    require_torch_or_tf,
    slow,
)

from .test_pipelines_common import ANY


@is_pipeline_test
@require_torch_or_tf
class TextToAudioPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING
    # for now only test text_to_waveform and not text_to_spectrogram

    @slow
    @require_torch
    def test_small_musicgen_pt(self):
        music_generator = pipeline(task="text-to-audio", model="facebook/musicgen-small", framework="pt")

        forward_params = {
            "do_sample": False,
            "max_new_tokens": 250,
        }

        outputs = music_generator("This is a test", forward_params=forward_params)
        self.assertEqual({"audio": ANY(np.ndarray), "sampling_rate": 32000}, outputs)

        # test two examples side-by-side
        outputs = music_generator(["This is a test", "This is a second test"], forward_params=forward_params)
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

        # test batching
        outputs = music_generator(
            ["This is a test", "This is a second test"], forward_params=forward_params, batch_size=2
        )
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

    @slow
    @require_torch
    def test_small_bark_pt(self):
        speech_generator = pipeline(task="text-to-audio", model="suno/bark-small", framework="pt")

        forward_params = {
            # Using `do_sample=False` to force deterministic output
            "do_sample": False,
            "semantic_max_new_tokens": 100,
        }

        outputs = speech_generator("This is a test", forward_params=forward_params)
        self.assertEqual(
            {"audio": ANY(np.ndarray), "sampling_rate": 24000},
            outputs,
        )

        # test two examples side-by-side
        outputs = speech_generator(
            ["This is a test", "This is a second test"],
            forward_params=forward_params,
        )
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

        # test other generation strategy
        forward_params = {
            "do_sample": True,
            "semantic_max_new_tokens": 100,
            "semantic_num_return_sequences": 2,
        }

        outputs = speech_generator("This is a test", forward_params=forward_params)
        audio = outputs["audio"]
        self.assertEqual(ANY(np.ndarray), audio)

        # test using a speaker embedding
        processor = AutoProcessor.from_pretrained("suno/bark-small")
        temp_inp = processor("hey, how are you?", voice_preset="v2/en_speaker_5")
        history_prompt = temp_inp["history_prompt"]
        forward_params["history_prompt"] = history_prompt

        outputs = speech_generator(
            ["This is a test", "This is a second test"],
            forward_params=forward_params,
            batch_size=2,
        )
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

    @slow
    @require_torch_gpu
    def test_conversion_additional_tensor(self):
        speech_generator = pipeline(task="text-to-audio", model="suno/bark-small", framework="pt", device=0)
        processor = AutoProcessor.from_pretrained("suno/bark-small")

        forward_params = {
            "do_sample": True,
            "semantic_max_new_tokens": 100,
        }

        # atm, must do to stay coherent with BarkProcessor
        preprocess_params = {
            "max_length": 256,
            "add_special_tokens": False,
            "return_attention_mask": True,
            "return_token_type_ids": False,
            "padding": "max_length",
        }
        outputs = speech_generator(
            "This is a test",
            forward_params=forward_params,
            preprocess_params=preprocess_params,
        )

        temp_inp = processor("hey, how are you?", voice_preset="v2/en_speaker_5")
        history_prompt = temp_inp["history_prompt"]
        forward_params["history_prompt"] = history_prompt

        # history_prompt is a torch.Tensor passed as a forward_param
        # if generation is successful, it means that it was passed to the right device
        outputs = speech_generator(
            "This is a test", forward_params=forward_params, preprocess_params=preprocess_params
        )
        self.assertEqual(
            {"audio": ANY(np.ndarray), "sampling_rate": 24000},
            outputs,
        )

    @slow
    @require_torch
    def test_vits_model_pt(self):
        speech_generator = pipeline(task="text-to-audio", model="facebook/mms-tts-eng", framework="pt")

        outputs = speech_generator("This is a test")
        self.assertEqual(outputs["sampling_rate"], 16000)

        audio = outputs["audio"]
        self.assertEqual(ANY(np.ndarray), audio)

        # test two examples side-by-side
        outputs = speech_generator(["This is a test", "This is a second test"])
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

        # test batching
        outputs = speech_generator(["This is a test", "This is a second test"], batch_size=2)
        self.assertEqual(ANY(np.ndarray), outputs[0]["audio"])

    def get_test_pipeline(self, model, tokenizer, processor):
        speech_generator = TextToAudioPipeline(model=model, tokenizer=tokenizer)
        return speech_generator, ["This is a test", "Another test"]

    def run_pipeline_test(self, speech_generator, _):
        outputs = speech_generator("This is a test")
        self.assertEqual(ANY(np.ndarray), outputs["audio"])

        forward_params = {"num_return_sequences": 2, "do_sample": True}
        outputs = speech_generator(["This is great !", "Something else"], forward_params=forward_params)
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)
