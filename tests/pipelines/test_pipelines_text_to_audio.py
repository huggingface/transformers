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
import torch

from transformers import (
    MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
    AutoProcessor,
    TextToAudioPipeline,
    pipeline,
)
from transformers.testing_utils import (
    is_pipeline_test,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.trainer_utils import set_seed

from .test_pipelines_common import ANY


@is_pipeline_test
@require_torch
class TextToAudioPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING
    # for now only test text_to_waveform and not text_to_spectrogram

    @require_torch
    def test_small_speecht5_pt(self):
        audio_generator = pipeline(task="text-to-audio", model="microsoft/speecht5_tts")
        num_channels = 1  # model generates mono audio
        forward_params = {
            "do_sample": True,
            "semantic_max_new_tokens": 5,
            "speaker_embeddings": torch.rand(1, 512) * 0.2 - 0.1,
        }

        outputs = audio_generator("This is a test", forward_params=forward_params)
        self.assertEqual({"audio": ANY(np.ndarray), "sampling_rate": 16000}, outputs)
        self.assertEqual(len(outputs["audio"].shape), num_channels)

        # test two examples side-by-side
        outputs = audio_generator(["This is a test", "This is a second test"], forward_params=forward_params)
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

        # test batching, this time with parameterization in the forward pass
        audio_generator = pipeline(task="text-to-audio", model="microsoft/speecht5_tts")
        forward_params = {
            "do_sample": False,
            "max_new_tokens": 5,
            "speaker_embeddings": torch.rand(1, 512) * 0.2 - 0.1,
        }
        outputs = audio_generator(
            ["This is a test", "This is a second test"], forward_params=forward_params, batch_size=2
        )
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

    @require_torch
    def test_small_musicgen_pt(self):
        music_generator = pipeline(
            task="text-to-audio", model="facebook/musicgen-small", do_sample=False, max_new_tokens=5
        )
        num_channels = 1  # model generates mono audio

        outputs = music_generator("This is a test")
        self.assertEqual({"audio": ANY(np.ndarray), "sampling_rate": 32000}, outputs)
        self.assertEqual(len(outputs["audio"].shape), num_channels)

        # test two examples side-by-side
        outputs = music_generator(["This is a test", "This is a second test"])
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

        # test batching, this time with parameterization in the forward pass
        music_generator = pipeline(task="text-to-audio", model="facebook/musicgen-small")
        forward_params = {"do_sample": False, "max_new_tokens": 5}
        outputs = music_generator(
            ["This is a test", "This is a second test"], forward_params=forward_params, batch_size=2
        )
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

    @slow
    @require_torch
    def test_medium_seamless_m4t_pt(self):
        speech_generator = pipeline(task="text-to-audio", model="facebook/hf-seamless-m4t-medium", max_new_tokens=5)

        for forward_params in [{"tgt_lang": "eng"}, {"return_intermediate_token_ids": True, "tgt_lang": "eng"}]:
            outputs = speech_generator("This is a test", forward_params=forward_params)
            self.assertEqual({"audio": ANY(np.ndarray), "sampling_rate": 16000}, outputs)

            # test two examples side-by-side
            outputs = speech_generator(["This is a test", "This is a second test"], forward_params=forward_params)
            audio = [output["audio"] for output in outputs]
            self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

            # test batching
            outputs = speech_generator(
                ["This is a test", "This is a second test"], forward_params=forward_params, batch_size=2
            )
            audio = [output["audio"] for output in outputs]
            self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

    @slow
    @require_torch
    def test_small_bark_pt(self):
        speech_generator = pipeline(task="text-to-audio", model="suno/bark-small")
        num_channels = 1  # model generates mono audio

        forward_params = {
            # Using `do_sample=False` to force deterministic output
            "do_sample": False,
            "semantic_max_new_tokens": 5,
        }

        outputs = speech_generator("This is a test", forward_params=forward_params)
        self.assertEqual(
            {"audio": ANY(np.ndarray), "sampling_rate": 24000},
            outputs,
        )
        self.assertEqual(len(outputs["audio"].shape), num_channels)

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
            "semantic_max_new_tokens": 5,
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
    @require_torch_accelerator
    def test_conversion_additional_tensor(self):
        speech_generator = pipeline(task="text-to-audio", model="suno/bark-small", device=torch_device)
        processor = AutoProcessor.from_pretrained("suno/bark-small")

        forward_params = {
            "do_sample": True,
            "semantic_max_new_tokens": 5,
        }

        # atm, must do to stay coherent with BarkProcessor
        preprocess_params = {
            "max_length": 256,
            "add_special_tokens": False,
            "return_attention_mask": True,
            "return_token_type_ids": False,
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

    @require_torch
    def test_vits_model_pt(self):
        speech_generator = pipeline(task="text-to-audio", model="facebook/mms-tts-eng")

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

    @require_torch
    def test_forward_model_kwargs(self):
        # use vits - a forward model
        speech_generator = pipeline(task="text-to-audio", model="kakao-enterprise/vits-vctk")

        # for reproducibility
        set_seed(555)
        outputs = speech_generator("This is a test", forward_params={"speaker_id": 5})
        audio = outputs["audio"]

        with self.assertRaises(TypeError):
            # assert error if generate parameter
            outputs = speech_generator("This is a test", forward_params={"speaker_id": 5, "do_sample": True})

        forward_params = {"speaker_id": 5}
        generate_kwargs = {"do_sample": True}

        with self.assertRaises(ValueError):
            # assert error if generate_kwargs with forward-only models
            outputs = speech_generator(
                "This is a test", forward_params=forward_params, generate_kwargs=generate_kwargs
            )
        self.assertTrue(np.abs(outputs["audio"] - audio).max() < 1e-5)

    @require_torch
    def test_generative_model_kwargs(self):
        # use musicgen - a generative model
        music_generator = pipeline(task="text-to-audio", model="facebook/musicgen-small")

        forward_params = {
            "do_sample": True,
            "max_new_tokens": 20,
        }

        # for reproducibility
        set_seed(555)
        outputs = music_generator("This is a test", forward_params=forward_params)
        audio = outputs["audio"]
        self.assertEqual(ANY(np.ndarray), audio)

        # make sure generate kwargs get priority over forward params
        forward_params = {
            "do_sample": False,
            "max_new_tokens": 20,
        }
        generate_kwargs = {"do_sample": True}

        # for reproducibility
        set_seed(555)
        outputs = music_generator("This is a test", forward_params=forward_params, generate_kwargs=generate_kwargs)
        self.assertListEqual(outputs["audio"].tolist(), audio.tolist())

    @slow
    @require_torch
    def test_csm_model_pt(self):
        speech_generator = pipeline(task="text-to-audio", model="sesame/csm-1b", device=torch_device)
        generate_kwargs = {"max_new_tokens": 10}
        num_channels = 1  # model generates mono audio

        outputs = speech_generator("This is a test", generate_kwargs=generate_kwargs)
        self.assertEqual(outputs["sampling_rate"], 24000)
        audio = outputs["audio"]
        self.assertEqual(ANY(np.ndarray), audio)
        # ensure audio and not discrete codes
        self.assertEqual(len(audio.shape), num_channels)

        # test two examples side-by-side
        outputs = speech_generator(["This is a test", "This is a second test"], generate_kwargs=generate_kwargs)
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)
        self.assertEqual(len(audio[0].shape), num_channels)

        # test batching
        batch_size = 2
        outputs = speech_generator(
            ["This is a test", "This is a second test"], generate_kwargs=generate_kwargs, batch_size=batch_size
        )
        self.assertEqual(len(outputs), batch_size)
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)
        self.assertEqual(len(outputs[0]["audio"].shape), num_channels)

    @slow
    @require_torch
    def test_dia_model(self):
        speech_generator = pipeline(task="text-to-audio", model="nari-labs/Dia-1.6B-0626", device=torch_device)
        generate_kwargs = {"max_new_tokens": 20}
        num_channels = 1  # model generates mono audio

        outputs = speech_generator("Dia is an open weights text to dialogue model.", generate_kwargs=generate_kwargs)
        self.assertEqual(outputs["sampling_rate"], 44100)
        audio = outputs["audio"]
        self.assertEqual(ANY(np.ndarray), audio)
        # ensure audio (with one channel) and not discrete codes
        self.assertEqual(len(audio.shape), num_channels)

        # test two examples side-by-side
        outputs = speech_generator(
            ["Dia is an open weights text to dialogue model.", "This is a second example."],
            generate_kwargs=generate_kwargs,
        )
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)
        self.assertEqual(len(audio[0].shape), num_channels)

        # test batching
        batch_size = 2
        outputs = speech_generator(
            ["Dia is an open weights text to dialogue model.", "This is a second example."],
            generate_kwargs=generate_kwargs,
            batch_size=2,
        )
        self.assertEqual(len(outputs), batch_size)
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)
        self.assertEqual(len(outputs[0]["audio"].shape), num_channels)

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        dtype="float32",
    ):
        model_test_kwargs = {}
        if model.can_generate():  # not all models in this pipeline can generate and, therefore, take `generate` kwargs
            model_test_kwargs["max_new_tokens"] = 5
        model.config._attn_implementation = "eager"
        speech_generator = TextToAudioPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            dtype=dtype,
            **model_test_kwargs,
        )

        return speech_generator, ["This is a test", "Another test"]

    def run_pipeline_test(self, speech_generator, _):
        outputs = speech_generator("This is a test")
        self.assertEqual(ANY(np.ndarray), outputs["audio"])

        forward_params = (
            {"num_return_sequences": 2, "do_sample": True} if speech_generator.model.can_generate() else {}
        )
        outputs = speech_generator(["This is great !", "Something else"], forward_params=forward_params)
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)
