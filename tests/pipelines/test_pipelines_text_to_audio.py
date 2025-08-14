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

import os
import tempfile
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
    is_torch_available,
    require_torch,
    require_torch_accelerator,
    require_torch_or_tf,
    slow,
    torch_device,
)
from transformers.trainer_utils import set_seed

from .test_pipelines_common import ANY


if is_torch_available():
    import torch


@is_pipeline_test
@require_torch_or_tf
class TextToAudioPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING
    # for now only test text_to_waveform and not text_to_spectrogram

    @slow
    @require_torch
    def test_small_musicgen(self):
        music_generator = pipeline(
            task="text-to-audio", model="facebook/musicgen-small", framework="pt", do_sample=False, max_new_tokens=5
        )

        outputs = music_generator("This is a test")
        self.assertEqual({"audio": ANY(np.ndarray), "sampling_rate": 32000}, outputs)

        # test two examples side-by-side
        outputs = music_generator(["This is a test", "This is a second test"])
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

        # test batching, this time with parameterization in the forward pass (BC test)
        music_generator = pipeline(task="text-to-audio", model="facebook/musicgen-small", framework="pt")
        forward_params = {"do_sample": False, "max_new_tokens": 5}
        outputs = music_generator(
            ["This is a test", "This is a second test"], forward_params=forward_params, batch_size=2
        )
        audio_bc = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio_bc)

        # same as above, but passing generate flags directly
        outputs = music_generator(
            ["This is a test", "This is a second test"],
            do_sample=False,
            max_new_tokens=5,
            batch_size=2,
        )
        audio_generate = [output["audio"] for output in outputs]
        self.assertTrue(np.abs(audio_bc[0] - audio_generate[0]).max() < 1e-5)
        self.assertTrue(np.abs(audio_bc[1] - audio_generate[1]).max() < 1e-5)

        # test saving audio
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 2 files and 1 path -> append a number suffix to the path
            music_generator.save_audio(outputs, os.path.join(tmp_dir, "audio.wav"))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "audio_0.wav")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "audio_1.wav")))

    @slow
    @require_torch
    def test_medium_seamless_m4t(self):
        speech_generator = pipeline(
            task="text-to-audio", model="facebook/hf-seamless-m4t-medium", framework="pt", max_new_tokens=5
        )

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

            # test saving audio
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 2 files and 1 path -> append a number suffix to the path
                speech_generator.save_audio(outputs, os.path.join(tmp_dir, "audio.wav"))
                self.assertTrue(os.path.exists(os.path.join(tmp_dir, "audio_0.wav")))
                self.assertTrue(os.path.exists(os.path.join(tmp_dir, "audio_1.wav")))

    @slow
    @require_torch
    def test_small_bark(self):
        speech_generator = pipeline(task="text-to-audio", model="suno/bark-small", framework="pt")

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
            # this makes the model return 2 sequences per batch item in the semantic model
            "semantic_num_return_sequences": 2,
        }

        outputs = speech_generator("This is a test", forward_params=forward_params)
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

        # test saving audio
        with tempfile.TemporaryDirectory() as tmp_dir:
            speech_generator.save_audio(outputs[0], os.path.join(tmp_dir, "audio.wav"))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "audio.wav")))

    @slow
    @require_torch
    def test_small_bark_set_speaker(self):
        speech_generator = pipeline(task="text-to-audio", model="suno/bark-small", framework="pt")

        # BC: test using a speaker embedding
        forward_params = {
            "do_sample": True,
            "semantic_max_new_tokens": 5,
            "semantic_num_return_sequences": 2,
        }

        processor = AutoProcessor.from_pretrained("suno/bark-small")
        temp_inp = processor("hey, how are you?", voice_preset="v2/en_speaker_5")
        history_prompt = temp_inp["history_prompt"]
        forward_params["history_prompt"] = history_prompt

        outputs = speech_generator(
            ["This is a test", "This is a second test"],
            forward_params=forward_params,
            batch_size=2,
        )
        audio_bc = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio_bc)

        # test using `voice` pipeline parameter
        audio = speech_generator("This is a test", voice="v2/en_speaker_5", do_sample=True)
        self.assertEqual(ANY(np.ndarray), audio["audio"])

    @slow
    @require_torch_accelerator
    def test_conversion_additional_tensor(self):
        speech_generator = pipeline(task="text-to-audio", model="suno/bark-small", framework="pt", device=torch_device)
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

    @slow
    @require_torch
    def test_vits_model(self):
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

        # test saving audio
        with tempfile.TemporaryDirectory() as tmp_dir:
            speech_generator.save_audio(outputs[0], os.path.join(tmp_dir, "audio.wav"))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "audio.wav")))

    @slow
    @require_torch
    def test_csm_model(self):
        """Tests CSM with the text-to-audio pipeline, including voice selection"""
        speech_generator = pipeline(task="text-to-audio", model="sesame/csm-1b", framework="pt")

        outputs = speech_generator("This is a test")
        self.assertEqual(outputs["sampling_rate"], 24000)

        audio = outputs["audio"]
        self.assertEqual(ANY(np.ndarray), audio)

        # test two examples side-by-side
        outputs = speech_generator(["This is a test", "This is a second test"], voice="1")
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

        # test batching
        outputs = speech_generator(["This is a test", "This is a second test"], batch_size=2, voice="2")
        self.assertEqual(ANY(np.ndarray), outputs[0]["audio"])

        # test saving audio
        with tempfile.TemporaryDirectory() as tmp_dir:
            speech_generator.save_audio(outputs[0], os.path.join(tmp_dir, "audio.wav"))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "audio.wav")))

    @slow
    @require_torch
    def test_qwen2_5_omni_model(self):
        """Tests Qwen2.5Omni with the text-to-audio pipeline, including voice selection"""
        speech_generator = pipeline(
            task="text-to-audio", model="Qwen/Qwen2.5-Omni-3B", framework="pt", torch_dtype=torch.bfloat16
        )
        qwen_kwargs = {
            "thinker_max_new_tokens": 5,
            "talker_max_new_tokens": 5,
        }

        outputs = speech_generator("This is a test", **qwen_kwargs)
        self.assertEqual(outputs["sampling_rate"], 16000)

        audio = outputs["audio"]
        self.assertEqual(ANY(np.ndarray), audio)

        # test two examples side-by-side
        outputs = speech_generator(["This is a test", "This is a second test"], voice="Ethan", **qwen_kwargs)
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

        # TODO: qwen2.5-omni does not support batched inference with audio output -> no batching test

        # test saving audio
        with tempfile.TemporaryDirectory() as tmp_dir:
            speech_generator.save_audio(outputs[0], os.path.join(tmp_dir, "audio.wav"))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "audio.wav")))

    @slow
    @require_torch
    def test_dia_model(self):
        """Tests Dia with the text-to-audio pipeline"""
        speech_generator = pipeline(task="text-to-audio", model="buttercrab/dia-v1-1.6b", framework="pt")

        outputs = speech_generator("This is a test")
        self.assertEqual(outputs["sampling_rate"], 44100)

        audio = outputs["audio"]
        self.assertEqual(ANY(np.ndarray), audio)

        # TODO: The checkpoint is quite large, so no batching tests

        # test saving audio
        with tempfile.TemporaryDirectory() as tmp_dir:
            speech_generator.save_audio(outputs, os.path.join(tmp_dir, "audio.wav"))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "audio.wav")))

    @require_torch
    def test_forward_model_kwargs(self):
        # use vits - a forward model
        speech_generator = pipeline(task="text-to-audio", model="kakao-enterprise/vits-vctk", framework="pt")

        # for reproducibility
        set_seed(555)
        outputs = speech_generator("This is a test", forward_params={"speaker_id": 5})
        audio = outputs["audio"]

        with self.assertRaises(TypeError):
            # assert error if generate parameter
            outputs = speech_generator("This is a test", forward_params={"speaker_id": 5, "do_sample": True})

        forward_params = {"speaker_id": 5}
        generate_kwargs = {"do_sample": True}

        with self.assertRaises(TypeError):
            # assert error if generate_kwargs with forward-only models
            outputs = speech_generator(
                "This is a test", forward_params=forward_params, generate_kwargs=generate_kwargs
            )
        self.assertTrue(np.abs(outputs["audio"] - audio).max() < 1e-5)

    @require_torch
    def test_generative_model_kwargs(self):
        # use musicgen - a generative model
        music_generator = pipeline(task="text-to-audio", model="facebook/musicgen-small", framework="pt")

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
        self.assertTrue(np.abs(outputs["audio"] - audio).max() < 1e-5)

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        torch_dtype="float32",
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
            torch_dtype=torch_dtype,
            **model_test_kwargs,
        )

        return speech_generator, ["This is a test", "Another test"]

    def run_pipeline_test(self, audio_generator, _):
        # TODO joao: after we fix the script to update test/utils/tiny_model_summary.json, we can delete most
        # integration tests above (after we add voice selection in this test)
        outputs = audio_generator("This is a test")
        self.assertEqual(ANY(np.ndarray), outputs["audio"])

        forward_params = {"num_return_sequences": 2, "do_sample": True} if audio_generator.model.can_generate() else {}
        outputs = audio_generator(["This is great !", "Something else"], **forward_params)
        audio = [output["audio"] for output in outputs]
        self.assertEqual([ANY(np.ndarray), ANY(np.ndarray)], audio)

        # unique to this pipeline: the pipeline can save audio
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_generator.save_audio(outputs[0], os.path.join(tmp_dir, "audio.wav"))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "audio.wav")))
