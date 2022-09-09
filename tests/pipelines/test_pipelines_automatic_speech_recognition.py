# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import pytest
from datasets import load_dataset

from huggingface_hub import snapshot_download
from transformers import (
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    AutoFeatureExtractor,
    AutoTokenizer,
    Speech2TextForConditionalGeneration,
    Wav2Vec2ForCTC,
)
from transformers.pipelines import AutomaticSpeechRecognitionPipeline, pipeline
from transformers.pipelines.audio_utils import chunk_bytes_iter
from transformers.pipelines.automatic_speech_recognition import chunk_iter
from transformers.testing_utils import (
    is_pipeline_test,
    is_torch_available,
    nested_simplify,
    require_pyctcdecode,
    require_tf,
    require_torch,
    require_torchaudio,
    slow,
)

from .test_pipelines_common import ANY, PipelineTestCaseMeta


if is_torch_available():
    import torch


# We can't use this mixin because it assumes TF support.
# from .test_pipelines_common import CustomInputPipelineCommonMixin


@is_pipeline_test
class AutomaticSpeechRecognitionPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = {
        k: v
        for k, v in (list(MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.items()) if MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING else [])
        + (MODEL_FOR_CTC_MAPPING.items() if MODEL_FOR_CTC_MAPPING else [])
    }

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        if tokenizer is None:
            # Side effect of no Fast Tokenizer class for these model, so skipping
            # But the slow tokenizer test should still run as they're quite small
            self.skipTest("No tokenizer available")
            return
            # return None, None

        speech_recognizer = AutomaticSpeechRecognitionPipeline(
            model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
        )

        # test with a raw waveform
        audio = np.zeros((34000,))
        audio2 = np.zeros((14000,))
        return speech_recognizer, [audio, audio2]

    def run_pipeline_test(self, speech_recognizer, examples):
        audio = np.zeros((34000,))
        outputs = speech_recognizer(audio)
        self.assertEqual(outputs, {"text": ANY(str)})

        # Striding
        audio = {"raw": audio, "stride": (0, 4000), "sampling_rate": speech_recognizer.feature_extractor.sampling_rate}
        if speech_recognizer.type == "ctc":
            outputs = speech_recognizer(audio)
            self.assertEqual(outputs, {"text": ANY(str)})

        else:
            # Non CTC models cannot use striding.
            with self.assertRaises(ValueError):
                outputs = speech_recognizer(audio)

        # Timestamps
        audio = np.zeros((34000,))
        if speech_recognizer.type == "ctc":
            outputs = speech_recognizer(audio, return_timestamps="char")
            self.assertIsInstance(outputs["chunks"], list)
            n = len(outputs["chunks"])
            self.assertEqual(
                outputs,
                {
                    "text": ANY(str),
                    "chunks": [{"text": ANY(str), "timestamp": (ANY(float), ANY(float))} for i in range(n)],
                },
            )

            outputs = speech_recognizer(audio, return_timestamps="word")
            self.assertIsInstance(outputs["chunks"], list)
            n = len(outputs["chunks"])
            self.assertEqual(
                outputs,
                {
                    "text": ANY(str),
                    "chunks": [{"text": ANY(str), "timestamp": (ANY(float), ANY(float))} for i in range(n)],
                },
            )
        else:
            # Non CTC models cannot use return_timestamps
            with self.assertRaises(ValueError):
                outputs = speech_recognizer(audio, return_timestamps="char")

    @require_torch
    @slow
    def test_pt_defaults(self):
        pipeline("automatic-speech-recognition", framework="pt")

    @require_torch
    def test_small_model_pt(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/s2t-small-mustc-en-fr-st",
            tokenizer="facebook/s2t-small-mustc-en-fr-st",
            framework="pt",
        )
        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        output = speech_recognizer(waveform)
        self.assertEqual(output, {"text": "(Applaudissements)"})

    @require_torch
    def test_small_model_pt_seq2seq(self):
        speech_recognizer = pipeline(
            model="hf-internal-testing/tiny-random-speech-encoder-decoder",
            framework="pt",
        )

        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        output = speech_recognizer(waveform)
        self.assertEqual(output, {"text": "あл ش 湯 清 ه ܬ া लᆨしث ल eか u w 全 u"})

    @slow
    @require_torch
    @require_pyctcdecode
    def test_large_model_pt_with_lm(self):
        dataset = load_dataset("Narsil/asr_dummy")
        filename = dataset["test"][3]["file"]

        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="patrickvonplaten/wav2vec2-large-xlsr-53-spanish-with-lm",
            framework="pt",
        )
        self.assertEqual(speech_recognizer.type, "ctc_with_lm")

        output = speech_recognizer(filename)
        self.assertEqual(
            output,
            {"text": "y en las ramas medio sumergidas revoloteaban algunos pájaros de quimérico y legendario plumaje"},
        )

        # Override back to pure CTC
        speech_recognizer.type = "ctc"
        output = speech_recognizer(filename)
        # plumajre != plumaje
        self.assertEqual(
            output,
            {
                "text": (
                    "y en las ramas medio sumergidas revoloteaban algunos pájaros de quimérico y legendario plumajre"
                )
            },
        )

        speech_recognizer.type = "ctc_with_lm"
        # Simple test with CTC with LM, chunking + timestamps
        output = speech_recognizer(filename, chunk_length_s=2.0, return_timestamps="word")
        self.assertEqual(
            output,
            {
                "text": (
                    "y en las ramas medio sumergidas revoloteaban algunos pájaros de quimérico y legendario plumajcri"
                ),
                "chunks": [
                    {"text": "y", "timestamp": (0.52, 0.54)},
                    {"text": "en", "timestamp": (0.6, 0.68)},
                    {"text": "las", "timestamp": (0.74, 0.84)},
                    {"text": "ramas", "timestamp": (0.94, 1.24)},
                    {"text": "medio", "timestamp": (1.32, 1.52)},
                    {"text": "sumergidas", "timestamp": (1.56, 2.22)},
                    {"text": "revoloteaban", "timestamp": (2.36, 3.0)},
                    {"text": "algunos", "timestamp": (3.06, 3.38)},
                    {"text": "pájaros", "timestamp": (3.46, 3.86)},
                    {"text": "de", "timestamp": (3.92, 4.0)},
                    {"text": "quimérico", "timestamp": (4.08, 4.6)},
                    {"text": "y", "timestamp": (4.66, 4.68)},
                    {"text": "legendario", "timestamp": (4.74, 5.26)},
                    {"text": "plumajcri", "timestamp": (5.34, 5.74)},
                ],
            },
        )

    @require_tf
    def test_small_model_tf(self):
        self.skipTest("Tensorflow not supported yet.")

    @require_torch
    def test_torch_small_no_tokenizer_files(self):
        # test that model without tokenizer file cannot be loaded
        with pytest.raises(OSError):
            pipeline(
                task="automatic-speech-recognition",
                model="patrickvonplaten/tiny-wav2vec2-no-tokenizer",
                framework="pt",
            )

    @require_torch
    @slow
    def test_torch_large(self):

        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/wav2vec2-base-960h",
            tokenizer="facebook/wav2vec2-base-960h",
            framework="pt",
        )
        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        output = speech_recognizer(waveform)
        self.assertEqual(output, {"text": ""})

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        filename = ds[40]["file"]
        output = speech_recognizer(filename)
        self.assertEqual(output, {"text": "A MAN SAID TO THE UNIVERSE SIR I EXIST"})

    @require_torch
    @slow
    def test_torch_speech_encoder_decoder(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/s2t-wav2vec2-large-en-de",
            feature_extractor="facebook/s2t-wav2vec2-large-en-de",
            framework="pt",
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        filename = ds[40]["file"]
        output = speech_recognizer(filename)
        self.assertEqual(output, {"text": 'Ein Mann sagte zum Universum : " Sir, ich existiert! "'})

    @slow
    @require_torch
    def test_simple_wav2vec2(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

        asr = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)
        output = asr(waveform)
        self.assertEqual(output, {"text": ""})

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        filename = ds[40]["file"]
        output = asr(filename)
        self.assertEqual(output, {"text": "A MAN SAID TO THE UNIVERSE SIR I EXIST"})

        filename = ds[40]["file"]
        with open(filename, "rb") as f:
            data = f.read()
        output = asr(data)
        self.assertEqual(output, {"text": "A MAN SAID TO THE UNIVERSE SIR I EXIST"})

    @slow
    @require_torch
    @require_torchaudio
    def test_simple_s2t(self):

        model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-mustc-en-it-st")
        tokenizer = AutoTokenizer.from_pretrained("facebook/s2t-small-mustc-en-it-st")
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/s2t-small-mustc-en-it-st")

        asr = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

        waveform = np.tile(np.arange(1000, dtype=np.float32), 34)

        output = asr(waveform)
        self.assertEqual(output, {"text": "(Applausi)"})

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        filename = ds[40]["file"]
        output = asr(filename)
        self.assertEqual(output, {"text": "Un uomo disse all'universo: \"Signore, io esisto."})

        filename = ds[40]["file"]
        with open(filename, "rb") as f:
            data = f.read()
        output = asr(data)
        self.assertEqual(output, {"text": "Un uomo disse all'universo: \"Signore, io esisto."})

    @slow
    @require_torch
    @require_torchaudio
    def test_xls_r_to_en(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/wav2vec2-xls-r-1b-21-to-en",
            feature_extractor="facebook/wav2vec2-xls-r-1b-21-to-en",
            framework="pt",
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        filename = ds[40]["file"]
        output = speech_recognizer(filename)
        self.assertEqual(output, {"text": "A man said to the universe: “Sir, I exist."})

    @slow
    @require_torch
    @require_torchaudio
    def test_xls_r_from_en(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="facebook/wav2vec2-xls-r-1b-en-to-15",
            feature_extractor="facebook/wav2vec2-xls-r-1b-en-to-15",
            framework="pt",
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        filename = ds[40]["file"]
        output = speech_recognizer(filename)
        self.assertEqual(output, {"text": "Ein Mann sagte zu dem Universum, Sir, ich bin da."})

    @slow
    @require_torch
    @require_torchaudio
    def test_speech_to_text_leveraged(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="patrickvonplaten/wav2vec2-2-bart-base",
            feature_extractor="patrickvonplaten/wav2vec2-2-bart-base",
            tokenizer=AutoTokenizer.from_pretrained("patrickvonplaten/wav2vec2-2-bart-base"),
            framework="pt",
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        filename = ds[40]["file"]

        output = speech_recognizer(filename)
        self.assertEqual(output, {"text": "a man said to the universe sir i exist"})

    @require_torch
    def test_chunking_fast(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="hf-internal-testing/tiny-random-wav2vec2",
            chunk_length_s=10.0,
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        n_repeats = 2
        audio_tiled = np.tile(audio, n_repeats)
        output = speech_recognizer([audio_tiled], batch_size=2)
        self.assertEqual(output, [{"text": ANY(str)}])
        self.assertEqual(output[0]["text"][:6], "ZBT ZC")

    @require_torch
    def test_return_timestamps_ctc_fast(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="hf-internal-testing/tiny-random-wav2vec2",
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        # Take short audio to keep the test readable
        audio = ds[40]["audio"]["array"][:800]

        output = speech_recognizer(audio, return_timestamps="char")
        self.assertEqual(
            output,
            {
                "text": "ZBT ZX G",
                "chunks": [
                    {"text": " ", "timestamp": (0.0, 0.012)},
                    {"text": "Z", "timestamp": (0.012, 0.016)},
                    {"text": "B", "timestamp": (0.016, 0.02)},
                    {"text": "T", "timestamp": (0.02, 0.024)},
                    {"text": " ", "timestamp": (0.024, 0.028)},
                    {"text": "Z", "timestamp": (0.028, 0.032)},
                    {"text": "X", "timestamp": (0.032, 0.036)},
                    {"text": " ", "timestamp": (0.036, 0.04)},
                    {"text": "G", "timestamp": (0.04, 0.044)},
                ],
            },
        )

        output = speech_recognizer(audio, return_timestamps="word")
        self.assertEqual(
            output,
            {
                "text": "ZBT ZX G",
                "chunks": [
                    {"text": "ZBT", "timestamp": (0.012, 0.024)},
                    {"text": "ZX", "timestamp": (0.028, 0.036)},
                    {"text": "G", "timestamp": (0.04, 0.044)},
                ],
            },
        )

    @require_torch
    @require_pyctcdecode
    def test_chunking_fast_with_lm(self):
        speech_recognizer = pipeline(
            model="hf-internal-testing/processor_with_lm",
            chunk_length_s=10.0,
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        n_repeats = 2
        audio_tiled = np.tile(audio, n_repeats)
        # Batch_size = 1
        output1 = speech_recognizer([audio_tiled], batch_size=1)
        self.assertEqual(output1, [{"text": ANY(str)}])
        self.assertEqual(output1[0]["text"][:6], "<s> <s")

        # batch_size = 2
        output2 = speech_recognizer([audio_tiled], batch_size=2)
        self.assertEqual(output2, [{"text": ANY(str)}])
        self.assertEqual(output2[0]["text"][:6], "<s> <s")

        # TODO There is an offby one error because of the ratio.
        # Maybe logits get affected by the padding on this random
        # model is more likely. Add some masking ?
        # self.assertEqual(output1, output2)

    @require_torch
    @require_pyctcdecode
    def test_with_lm_fast(self):
        speech_recognizer = pipeline(
            model="hf-internal-testing/processor_with_lm",
        )
        self.assertEqual(speech_recognizer.type, "ctc_with_lm")

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        n_repeats = 2
        audio_tiled = np.tile(audio, n_repeats)

        output = speech_recognizer([audio_tiled], batch_size=2)
        self.assertEqual(output, [{"text": ANY(str)}])
        self.assertEqual(output[0]["text"][:6], "<s> <s")

        # Making sure the argument are passed to the decoder
        # Since no change happens in the result, check the error comes from
        # the `decode_beams` function.
        with self.assertRaises(TypeError) as e:
            output = speech_recognizer([audio_tiled], decoder_kwargs={"num_beams": 2})
            self.assertContains(e.msg, "TypeError: decode_beams() got an unexpected keyword argument 'num_beams'")
        output = speech_recognizer([audio_tiled], decoder_kwargs={"beam_width": 2})

    @require_torch
    @require_pyctcdecode
    def test_with_local_lm_fast(self):
        local_dir = snapshot_download("hf-internal-testing/processor_with_lm")
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model=local_dir,
        )
        self.assertEqual(speech_recognizer.type, "ctc_with_lm")

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        n_repeats = 2
        audio_tiled = np.tile(audio, n_repeats)

        output = speech_recognizer([audio_tiled], batch_size=2)

        self.assertEqual(output, [{"text": ANY(str)}])
        self.assertEqual(output[0]["text"][:6], "<s> <s")

    @require_torch
    @slow
    def test_chunking_and_timestamps(self):
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            framework="pt",
            chunk_length_s=10.0,
        )

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        n_repeats = 10
        audio_tiled = np.tile(audio, n_repeats)
        output = speech_recognizer([audio_tiled], batch_size=2)
        self.assertEqual(output, [{"text": ("A MAN SAID TO THE UNIVERSE SIR I EXIST " * n_repeats).strip()}])

        output = speech_recognizer(audio, return_timestamps="char")
        self.assertEqual(audio.shape, (74_400,))
        self.assertEqual(speech_recognizer.feature_extractor.sampling_rate, 16_000)
        # The audio is 74_400 / 16_000 = 4.65s long.
        self.assertEqual(
            output,
            {
                "text": "A MAN SAID TO THE UNIVERSE SIR I EXIST",
                "chunks": [
                    {"text": "A", "timestamp": (0.6, 0.62)},
                    {"text": " ", "timestamp": (0.62, 0.66)},
                    {"text": "M", "timestamp": (0.68, 0.7)},
                    {"text": "A", "timestamp": (0.78, 0.8)},
                    {"text": "N", "timestamp": (0.84, 0.86)},
                    {"text": " ", "timestamp": (0.92, 0.98)},
                    {"text": "S", "timestamp": (1.06, 1.08)},
                    {"text": "A", "timestamp": (1.14, 1.16)},
                    {"text": "I", "timestamp": (1.16, 1.18)},
                    {"text": "D", "timestamp": (1.2, 1.24)},
                    {"text": " ", "timestamp": (1.24, 1.28)},
                    {"text": "T", "timestamp": (1.28, 1.32)},
                    {"text": "O", "timestamp": (1.34, 1.36)},
                    {"text": " ", "timestamp": (1.38, 1.42)},
                    {"text": "T", "timestamp": (1.42, 1.44)},
                    {"text": "H", "timestamp": (1.44, 1.46)},
                    {"text": "E", "timestamp": (1.46, 1.5)},
                    {"text": " ", "timestamp": (1.5, 1.56)},
                    {"text": "U", "timestamp": (1.58, 1.62)},
                    {"text": "N", "timestamp": (1.64, 1.68)},
                    {"text": "I", "timestamp": (1.7, 1.72)},
                    {"text": "V", "timestamp": (1.76, 1.78)},
                    {"text": "E", "timestamp": (1.84, 1.86)},
                    {"text": "R", "timestamp": (1.86, 1.9)},
                    {"text": "S", "timestamp": (1.96, 1.98)},
                    {"text": "E", "timestamp": (1.98, 2.02)},
                    {"text": " ", "timestamp": (2.02, 2.06)},
                    {"text": "S", "timestamp": (2.82, 2.86)},
                    {"text": "I", "timestamp": (2.94, 2.96)},
                    {"text": "R", "timestamp": (2.98, 3.02)},
                    {"text": " ", "timestamp": (3.06, 3.12)},
                    {"text": "I", "timestamp": (3.5, 3.52)},
                    {"text": " ", "timestamp": (3.58, 3.6)},
                    {"text": "E", "timestamp": (3.66, 3.68)},
                    {"text": "X", "timestamp": (3.68, 3.7)},
                    {"text": "I", "timestamp": (3.9, 3.92)},
                    {"text": "S", "timestamp": (3.94, 3.96)},
                    {"text": "T", "timestamp": (4.0, 4.02)},
                    {"text": " ", "timestamp": (4.06, 4.1)},
                ],
            },
        )
        output = speech_recognizer(audio, return_timestamps="word")
        self.assertEqual(
            output,
            {
                "text": "A MAN SAID TO THE UNIVERSE SIR I EXIST",
                "chunks": [
                    {"text": "A", "timestamp": (0.6, 0.62)},
                    {"text": "MAN", "timestamp": (0.68, 0.86)},
                    {"text": "SAID", "timestamp": (1.06, 1.24)},
                    {"text": "TO", "timestamp": (1.28, 1.36)},
                    {"text": "THE", "timestamp": (1.42, 1.5)},
                    {"text": "UNIVERSE", "timestamp": (1.58, 2.02)},
                    {"text": "SIR", "timestamp": (2.82, 3.02)},
                    {"text": "I", "timestamp": (3.5, 3.52)},
                    {"text": "EXIST", "timestamp": (3.66, 4.02)},
                ],
            },
        )
        output = speech_recognizer(audio, return_timestamps="word", chunk_length_s=2.0)
        self.assertEqual(
            output,
            {
                "text": "A MAN SAID TO THE UNIVERSE SIR I EXIST",
                "chunks": [
                    {"text": "A", "timestamp": (0.6, 0.62)},
                    {"text": "MAN", "timestamp": (0.68, 0.86)},
                    {"text": "SAID", "timestamp": (1.06, 1.24)},
                    {"text": "TO", "timestamp": (1.3, 1.36)},
                    {"text": "THE", "timestamp": (1.42, 1.48)},
                    {"text": "UNIVERSE", "timestamp": (1.58, 2.02)},
                    # Tiny change linked to chunking.
                    {"text": "SIR", "timestamp": (2.84, 3.02)},
                    {"text": "I", "timestamp": (3.5, 3.52)},
                    {"text": "EXIST", "timestamp": (3.66, 4.02)},
                ],
            },
        )

    @require_torch
    @slow
    def test_chunking_with_lm(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="patrickvonplaten/wav2vec2-base-100h-with-lm",
            chunk_length_s=10.0,
        )
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        n_repeats = 10
        audio = np.tile(audio, n_repeats)
        output = speech_recognizer([audio], batch_size=2)
        expected_text = "A MAN SAID TO THE UNIVERSE SIR I EXIST " * n_repeats
        expected = [{"text": expected_text.strip()}]
        self.assertEqual(output, expected)

    @require_torch
    def test_chunk_iterator(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        inputs = torch.arange(100).long()

        outs = list(chunk_iter(inputs, feature_extractor, 100, 0, 0))
        self.assertEqual(len(outs), 1)
        self.assertEqual([o["stride"] for o in outs], [(100, 0, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 100)])
        self.assertEqual([o["is_last"] for o in outs], [True])

        # two chunks no stride
        outs = list(chunk_iter(inputs, feature_extractor, 50, 0, 0))
        self.assertEqual(len(outs), 2)
        self.assertEqual([o["stride"] for o in outs], [(50, 0, 0), (50, 0, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 50), (1, 50)])
        self.assertEqual([o["is_last"] for o in outs], [False, True])

        # two chunks incomplete last
        outs = list(chunk_iter(inputs, feature_extractor, 80, 0, 0))
        self.assertEqual(len(outs), 2)
        self.assertEqual([o["stride"] for o in outs], [(80, 0, 0), (20, 0, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 80), (1, 20)])
        self.assertEqual([o["is_last"] for o in outs], [False, True])

        # one chunk since first is also last, because it contains only data
        # in the right strided part we just mark that part as non stride
        # This test is specifically crafted to trigger a bug if next chunk
        # would be ignored by the fact that all the data would be
        # contained in the strided left data.
        outs = list(chunk_iter(inputs, feature_extractor, 105, 5, 5))
        self.assertEqual(len(outs), 1)
        self.assertEqual([o["stride"] for o in outs], [(100, 0, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 100)])
        self.assertEqual([o["is_last"] for o in outs], [True])

    @require_torch
    def test_chunk_iterator_stride(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        inputs = torch.arange(100).long()
        input_values = feature_extractor(inputs, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")[
            "input_values"
        ]

        outs = list(chunk_iter(inputs, feature_extractor, 100, 20, 10))
        self.assertEqual(len(outs), 2)
        self.assertEqual([o["stride"] for o in outs], [(100, 0, 10), (30, 20, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 100), (1, 30)])
        self.assertEqual([o["is_last"] for o in outs], [False, True])

        outs = list(chunk_iter(inputs, feature_extractor, 80, 20, 10))
        self.assertEqual(len(outs), 2)
        self.assertEqual([o["stride"] for o in outs], [(80, 0, 10), (50, 20, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 80), (1, 50)])
        self.assertEqual([o["is_last"] for o in outs], [False, True])

        outs = list(chunk_iter(inputs, feature_extractor, 90, 20, 0))
        self.assertEqual(len(outs), 2)
        self.assertEqual([o["stride"] for o in outs], [(90, 0, 0), (30, 20, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 90), (1, 30)])

        inputs = torch.LongTensor([i % 2 for i in range(100)])
        input_values = feature_extractor(inputs, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")[
            "input_values"
        ]
        outs = list(chunk_iter(inputs, feature_extractor, 30, 5, 5))
        self.assertEqual(len(outs), 5)
        self.assertEqual([o["stride"] for o in outs], [(30, 0, 5), (30, 5, 5), (30, 5, 5), (30, 5, 5), (20, 5, 0)])
        self.assertEqual([o["input_values"].shape for o in outs], [(1, 30), (1, 30), (1, 30), (1, 30), (1, 20)])
        self.assertEqual([o["is_last"] for o in outs], [False, False, False, False, True])
        # (0, 25)
        self.assertEqual(nested_simplify(input_values[:, :30]), nested_simplify(outs[0]["input_values"]))
        # (25, 45)
        self.assertEqual(nested_simplify(input_values[:, 20:50]), nested_simplify(outs[1]["input_values"]))
        # (45, 65)
        self.assertEqual(nested_simplify(input_values[:, 40:70]), nested_simplify(outs[2]["input_values"]))
        # (65, 85)
        self.assertEqual(nested_simplify(input_values[:, 60:90]), nested_simplify(outs[3]["input_values"]))
        # (85, 100)
        self.assertEqual(nested_simplify(input_values[:, 80:100]), nested_simplify(outs[4]["input_values"]))

    @require_torch
    def test_stride(self):
        speech_recognizer = pipeline(
            task="automatic-speech-recognition",
            model="hf-internal-testing/tiny-random-wav2vec2",
        )
        waveform = np.tile(np.arange(1000, dtype=np.float32), 10)
        output = speech_recognizer({"raw": waveform, "stride": (0, 0), "sampling_rate": 16_000})
        self.assertEqual(output, {"text": "OB XB  B EB BB  B EB B OB X"})

        # 0 effective ids Just take the middle one
        output = speech_recognizer({"raw": waveform, "stride": (5000, 5000), "sampling_rate": 16_000})
        self.assertEqual(output, {"text": ""})

        # Only 1 arange.
        output = speech_recognizer({"raw": waveform, "stride": (0, 9000), "sampling_rate": 16_000})
        self.assertEqual(output, {"text": "OB"})

        # 2nd arange
        output = speech_recognizer({"raw": waveform, "stride": (1000, 8000), "sampling_rate": 16_000})
        self.assertEqual(output, {"text": "XB"})


def require_ffmpeg(test_case):
    """
    Decorator marking a test that requires FFmpeg.

    These tests are skipped when FFmpeg isn't installed.

    """
    import subprocess

    try:
        subprocess.check_output(["ffmpeg", "-h"], stderr=subprocess.DEVNULL)
        return test_case
    except Exception:
        return unittest.skip("test requires ffmpeg")(test_case)


def bytes_iter(chunk_size, chunks):
    for i in range(chunks):
        yield bytes(range(i * chunk_size, (i + 1) * chunk_size))


@require_ffmpeg
class AudioUtilsTest(unittest.TestCase):
    def test_chunk_bytes_iter_too_big(self):
        iter_ = iter(chunk_bytes_iter(bytes_iter(chunk_size=3, chunks=2), 10, stride=(0, 0)))
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02\x03\x04\x05", "stride": (0, 0)})
        with self.assertRaises(StopIteration):
            next(iter_)

    def test_chunk_bytes_iter(self):
        iter_ = iter(chunk_bytes_iter(bytes_iter(chunk_size=3, chunks=2), 3, stride=(0, 0)))
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02", "stride": (0, 0)})
        self.assertEqual(next(iter_), {"raw": b"\x03\x04\x05", "stride": (0, 0)})
        with self.assertRaises(StopIteration):
            next(iter_)

    def test_chunk_bytes_iter_stride(self):
        iter_ = iter(chunk_bytes_iter(bytes_iter(chunk_size=3, chunks=2), 3, stride=(1, 1)))
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02", "stride": (0, 1)})
        self.assertEqual(next(iter_), {"raw": b"\x01\x02\x03", "stride": (1, 1)})
        self.assertEqual(next(iter_), {"raw": b"\x02\x03\x04", "stride": (1, 1)})
        # This is finished, but the chunk_bytes doesn't know it yet.
        self.assertEqual(next(iter_), {"raw": b"\x03\x04\x05", "stride": (1, 1)})
        self.assertEqual(next(iter_), {"raw": b"\x04\x05", "stride": (1, 0)})
        with self.assertRaises(StopIteration):
            next(iter_)

    def test_chunk_bytes_iter_stride_stream(self):
        iter_ = iter(chunk_bytes_iter(bytes_iter(chunk_size=3, chunks=2), 5, stride=(1, 1), stream=True))
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02", "stride": (0, 0), "partial": True})
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02\x03\x04", "stride": (0, 1), "partial": False})
        self.assertEqual(next(iter_), {"raw": b"\x03\x04\x05", "stride": (1, 0), "partial": False})
        with self.assertRaises(StopIteration):
            next(iter_)

        iter_ = iter(chunk_bytes_iter(bytes_iter(chunk_size=3, chunks=3), 5, stride=(1, 1), stream=True))
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02", "stride": (0, 0), "partial": True})
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02\x03\x04", "stride": (0, 1), "partial": False})
        self.assertEqual(next(iter_), {"raw": b"\x03\x04\x05\x06\x07", "stride": (1, 1), "partial": False})
        self.assertEqual(next(iter_), {"raw": b"\x06\x07\x08", "stride": (1, 0), "partial": False})
        with self.assertRaises(StopIteration):
            next(iter_)

        iter_ = iter(chunk_bytes_iter(bytes_iter(chunk_size=3, chunks=3), 10, stride=(1, 1), stream=True))
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02", "stride": (0, 0), "partial": True})
        self.assertEqual(next(iter_), {"raw": b"\x00\x01\x02\x03\x04\x05", "stride": (0, 0), "partial": True})
        self.assertEqual(
            next(iter_), {"raw": b"\x00\x01\x02\x03\x04\x05\x06\x07\x08", "stride": (0, 0), "partial": True}
        )
        self.assertEqual(
            next(iter_), {"raw": b"\x00\x01\x02\x03\x04\x05\x06\x07\x08", "stride": (0, 0), "partial": False}
        )
        with self.assertRaises(StopIteration):
            next(iter_)
