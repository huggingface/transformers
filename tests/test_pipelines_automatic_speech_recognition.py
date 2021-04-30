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

from transformers import AutoFeatureExtractor, AutoTokenizer, Speech2TextForConditionalGeneration, Wav2Vec2ForCTC
from transformers.pipelines import AutomaticSpeechRecognitionPipeline, pipeline
from transformers.testing_utils import require_datasets, require_torch, require_torchaudio, slow


# We can't use this mixin because it assumes TF support.
# from .test_pipelines_common import CustomInputPipelineCommonMixin


class AutomaticSpeechRecognitionPipelineTests(unittest.TestCase):
    @require_torch
    @slow
    def test_pt_defaults(self):
        pipeline("automatic-speech-recognition", framework="pt")

    @require_torch
    def test_torch_small(self):
        import numpy as np

        nlp = pipeline(
            task="automatic-speech-recognition",
            model="facebook/s2t-small-mustc-en-fr-st",
            tokenizer="facebook/s2t-small-mustc-en-fr-st",
            framework="pt",
        )
        waveform = np.zeros((34000,))
        output = nlp(waveform)
        self.assertEqual(output, {"text": "C'est ce que j'ai fait à ce moment-là."})

    @require_datasets
    @require_torch
    @slow
    def test_torch_large(self):
        import numpy as np

        nlp = pipeline(
            task="automatic-speech-recognition",
            model="facebook/wav2vec2-base-960h",
            tokenizer="facebook/wav2vec2-base-960h",
            framework="pt",
        )
        waveform = np.zeros((34000,))
        output = nlp(waveform)
        self.assertEqual(output, {"text": ""})

        from datasets import load_dataset

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        filename = ds[0]["file"]
        output = nlp(filename)
        self.assertEqual(output, {"text": "A MAN SAID TO THE UNIVERSE SIR I EXIST"})

    @slow
    @require_torch
    @require_datasets
    def test_simple_wav2vec2(self):
        import numpy as np
        from datasets import load_dataset

        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

        asr = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

        waveform = np.zeros((34000,))
        output = asr(waveform)
        self.assertEqual(output, {"text": ""})

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        filename = ds[0]["file"]
        output = asr(filename)
        self.assertEqual(output, {"text": "A MAN SAID TO THE UNIVERSE SIR I EXIST"})

        filename = ds[0]["file"]
        with open(filename, "rb") as f:
            data = f.read()
        output = asr(data)
        self.assertEqual(output, {"text": "A MAN SAID TO THE UNIVERSE SIR I EXIST"})

    @slow
    @require_torch
    @require_torchaudio
    @require_datasets
    def test_simple_s2t(self):
        import numpy as np
        from datasets import load_dataset

        model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-mustc-en-it-st")
        tokenizer = AutoTokenizer.from_pretrained("facebook/s2t-small-mustc-en-it-st")
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/s2t-small-mustc-en-it-st")

        asr = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

        waveform = np.zeros((34000,))

        output = asr(waveform)
        self.assertEqual(output, {"text": "E questo è il motivo per cui non ci siamo mai incontrati."})

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
        filename = ds[0]["file"]
        output = asr(filename)
        self.assertEqual(output, {"text": "Un uomo disse all'universo: \"Signore, io esisto."})

        filename = ds[0]["file"]
        with open(filename, "rb") as f:
            data = f.read()
        output = asr(data)
        self.assertEqual(output, {"text": "Un uomo disse all'universo: \"Signore, io esisto."})
