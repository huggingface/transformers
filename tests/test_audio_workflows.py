# Copyright 2026 The HuggingFace Inc. team.
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

"""Behavioral contracts shared by audio workflows, exercised at their public call interface."""

import importlib
import unittest

import numpy as np

from transformers.testing_utils import require_librosa, require_torch, require_torchaudio


def processor_classes(model, name):
    for numpy in (False, True):
        module = importlib.import_module(
            f"transformers.models.{model}.audio_processing_{'numpy_' if numpy else ''}{model}"
        )
        yield getattr(module, name + ("Numpy" if numpy else ""))


@require_torch
class AudioWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1

    def test_raw_token_framing_and_mask(self):
        for cls in processor_classes("gemma4_unified", "Gemma4UnifiedAudioProcessor"):
            with self.subTest(cls=cls):
                processor = cls()
                output = processor(
                    [self.audio[:701], self.audio[:1300]],
                    audio_samples_per_token=320,
                    padding="max_length",
                    max_length=5,
                    return_tensors="np",
                )
                self.assertEqual(output["audio_features"].shape, (2, 5, 320))
                np.testing.assert_array_equal(output["audio_features_mask"], [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
                np.testing.assert_array_equal(output["audio_features"][0].reshape(-1)[:701], self.audio[:701])
                self.assertEqual(processor.audio_samples_per_token, 640)
                self.assertNotIn("audio_features_mask", processor(self.audio, return_padding_mask=False))

    @require_torchaudio
    @require_librosa
    def test_chroma_overrides_match_fresh_processor(self):
        for cls in processor_classes("musicgen_melody", "MusicgenMelodyAudioProcessor"):
            with self.subTest(cls=cls):
                processor = cls()
                config = processor.to_dict()
                baseline = processor(self.audio, return_tensors="np")["audio_features"]
                options = {"n_fft": 2048, "hop_length": 512, "n_chroma": 24, "chunk_length": 1}
                override = processor(self.audio, padding="max_length", return_tensors="np", **options)
                fresh = cls(**options)(self.audio, padding="max_length", return_tensors="np")
                np.testing.assert_array_equal(override["audio_features"], fresh["audio_features"])
                self.assertEqual(override["audio_features"].shape, (1, 63, 24))
                self.assertEqual(processor.to_dict(), config)
                np.testing.assert_array_equal(processor(self.audio, return_tensors="np")["audio_features"], baseline)

    def test_univnet_compression_and_length_options(self):
        for cls in processor_classes("univnet", "UnivNetAudioProcessor"):
            with self.subTest(cls=cls):
                processor = cls()
                baseline = processor(self.audio, return_tensors="np")["audio_features"]
                amplified = processor(self.audio, compression_factor=2.0, return_tensors="np")["audio_features"]
                np.testing.assert_allclose(amplified, baseline + np.log(2), atol=2e-6, rtol=0)
                clipped = processor(self.audio, compression_clip_val=1.0, return_tensors="np")["audio_features"]
                self.assertTrue(np.all(clipped >= 0))
                output = processor(self.audio, max_length_s=1, padding="max_length", return_tensors="np")
                self.assertEqual(output["audio_features"].shape[1], 24000 // 256)
                self.assertEqual(processor.max_length_s, 10)

    def test_coordinated_workflow_overrides_match_fresh_processor(self):
        cases = (
            ("cohere_asr", "CohereAsrAudioProcessor", {"max_audio_clip_s": 0.5, "overlap_chunk_second": 0.1}),
            (
                "granite_speech",
                "GraniteSpeechAudioProcessor",
                {"projector_window_size": 10, "projector_downsample_rate": 2},
            ),
            ("granite_speech5", "GraniteSpeech5AudioProcessor", {"frame_stacking": 3, "delta_win_length": 5}),
            ("xcodec2", "Xcodec2AudioProcessor", {"stride": 1, "pad_to_multiple_of": 640}),
            ("neucodec", "NeuCodecAudioProcessor", {"stride": 1, "pad_to_multiple_of": 640}),
        )
        for model, name, options in cases:
            for cls in processor_classes(model, name):
                with self.subTest(cls=cls):
                    processor = cls()
                    config = processor.to_dict()
                    audio = [self.audio[:7100], self.audio]
                    override = processor(audio, dither=0.0, return_tensors="np", **options)
                    fresh = cls(**options)(audio, dither=0.0, return_tensors="np")
                    self.assertEqual(set(override), set(fresh))
                    for key in override:
                        if key == "audio_chunk_index":
                            self.assertEqual(override[key], fresh[key])
                        else:
                            np.testing.assert_array_equal(override[key], fresh[key])
                    self.assertEqual(processor.to_dict(), config)
                    self.assertEqual(cls.from_dict(config).to_dict(), config)
                    if model == "cohere_asr":
                        self.assertGreater(len(override["audio_chunk_index"]), len(audio))
                        self.assertEqual(len(override["audio_chunk_index"]), override["audio_features"].shape[0])

    def test_optional_masks_do_not_remove_required_metadata(self):
        for model, name in (
            ("phi4_multimodal", "Phi4MultimodalAudioProcessor"),
            ("kyutai_speech_to_text", "KyutaiSpeechToTextAudioProcessor"),
            ("xcodec2", "Xcodec2AudioProcessor"),
            ("neucodec", "NeuCodecAudioProcessor"),
        ):
            for cls in processor_classes(model, name):
                with self.subTest(cls=cls):
                    processor = cls()
                    audio = [self.audio[:7001], self.audio]
                    default = processor(audio, return_tensors="np")
                    unmasked = processor(audio, return_padding_mask=False, return_tensors="np")
                    removed = "audio_features_mask" if model == "phi4_multimodal" else "audio_values_mask"
                    self.assertNotIn(removed, unmasked)
                    self.assertEqual(set(unmasked), set(default) - {removed})
                    for key in unmasked:
                        np.testing.assert_array_equal(unmasked[key], default[key])

    def test_seamless_stride_keeps_mask_aligned(self):
        for cls in processor_classes("seamless_m4t", "SeamlessM4tAudioProcessor"):
            for stride in (1, 2, 3):
                with self.subTest(cls=cls, stride=stride):
                    output = cls()([self.audio[:7100], self.audio], stride=stride, return_tensors="np")
                    self.assertEqual(output["audio_features"].shape[:2], output["audio_features_mask"].shape)
                    self.assertEqual(output["audio_features"].shape[-1], 80 * stride)
                    self.assertTrue(output["audio_features_mask"][0].sum() < output["audio_features_mask"][1].sum())

    def test_unsupported_workflow_switches_fail_at_validation(self):
        cases = (
            (
                "audio_spectrogram_transformer",
                "AudioSpectrogramTransformerAudioProcessor",
                {"do_batch_spectrogram": True},
            ),
            ("speech_to_text", "SpeechToTextAudioProcessor", {"do_batch_spectrogram": True}),
            ("kyutai_speech_to_text", "KyutaiSpeechToTextAudioProcessor", {"do_extract_spectrogram": True}),
            ("nemotron_asr_streaming", "NemotronAsrStreamingAudioProcessor", {"do_extract_spectrogram": False}),
            ("xcodec2", "Xcodec2AudioProcessor", {"padding_side": "left"}),
        )
        for model, name, options in cases:
            for cls in processor_classes(model, name):
                with self.subTest(cls=cls):
                    with self.assertRaises(ValueError):
                        cls()(self.audio, **options)
