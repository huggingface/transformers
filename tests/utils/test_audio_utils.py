# Copyright 2023 HuggingFace Inc.
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
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pytest
from huggingface_hub import sync_bucket

from transformers.audio_utils import (
    TORCHCODEC_ONLY_FILETYPES,
    _format_from_source,
    amplitude_to_db,
    chroma_filter_bank,
    get_audio_filetype,
    hertz_to_mel,
    load_audio,
    load_audio_librosa,
    load_audio_torchcodec,
    mel_to_hertz,
    power_to_db,
    window_function,
)
from transformers.testing_utils import (
    is_librosa_available,
    require_librosa,
    require_torchaudio,
    require_torchcodec,
    slow,
)
from transformers.utils import is_torch_tensor


if is_librosa_available():
    from librosa.filters import chroma


def normalize_waveform(arr: np.ndarray) -> np.ndarray:
    """Normalizes an array by dividing by its L2 norm."""
    return arr / np.linalg.norm(arr)


def compute_rmse(arr1, arr2) -> np.ndarray:
    """
    Computes the RMSE between two audio arrays after L2 normalization.
    Accepts both ``torch.Tensor`` and ``np.ndarray`` inputs;
    arrays are truncated to the shorter length before comparison.
    """
    if is_torch_tensor(arr1):
        arr1 = arr1.cpu().numpy()
    if is_torch_tensor(arr2):
        arr2 = arr2.cpu().numpy()
    arr1 = np.asarray(arr1).squeeze()
    arr2 = np.asarray(arr2).squeeze()
    max_length = min(arr1.shape[-1], arr2.shape[-1])
    arr1 = arr1[..., :max_length]
    arr2 = arr2[..., :max_length]
    return np.sqrt(((normalize_waveform(arr1) - normalize_waveform(arr2)) ** 2).mean())


class AudioUtilsFunctionTester(unittest.TestCase):
    # will be set in `def _load_datasamples`
    _dataset = None

    def test_hertz_to_mel(self):
        self.assertEqual(hertz_to_mel(0.0), 0.0)
        self.assertAlmostEqual(hertz_to_mel(100), 150.48910241)

        inputs = np.array([100, 200])
        expected = np.array([150.48910241, 283.22989816])
        self.assertTrue(np.allclose(hertz_to_mel(inputs), expected))

        self.assertEqual(hertz_to_mel(0.0, "slaney"), 0.0)
        self.assertEqual(hertz_to_mel(100, "slaney"), 1.5)

        inputs = np.array([60, 100, 200, 1000, 1001, 2000])
        expected = np.array([0.9, 1.5, 3.0, 15.0, 15.01453781, 25.08188016])
        self.assertTrue(np.allclose(hertz_to_mel(inputs, "slaney"), expected))

        inputs = np.array([60, 100, 200, 1000, 1001, 2000])
        expected = np.array([92.6824, 150.4899, 283.2313, 999.9907, 1000.6534, 1521.3674])
        self.assertTrue(np.allclose(hertz_to_mel(inputs, "kaldi"), expected))

        with pytest.raises(ValueError):
            hertz_to_mel(100, mel_scale=None)

    def test_mel_to_hertz(self):
        self.assertEqual(mel_to_hertz(0.0), 0.0)
        self.assertAlmostEqual(mel_to_hertz(150.48910241), 100)

        inputs = np.array([150.48910241, 283.22989816])
        expected = np.array([100, 200])
        self.assertTrue(np.allclose(mel_to_hertz(inputs), expected))

        self.assertEqual(mel_to_hertz(0.0, "slaney"), 0.0)
        self.assertEqual(mel_to_hertz(1.5, "slaney"), 100)

        inputs = np.array([0.9, 1.5, 3.0, 15.0, 15.01453781, 25.08188016])
        expected = np.array([60, 100, 200, 1000, 1001, 2000])
        self.assertTrue(np.allclose(mel_to_hertz(inputs, "slaney"), expected))

        inputs = np.array([92.6824, 150.4899, 283.2313, 999.9907, 1000.6534, 1521.3674])
        expected = np.array([60, 100, 200, 1000, 1001, 2000])
        self.assertTrue(np.allclose(mel_to_hertz(inputs, "kaldi"), expected))

        with pytest.raises(ValueError):
            mel_to_hertz(100, mel_scale=None)

    def test_window_function(self):
        window = window_function(16, "hann")
        self.assertEqual(len(window), 16)

        # fmt: off
        expected = np.array([
            0.0, 0.03806023, 0.14644661, 0.30865828, 0.5, 0.69134172, 0.85355339, 0.96193977,
            1.0, 0.96193977, 0.85355339, 0.69134172, 0.5, 0.30865828, 0.14644661, 0.03806023,
        ])
        # fmt: on
        self.assertTrue(np.allclose(window, expected))

    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        if self._dataset is None:
            self._dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        speech_samples = self._dataset.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    def test_power_to_db(self):
        spectrogram = np.zeros((2, 3))
        spectrogram[0, 0] = 2.0
        spectrogram[0, 1] = 0.5
        spectrogram[0, 2] = 0.707
        spectrogram[1, 1] = 1.0

        output = power_to_db(spectrogram, reference=1.0)
        expected = np.array([[3.01029996, -3.01029996, -1.50580586], [-100.0, 0.0, -100.0]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(spectrogram, reference=2.0)
        expected = np.array([[0.0, -6.02059991, -4.51610582], [-103.01029996, -3.01029996, -103.01029996]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(spectrogram, min_value=1e-6)
        expected = np.array([[3.01029996, -3.01029996, -1.50580586], [-60.0, 0.0, -60.0]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(spectrogram, db_range=80)
        expected = np.array([[3.01029996, -3.01029996, -1.50580586], [-76.98970004, 0.0, -76.98970004]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(spectrogram, reference=2.0, db_range=80)
        expected = np.array([[0.0, -6.02059991, -4.51610582], [-80.0, -3.01029996, -80.0]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(spectrogram, reference=2.0, min_value=1e-6, db_range=80)
        expected = np.array([[0.0, -6.02059991, -4.51610582], [-63.01029996, -3.01029996, -63.01029996]])
        self.assertTrue(np.allclose(output, expected))

        with pytest.raises(ValueError):
            power_to_db(spectrogram, reference=0.0)
        with pytest.raises(ValueError):
            power_to_db(spectrogram, min_value=0.0)
        with pytest.raises(ValueError):
            power_to_db(spectrogram, db_range=-80)

    def test_amplitude_to_db(self):
        spectrogram = np.zeros((2, 3))
        spectrogram[0, 0] = 2.0
        spectrogram[0, 1] = 0.5
        spectrogram[0, 2] = 0.707
        spectrogram[1, 1] = 1.0

        output = amplitude_to_db(spectrogram, reference=1.0)
        expected = np.array([[6.02059991, -6.02059991, -3.01161172], [-100.0, 0.0, -100.0]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(spectrogram, reference=2.0)
        expected = np.array([[0.0, -12.04119983, -9.03221164], [-106.02059991, -6.02059991, -106.02059991]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(spectrogram, min_value=1e-3)
        expected = np.array([[6.02059991, -6.02059991, -3.01161172], [-60.0, 0.0, -60.0]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(spectrogram, db_range=80)
        expected = np.array([[6.02059991, -6.02059991, -3.01161172], [-73.97940009, 0.0, -73.97940009]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(spectrogram, reference=2.0, db_range=80)
        expected = np.array([[0.0, -12.04119983, -9.03221164], [-80.0, -6.02059991, -80.0]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(spectrogram, reference=2.0, min_value=1e-3, db_range=80)
        expected = np.array([[0.0, -12.04119983, -9.03221164], [-66.02059991, -6.02059991, -66.02059991]])
        self.assertTrue(np.allclose(output, expected))

        with pytest.raises(ValueError):
            amplitude_to_db(spectrogram, reference=0.0)
        with pytest.raises(ValueError):
            amplitude_to_db(spectrogram, min_value=0.0)
        with pytest.raises(ValueError):
            amplitude_to_db(spectrogram, db_range=-80)

    def test_chroma_equivalence(self):
        num_frequency_bins = 25
        num_chroma = 6
        sampling_rate = 24000

        # test default parameters
        original_chroma = chroma(sr=sampling_rate, n_chroma=num_chroma, n_fft=num_frequency_bins)
        utils_chroma = chroma_filter_bank(
            num_frequency_bins=num_frequency_bins, num_chroma=num_chroma, sampling_rate=sampling_rate
        )

        self.assertTrue(np.allclose(original_chroma, utils_chroma))

        # test no weighting_parameters
        original_chroma = chroma(sr=sampling_rate, n_chroma=num_chroma, n_fft=num_frequency_bins, octwidth=None)
        utils_chroma = chroma_filter_bank(
            num_frequency_bins=num_frequency_bins,
            num_chroma=num_chroma,
            sampling_rate=sampling_rate,
            weighting_parameters=None,
        )

        self.assertTrue(np.allclose(original_chroma, utils_chroma))

        # test with L1 norm
        original_chroma = chroma(sr=sampling_rate, n_chroma=num_chroma, n_fft=num_frequency_bins, norm=1.0)
        utils_chroma = chroma_filter_bank(
            num_frequency_bins=num_frequency_bins, num_chroma=num_chroma, sampling_rate=sampling_rate, power=1.0
        )

        self.assertTrue(np.allclose(original_chroma, utils_chroma))

        # test starting at 'A' chroma, power = None, tuning = 0, different weighting_parameters
        original_chroma = chroma(
            sr=sampling_rate,
            n_chroma=num_chroma,
            n_fft=num_frequency_bins,
            norm=None,
            base_c=None,
            octwidth=1.0,
            ctroct=4.0,
        )
        utils_chroma = chroma_filter_bank(
            num_frequency_bins=num_frequency_bins,
            num_chroma=num_chroma,
            sampling_rate=sampling_rate,
            power=None,
            start_at_c_chroma=False,
            weighting_parameters=(4.0, 1.0),
        )

        self.assertTrue(np.allclose(original_chroma, utils_chroma))


@slow
class LoadAudioTester(unittest.TestCase):
    _BUCKET_URI = "hf://buckets/hf-internal-testing/all-audio-formats"
    _BUCKET_RESOLVE_URL = "https://huggingface.co/buckets/hf-internal-testing/all-audio-formats/resolve"
    _BUCKET_FILETYPES = {
        # Audio formats
        "audio.3gp": "3gp",
        "audio.aac": "aac",
        "audio.ac3": "ac3",
        "audio.aiff": "aiff",
        "audio.amr": "amr",
        "audio.au": "au",
        "audio.caf": "caf",
        "audio.flac": "flac",
        "audio.m4a": "m4a",
        "audio.mp2": "mp2",
        "audio.mp3": "mp3",
        "audio.ogg": "ogg",
        "audio.opus": "opus",
        "audio.rf64": "rf64",
        "audio.sf": "sf",
        "audio.sox": "sox",
        "audio.voc": "voc",
        "audio.w64": "w64",
        "audio.wav": "wav",
        "audio.wavex": "wav",
        "audio.wma": "wma",
        "audio.wv": "wv",
        # Video formats
        "video.3gp": "3gp",
        "video.avi": "avi",
        "video.flv": "flv",
        "video.hevc.mp4": "mp4",
        "video.m4v": "mp4",
        "video.mkv": "mkv",
        "video.mov": "mov",
        "video.mp4": "mp4",
        "video.mpg": "mpg",
        "video.ogv": "ogv",
        "video.ts": "ts",
        "video.webm": "webm",
        "video.wmv": "wmv",
    }

    @classmethod
    def setUpClass(cls):
        cls.data_dir = tempfile.mkdtemp(prefix="all-audio-")
        try:
            sync_bucket(cls._BUCKET_URI, cls.data_dir, quiet=True)
        except Exception as e:
            shutil.rmtree(cls.data_dir, ignore_errors=True)
            raise unittest.SkipTest(f"could not sync the all-audio bucket: {e}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.data_dir, ignore_errors=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.data_dir, name)

    def _sources(self, name: str) -> dict[str, str]:
        """The same synced file expressed as every source form `load_audio` accepts."""
        with open(self._path(name), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        # bucket files are named `audio.*` / `video.*`, which gives the data: URI top-level type
        media_type = f"{name.split('.', 1)[0]}/{self._BUCKET_FILETYPES[name]}"
        return {
            "path": self._path(name),
            "url": f"{self._BUCKET_RESOLVE_URL}/{name}?download=true",
            "base64": b64,
            "data_uri": f"data:{media_type};base64,{b64}",
        }

    # ---- magic-byte filetype detection --------------------------------------------------------

    def test_get_audio_filetype_identifies_every_bucket_format(self):
        for name, expected in self._BUCKET_FILETYPES.items():
            with open(self._path(name), "rb") as f:
                self.assertEqual(get_audio_filetype(f.read()), expected, name)

    def test_get_audio_filetype_rejects_non_audio_bytes(self):
        for blob in (b"", b"\xff", b"not audio at all", b"\x89PNG\r\n\x1a\n", b"%PDF-1.4"):
            with self.assertRaises(ValueError):
                get_audio_filetype(blob)

    def test_format_from_source(self):
        # format read straight from the source string, without resolving/decoding it
        cases = {
            "data:audio/wav;base64,AAAA": "wav",  # complete data: URI -> media subtype
            "data:video/mp4;base64,AAAA": "mp4",
            "data:audio/x-wav;base64,AAAA": "wav",  # the `x-` vendor prefix is stripped
            "data:;base64,AAAA": None,  # data: URI without a media type -> no hint
            "https://host/path/clip.MP4?download=true": "mp4",  # URL: extension only, query dropped
            "/some/dir/song.flac": "flac",  # local path -> extension
            "cm9hcgo=": None,  # raw base64 -> no hint
        }
        for source, expected in cases.items():
            self.assertEqual(_format_from_source(source), expected, source)

    # ---- source forms: local path, URL, base64, data: URI -------------------------------------

    @require_librosa
    def test_load_audio_from_every_source_form(self):
        # a single sync gives us the file as a local path, a public URL, raw base64 and a data: URI
        with patch("transformers.audio_utils.is_torchcodec_available", return_value=False):
            for form, source in self._sources("audio.flac").items():
                audio = load_audio(source, sampling_rate=16000)
                self.assertIsInstance(audio, np.ndarray, form)
                self.assertGreater(audio.size, 0, form)

    @require_librosa
    def test_invalid_base64_raises_value_error(self):
        # a non-URL, non-path string is treated as base64 and must fail cleanly
        with patch("transformers.audio_utils.is_torchcodec_available", return_value=False):
            with self.assertRaises(ValueError) as cm:
                load_audio("clearly_not_valid_base64_@@@")
        self.assertIn("base64", str(cm.exception))

    # ---- librosa fallback (torchcodec unavailable) --------------------------------------------

    @require_librosa
    def test_librosa_decodes_soundfile_formats(self):
        with patch("transformers.audio_utils.is_torchcodec_available", return_value=False):
            for name, filetype in self._BUCKET_FILETYPES.items():
                if filetype in TORCHCODEC_ONLY_FILETYPES:
                    continue
                audio = load_audio(self._path(name), sampling_rate=16000)
                self.assertIsInstance(audio, np.ndarray, name)
                self.assertGreater(audio.size, 0, name)

    @require_librosa
    def test_librosa_rejects_torchcodec_only_formats_with_helpful_error(self):
        with patch("transformers.audio_utils.is_torchcodec_available", return_value=False):
            for name, filetype in self._BUCKET_FILETYPES.items():
                if filetype not in TORCHCODEC_ONLY_FILETYPES:
                    continue
                with self.assertRaises(RuntimeError) as cm:
                    load_audio(self._path(name), sampling_rate=16000)
                # actionable hint instead of librosa's cryptic "Format not recognised"
                self.assertIn("torchcodec", str(cm.exception), name)

    def test_video_url_rejected_up_front_without_download(self):
        # the format is read from the URL extension alone -> clear error, no bytes fetched
        with (
            patch("transformers.audio_utils.is_torchcodec_available", return_value=False),
            patch("transformers.audio_utils._fetch_audio_bytes") as fetch,
        ):
            with self.assertRaises(RuntimeError) as cm:
                load_audio("https://host/clip.mp4?download=true")
        message = str(cm.exception)
        self.assertIn("mp4", message)
        self.assertIn("torchcodec", message)
        fetch.assert_not_called()

    @require_librosa
    def test_deprecated_load_audio_librosa_warns_and_pins_librosa(self):
        # BC: even with torchcodec available, the alias must still decode with librosa
        with patch("transformers.audio_utils.is_torchcodec_available", return_value=True):
            with self.assertWarns(FutureWarning):
                audio = load_audio_librosa(self._path("audio.wav"), sampling_rate=16000)
        self.assertIsInstance(audio, np.ndarray)

    @require_torchcodec
    def test_deprecated_load_audio_torchcodec_warns_and_pins_torchcodec(self):
        # BC: the alias must use torchcodec even when the auto-selection would pick librosa
        with patch("transformers.audio_utils.is_torchcodec_available", return_value=False):
            with self.assertWarns(FutureWarning):
                audio = load_audio_torchcodec(self._path("audio.wav"), sampling_rate=16000)
        self.assertIsInstance(audio, np.ndarray)

    # ---- torchcodec dispatch ------------------------------------------------------------------

    @require_torchcodec
    def test_torchcodec_decodes_every_bucket_format(self):
        for name in self._BUCKET_FILETYPES:
            audio = load_audio(self._path(name), sampling_rate=16000)
            self.assertIsInstance(audio, np.ndarray, name)
            self.assertGreater(audio.size, 0, name)

    # ---- torchaudio dispatch ------------------------------------------------------------------

    @require_torchaudio
    def test_torchaudio_decodes_soundfile_formats(self):
        # backend="torchaudio" decodes with torchaudio.load and resamples with
        # torchaudio.functional.resample
        for name, filetype in self._BUCKET_FILETYPES.items():
            if filetype in TORCHCODEC_ONLY_FILETYPES:
                continue
            audio = load_audio(self._path(name), sampling_rate=16000, backend="torchaudio")
            self.assertIsInstance(audio, np.ndarray, name)
            self.assertEqual(audio.ndim, 1, name)  # mono
            self.assertEqual(audio.dtype, np.float32, name)
            self.assertGreater(audio.size, 0, name)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            load_audio(self._path("audio.wav"), sampling_rate=16000, backend="not_a_backend")
