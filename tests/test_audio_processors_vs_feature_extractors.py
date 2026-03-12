# Copyright 2025 The HuggingFace Inc. team.
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

"""
Tests comparing the new AudioProcessor classes against the legacy FeatureExtractor classes.

For each model, we:
1. Instantiate the FeatureExtractor via from_pretrained (from the Hub)
2. Instantiate the corresponding AudioProcessor directly
3. Run both on the same batched audio input
4. Assert torch.equal on the main output tensors
"""

import importlib
import os
import sys

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Feature extractor classes are loaded from ~/transformers/src (upstream).
# We temporarily swap sys.path and clear cached transformers modules so that
# ``import transformers.models.X.feature_extraction_X`` resolves to the
# upstream checkout rather than the locally-installed (audio-processors) copy.
# ---------------------------------------------------------------------------
_UPSTREAM_SRC = os.path.expanduser("~/transformers/src")

_fe_class_specs = [
    ("transformers.models.audio_spectrogram_transformer.feature_extraction_audio_spectrogram_transformer", "ASTFeatureExtractor"),
    ("transformers.models.clap.feature_extraction_clap", "ClapFeatureExtractor"),
    ("transformers.models.clvp.feature_extraction_clvp", "ClvpFeatureExtractor"),
    ("transformers.models.dac.feature_extraction_dac", "DacFeatureExtractor"),
    ("transformers.models.dia.feature_extraction_dia", "DiaFeatureExtractor"),
    ("transformers.models.encodec.feature_extraction_encodec", "EncodecFeatureExtractor"),
    ("transformers.models.granite_speech.feature_extraction_granite_speech", "GraniteSpeechFeatureExtractor"),
    ("transformers.models.kyutai_speech_to_text.feature_extraction_kyutai_speech_to_text", "KyutaiSpeechToTextFeatureExtractor"),
    ("transformers.models.lasr.feature_extraction_lasr", "LasrFeatureExtractor"),
    ("transformers.models.musicgen_melody.feature_extraction_musicgen_melody", "MusicgenMelodyFeatureExtractor"),
    ("transformers.models.parakeet.feature_extraction_parakeet", "ParakeetFeatureExtractor"),
    ("transformers.models.phi4_multimodal.feature_extraction_phi4_multimodal", "Phi4MultimodalFeatureExtractor"),
    ("transformers.models.pop2piano.feature_extraction_pop2piano", "Pop2PianoFeatureExtractor"),
    ("transformers.models.seamless_m4t.feature_extraction_seamless_m4t", "SeamlessM4TFeatureExtractor"),
    ("transformers.models.speech_to_text.feature_extraction_speech_to_text", "Speech2TextFeatureExtractor"),
    ("transformers.models.speecht5.feature_extraction_speecht5", "SpeechT5FeatureExtractor"),
    ("transformers.models.univnet.feature_extraction_univnet", "UnivNetFeatureExtractor"),
    ("transformers.models.vibevoice_acoustic_tokenizer.feature_extraction_vibevoice_acoustic_tokenizer", "VibeVoiceAcousticTokenizerFeatureExtractor"),
    ("transformers.models.voxtral_realtime.feature_extraction_voxtral_realtime", "VoxtralRealtimeFeatureExtractor"),
    ("transformers.models.wav2vec2.feature_extraction_wav2vec2", "Wav2Vec2FeatureExtractor"),
    ("transformers.models.whisper.feature_extraction_whisper", "WhisperFeatureExtractor"),
]


def _load_upstream_classes(class_specs):
    """Load feature extractor classes from ~/transformers/src.

    Temporarily replaces the transformers package in sys.modules so that
    imports resolve to the upstream checkout.
    """
    # 1. Save and remove all cached transformers modules
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "transformers" or key.startswith("transformers."):
            saved_modules[key] = sys.modules.pop(key)

    # 2. Prepend upstream src to sys.path
    sys.path.insert(0, _UPSTREAM_SRC)

    results = {}
    try:
        for module_path, class_name in class_specs:
            mod = importlib.import_module(module_path)
            results[class_name] = getattr(mod, class_name)
    finally:
        # 3. Remove upstream from sys.path
        sys.path.remove(_UPSTREAM_SRC)
        # 4. Clear all upstream-loaded transformers modules
        for key in list(sys.modules.keys()):
            if key == "transformers" or key.startswith("transformers."):
                del sys.modules[key]
        # 5. Restore the local project's transformers modules
        sys.modules.update(saved_modules)

    return results


def _load_upstream_class(module_path, class_name):
    """Load a single class from ~/transformers/src."""
    return _load_upstream_classes([(module_path, class_name)])[class_name]


# Load all FE classes from upstream in one batch
_fe_classes = _load_upstream_classes(_fe_class_specs)
ASTFeatureExtractor = _fe_classes["ASTFeatureExtractor"]
ClapFeatureExtractor = _fe_classes["ClapFeatureExtractor"]
ClvpFeatureExtractor = _fe_classes["ClvpFeatureExtractor"]
DacFeatureExtractor = _fe_classes["DacFeatureExtractor"]
DiaFeatureExtractor = _fe_classes["DiaFeatureExtractor"]
EncodecFeatureExtractor = _fe_classes["EncodecFeatureExtractor"]
GraniteSpeechFeatureExtractor = _fe_classes["GraniteSpeechFeatureExtractor"]
KyutaiSpeechToTextFeatureExtractor = _fe_classes["KyutaiSpeechToTextFeatureExtractor"]
LasrFeatureExtractor = _fe_classes["LasrFeatureExtractor"]
MusicgenMelodyFeatureExtractor = _fe_classes["MusicgenMelodyFeatureExtractor"]
ParakeetFeatureExtractor = _fe_classes["ParakeetFeatureExtractor"]
Phi4MultimodalFeatureExtractor = _fe_classes["Phi4MultimodalFeatureExtractor"]
Pop2PianoFeatureExtractor = _fe_classes["Pop2PianoFeatureExtractor"]
SeamlessM4TFeatureExtractor = _fe_classes["SeamlessM4TFeatureExtractor"]
Speech2TextFeatureExtractor = _fe_classes["Speech2TextFeatureExtractor"]
SpeechT5FeatureExtractor = _fe_classes["SpeechT5FeatureExtractor"]
UnivNetFeatureExtractor = _fe_classes["UnivNetFeatureExtractor"]
VibeVoiceAcousticTokenizerFeatureExtractor = _fe_classes["VibeVoiceAcousticTokenizerFeatureExtractor"]
VoxtralRealtimeFeatureExtractor = _fe_classes["VoxtralRealtimeFeatureExtractor"]
Wav2Vec2FeatureExtractor = _fe_classes["Wav2Vec2FeatureExtractor"]
WhisperFeatureExtractor = _fe_classes["WhisperFeatureExtractor"]

# Audio processor imports (from local project)
from transformers.models.audio_spectrogram_transformer.audio_processing_audio_spectrogram_transformer import (
    AudioSpectrogramTransformerAudioProcessor,
)
from transformers.models.clap.audio_processing_clap import ClapAudioProcessor
from transformers.models.clvp.audio_processing_clvp import ClvpAudioProcessor
from transformers.models.dac.audio_processing_dac import DacAudioProcessor
from transformers.models.dia.audio_processing_dia import DiaAudioProcessor
from transformers.models.encodec.audio_processing_encodec import EncodecAudioProcessor
from transformers.models.granite_speech.audio_processing_granite_speech import GraniteSpeechAudioProcessor
from transformers.models.kyutai_speech_to_text.audio_processing_kyutai_speech_to_text import (
    KyutaiSpeechToTextAudioProcessor,
)
from transformers.models.lasr.audio_processing_lasr import LasrAudioProcessor
from transformers.models.musicgen_melody.audio_processing_musicgen_melody import MusicgenMelodyAudioProcessor
from transformers.models.parakeet.audio_processing_parakeet import ParakeetAudioProcessor
from transformers.models.phi4_multimodal.audio_processing_phi4_multimodal import Phi4MultimodalAudioProcessor
from transformers.models.pop2piano.audio_processing_pop2piano import Pop2PianoAudioProcessor
from transformers.models.seamless_m4t.audio_processing_seamless_m4t import SeamlessM4tAudioProcessor
from transformers.models.speech_to_text.audio_processing_speech_to_text import SpeechToTextAudioProcessor
from transformers.models.speecht5.audio_processing_speecht5 import SpeechT5AudioProcessor
from transformers.models.univnet.audio_processing_univnet import UnivNetAudioProcessor
from transformers.models.vibevoice_acoustic_tokenizer.audio_processing_vibevoice_acoustic_tokenizer import (
    VibevoiceAcousticTokenizerAudioProcessor,
)
from transformers.models.voxtral_realtime.audio_processing_voxtral_realtime import VoxtralRealtimeAudioProcessor
from transformers.models.wav2vec2.audio_processing_wav2vec2 import Wav2Vec2AudioProcessor
from transformers.models.whisper.audio_processing_whisper import WhisperAudioProcessor


# Sentinel to exclude a key from default kwargs
_EXCLUDE = object()

# Each entry is a dict with model config. Keys:
#   name, hub_repo, fe_class, ap_class, fe_output_key, sample_rate
#   fe_kwargs (optional): extra kwargs for the FE call (use _EXCLUDE to remove a default key)
#   ap_kwargs (optional): extra kwargs for the AP call
MODEL_CONFIGS = [
    {
        "name": "audio_spectrogram_transformer",
        "hub_repo": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "fe_class": ASTFeatureExtractor,
        "ap_class": AudioSpectrogramTransformerAudioProcessor,
        "fe_output_key": "input_values",
        "sample_rate": 16000,
        "atol": 1e-6,
    },
    {
        "name": "clap",
        "hub_repo": "laion/clap-htsat-unfused",
        "fe_class": ClapFeatureExtractor,
        "ap_class": ClapAudioProcessor,
        "fe_output_key": "input_features",
        "sample_rate": 48000,
    },
    {
        "name": "clvp",
        "hub_repo": "susnato/clvp_dev",
        "fe_class": ClvpFeatureExtractor,
        "ap_class": ClvpAudioProcessor,
        "fe_output_key": "input_features",
        "sample_rate": 22050,
        "ap_init_kwargs": {
            "mel_norms": [-7.0095, -6.0832, -4.644, -3.3562, -2.4548, -2.0097, -1.6036, -1.8641, -2.3728, -2.3455, -2.5947, -2.6695, -2.7129, -2.8555, -3.0251, -3.0889, -3.4261, -3.6759, -4.078, -4.4624, -4.7812, -5.0075, -5.1284, -5.2717, -5.4006, -5.4993, -5.531, -5.5878, -5.6726, -5.7016, -5.7943, -5.8831, -5.9537, -5.9989, -6.0305, -6.0539, -6.0748, -6.1163, -6.1481, -6.2476, -6.3195, -6.4457, -6.5377, -6.611, -6.6481, -6.6671, -6.6539, -6.6499, -6.6794, -6.7833, -6.9307, -7.0818, -7.1894, -7.2439, -7.3168, -7.3779, -7.4491, -7.5233, -7.6224, -7.7473, -7.8994, -8.0604, -8.2181, -8.3998, -8.5556, -8.7161, -8.8481, -8.9582, -9.0371, -9.0867, -9.1546, -9.2038, -9.2334, -9.2292, -9.2304, -9.268, -9.3156, -9.3716, -9.4165, -9.4822],
        },
    },
    {
        "name": "dac",
        "hub_repo": "descript/dac_16khz",
        "fe_class": DacFeatureExtractor,
        "ap_class": DacAudioProcessor,
        "fe_output_key": "input_values",
        "sample_rate": 16000,
    },
    {
        "name": "dia",
        "hub_repo": "nari-labs/Dia-1.6B-0626",
        "fe_class": DiaFeatureExtractor,
        "ap_class": DiaAudioProcessor,
        "fe_output_key": "input_values",
        "sample_rate": 44100,
    },
    {
        "name": "encodec",
        "hub_repo": "facebook/encodec_24khz",
        "fe_class": EncodecFeatureExtractor,
        "ap_class": EncodecAudioProcessor,
        "fe_output_key": "input_values",
        "sample_rate": 24000,
    },
    # {
    #     "name": "gemma3n",
    #     "hub_repo": "google/gemma-3n-e4b-it",
    #     "fe_class": Gemma3nAudioFeatureExtractor,
    #     "ap_class": Gemma3nAudioProcessor,
    #     "fe_output_key": "input_features",
    #     "sample_rate": 16000,
    #     # AP now implements custom FFT with HTK preemphasis and FFT overdrive
    # },
    {
        "name": "granite_speech",
        "hub_repo": "ibm-granite/granite-speech-3.2-8b",
        "fe_class": GraniteSpeechFeatureExtractor,
        "ap_class": GraniteSpeechAudioProcessor,
        "fe_output_key": "input_features",
        "sample_rate": 16000,
        "fe_kwargs": {"sampling_rate": _EXCLUDE, "return_tensors": _EXCLUDE, "padding": _EXCLUDE},
    },
    {
        "name": "kyutai_speech_to_text",
        "hub_repo": "kyutai/stt-2.6b-en-trfs",
        "fe_class": KyutaiSpeechToTextFeatureExtractor,
        "ap_class": KyutaiSpeechToTextAudioProcessor,
        "fe_output_key": "input_values",
        "sample_rate": 24000,
        # AP now implements 1-second delay padding
    },
    {
        "name": "lasr",
        "hub_repo": None,
        "fe_class": LasrFeatureExtractor,
        "ap_class": LasrAudioProcessor,
        "fe_output_key": "input_features",
        "sample_rate": 16000,
    },
    {
        "name": "musicgen_melody",
        "hub_repo": "facebook/musicgen-melody",
        "fe_class": MusicgenMelodyFeatureExtractor,
        "ap_class": MusicgenMelodyAudioProcessor,
        "fe_output_key": "input_features",
        "sample_rate": 32000,
    },
    {
        "name": "parakeet",
        "hub_repo": "nvidia/parakeet-ctc-1.1b",
        "fe_class": ParakeetFeatureExtractor,
        "ap_class": ParakeetAudioProcessor,
        "fe_output_key": "input_features",
        "sample_rate": 16000,
        # AP now implements preemphasis, natural log, and slaney mel filters
    },
    {
        "name": "phi4_multimodal",
        "hub_repo": "microsoft/Phi-4-multimodal-instruct",
        "fe_class": Phi4MultimodalFeatureExtractor,
        "ap_class": Phi4MultimodalAudioProcessor,
        "fe_output_key": "audio_input_features",
        "sample_rate": 16000,
    },
    # {
    #     "name": "pop2piano",
    #     "hub_repo": "sweetcocoa/pop2piano",
    #     "fe_class": Pop2PianoFeatureExtractor,
    #     "ap_class": Pop2PianoAudioProcessor,
    #     "fe_output_key": "input_features",
    #     "sample_rate": 22050,
    #     "fe_kwargs": {"sampling_rate": [22050, 22050]},
    #     # Skipped: Requires essentia library
    # },
    {
        "name": "seamless_m4t",
        "hub_repo": "facebook/hf-seamless-m4t-medium",
        "fe_class": SeamlessM4TFeatureExtractor,
        "ap_class": SeamlessM4tAudioProcessor,
        "fe_output_key": "input_features",
        "sample_rate": 16000,
        # AP now implements Kaldi-style features with stride concatenation
    },
    {
        "name": "speech_to_text",
        "hub_repo": "facebook/s2t-small-librispeech-asr",
        "fe_class": Speech2TextFeatureExtractor,
        "ap_class": SpeechToTextAudioProcessor,
        "fe_output_key": "input_features",
        "sample_rate": 16000,
    },
    {
        "name": "speecht5",
        "hub_repo": "microsoft/speecht5_asr",
        "fe_class": SpeechT5FeatureExtractor,
        "ap_class": SpeechT5AudioProcessor,
        "fe_output_key": "input_values",
        "sample_rate": 16000,
    },
    # {
    #     "name": "univnet",
    #     "hub_repo": "dg845/univnet-dev",
    #     "fe_class": UnivNetFeatureExtractor,
    #     "ap_class": UnivNetAudioProcessor,
    #     "fe_output_key": "input_features",
    #     "sample_rate": 24000,
    # },
    {
        "name": "vibevoice_acoustic_tokenizer",
        "hub_repo": "microsoft/VibeVoice-AcousticTokenizer",
        "fe_class": VibeVoiceAcousticTokenizerFeatureExtractor,
        "ap_class": VibevoiceAcousticTokenizerAudioProcessor,
        "fe_output_key": "input_values",
        "sample_rate": 24000,
        "fe_kwargs": {"return_tensors": _EXCLUDE, "padding": _EXCLUDE},
    },
    {
        "name": "voxtral_realtime",
        "hub_repo": "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "fe_class": VoxtralRealtimeFeatureExtractor,
        "ap_class": VoxtralRealtimeAudioProcessor,
        "fe_output_key": "input_features",
        "sample_rate": 16000,
    },
    {
        "name": "wav2vec2",
        "hub_repo": "facebook/wav2vec2-large-960h-lv60-self",
        "fe_class": Wav2Vec2FeatureExtractor,
        "ap_class": Wav2Vec2AudioProcessor,
        "fe_output_key": "input_values",
        "sample_rate": 16000,
    },
    {
        "name": "whisper",
        "hub_repo": "openai/whisper-small",
        "fe_class": WhisperFeatureExtractor,
        "ap_class": WhisperAudioProcessor,
        "fe_output_key": "input_features",
        "sample_rate": 16000,
        "ap_kwargs": {"max_length": None, "truncation": False},
    },
]


def _make_audio_batch(sample_rate: int, seed: int = 42) -> list[np.ndarray]:
    """Create a deterministic batched audio input: two clips of different lengths."""
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal(sample_rate).astype(np.float32),  # 1 second
        rng.standard_normal(sample_rate * 2).astype(np.float32),  # 2 seconds
    ]


@pytest.mark.parametrize(
    "config",
    MODEL_CONFIGS,
    ids=[c["name"] if isinstance(c, dict) else c.values[0]["name"] for c in MODEL_CONFIGS],
)
def test_audio_processor_matches_feature_extractor(config):
    hub_repo = config["hub_repo"]
    fe_class = config["fe_class"]
    ap_class = config["ap_class"]
    fe_output_key = config["fe_output_key"]
    sample_rate = config["sample_rate"]

    # Instantiate feature extractor from the Hub (or with defaults if hub_repo is None)
    if hub_repo is not None:
        fe = fe_class.from_pretrained(hub_repo)
    else:
        fe = fe_class()

    # Instantiate audio processor directly
    ap_init_kwargs = config.get("ap_init_kwargs", {})
    ap = ap_class(**ap_init_kwargs)

    # Create batched audio input (deterministic)
    audio_batch = _make_audio_batch(sample_rate)

    # Default kwargs
    default_fe_kwargs = {
        "sampling_rate": sample_rate,
        "return_tensors": "pt",
        "padding": True,
    }
    default_ap_kwargs = {
        "sampling_rate": sample_rate,
        "return_tensors": "pt",
        "padding": True,
    }

    # Apply per-model overrides (use _EXCLUDE sentinel to remove default keys)
    fe_kwargs = {**default_fe_kwargs, **config.get("fe_kwargs", {})}
    fe_kwargs = {k: v for k, v in fe_kwargs.items() if v is not _EXCLUDE}
    ap_kwargs = {**default_ap_kwargs, **config.get("ap_kwargs", {})}
    ap_kwargs = {k: v for k, v in ap_kwargs.items() if v is not _EXCLUDE}

    # Run feature extractor (copy inputs since some FEs mutate the list in-place)
    fe_output = fe([x.copy() for x in audio_batch], **fe_kwargs)

    # Run audio processor
    ap_output = ap([x.copy() for x in audio_batch], **ap_kwargs)

    fe_to_ap_key_map = {
        "input_features": "audio_features",
        "input_values": "audio_values",
        "audio_input_features": "audio_features",
    }

    # Mapping for attention mask and padding mask keys depending on the primary input key
    mask_key_map = {
        "input_values": "audio_values_mask",
        "input_features": "audio_features_mask",
    }

    # Find out if this output contains input_values or input_features (to key mask mapping)
    has_input_values = "input_values" in fe_output
    has_input_features = "input_features" in fe_output

    for fe_key in fe_output.keys():
        # Remap the primary data keys
        ap_key = fe_to_ap_key_map.get(fe_key, fe_key)

        # Special handling for attention_mask and padding_mask mapping
        if fe_key in ("attention_mask", "padding_mask"):
            if has_input_values:
                ap_key = mask_key_map["input_values"]
            elif has_input_features:
                ap_key = mask_key_map["input_features"]
            else:
                ap_key = fe_key  # fallback/default

        assert ap_key in ap_output, f"Key {ap_key} (from FE key {fe_key}) not found in audio processor output"
        fe_tensor = fe_output[fe_key]
        ap_tensor = ap_output[ap_key]

        if not isinstance(fe_tensor, torch.Tensor):
            fe_tensor = torch.tensor(fe_tensor)
        if not isinstance(ap_tensor, torch.Tensor):
            ap_tensor = torch.tensor(ap_tensor)

        assert fe_tensor.shape == ap_tensor.shape, (
            f"Shape mismatch for key '{fe_key}' (ap key '{ap_key}'): fe {fe_tensor.shape} vs ap {ap_tensor.shape}"
        )
        atol = config.get("atol", 0.0)
        if atol > 0:
            assert torch.allclose(fe_tensor, ap_tensor, atol=atol, rtol=0), (
                f"Value mismatch for key '{fe_key}' (ap key '{ap_key}'): max abs diff = {(fe_tensor - ap_tensor).abs().max().item():.6e}, atol={atol}"
            )
        else:
            assert torch.equal(fe_tensor, ap_tensor), (
                f"Value mismatch for key '{fe_key}' (ap key '{ap_key}'): max abs diff = {(fe_tensor - ap_tensor).abs().max().item():.6e}"
            )


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

# Pairs of (fe_module_path, fe_class_name, ap_class)
_COMPAT_PAIRS = [
    ("transformers.models.whisper.feature_extraction_whisper", "WhisperFeatureExtractor", WhisperAudioProcessor),
    ("transformers.models.clap.feature_extraction_clap", "ClapFeatureExtractor", ClapAudioProcessor),
    ("transformers.models.encodec.feature_extraction_encodec", "EncodecFeatureExtractor", EncodecAudioProcessor),
    ("transformers.models.dac.feature_extraction_dac", "DacFeatureExtractor", DacAudioProcessor),
    ("transformers.models.wav2vec2.feature_extraction_wav2vec2", "Wav2Vec2FeatureExtractor", Wav2Vec2AudioProcessor),
]


@pytest.mark.parametrize(
    "module_path, class_name, ap_class",
    _COMPAT_PAIRS,
    ids=[p[1] for p in _COMPAT_PAIRS],
)
class TestFeatureExtractorBackwardCompat:
    """Tests that deprecated FeatureExtractor wrappers work correctly."""

    def test_importable_and_warns(self, module_path, class_name, ap_class):
        """Old class names are importable and emit FutureWarning."""
        import importlib

        mod = importlib.import_module(module_path)
        fe_cls = getattr(mod, class_name)
        with pytest.warns(FutureWarning, match="deprecated"):
            fe_cls()

    def test_isinstance_check(self, module_path, class_name, ap_class):
        """Deprecated FE instances pass isinstance checks against AudioProcessor."""
        import importlib
        import warnings

        mod = importlib.import_module(module_path)
        fe_cls = getattr(mod, class_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fe = fe_cls()
        assert isinstance(fe, ap_class)
        assert issubclass(fe_cls, ap_class)

    def test_issubclass(self, module_path, class_name, ap_class):
        """Deprecated FE class is a subclass of AudioProcessor."""
        import importlib

        mod = importlib.import_module(module_path)
        fe_cls = getattr(mod, class_name)
        assert issubclass(fe_cls, ap_class)


class TestBatchFeatureLegacyKeys:
    """Tests that old output key names are accessible via BatchFeature."""

    def setup_method(self):
        from transformers.audio_processing_base import BatchFeature as AudioBatchFeature

        # Reset warned keys so each test gets fresh warnings
        AudioBatchFeature._warned_keys.clear()

    def test_input_features_resolves_to_audio_features(self):
        from transformers.audio_processing_base import BatchFeature as AudioBatchFeature

        bf = AudioBatchFeature({"audio_features": np.array([1, 2, 3])})
        with pytest.warns(FutureWarning, match="input_features"):
            result = bf["input_features"]
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_input_values_resolves_to_audio_values(self):
        from transformers.audio_processing_base import BatchFeature as AudioBatchFeature

        bf = AudioBatchFeature({"audio_values": np.array([4, 5, 6])})
        with pytest.warns(FutureWarning, match="input_values"):
            result = bf["input_values"]
        assert np.array_equal(result, np.array([4, 5, 6]))

    def test_attention_mask_resolves_to_audio_features_mask(self):
        from transformers.audio_processing_base import BatchFeature as AudioBatchFeature

        bf = AudioBatchFeature({"audio_features": np.array([1]), "audio_features_mask": np.array([1, 1, 0])})
        with pytest.warns(FutureWarning, match="attention_mask"):
            result = bf["attention_mask"]
        assert np.array_equal(result, np.array([1, 1, 0]))

    def test_attention_mask_resolves_to_audio_values_mask(self):
        from transformers.audio_processing_base import BatchFeature as AudioBatchFeature

        bf = AudioBatchFeature({"audio_values": np.array([1]), "audio_values_mask": np.array([0, 1, 1])})
        with pytest.warns(FutureWarning, match="attention_mask"):
            result = bf["attention_mask"]
        assert np.array_equal(result, np.array([0, 1, 1]))

    def test_contains_legacy_key(self):
        from transformers.audio_processing_base import BatchFeature as AudioBatchFeature

        bf = AudioBatchFeature({"audio_features": np.array([1])})
        assert "input_features" in bf
        assert "audio_features" in bf
        assert "nonexistent_key" not in bf

    def test_warning_fires_once(self):
        from transformers.audio_processing_base import BatchFeature as AudioBatchFeature

        bf = AudioBatchFeature({"audio_features": np.array([1, 2, 3])})
        with pytest.warns(FutureWarning, match="input_features"):
            bf["input_features"]
        # Second access should not warn
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bf["input_features"]
            future_warnings = [x for x in w if issubclass(x.category, FutureWarning) and "input_features" in str(x.message)]
            assert len(future_warnings) == 0
