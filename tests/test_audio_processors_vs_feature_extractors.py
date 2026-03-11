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

import numpy as np
import pytest
import torch

from transformers.models.audio_spectrogram_transformer.audio_processing_audio_spectrogram_transformer import (
    AudioSpectrogramTransformerAudioProcessor,
)
from transformers.models.audio_spectrogram_transformer.feature_extraction_audio_spectrogram_transformer import (
    ASTFeatureExtractor,
)
from transformers.models.clap.audio_processing_clap import ClapAudioProcessor
from transformers.models.clap.feature_extraction_clap import ClapFeatureExtractor
from transformers.models.clvp.audio_processing_clvp import ClvpAudioProcessor
from transformers.models.clvp.feature_extraction_clvp import ClvpFeatureExtractor
from transformers.models.dac.audio_processing_dac import DacAudioProcessor
from transformers.models.dac.feature_extraction_dac import DacFeatureExtractor
from transformers.models.dia.audio_processing_dia import DiaAudioProcessor
from transformers.models.dia.feature_extraction_dia import DiaFeatureExtractor
from transformers.models.encodec.audio_processing_encodec import EncodecAudioProcessor
from transformers.models.encodec.feature_extraction_encodec import EncodecFeatureExtractor
from transformers.models.gemma3n.audio_processing_gemma3n import Gemma3nAudioProcessor
from transformers.models.gemma3n.feature_extraction_gemma3n import Gemma3nAudioFeatureExtractor
from transformers.models.granite_speech.audio_processing_granite_speech import GraniteSpeechAudioProcessor
from transformers.models.granite_speech.feature_extraction_granite_speech import GraniteSpeechFeatureExtractor
from transformers.models.kyutai_speech_to_text.audio_processing_kyutai_speech_to_text import (
    KyutaiSpeechToTextAudioProcessor,
)
from transformers.models.kyutai_speech_to_text.feature_extraction_kyutai_speech_to_text import (
    KyutaiSpeechToTextFeatureExtractor,
)
from transformers.models.lasr.audio_processing_lasr import LasrAudioProcessor
from transformers.models.lasr.feature_extraction_lasr import LasrFeatureExtractor
from transformers.models.musicgen_melody.audio_processing_musicgen_melody import MusicgenMelodyAudioProcessor
from transformers.models.musicgen_melody.feature_extraction_musicgen_melody import MusicgenMelodyFeatureExtractor
from transformers.models.parakeet.audio_processing_parakeet import ParakeetAudioProcessor
from transformers.models.parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor
from transformers.models.phi4_multimodal.audio_processing_phi4_multimodal import Phi4MultimodalAudioProcessor
from transformers.models.phi4_multimodal.feature_extraction_phi4_multimodal import Phi4MultimodalFeatureExtractor
from transformers.models.pop2piano.audio_processing_pop2piano import Pop2PianoAudioProcessor
from transformers.models.pop2piano.feature_extraction_pop2piano import Pop2PianoFeatureExtractor
from transformers.models.seamless_m4t.audio_processing_seamless_m4t import SeamlessM4tAudioProcessor
from transformers.models.seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
from transformers.models.speech_to_text.audio_processing_speech_to_text import SpeechToTextAudioProcessor
from transformers.models.speech_to_text.feature_extraction_speech_to_text import Speech2TextFeatureExtractor
from transformers.models.speecht5.audio_processing_speecht5 import SpeechT5AudioProcessor
from transformers.models.speecht5.feature_extraction_speecht5 import SpeechT5FeatureExtractor
from transformers.models.univnet.audio_processing_univnet import UnivNetAudioProcessor
from transformers.models.univnet.feature_extraction_univnet import UnivNetFeatureExtractor
from transformers.models.vibevoice_acoustic_tokenizer.audio_processing_vibevoice_acoustic_tokenizer import (
    VibevoiceAcousticTokenizerAudioProcessor,
)
from transformers.models.vibevoice_acoustic_tokenizer.feature_extraction_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerFeatureExtractor,
)
from transformers.models.voxtral_realtime.audio_processing_voxtral_realtime import VoxtralRealtimeAudioProcessor
from transformers.models.voxtral_realtime.feature_extraction_voxtral_realtime import VoxtralRealtimeFeatureExtractor
from transformers.models.wav2vec2.audio_processing_wav2vec2 import Wav2Vec2AudioProcessor
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from transformers.models.whisper.audio_processing_whisper import WhisperAudioProcessor
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor


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

    for fe_key in fe_output.keys():
        if fe_key == "attention_mask" or fe_key == "padding_mask" or fe_key == "input_features_mask":
            continue
        ap_key = fe_to_ap_key_map.get(fe_key, fe_key)
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
        assert torch.equal(fe_tensor, ap_tensor), (
            f"Value mismatch for key '{fe_key}' (ap key '{ap_key}'): max abs diff = {(fe_tensor - ap_tensor).abs().max().item():.6e}"
        )
