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

import ast
import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

import transformers
from transformers import AutoProcessor, PreTrainedTokenizerFast
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.utils import is_torch_available, is_torchaudio_available


pytestmark = pytest.mark.skipif(
    not (is_torch_available() and is_torchaudio_available()),
    reason="Audio processor tests require PyTorch and torchaudio",
)


@pytest.fixture
def tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<blank>": 1}, unk_token="<unk>")),
        unk_token="<unk>",
        pad_token="<blank>",
        extra_special_tokens={
            name: f"<{name}>"
            for name in (
                "image_token",
                "audio_token",
                "video_token",
                "vision_bos_token",
                "vision_eos_token",
                "audio_bos_token",
                "audio_eos_token",
            )
        },
    )


@pytest.mark.parametrize(
    "processor_name,audio_name",
    [
        ("WhisperProcessor", "WhisperAudioProcessor"),
        ("Wav2Vec2Processor", "Wav2Vec2AudioProcessor"),
        ("Wav2Vec2BertProcessor", "SeamlessM4tAudioProcessor"),
        ("SeamlessM4TProcessor", "SeamlessM4tAudioProcessor"),
        ("Speech2TextProcessor", "SpeechToTextAudioProcessor"),
        ("MusicgenProcessor", "EncodecAudioProcessor"),
        ("ClapProcessor", "ClapAudioProcessor"),
        ("Qwen2AudioProcessor", "WhisperAudioProcessor"),
        ("PeAudioProcessor", "PeAudioAudioProcessor"),
        ("NemotronAsrStreamingProcessor", "NemotronAsrStreamingAudioProcessor"),
        ("Nemotron3_5AsrProcessor", "NemotronAsrStreamingAudioProcessor"),
    ],
)
def test_audio_component_migration(processor_name, audio_name, tokenizer, tmp_path):
    processor_class = getattr(transformers, processor_name)
    audio_class = getattr(transformers, audio_name)
    audio_processor = audio_class()
    processor = processor_class(audio_processor=audio_processor, tokenizer=tokenizer)
    assert "audio_processor" in inspect.signature(processor_class).parameters
    assert processor.get_attributes() == ["audio_processor", "tokenizer"]
    with pytest.warns(FutureWarning, match="audio_processor"):
        legacy = processor_class(feature_extractor=audio_processor, tokenizer=tokenizer)
    assert legacy.audio_processor is audio_processor
    with pytest.warns(FutureWarning, match="audio_processor"):
        assert processor.feature_extractor is audio_processor
    replacement = audio_class()
    with pytest.warns(FutureWarning, match="audio_processor"):
        processor.feature_extractor = replacement
    assert processor.audio_processor is replacement
    with pytest.warns(FutureWarning, match="audio_processor"):
        canonical = processor_class(
            audio_processor=audio_processor, feature_extractor=replacement, tokenizer=tokenizer
        )
    assert canonical.audio_processor is audio_processor
    assert processor_class(audio_processor, tokenizer).audio_processor is audio_processor

    for legacy_config in (False, True):
        processor.save_pretrained(tmp_path)
        path = tmp_path / "processor_config.json"
        config = json.loads(path.read_text())
        assert "audio_processor" in config
        assert "feature_extractor" not in config
        if legacy_config:
            config["feature_extractor"] = config.pop("audio_processor")
            path.write_text(json.dumps(config))
        with warnings.catch_warnings(record=True) as caught:
            restored = AutoProcessor.from_pretrained(tmp_path)
        assert not any("AutoFeatureExtractor" in str(w.message) for w in caught)
        assert restored.audio_processor.to_dict() == processor.audio_processor.to_dict()
        restored.save_pretrained(tmp_path)
        config = json.loads(path.read_text())
        assert "audio_processor" in config and "feature_extractor" not in config
        config["feature_extractor"] = {**config["audio_processor"], "sampling_rate": 8000}
        path.write_text(json.dumps(config))
        restored = AutoProcessor.from_pretrained(tmp_path)
        assert restored.audio_processor.sampling_rate == processor.audio_processor.sampling_rate


def test_image_and_audio_components(tokenizer, tmp_path):
    processor = transformers.Qwen2_5OmniProcessor(
        image_processor=transformers.Qwen2VLImageProcessor(),
        video_processor=transformers.Qwen2VLVideoProcessor(),
        audio_processor=transformers.WhisperAudioProcessor(),
        tokenizer=tokenizer,
    )
    processor.save_pretrained(tmp_path)
    restored = AutoProcessor.from_pretrained(tmp_path)
    assert restored.get_attributes() == ["image_processor", "video_processor", "audio_processor", "tokenizer"]
    assert restored.audio_processor.to_dict() == processor.audio_processor.to_dict()
    assert restored.image_processor.to_dict() == processor.image_processor.to_dict()
    with warnings.catch_warnings(record=True) as caught:
        kwargs = restored._merge_kwargs(ProcessingKwargs, audio_kwargs={"return_padding_mask": False})
    assert not caught
    assert kwargs["audio_kwargs"]["return_padding_mask"] is False


def test_whisper_call_matches_legacy_constructor(tokenizer):
    audio_processor = transformers.WhisperAudioProcessor()
    processor = transformers.WhisperProcessor(audio_processor=audio_processor, tokenizer=tokenizer)
    with pytest.warns(FutureWarning):
        legacy = transformers.WhisperProcessor(feature_extractor=audio_processor, tokenizer=tokenizer)
    audio = np.zeros(16000, dtype=np.float32)
    actual = processor(audio, sampling_rate=16000, return_tensors="np")
    expected = legacy(audio, sampling_rate=16000, return_tensors="np")
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_legacy_non_audio_component_is_not_renamed(tokenizer):
    processor = transformers.MarkupLMProcessor(
        feature_extractor=transformers.MarkupLMFeatureExtractor(), tokenizer=tokenizer
    )
    with warnings.catch_warnings(record=True) as caught:
        assert processor.feature_extractor is not None
    assert not caught
    assert processor.get_attributes() == ["feature_extractor", "tokenizer"]
    assert "feature_extractor" in processor.to_dict()
    assert "audio_processor" not in processor.to_dict()


def test_inherited_constructor_and_class_attribute_components(tokenizer):
    class InheritedProcessor(transformers.WhisperProcessor):
        pass

    class LegacyDeclarationProcessor(ProcessorMixin):
        audio_processor_class = "WhisperAudioProcessor"
        tokenizer_class = "AutoTokenizer"

    audio_processor = transformers.WhisperAudioProcessor()
    for cls in (InheritedProcessor, LegacyDeclarationProcessor):
        with pytest.warns(FutureWarning, match="audio_processor"):
            processor = cls(feature_extractor=audio_processor, tokenizer=tokenizer)
        assert processor.audio_processor is audio_processor
        assert "feature_extractor" not in processor.to_dict()


def test_all_audio_processor_signatures_are_migrated():
    models = Path(__file__).resolve().parents[2] / "src/transformers/models"
    for path in models.glob("*/processing_*.py"):
        if path.parent.name in {"auto", "markuplm"}:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
                assert "feature_extractor" not in names, str(path)
            if isinstance(node, ast.Attribute):
                assert node.attr != "feature_extractor", str(path)
