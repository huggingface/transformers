# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from transformers import (
    TokenizersBackend,
    ZacFeatureExtractor,
    Zonos3MiniFeatureExtractor,
    Zonos3MiniProcessor,
)
from transformers.testing_utils import require_torch, require_torchaudio, slow
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

    from transformers import ZacModel


# TODO: move to final Zyphra repo
ZONOS3_MINI_CHECKPOINT = os.environ.get("ZONOS3_MINI_TEST_DIR", "Zyphra/ZONOS3-Mini-transformers")

NUM_CODEBOOKS = 10
AUDIO_PAD, TEXT_PAD, AUDIO_EOS = 4097, 8196, 4096
SILENCE_FRAME = [1918, 746, 3544, 234, 1267, 2914, 3944, 2849, 326, 1337]
CHAT_TEMPLATE = """
{%- set training = messages | length == 2 -%}
{%- if messages | length > 2 or messages[0]['role'] != 'user' or (training and messages[1]['role'] != 'assistant') -%}
    {{- raise_exception('ZONOS3-Mini takes one `user` message per conversation, optionally followed by the `assistant` target audio to train on.') -}}
{%- endif -%}
{%- set content = messages[0]['content'] -%}
{%- if content is string -%}
    {%- set texts = [content] -%}
    {%- set num_audio = 0 -%}
{%- else -%}
    {%- set texts = content | selectattr('type', 'equalto', 'text') | map(attribute='text') | list -%}
    {%- set num_audio = content | selectattr('type', 'equalto', 'audio') | list | length -%}
{%- endif -%}
{%- if texts | length != 1 or num_audio > 1 -%}
    {{- raise_exception('The `user` turn holds one `text` item and at most one `audio` item (the voice).') -}}
{%- endif -%}
{{- texts[0] -}}
"""


def tiny_tokenizer() -> "TokenizersBackend":
    """A word-level tokenizer with the ZONOS3-Mini special tokens and its BOS / EOS template."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    words = ["hello", "world", "!", ".", "the", "weather", "is", "lovely", "a", "b", "c", "d", "e", "f"]
    vocab = {"<|zonos_pad|>": 0, "<|zonos_unk|>": 1, "<|zonos_bos|>": 2, "<|zonos_eos|>": 3}
    vocab.update({word: i + 4 for i, word in enumerate(words)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<|zonos_unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|zonos_bos|> $A <|zonos_eos|>", special_tokens=[("<|zonos_bos|>", 2), ("<|zonos_eos|>", 3)]
    )
    return TokenizersBackend(
        tokenizer_object=tokenizer,
        bos_token="<|zonos_bos|>",
        eos_token="<|zonos_eos|>",
        pad_token="<|zonos_pad|>",
        unk_token="<|zonos_unk|>",
        padding_side="left",
    )


@require_torch
@require_torchaudio
class Zonos3MiniProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Zonos3MiniProcessor

    @classmethod
    def _setup_tokenizer(cls):
        return tiny_tokenizer()

    @classmethod
    def _setup_feature_extractor(cls):
        return Zonos3MiniFeatureExtractor()

    @classmethod
    def _setup_codec_feature_extractor(cls):
        return ZacFeatureExtractor(lane_frames=4, lane_context_frames=2, hop_length=735)

    @classmethod
    def prepare_processor_dict(cls):
        # a tiny random ZAC-1 with the real sampling rates, saved so the processor's pointer to it resolves
        if not hasattr(cls, "_audio_tokenizer_dir"):
            from ..zac.test_modeling_zac import ZacModelTester

            cls._audio_tokenizer_dir = tempfile.mkdtemp()
            ZacModel(ZacModelTester(None).get_config()).save_pretrained(cls._audio_tokenizer_dir)
        return {
            "audio_tokenizer": ZacModel.from_pretrained(cls._audio_tokenizer_dir),
            "chat_template": CHAT_TEMPLATE.strip() + "\n",
            "languages": {"en_us": "en", "de": "de"},
            "condition_names": ["loudness", "fmax", "speaking_rate"],
            "condition_edges": {"loudness": [-30.0, -20.0], "fmax": [8000.0], "speaking_rate": [10.0, 15.0]},
            "condition_bucket_counts": [10, 10, 30],
            "silence_frame": SILENCE_FRAME,
        }

    def get_component(self, attribute, **kwargs):
        # the codec feature extractor is saved inside `processor_config.json`, not as its own preprocessor config
        if attribute == "codec_feature_extractor":
            return self.get_processor().codec_feature_extractor
        return super().get_component(attribute, **kwargs)

    def prepare_audio_inputs(self, batch_size: int | None = None):
        # voice clips of 1 s at 44.1 kHz (ZAC-1 rejects clips shorter than 0.5 s)
        audio = [np.random.default_rng(0).standard_normal(44100).astype(np.float32) * 0.1]
        return audio if batch_size is None else audio * batch_size

    def setUp(self):
        super().setUp()
        if self._testMethodName.startswith(("test_apply_chat_template", "test_tokenizer", "test_processor_text")):
            if self._testMethodName not in ("test_apply_chat_template_text_rows",):
                self.skipTest(
                    reason="ZONOS3-Mini prompts are rows of 10 audio columns and a text column built by the processor, "
                    "from one `user` message; the common text-processor tests assume token sequences."
                )
        if self._testMethodName in (
            "test_subprocessor_defaults_0_text",
            "test_kwargs_overrides_default_subprocessor_kwargs_2_audio",
            "test_subprocessor_defaults_preserved_by_kwargs_2_audio",
            "test_structured_kwargs_nested_from_dict_2_audio",
            "test_unstructured_kwargs_2_audio",
            "test_unstructured_kwargs_batched_2_audio",
        ):
            self.skipTest(
                reason="These tests compare `input_ids` with the tokenizer's output padded to `max_length`; ZONOS3-Mini's "
                "`input_ids` are prompt rows whose text column is one of 11."
            )
        if self._testMethodName in ("test_processor_with_multiple_inputs", "test_chat_template_audio_sampling_rate"):
            self.skipTest(
                reason="In the default reference mode, the voice clip is encoded into codec rows of `input_ids`, so no "
                "feature tensor is returned (`test_voice_clip_modes` covers both modes)."
            )

    @unittest.skip(
        reason="The codec feature extractor is saved inside `processor_config.json`, so it cannot be loaded on its own "
        "through an auto class; the round trip is covered by `test_processor_from_and_save_pretrained`."
    )
    def test_processor_from_and_save_pretrained_as_nested_dict(self):
        pass

    def test_processor_to_json_string(self):
        # the audio tokenizer is saved as a pointer to its checkpoint, not as a value
        processor = self.get_processor()
        self.assertEqual(
            json.loads(processor.to_json_string())["audio_tokenizer"],
            {
                "audio_tokenizer_class": "ZacModel",
                "audio_tokenizer_name_or_path": processor.audio_tokenizer.name_or_path,
            },
        )

    def test_model_input_names(self):
        # every output is a model input, but the speaker, guidance and label entries are only returned when requested
        processor = self.get_processor()
        inputs = processor(text="hello world", return_tensors="pt")
        self.assertSetEqual(set(inputs), {"input_ids", "attention_mask", "condition_ids"})
        self.assertLessEqual(set(inputs), set(processor.model_input_names))

    def test_voice_clip_modes(self):
        """A voice clip becomes reference rows (default) or speaker features with a speaker row (embedding mode)."""
        processor = self.get_processor()
        clip = self.prepare_audio_inputs()
        reference = processor(text="hello world", audio=clip, sampling_rate=44100, return_tensors="pt")
        plain = processor(text="hello world", return_tensors="pt")
        self.assertSetEqual(set(reference), set(plain))
        self.assertGreater(reference["input_ids"].shape[1], plain["input_ids"].shape[1])
        embedding = processor(
            text="hello world", audio=clip, sampling_rate=44100, speaker_mode="embedding", return_tensors="pt"
        )
        self.assertIn("speaker_input_features", embedding)
        self.assertEqual(embedding["speaker_token_mask"][0].nonzero().flatten().tolist(), [0])
        self.assertLessEqual(set(embedding), set(processor.model_input_names))

    def _reference_codes(self, num_frames=5):
        return torch.arange(num_frames * NUM_CODEBOOKS).reshape(num_frames, NUM_CODEBOOKS) % 4096

    def test_reference_prompt_layout(self):
        processor = self.get_processor()
        codes = self._reference_codes()
        inputs = processor(text="hello world", speaker_reference_codes=[codes], return_tensors="pt")
        rows = inputs["input_ids"][0]
        prefix = len(codes) + NUM_CODEBOOKS - 1
        # reference prefix: the delayed codes (codebook j shifted by j rows), text column padded
        for j in range(NUM_CODEBOOKS):
            torch.testing.assert_close(rows[j : j + len(codes), j], codes[:, j])
        self.assertTrue((rows[:prefix, -1] == TEXT_PAD).all())
        # 3 condition rows, all unknown: `offset_i + count_i`
        self.assertEqual(inputs["condition_ids"][0, prefix : prefix + 3].tolist(), [10, 21, 52])
        self.assertTrue((rows[prefix : prefix + 3, :NUM_CODEBOOKS] == AUDIO_PAD).all())
        # BOS hello world EOS, then 3 silence rows
        self.assertEqual(rows[prefix + 3 : prefix + 7, -1].tolist(), [2, 4, 5, 3])
        self.assertEqual(rows[prefix + 7 :, 0].tolist(), [SILENCE_FRAME[0]] * 3)
        self.assertEqual(len(rows), prefix + 3 + 4 + 3)

    def test_embedding_prompt_layout(self):
        processor = self.get_processor()
        embedding = torch.randn(1, 192)
        inputs = processor(
            text="hello", speaker_embeddings=embedding, condition_buckets=[1, None, 2], return_tensors="pt"
        )
        self.assertEqual(inputs["speaker_token_mask"][0].nonzero().flatten().tolist(), [0])
        torch.testing.assert_close(inputs["speaker_embeddings"], embedding)
        self.assertEqual(inputs["condition_ids"][0, 1:4].tolist(), [1, 21, 24])

    def test_condition_values_are_bucketed(self):
        processor = self.get_processor()
        inputs = processor(
            text="hello", condition_values={"loudness": -25.0, "speaking_rate": 20.0}, return_tensors="pt"
        )
        self.assertEqual(inputs["condition_ids"][0, :3].tolist(), [0 + 1, 11 + 10, 22 + 2])

    def test_left_padding(self):
        processor = self.get_processor()
        inputs = processor(text=["hello", "hello world ! the weather"], return_tensors="pt")
        self.assertEqual(inputs["attention_mask"].sum(-1).tolist(), [3 + 3 + 3, 3 + 7 + 3])
        padding = inputs["input_ids"][0, :4]
        self.assertTrue((padding[:, :NUM_CODEBOOKS] == AUDIO_PAD).all() and (padding[:, -1] == TEXT_PAD).all())
        self.assertTrue((inputs["condition_ids"][0, :4] == -1).all())

    def test_speaker_guidance_twin(self):
        processor = self.get_processor()
        codes = self._reference_codes()
        inputs = processor(text="hello", speaker_reference_codes=[codes], speaker_guidance=True, return_tensors="pt")
        self.assertEqual(inputs["uncond_input_ids"].shape, inputs["input_ids"].shape)
        # the twin is the prompt without its speaker part, left-padded to the same length
        prefix = len(codes) + NUM_CODEBOOKS - 1
        num_rows = inputs["uncond_attention_mask"][0].sum()
        torch.testing.assert_close(inputs["uncond_input_ids"][0, -num_rows:], inputs["input_ids"][0, prefix:])
        with self.assertRaises(ValueError):
            processor(text="hello", speaker_guidance=True, return_tensors="pt")

    def test_rejects_invalid_inputs(self):
        processor = self.get_processor()
        codes = self._reference_codes()
        with self.assertRaises(ValueError):
            processor(text="hello", speaker_reference_codes=[codes], speaker_embeddings=torch.randn(1, 192))
        with self.assertRaises(ValueError):
            processor(text="hello", language="xx")
        with self.assertRaises(ValueError):
            processor(text="hello", speaker_reference_codes=[codes + 4096])
        with self.assertRaises(ValueError):
            processor(text="hello", speaker_mode="embedding")
        with self.assertRaises(ValueError):
            processor(text="hello \ue000")

    def test_decode_take(self):
        """`decode` decodes the frames from the last `overlap_frames` silence frames up to the end-of-audio."""
        processor = self.get_processor()
        decoded = []

        class Codec:
            device = torch.device("cpu")
            config = type("Config", (), {"hop_length": 2})

            @staticmethod
            def decode(audio_codes):
                decoded.append(audio_codes)
                return type("Output", (), {"audio_values": torch.zeros(1, 1, audio_codes.shape[-1] * 2)})

        processor.audio_tokenizer = Codec()
        inputs = processor(text="hello", return_tensors="pt")
        frames = torch.arange(6 * NUM_CODEBOOKS).reshape(6, NUM_CODEBOOKS)
        frames[-1] = AUDIO_EOS
        # prompt silence rows (3), then the delayed generated frames
        generated = processor._delayed_rows(frames, len(frames) + NUM_CODEBOOKS - 1)
        silence = inputs["input_ids"][0, -3:]
        sequences = torch.cat([inputs["input_ids"][0], generated])
        audio = processor.decode(sequences)[0]
        # the 3 silence frames, then 5 frames before the end-of-audio frame
        self.assertEqual(decoded[0].shape, (1, NUM_CODEBOOKS, 3 + 5))
        torch.testing.assert_close(decoded[0][0, :, 3:].T, frames[:5])
        self.assertEqual(decoded[0][0, 0, :3].tolist(), silence[:, 0].tolist())
        self.assertEqual(audio.shape, (2 * (3 + 5),))
        # nothing before the end-of-audio: an empty waveform
        frames[0] = AUDIO_EOS
        generated = processor._delayed_rows(frames, len(frames) + NUM_CODEBOOKS - 1)
        audio = processor.decode(torch.cat([inputs["input_ids"][0], generated]))[0]
        self.assertEqual(audio.numel(), 0)

    def test_apply_chat_template_text_rows(self):
        processor = self.get_processor()
        conversation = [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}]
        inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True, return_tensors="pt")
        self.assertEqual(inputs["input_ids"].shape, (1, 3 + 4 + 3, NUM_CODEBOOKS + 1))
        with self.assertRaises(Exception):
            processor.apply_chat_template(conversation * 2, tokenize=True)

    @slow
    def test_prompt_rows_match_reference(self):
        """Reproducer: zonos3_repro/reproducer.py `prompt` (the remote code of Zyphra/ZONOS3-Mini-hf@e39d30d)."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        with open(
            Path(__file__).parent.parent.parent / "fixtures/zonos3_mini/expected_prompt.json", encoding="utf-8"
        ) as f:
            expected = json.load(f)
        processor = Zonos3MiniProcessor.from_pretrained(ZONOS3_MINI_CHECKPOINT)
        voice_path = (
            os.path.join(ZONOS3_MINI_CHECKPOINT, "voices", f"{expected['voice']}.safetensors")
            if os.path.isdir(ZONOS3_MINI_CHECKPOINT)
            else hf_hub_download(ZONOS3_MINI_CHECKPOINT, f"voices/{expected['voice']}.safetensors")
        )
        voice = load_file(voice_path)
        reference = processor(
            text=expected["text"],
            speaker_reference_codes=[voice["codes"]],
            text_normalization=False,
            return_tensors="pt",
        )
        torch.testing.assert_close(reference["input_ids"][0], torch.tensor(expected["reference"]["input_ids"]))
        embedding = processor(
            text=expected["text"],
            speaker_embeddings=voice["embedding"][None],
            text_normalization=False,
            return_tensors="pt",
        )
        torch.testing.assert_close(embedding["input_ids"][0], torch.tensor(expected["embedding"]["input_ids"]))
        self.assertEqual(embedding["speaker_token_mask"][0].nonzero().flatten().tolist(), [0])
