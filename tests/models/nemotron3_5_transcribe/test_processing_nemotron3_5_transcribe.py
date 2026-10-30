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

import unittest

import numpy as np
from parameterized import parameterized

from transformers import Nemotron3_5TranscribeProcessor, PreTrainedTokenizerFast
from transformers.testing_utils import (
    require_librosa,
    require_scipy,
    require_tokenizers,
    require_torch,
    require_torchaudio,
)
from transformers.utils import is_tokenizers_available, is_torch_available

from ...test_processing_common import MODALITY_INPUT_DATA, ProcessorTesterMixin


if is_torch_available():
    import torch

if is_tokenizers_available():
    from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers


@require_torch
class Nemotron3_5TranscribeProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Nemotron3_5TranscribeProcessor
    model_id = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        # TODO: @eustlb, drop the revision once NVIDIA merges the Transformers files (hub PR #5) into main
        return super()._setup_from_pretrained(model_id, revision="refs/pr/5", **kwargs)

    @require_librosa
    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        if return_tensors == "np":
            self.skipTest("Nemotron3_5Transcribe only supports PyTorch tensors")
        self._test_apply_chat_template(
            "audio", batch_size, return_tensors, "audio_input_name", "feature_extractor", MODALITY_INPUT_DATA["audio"]
        )

    def test_transcription_request_prompt(self):
        """Same prompt as the original `nemotron3p5` prompt formatter: empty system turn and no thinking."""
        processor = self.get_processor()
        inputs = processor.apply_transcription_request(np.zeros(16000, dtype=np.float32), prompt="Transcribe.")
        prompt = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        audio_tokens = processor.audio_token * 13
        expected = (
            "<|im_start|>system\n<|im_end|>\n"
            f"<|im_start|>user\nTranscribe. {audio_tokens}<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>"
        )
        self.assertEqual(prompt, expected)

        inputs = processor.apply_transcription_request(np.zeros(16000, dtype=np.float32))
        prompt = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        expected = expected.replace("Transcribe.", processor.default_transcription_prompt)
        self.assertEqual(prompt, expected)

    def test_audio_tokens_and_padding(self):
        """
        One audio token per started group of `subsampling_factor` spectrogram frames of 160 samples; the spectrogram is
        padded to whole groups, the mask keeps the valid frames, and the text is left-padded for generation.
        """
        processor = self.get_processor()
        num_samples = [12800, 12959, 12960, 32037]
        inputs = processor.apply_transcription_request([np.zeros(n, dtype=np.float32) for n in num_samples])
        self.assertEqual(inputs["input_features"].shape[1] % processor.subsampling_factor, 0)
        self.assertEqual(inputs["input_features_mask"].sum(-1).tolist(), [80, 80, 81, 200])
        self.assertEqual((inputs["input_ids"] == processor.audio_token_id).sum(-1).tolist(), [10, 10, 11, 25])
        self.assertEqual(inputs["attention_mask"][:, 0].tolist(), [0, 0, 0, 1])

    @unittest.skip(
        reason="Nemotron3_5TranscribeProcessor always returns PyTorch tensors, but this test expects lists."
    )
    def test_apply_chat_template_tool_calls_no_content(self):
        pass

    @unittest.skip(reason="Cannot change the default of output_labels in __call__ without breaking BC.")
    def test_call_arguments_default_none(self):
        pass

    def test_content_parts_render_as_training_prompt(self):
        """As NeMo's training prompt: the text parts joined by a space, then one `<|audio|>` per audio part."""
        processor = self.get_processor()
        audio, text = {"type": "audio", "path": "unused.wav"}, {"type": "text", "text": "Transcribe."}
        for content in ([text, audio], [audio, text]):
            conversation = [{"role": "user", "content": content}]
            prompt = processor.apply_chat_template(conversation, tokenize=False, enable_thinking=False)
            self.assertIn("<|im_start|>user\nTranscribe. <|audio|><|im_end|>", prompt)

    def test_extract_speaker_dict(self):
        processor = self.get_processor()
        text = f"<spk:0> Hi, how are you? <spk:1> Fine, thanks.{processor.tokenizer.eos_token}"
        expected = [{"Speaker": 0, "Content": "Hi, how are you?"}, {"Speaker": 1, "Content": "Fine, thanks."}]
        self.assertEqual(processor.extract_speaker_dict(text), expected)

        # a batch, decoded with the speaker tags (special tokens) and the generation padding after end-of-text
        generated = [
            processor.tokenizer(text, add_special_tokens=False)["input_ids"],
            processor.tokenizer(f"just words{processor.tokenizer.eos_token}", add_special_tokens=False)["input_ids"],
        ]
        generated[1] += [processor.tokenizer.pad_token_id] * (len(generated[0]) - len(generated[1]))
        decoded = processor.decode(generated, skip_special_tokens=False)
        self.assertEqual(
            processor.extract_speaker_dict(decoded), [expected, [{"Speaker": None, "Content": "just words"}]]
        )


# pieces of the tiny alignment vocabulary, the CTC blank is the next class
PIECES = [
    "<unk>",
    "<ev:laughter>",
    "▁",
    "a",
    "b",
    "▁a",
    "▁b",
    "你",
    "好",
    "▁是",
]  # "▁是" is added: "是" is not a piece
BLANK = len(PIECES)


@require_torch
@require_scipy
@require_torchaudio
@require_tokenizers
class Nemotron3_5TranscribeWordTimestampAlignmentTest(unittest.TestCase):
    """`align_word_timestamps` on hand-built head logits, with a 9-piece alignment vocabulary built like the real one."""

    @classmethod
    def setUpClass(cls):
        # TODO: @eustlb, drop the revision once NVIDIA merges the Transformers files (hub PR #5) into main
        processor = Nemotron3_5TranscribeProcessor.from_pretrained(
            "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B", revision="refs/pr/5"
        )
        tokenizer = Tokenizer(
            models.BPE(
                vocab={piece: i for i, piece in enumerate(PIECES[:-1])},
                merges=[("▁", "a"), ("▁", "b")],
                unk_token="<unk>",
                fuse_unk=True,
            )
        )
        tokenizer.normalizer = normalizers.Replace(" ", "▁")
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first", split=True)
        tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="first", split=True)
        tokenizer.add_tokens([AddedToken("<ev:laughter>", normalized=False), AddedToken("▁是", normalized=True)])
        alignment_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
        cls.processor = Nemotron3_5TranscribeProcessor(
            processor.feature_extractor, processor.tokenizer, alignment_tokenizer
        )

    @staticmethod
    def make_inputs(frames, speaker_activity=None, num_spectrogram_frames=None):
        """
        `align_word_timestamps` tensors: logits of 10 on the classes of each frame (`frames[t]`: a piece, a tuple of
        pieces, or `None` for the blank), 8 spectrogram frames per frame, valid up to `num_spectrogram_frames` (all by
        default), and diarization logits of +/-10 following the speaker activities `speaker_activity`
        `(frames, speakers)` of 0 or 1.
        """
        logits = torch.zeros(1, len(frames), BLANK + 1)
        for t, pieces in enumerate(frames):
            pieces = (pieces,) if not isinstance(pieces, tuple) else pieces
            for piece in pieces:
                logits[0, t, BLANK if piece is None else PIECES.index(piece)] = 10.0
        speaker_logits = None
        if speaker_activity is not None:
            speaker_activity = torch.tensor(speaker_activity, dtype=torch.float32)[None]
            speaker_logits = (speaker_activity * 20 - 10).repeat_interleave(8, dim=1)
        return {
            "word_timestamp_logits": logits,
            "input_features_mask": (torch.arange(8 * len(frames)) < (num_spectrogram_frames or 8 * len(frames)))[
                None
            ].long(),
            "speaker_logits": speaker_logits,
        }

    def assertWordsAlmostEqual(self, chunks, expected):
        self.assertEqual([(c["text"], c["speaker"]) for c in chunks], [(w, s) for w, _, _, s in expected])
        for chunk, (_, start, end, _) in zip(chunks, expected):
            self.assertAlmostEqual(chunk["timestamp"][0], start, places=9)
            self.assertAlmostEqual(chunk["timestamp"][1], end, places=9)

    def test_known_path(self):
        """The single best path gives the word frames; without diarization a word ends at the next word's onset."""
        frames = [None, None, "▁a", "▁a", None, "▁b", "▁b", None, None, None]
        inputs = self.make_inputs(frames)
        result = self.processor.align_word_timestamps("a b<|im_end|>", **inputs)
        step = 0.8 / 10
        self.assertEqual(
            result["chunks"],
            [
                {"text": "a", "timestamp": (2 * step, 5 * step), "speaker": 0},
                {"text": "b", "timestamp": (5 * step, 7 * step), "speaker": 0},
            ],
        )
        self.assertEqual(result["diarization"], [])
        self.assertEqual(result["speaker_tag_to_diarization_speaker"], {0: None})
        self.assertEqual(result["failed_speakers"], [])
        # a list of transcripts gives a list of results
        self.assertEqual(self.processor.align_word_timestamps(["a b<|im_end|>"], **inputs), [result])

    def test_frame_duration_from_mask(self):
        """The frame duration is `valid spectrogram frames * hop_length / sampling_rate / num_frames`."""
        inputs = self.make_inputs([None, None, None, "▁a", None])
        result = self.processor.align_word_timestamps("a", **inputs)
        self.assertAlmostEqual(result["chunks"][0]["timestamp"][0], 3 * 0.08)
        # 35 valid spectrogram frames: 0.35 s over 5 frames, the last one partial
        inputs = self.make_inputs([None, None, None, "▁a", None], num_spectrogram_frames=35)
        result = self.processor.align_word_timestamps("a", **inputs)
        self.assertAlmostEqual(result["chunks"][0]["timestamp"][0], 3 * 0.07)

    def test_upsampled_head_frames(self):
        """Head logits with 2 frames per encoder frame, read from their shape, give 40 ms frames."""
        inputs = self.make_inputs([None, None, None, "▁a", None, None])
        inputs["input_features_mask"] = inputs["input_features_mask"][:, :24]  # 3 encoder frames, 6 head frames
        result = self.processor.align_word_timestamps("a", **inputs)
        self.assertAlmostEqual(result["chunks"][0]["timestamp"][0], 3 * 0.04)
        inputs["input_features_mask"] = torch.ones(1, 32, dtype=torch.long)  # 4 encoder frames
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            self.processor.align_word_timestamps("a", **inputs)

    def test_overlapping_speakers(self):
        """
        Each speaker stream is aligned on the frames where its diarization speaker is active, so overlapping speech
        gets overlapping words; tag 0 speaks first but is diarization speaker 1. The first word of a diarization
        segment then moves its onset to the segment onset and ends with the segment.
        """
        frames = [None, None, "▁a", "▁a", ("▁a", "▁b"), ("▁a", "▁b"), "▁b", "▁b", None, None]
        early, late = [1.0] * 6 + [0.0] * 4, [0.0] * 4 + [1.0] * 6
        inputs = self.make_inputs(frames, speaker_activity=list(zip(late, early)))
        result = self.processor.align_word_timestamps("<spk:0> a <spk:1> b", **inputs)
        self.assertEqual(result["speaker_tag_to_diarization_speaker"], {0: 1, 1: 0})
        self.assertEqual(
            result["diarization"],
            [{"speaker": 1, "timestamp": (0.0, 0.48)}, {"speaker": 0, "timestamp": (0.32, 0.8)}],
        )
        # before refinement: "a" on frames 2-5 (0.16-0.48 s), "b" on frames 4-7 (0.32-0.64 s)
        self.assertWordsAlmostEqual(result["chunks"], [("a", 0.0, 0.48, 0), ("b", 0.32, 0.8, 1)])
        self.assertEqual(result["failed_speakers"], [])

    def test_failed_speaker(self):
        """A stream that has no path on the frames of its diarization speaker is reported, its words are dropped."""
        frames = [None, None, "▁a", "▁a", None, None, None, None, "▁b", None, "▁b", None]
        everywhere, end = [1.0] * 12, [0.0] * 8 + [1.0] * 3 + [0.0]
        inputs = self.make_inputs(frames, speaker_activity=list(zip(everywhere, end)))
        # "b b b" needs 5 frames (a blank between repeated tokens), its speaker is active on 3
        result = self.processor.align_word_timestamps("<spk:0> a <spk:1> b b b", **inputs)
        self.assertEqual(result["speaker_tag_to_diarization_speaker"], {0: 0, 1: 1})
        self.assertEqual(result["failed_speakers"], [1])
        self.assertEqual([chunk["text"] for chunk in result["chunks"]], ["a"])

    def test_character_units(self):
        """Chinese is aligned character by character; another language keeps the word."""
        inputs = self.make_inputs([None, "你", None, "好", None])
        result = self.processor.align_word_timestamps("你好", **inputs)
        self.assertWordsAlmostEqual(result["chunks"], [("你", 0.08, 0.24, 0), ("好", 0.24, 0.32, 0)])

        # a character without its own piece is its pieces without the bare "▁": "是" is "▁是"
        inputs = self.make_inputs([None, "▁是", None])
        result = self.processor.align_word_timestamps("是", **inputs)
        self.assertWordsAlmostEqual(result["chunks"], [("是", 0.08, 0.16, 0)])

        # as a word, "你好" is "▁", "你", "好"; the language is per transcript
        inputs = self.make_inputs([None, "▁", "你", "好", None])
        result = self.processor.align_word_timestamps("你好", **inputs, language="fr")
        self.assertWordsAlmostEqual(result["chunks"], [("你好", 0.08, 0.32, 0)])
        by_character = self.make_inputs([None, "你", None, "好", None])
        batch = {k: torch.cat([by_character[k], inputs[k]]) for k in ("word_timestamp_logits", "input_features_mask")}
        results = self.processor.align_word_timestamps(["你好", "你好"], **batch, language=[None, "fr"])
        self.assertWordsAlmostEqual(results[0]["chunks"], [("你", 0.08, 0.24, 0), ("好", 0.24, 0.32, 0)])
        self.assertWordsAlmostEqual(results[1]["chunks"], [("你好", 0.08, 0.32, 0)])

    def test_event_tags_are_aligned(self):
        """Event tags are words of their speaker stream; a word starting with a tag is preceded by the bare `▁` piece."""
        inputs = self.make_inputs([None, "▁a", None, "▁", "<ev:laughter>", None])
        result = self.processor.align_word_timestamps("<spk:3> a <ev:laughter>", **inputs)
        self.assertWordsAlmostEqual(result["chunks"], [("a", 0.08, 0.24, 3), ("<ev:laughter>", 0.24, 0.40, 3)])
