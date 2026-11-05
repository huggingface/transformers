# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the LFM3-Audio processor."""

import tempfile
import unittest

import numpy as np

from transformers import (
    Lfm3AudioFeatureExtractor,
    Lfm3AudioProcessor,
    LiquidCodecConfig,
    LiquidCodecModel,
)
from transformers.testing_utils import (
    require_torch,
)
from transformers.utils import is_tokenizers_available

from ...test_processing_common import ProcessorTesterMixin


if is_tokenizers_available():
    from tokenizers import Tokenizer, models, pre_tokenizers

    from transformers import PreTrainedTokenizerFast


# the placeholder and span tokens the LFM3 tokenizers define (`test_hub_tokenizer_defines_the_placeholder_tokens`)
PLACEHOLDER_TOKENS = [
    "<|audio|>",
    "<|audio_out|>",
    "<|audio_start|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image|>",
    "<|video|>",
]


def tiny_tokenizer(special_tokens=PLACEHOLDER_TOKENS):
    vocab = {f"<{i}>": i for i in range(50)}
    words = [
        "<|startoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "\n",
        "system",
        "user",
        "assistant",
        "Perform",
        "ASR",
        "in",
        "English",
        ".",
        "Hello",
        "<|mixed_start|>",
        "<|text_end|>",
        "<|pad|>",
        *special_tokens,
    ]
    for word in words:
        vocab.setdefault(word, len(vocab))
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<0>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|startoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
        unk_token="<0>",
    )
    fast.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                "<|mixed_start|>",
                "<|text_end|>",
                *special_tokens,
            ]
        }
    )
    return fast


TINY_CODEC_BACKBONE = {
    "vocab_size": 1,
    "pad_token_id": None,
    "bos_token_id": None,
    "eos_token_id": None,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "use_cache": False,
}


def tiny_codec():
    return LiquidCodecModel(
        LiquidCodecConfig(
            encoder_config=TINY_CODEC_BACKBONE,
            decoder_config=TINY_CODEC_BACKBONE,
            encoder_conv_dim=8,
            quantizer_dim=16,
            n_q=4,
            fsq_levels=[3, 3],
        )
    ).eval()


# the hub chat template reduced to ChatML turns with one placeholder per clip
CHAT_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
    "{%- if message['content'] is string -%}{{- message['content'] -}}"
    "{%- else -%}{%- for content in message['content'] -%}"
    "{%- if content['type'] == 'audio' -%}{{- '<|audio|>' -}}"
    "{%- elif content['type'] == 'audio_out' -%}{{- '<|audio_out|>' -}}"
    "{%- elif content['type'] == 'text' -%}{{- content['text'] -}}"
    "{%- endif -%}{%- endfor -%}{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
)

# a template whose `{% generation %}` blocks cover the assistant turns, speech placeholders included
GENERATION_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
    "{%- if message['role'] == 'assistant' -%}{%- generation -%}"
    "{%- for content in message['content'] -%}"
    "{%- if content['type'] == 'audio_out' -%}{{- '<|audio_out|>' -}}{%- else -%}{{- content['text'] -}}{%- endif -%}"
    "{%- endfor -%}{{- '<|im_end|>\\n' -}}{%- endgeneration -%}"
    "{%- else -%}"
    "{%- for content in message['content'] -%}"
    "{%- if content['type'] == 'audio_out' -%}{{- '<|audio_out|>' -}}"
    "{%- elif content['type'] == 'audio' -%}{{- '<|audio|>' -}}{%- else -%}{{- content['text'] -}}{%- endif -%}"
    "{%- endfor -%}{{- '<|im_end|>\\n' -}}"
    "{%- endif -%}{%- endfor -%}"
)


def assert_offsets_rebuild_the_prompts(test, processor, texts, outputs):
    """Every record slices its `text` from the caller's prompt, applying them in order rebuilds the tokenized prompt,
    and every `new_span` slices its `replacement` from the rebuilt one."""
    for text, row, ids in zip(texts, outputs["text_replacement_offsets"], outputs["input_ids"]):
        pieces, last = [], 0
        for record in row:
            start, end = record["span"]
            test.assertGreaterEqual(start, last)
            test.assertEqual(text[start:end], record["text"])
            pieces.extend((text[last:start], record["replacement"]))
            last = end
        pieces.append(text[last:])
        rebuilt = "".join(pieces)
        for record in row:
            test.assertEqual(rebuilt[record["new_span"][0] : record["new_span"][1]], record["replacement"])
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        pad = processor.tokenizer.pad_token_id
        test.assertEqual(processor.tokenizer(rebuilt)["input_ids"], [token for token in ids if token != pad])


def expected_assistant_mask(processor, ids):
    """The tokens after each assistant header up to and including its `<|im_end|>`."""
    start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant = processor.tokenizer.convert_tokens_to_ids("assistant")
    mask, inside = [], False
    for index, token in enumerate(ids):
        mask.append(int(inside))
        if index and ids[index - 1] == start and token == assistant:
            inside = True
        elif token == end:
            inside = False
    return mask


@require_torch
class Lfm3AudioProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Lfm3AudioProcessor

    @classmethod
    def _setup_tokenizer(cls):
        return tiny_tokenizer()

    @classmethod
    def _setup_feature_extractor(cls):
        return Lfm3AudioFeatureExtractor()

    @staticmethod
    def prepare_processor_dict():
        return {"chat_template": CHAT_TEMPLATE}

    def setUp(self):
        self.tokenizer = tiny_tokenizer()
        self.processor = Lfm3AudioProcessor(Lfm3AudioFeatureExtractor(), self.tokenizer, chat_template=CHAT_TEMPLATE)

    def test_model_input_names(self):
        processor = Lfm3AudioProcessor(Lfm3AudioFeatureExtractor(), self.tokenizer, audio_tokenizer=tiny_codec())
        inputs = processor(
            text="<|audio|> <|audio_out|>",
            audio=[np.zeros(4000, dtype=np.float32)],
            audio_out=[np.zeros(1920, dtype=np.float32)],
            sampling_rate=16000,
        )
        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_truncated_speech_placeholders_are_rejected(self):
        processor = Lfm3AudioProcessor(Lfm3AudioFeatureExtractor(), self.tokenizer, audio_tokenizer=tiny_codec())
        clips = [np.zeros(1920 * 3, dtype=np.float32)]
        with self.assertRaisesRegex(ValueError, "after tokenization"):
            processor(text="<|audio_start|><|audio_out|> Hello", audio_out=clips, truncation=True, max_length=2)
        inputs = processor(text="<|audio_start|><|audio_out|> Hello", audio_out=clips, truncation=True, max_length=6)
        self.assertEqual(inputs["input_ids"][0].count(processor.audio_out_token_id), 4)

    def test_assistant_mask_covers_turns_after_speech_frames(self):
        processor = Lfm3AudioProcessor(
            Lfm3AudioFeatureExtractor(),
            self.tokenizer,
            audio_tokenizer=tiny_codec(),
            chat_template=GENERATION_TEMPLATE,
        )
        conversation = [
            {"role": "user", "content": [{"type": "audio_out"}, {"type": "text", "text": " Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello "}, {"type": "audio_out"}]},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello ."}]},
        ]
        speech = [np.zeros(1920 * 2, dtype=np.float32), np.zeros(1920 * 3, dtype=np.float32)]
        outputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            processor_kwargs={"audio_out": speech},
        )
        ids = list(outputs["input_ids"][0])
        self.assertEqual(ids.count(processor.audio_out_token_id), 3 + 4)
        self.assertEqual(list(outputs["assistant_masks"][0]), expected_assistant_mask(processor, ids))

    def test_serialization_references_the_codec_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tiny_codec().save_pretrained(tmp)
            codec = LiquidCodecModel.from_pretrained(tmp)
            processor = Lfm3AudioProcessor(Lfm3AudioFeatureExtractor(), self.tokenizer, audio_tokenizer=codec)
            output = processor.to_dict()
        self.assertEqual(
            output["audio_tokenizer"],
            {"audio_tokenizer_class": "LiquidCodecModel", "audio_tokenizer_name_or_path": tmp},
        )
        self.assertIs(processor.audio_tokenizer, codec)

    def test_placeholder_tokens_come_from_the_tokenizer(self):
        self.assertEqual(self.processor.audio_token_id, self.tokenizer.convert_tokens_to_ids("<|audio|>"))
        self.assertEqual(self.processor.audio_out_token_id, self.tokenizer.convert_tokens_to_ids("<|audio_out|>"))
        self.assertNotEqual(self.processor.audio_token_id, self.tokenizer.unk_token_id)
        # a tokenizer lacking them is left as it is: the tokenizer files define the vocabulary, not the processor
        tokenizer = tiny_tokenizer(special_tokens=[])
        size = len(tokenizer)
        Lfm3AudioProcessor(Lfm3AudioFeatureExtractor(), tokenizer)
        self.assertEqual(len(tokenizer), size)

    def test_expands_one_placeholder_per_encoder_token(self):
        clips = [np.zeros(16000, dtype=np.float32), np.zeros(4000, dtype=np.float32)]
        inputs = self.processor(
            text=["Hello <|audio|> Hello", "<|audio|>"], audio=clips, sampling_rate=16000, padding=True
        )
        counts = [int((np.asarray(row) == self.processor.audio_token_id).sum()) for row in inputs["input_ids"]]
        extractor = self.processor.feature_extractor
        expected = [
            extractor.num_encoder_tokens(extractor.num_frames(16000)),
            extractor.num_encoder_tokens(extractor.num_frames(4000)),
        ]
        self.assertEqual(counts, expected)
        self.assertEqual(np.asarray(inputs["input_features"]).shape[:2], (2, 128))
        self.assertEqual(np.asarray(inputs["feature_attention_mask"]).shape[0], 2)

    def test_clip_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.processor(text="<|audio|> <|audio|>", audio=[np.zeros(4000, dtype=np.float32)], sampling_rate=16000)
        with self.assertRaises(ValueError):
            self.processor(text="<|audio|>", sampling_rate=16000)

    def test_output_clips_become_codec_frames(self):
        codec = LiquidCodecModel(
            LiquidCodecConfig(
                encoder_config=TINY_CODEC_BACKBONE,
                decoder_config=TINY_CODEC_BACKBONE,
                encoder_conv_dim=8,
                quantizer_dim=16,
                n_q=4,
                fsq_levels=[3, 3],
            )
        ).eval()
        processor = Lfm3AudioProcessor(
            Lfm3AudioFeatureExtractor(), self.tokenizer, audio_tokenizer=codec, chat_template=CHAT_TEMPLATE
        )
        clips = [np.zeros(1920 * 2, dtype=np.float32), np.zeros(1920 * 3 + 1, dtype=np.float32)]
        inputs = processor(
            text="<|audio_start|><|audio_out|> Hello <|audio_out|>", audio_out=clips, return_tensors="pt"
        )
        # 2 + 3 codec frames, each clip closed by an end-of-speech frame
        self.assertEqual(int((inputs["input_ids"] == processor.audio_out_token_id).sum()), 8)
        self.assertEqual(tuple(inputs["audio_out_codes"].shape), (8, 4))
        self.assertEqual(processor.audio_eos_code, 9)
        self.assertTrue((inputs["audio_out_codes"][[2, 7]] == 9).all())
        self.assertTrue((inputs["audio_out_codes"][[0, 1, 3, 4, 5, 6]] < 9).all())
        batch = processor(text=("<|audio_out|>", "<|audio_out|>"), audio_out=clips, padding=True, return_tensors="np")
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertIsInstance(batch["audio_out_codes"], np.ndarray)
        for kwargs in ({"text_kwargs": {"return_tensors": "np"}}, {"common_kwargs": {"return_tensors": "np"}}):
            with self.subTest(kwargs=kwargs):
                nested = processor(text=("<|audio_out|>", "<|audio_out|>"), audio_out=clips, padding=True, **kwargs)
                self.assertIsInstance(nested["input_ids"], np.ndarray)
                self.assertIsInstance(nested["audio_out_codes"], np.ndarray)
                np.testing.assert_array_equal(nested["audio_out_codes"], batch["audio_out_codes"])
        with self.assertRaises(ValueError):
            processor(text="<|audio_out|>", audio_out=clips)
        with self.assertRaises(ValueError):
            self.processor(text="<|audio_out|>", audio_out=clips[:1])
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "<|audio_start|>"}, {"type": "audio_out"}]}
        ]
        self.assertTrue(
            processor.apply_chat_template(messages, tokenize=False).endswith(
                "<|audio_start|><|audio_out|><|im_end|>\n"
            )
        )
