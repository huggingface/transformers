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

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from transformers import (
    AutoTokenizer,
    Lfm3AudioProcessor,
    LiquidCodecConfig,
    LiquidCodecFeatureExtractor,
    LiquidCodecModel,
)
from transformers.testing_utils import require_librosa, require_soundfile, require_torch, torch_device
from transformers.utils import is_soundfile_available, is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf


@require_torch
class Lfm3AudioProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Lfm3AudioProcessor
    # tokenizer and chat template of the checkpoint (`from_pretrained(model_id)` would load its codec too)
    checkpoint_name = "eustlb/LFM3-Audio-2B"
    revision = "refs/pr/1"

    @classmethod
    def _setup_tokenizer(cls):
        return AutoTokenizer.from_pretrained(cls.checkpoint_name, revision=cls.revision)

    @classmethod
    def _setup_codec_feature_extractor(cls):
        return LiquidCodecFeatureExtractor()

    @classmethod
    def prepare_processor_dict(cls):
        # tiny random codec, real frame size (1920 samples); saved so the processor's pointer resolves
        if not hasattr(cls, "codec_dirname"):
            cls.codec_dirname = tempfile.mkdtemp()
            backbone = {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "use_cache": False,
                "quantization_dim": 16,
            }
            config = LiquidCodecConfig(
                encoder_config={**backbone, "embedder_hidden_size": 8},
                decoder_config=backbone,
                quantization_dim=16,
                num_quantizers=4,
                quantization_levels=[3, 3],
            )
            torch.manual_seed(0)
            LiquidCodecModel(config).save_pretrained(cls.codec_dirname)
            chat_template = hf_hub_download(cls.checkpoint_name, "chat_template.jinja", revision=cls.revision)
            cls.chat_template = Path(chat_template).read_text(encoding="utf-8")
        return {
            "audio_tokenizer": LiquidCodecModel.from_pretrained(cls.codec_dirname),
            "chat_template": cls.chat_template,
        }

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls.codec_dirname, ignore_errors=True)

    def get_component(self, attribute, **kwargs):
        # `codec_feature_extractor` is saved in the processor config: no file an auto class could load it from
        if attribute == "codec_feature_extractor":
            return self.get_processor().codec_feature_extractor
        return super().get_component(attribute, **kwargs)

    def test_processor_from_and_save_pretrained_as_nested_dict(self):
        # the parent test also reloads each sub-processor from its own file, which `codec_feature_extractor` lacks
        processor = self.get_processor()
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor.save_pretrained(tmpdirname)
            reloaded = self.processor_class.from_pretrained(tmpdirname)
        self.assertEqual(reloaded.to_dict(), processor.to_dict())

    def test_processor_to_json_string(self):
        # codec saved as a pointer to its checkpoint
        processor = self.get_processor()
        self.assertEqual(
            json.loads(processor.to_json_string())["audio_tokenizer"],
            {"audio_tokenizer_class": "LiquidCodecModel", "audio_tokenizer_name_or_path": self.codec_dirname},
        )

    def test_audio_tokenizer_placement(self):
        # `device_map` / `dtype` reach the codec only
        processor = self.processor_class.from_pretrained(self.tmpdirname, device_map=torch_device, dtype=torch.float16)
        codec = processor.audio_tokenizer
        self.assertEqual(codec.device.type, torch.device(torch_device).type)
        self.assertEqual(codec.dtype, torch.float16)
        # inputs go to its device; frames and waveforms come back on CPU
        waveform = np.random.randn(1920 * 2).astype(np.float32)
        inputs = processor(text=processor.audio_out_token, audio_out=[waveform], return_tensors="pt")
        self.assertEqual((inputs["audio_input_ids"].device.type, inputs["audio_input_ids"].shape), ("cpu", (2 + 1, 4)))
        audio = processor.decode(torch.randint(0, 9, (1, 4, 2)), torch.ones(1, 2, dtype=torch.bool))
        self.assertEqual((audio[0].device.type, audio[0].shape), ("cpu", (2 * 1920,)))

    def test_model_input_names(self):
        # `audio_input_ids` need `audio_out`, which the common test does not pass
        processor = self.get_processor()
        inputs = processor(
            text=self.prepare_text_inputs(modalities="audio") + processor.audio_out_token,
            audio=self.prepare_audio_inputs(),
            audio_out=[torch.randint(0, 9, (4, 2))],
            return_tensors="pt",
        )
        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_replacement_offsets(self):
        # common test: image / video only. Here 2 placeholder kinds, 2 passes, merged offsets
        processor = self.get_processor()
        audio_token, audio_out_token = processor.audio_token, processor.audio_out_token
        text = f"Hello {audio_out_token} {audio_token} Hello {audio_out_token}{audio_token} ."
        inputs = processor(
            text=text,
            # 0.25 s and 1 s: 3 and 13 speech encoder frames
            audio=[np.zeros(4000, dtype=np.float32), np.zeros(16000, dtype=np.float32)],
            audio_out=[torch.randint(0, 9, (4, 2)), torch.randint(0, 9, (4, 5))],
            sampling_rate=16000,
            return_text_replacement_offsets=True,
        )
        replacements = [audio_out_token * 2, audio_token * 3, audio_out_token * 5, audio_token * 13]
        expanded_text = f"Hello {replacements[0]} {replacements[1]} Hello {replacements[2]}{replacements[3]} ."
        self.assertListEqual(inputs["input_ids"][0], processor.tokenizer(expanded_text)["input_ids"])

        offsets = inputs["text_replacement_offsets"][0]
        self.assertListEqual([entry["type"] for entry in offsets], ["audio_out", "audio", "audio_out", "audio"])
        self.assertListEqual([entry["replacement"] for entry in offsets], replacements)
        for entry in offsets:
            placeholder = audio_token if entry["type"] == "audio" else audio_out_token
            self.assertEqual(text[slice(*entry["span"])], placeholder)
            self.assertEqual(expanded_text[slice(*entry["new_span"])], entry["replacement"])

    def test_apply_chat_template_generation_mode(self):
        processor = self.get_processor()
        conversation = [{"role": "user", "content": "Hello"}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # the mode opens the answer with its marker; `"text"` with none
        markers = {"text": "", "audio": "<|audio_start|>", "interleaved": "<|mixed_start|>"}
        for mode, marker in markers.items():
            rendered = processor.apply_chat_template(conversation, add_generation_prompt=True, generation_mode=mode)
            self.assertEqual(rendered, prompt + marker)
        # one mode per conversation, or one for all; caller's messages untouched
        conversations = [conversation] * 3
        rendered = processor.apply_chat_template(
            conversations, add_generation_prompt=True, generation_mode=list(markers)
        )
        self.assertListEqual(rendered, [prompt + marker for marker in markers.values()])
        rendered = processor.apply_chat_template(conversations, add_generation_prompt=True, generation_mode="audio")
        self.assertListEqual(rendered, [prompt + "<|audio_start|>"] * 3)
        self.assertNotIn("generation_mode", conversation[-1])
        # tokenized: left-padded, each prompt ends with its marker
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            generation_mode=list(markers),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        newline_id, mixed_start_id = processor.tokenizer.convert_tokens_to_ids(["Ċ", "<|mixed_start|>"])
        self.assertListEqual(
            inputs["input_ids"][:, -1].tolist(), [newline_id, processor.audio_start_token_id, mixed_start_id]
        )
        self.assertListEqual(inputs["attention_mask"][:, 0].tolist(), [0, 1, 1])

    def test_apply_chat_template_generation_mode_errors(self):
        processor = self.get_processor()
        conversation = [{"role": "user", "content": "Hello"}]
        for kwargs, message in (
            ({"generation_mode": "audio"}, "add_generation_prompt"),
            ({"generation_mode": "speech", "add_generation_prompt": True}, "must be"),
            ({"generation_mode": ["audio"], "add_generation_prompt": True}, "one per conversation"),
            (
                {"generation_mode": "audio", "add_generation_prompt": True, "chat_template": "{{ messages }}"},
                "does not support",
            ),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                processor.apply_chat_template([conversation] * 3, **kwargs)
        # a list of modes needs a batch of conversations
        with self.assertRaisesRegex(ValueError, "one per conversation"):
            processor.apply_chat_template(conversation, add_generation_prompt=True, generation_mode=["audio"])

    def test_apply_chat_template_assistant_mask_with_audio_out(self):
        # common test: image placeholders only. Speech frames (2nd pass) shift the assistant spans too
        processor = self.get_processor()
        conversation = [
            {"role": "user", "content": [{"type": "audio_out"}, {"type": "text", "text": " Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello "}, {"type": "audio_out"}]},
            {"role": "user", "content": [{"type": "audio", "audio": np.zeros(16000, dtype=np.float32)}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]},
        ]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            processor_kwargs={"audio_out": [torch.randint(0, 9, (4, 3)), torch.randint(0, 9, (4, 4))]},
        )
        assistant_ids = inputs["input_ids"][0][inputs["assistant_masks"][0].bool()]
        speech_answer = f"Hello {processor.audio_start_token}{processor.audio_out_token * 4}<|im_end|>\n"
        self.assertEqual(processor.tokenizer.decode(assistant_ids), speech_answer + "Hello.<|im_end|>\n")

    def test_call_expands_audio_placeholders(self):
        processor = self.get_processor()
        clips = [np.zeros(16000, dtype=np.float32), np.zeros(4000, dtype=np.float32)]
        inputs = processor(
            text=[f"Hello {processor.audio_token} Hello", processor.audio_token],
            audio=clips,
            sampling_rate=16000,
            return_tensors="pt",
        )
        # 1 s: 99 spectrogram frames -> 13 encoder frames (8 each, last partial); 0.25 s: 24 -> 3
        self.assertListEqual((inputs["input_ids"] == processor.audio_token_id).sum(-1).tolist(), [13, 3])
        self.assertListEqual(inputs["input_features_mask"].sum(-1).tolist(), [99, 24])

    def test_call_with_audio_out_codes(self):
        processor = self.get_processor()
        audio_out_token = processor.audio_out_token
        codes = [torch.randint(0, 9, (4, 5)), torch.randint(0, 9, (4, 3))]
        # codes used as is: one placeholder per frame, all segments in one `(num_frames, 4)` tensor
        inputs = processor(text=f"{audio_out_token} Hello {audio_out_token}", audio_out=codes, return_tensors="pt")
        self.assertEqual((inputs["input_ids"] == processor.audio_out_token_id).sum().item(), 5 + 3)
        self.assertTrue(torch.equal(inputs["audio_input_ids"], torch.cat(codes, dim=-1).T))
        # one segment per sample of a batch
        batch = processor(text=[f"{audio_out_token} Hello", audio_out_token], audio_out=codes, return_tensors="pt")
        self.assertListEqual((batch["input_ids"] == processor.audio_out_token_id).sum(-1).tolist(), [5, 3])
        self.assertTrue(torch.equal(batch["audio_input_ids"], inputs["audio_input_ids"]))
        # one segment, not in a list
        single = processor(text=audio_out_token, audio_out=codes[0], return_tensors="pt")
        self.assertTrue(torch.equal(single["audio_input_ids"], codes[0].T))
        # `audio_input_ids` follow the text's `return_tensors`
        for kwargs in (
            {"return_tensors": "np"},
            {"text_kwargs": {"return_tensors": "np"}},
            {"common_kwargs": {"return_tensors": "np"}},
        ):
            with self.subTest(kwargs=kwargs):
                outputs = processor(text=[audio_out_token, audio_out_token], audio_out=codes, **kwargs)
                self.assertIsInstance(outputs["input_ids"], np.ndarray)
                self.assertIsInstance(outputs["audio_input_ids"], np.ndarray)
        # one segment per placeholder, and a text to hold them
        with self.assertRaisesRegex(ValueError, "2 entries in `audio_out`"):
            processor(text=audio_out_token, audio_out=codes)
        with self.assertRaisesRegex(ValueError, "0 entries in `audio_out`"):
            processor(text=audio_out_token)
        with self.assertRaisesRegex(ValueError, "requires `text`"):
            processor(audio_out=codes)

    def test_call_with_audio_out_audio(self):
        processor = self.get_processor()
        audio_out_token = processor.audio_out_token
        # 24 kHz waveforms of different lengths, next to codes
        generator = np.random.default_rng(0)
        waveforms = [generator.standard_normal(length, dtype=np.float32) for length in (24000, 1920 * 3 + 100)]
        codes = torch.randint(0, 9, (4, 2))
        inputs = processor(
            text=f"{audio_out_token} Hello {audio_out_token} Hello {audio_out_token}",
            audio_out=[waveforms[0], codes, waveforms[1]],
            return_tensors="pt",
        )
        # each waveform encoded as alone, per started 1920-sample frame, + end-of-speech frame
        audio_eos_code = processor.audio_tokenizer.config.codebook_size
        self.assertEqual(processor.audio_eos_code, audio_eos_code)
        expected = []
        for waveform in waveforms:
            features = processor.codec_feature_extractor(waveform, sampling_rate=24000, return_tensors="pt")
            with torch.no_grad():
                frames = processor.audio_tokenizer.encode(features["input_features"]).audio_codes[0].T
            expected.append(torch.cat([frames, torch.full_like(frames[:1], audio_eos_code)]))
        self.assertListEqual([len(frames) for frames in expected], [13 + 1, 4 + 1])
        self.assertTrue(torch.equal(inputs["audio_input_ids"], torch.cat([expected[0], codes.T, expected[1]])))
        self.assertEqual((inputs["input_ids"] == processor.audio_out_token_id).sum().item(), 14 + 2 + 5)

        # without the codec, codes are still taken and waveforms refused
        processor.audio_tokenizer = None
        self.assertEqual(len(processor(text=audio_out_token, audio_out=[codes])["audio_input_ids"]), 2)
        with self.assertRaisesRegex(ValueError, "requires the codec"):
            processor(text=audio_out_token, audio_out=waveforms[:1])

    def test_call_rejects_truncated_audio_out(self):
        processor = self.get_processor()
        # the start of text token, 4 speech frames and one text token
        text, codes = f"{processor.audio_out_token} Hello", [torch.randint(0, 9, (4, 4))]
        with self.assertRaisesRegex(ValueError, "after tokenization"):
            processor(text=text, audio_out=codes, truncation=True, max_length=2)
        inputs = processor(text=text, audio_out=codes, truncation=True, max_length=5)
        self.assertEqual(inputs["input_ids"][0].count(processor.audio_out_token_id), 4)

    def test_output_labels(self):
        processor = self.get_processor()
        speech, codes = np.random.randn(24000).astype(np.float32), torch.randint(0, 9, (4, 3))
        # speech as codes or waveform; 2nd conversation shorter, padded
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": np.zeros(16000, dtype=np.float32)},
                        {"type": "audio_out", "codes": codes},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello "}, {"type": "audio_out", "audio": speech}],
                },
            ],
            [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
            ],
        ]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"output_labels": True},
        )
        # `audio_labels`: 3 frames of codes, 13 + end of speech for the waveform
        self.assertEqual(inputs["audio_input_ids"].shape, (3 + 14, 4))
        self.assertTrue(torch.equal(inputs["audio_input_ids"][:3], codes.T))
        self.assertTrue(torch.equal(inputs["audio_labels"], inputs["audio_input_ids"]))
        # `labels` ignore both placeholder kinds, `<|audio_start|>` (prefilled at inference) and padding
        input_ids, labels = inputs["input_ids"], inputs["labels"]
        is_ignored = {
            "padding": inputs["attention_mask"] == 0,
            "audio": input_ids == processor.audio_token_id,
            "audio_out": input_ids == processor.audio_out_token_id,
            "audio_start": input_ids == processor.audio_start_token_id,
        }
        self.assertTrue(is_ignored["padding"].any())
        self.assertListEqual(
            [is_ignored[key].sum().item() for key in ("audio", "audio_out", "audio_start")], [13, 17, 1]
        )
        is_ignored = torch.stack(list(is_ignored.values())).any(0)
        self.assertTrue((labels[is_ignored] == -100).all())
        self.assertTrue(torch.equal(labels[~is_ignored], input_ids[~is_ignored]))

        # no `audio_labels` without speech frames, and tensors are required
        self.assertNotIn("audio_labels", processor(text="Hello", output_labels=True, return_tensors="pt"))
        with self.assertRaisesRegex(ValueError, "return_tensors"):
            processor(text="Hello", output_labels=True)

    @require_librosa
    @require_soundfile
    def test_apply_chat_template_audio_out_path(self):
        # speech answer from a file: loaded at the codec's 24 kHz
        processor = self.get_processor()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "answer.wav")
            sf.write(path, np.random.randn(16000).astype(np.float32) * 0.1, 16000)
            conversation = [
                {"role": "user", "content": [{"type": "audio", "path": path}]},
                {"role": "assistant", "content": [{"type": "audio_out", "path": path}]},
            ]
            inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True, return_tensors="pt")
        # 1 s: 13 encoder frames (16 kHz); 13 codec frames (24 kHz) + end of speech
        self.assertEqual((inputs["input_ids"] == processor.audio_token_id).sum().item(), 13)
        self.assertEqual((inputs["input_ids"] == processor.audio_out_token_id).sum().item(), 13 + 1)
        self.assertEqual(inputs["audio_input_ids"].shape, (13 + 1, 4))

    def test_decode(self):
        processor = self.get_processor()
        codec = processor.audio_tokenizer
        # padded batch as `generate` returns it; mask false on end-of-speech and padding: segments of 2, 1 | 4 frames
        codes = torch.randint(0, 9, (2, 4, 6))
        mask = torch.tensor([[1, 1, 0, 1, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        audio = processor.decode(codes, mask)
        self.assertListEqual([waveform.shape[-1] for waveform in audio], [2 * 1920, 1920, 4 * 1920])
        # each segment decoded as alone
        segments = [codes[:1, :, :2], codes[:1, :, 3:4], codes[1:, :, :4]]
        for waveform, segment in zip(audio, segments):
            with torch.no_grad():
                torch.testing.assert_close(waveform, codec.decode(segment).audio_values[0])
        by_keyword = processor.decode(audio_sequences=codes, audio_sequences_mask=mask)
        self.assertTrue(all(torch.equal(waveform, other) for waveform, other in zip(audio, by_keyword, strict=True)))
        self.assertListEqual(processor.decode(codes, torch.zeros_like(mask)), [])
        # token ids still decode to text
        ids = processor.tokenizer("Hello.", add_special_tokens=False)["input_ids"]
        self.assertEqual(processor.decode(ids), "Hello.")

        processor.audio_tokenizer = None
        with self.assertRaisesRegex(ValueError, "requires the codec"):
            processor.decode(codes, mask)

    @require_soundfile
    def test_save_audio(self):
        processor = self.get_processor()
        audio = [torch.randn(1920), torch.randn(3840)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = [os.path.join(tmp_dir, "first.wav"), os.path.join(tmp_dir, "second.wav")]
            processor.save_audio(audio, paths)
            self.assertListEqual([sf.info(path).frames for path in paths], [1920, 3840])
            self.assertEqual(sf.info(paths[0]).samplerate, processor.audio_tokenizer.config.sampling_rate)
            # one waveform and its path
            processor.save_audio(audio[1], paths[0])
            self.assertEqual(sf.info(paths[0]).frames, 3840)
            with self.assertRaisesRegex(ValueError, "saving paths"):
                processor.save_audio(audio, paths[0])

    def test_live_transcription_chunks(self):
        processor = self.get_processor()
        # chunk: 12 encoder frames x 8 spectrogram frames = 96 hops of 160 samples, + 240 to fill the last window
        self.assertEqual(processor.audio_chunk_hop_length, 15360)
        self.assertEqual(processor.num_samples_per_audio_chunk, 15600)
        # 30 s context (480000 samples) = 31 chunk hops
        self.assertEqual(processor.max_audio_chunks_per_context, 31)
        chunk = np.zeros(processor.num_samples_per_audio_chunk, dtype=np.float32)
        inputs = processor(text=processor.audio_token, audio=chunk, sampling_rate=16000, is_last_audio_chunk=True)
        self.assertEqual(inputs["input_ids"][0].count(processor.audio_token_id), 12)
        self.assertIs(inputs["is_last_audio_chunk"], True)
        self.assertNotIn(
            "is_last_audio_chunk", processor(text=processor.audio_token, audio=chunk, sampling_rate=16000)
        )
