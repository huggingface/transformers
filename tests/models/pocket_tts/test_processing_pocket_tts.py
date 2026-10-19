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
import tempfile
import unittest
from pathlib import Path

from transformers import PocketTTSFeatureExtractor, PocketTTSProcessor
from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

    from transformers import PocketTTSAudioTokenizerConfig, PocketTTSAudioTokenizerModel


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures/pocket_tts"
# The public tokenizer the original reads (`kyutai/pocket-tts-without-voice-cloning`), with the converter's placeholders.
TOKENIZER_SOURCE = (
    "kyutai/pocket-tts-without-voice-cloning",
    "languages/english/tokenizer.json",
    "00eac05ed3d16bdc3f6b5d598874019c34a89214",
)


def conversation(text, audio=None, voice=False):
    """A generation conversation: the voice as a recording (`audio`), as precomputed latents (`voice=True`, an empty
    `audio` item), or neither (the voice comes as a cache)."""
    content = [{"type": "text", "text": text}]
    if audio is not None:
        content.insert(0, {"type": "audio", "audio": audio})
    elif voice:
        content.insert(0, {"type": "audio"})
    return [{"role": "user", "content": content}]


@require_torch
class PocketTTSProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    """
    reproducer: reproducer_processing.py — the original's `prepare_text_prompt` (prepared text, token ids,
    `frames_after_eos`) and `split_into_best_sentences`, for English and for French (`remove_semicolons`,
    `replace_characters`). The chat templates (`chat_template_<language>.jinja`) are the converter's.
    """

    processor_class = PocketTTSProcessor
    audio_input_name = "audio_prompt_latents"

    @classmethod
    def _setup_tokenizer(cls):
        from huggingface_hub import hf_hub_download
        from tokenizers import AddedToken, Tokenizer

        from transformers import TokenizersBackend

        repo_id, filename, revision = TOKENIZER_SOURCE
        tokenizer = Tokenizer.from_file(hf_hub_download(repo_id, filename, revision=revision))
        tokens = ["<|audio_prompt|>", "<|audio_latent|>", "<|audio_prompt_bos|>", "<|audio_bos|>"]
        tokenizer.add_special_tokens([AddedToken(token, special=True, normalized=False) for token in tokens])
        return TokenizersBackend(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            # the call defaults the converter writes into the checkpoint's `tokenizer_config.json`
            padding=True,
            padding_side="left",
            model_specific_special_tokens=dict(
                zip(["audio_prompt_token", "audio_latent_token", "audio_prompt_bos_token", "audio_bos_token"], tokens)
            ),
        )

    @classmethod
    def _setup_feature_extractor(cls):
        return PocketTTSFeatureExtractor()

    @classmethod
    def prepare_processor_dict(cls):
        # a tiny random audio tokenizer with the real frame size, saved so the processor's pointer to it resolves
        if not hasattr(cls, "_audio_tokenizer_dir"):
            cls._audio_tokenizer_dir = tempfile.mkdtemp()
            config = PocketTTSAudioTokenizerConfig(
                hidden_size=16,
                num_filters=4,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                intermediate_size=32,
                upsample_groups=16,
            )
            PocketTTSAudioTokenizerModel(config).save_pretrained(cls._audio_tokenizer_dir)
        return {
            "audio_tokenizer": PocketTTSAudioTokenizerModel.from_pretrained(cls._audio_tokenizer_dir),
            "chat_template": (FIXTURES_DIR / "chat_template_english.jinja").read_text(encoding="utf-8"),
        }

    def setUp(self):
        super().setUp()
        if self._testMethodName.startswith(
            ("test_apply_chat_template_audio", "test_apply_chat_template_assistant_mask")
        ):
            self.skipTest(
                reason="The common conversations have a system turn or several turns; a Pocket TTS conversation is one "
                "`user` message (`test_chat_template_rejects_other_conversations`)."
            )
        self.fixture = json.load(open(FIXTURES_DIR / "expected_results_processing.json", encoding="utf-8"))

    def prepare_text_inputs(self, batch_size=None, modalities=None):
        # prompts as the chat template renders them: the voice prompt first, the first audio position last
        text = super().prepare_text_inputs(batch_size=batch_size)
        voice_prompt, audio_bos = "<|audio_prompt_bos|>" + self.audio_token, "<|audio_bos|>"
        if isinstance(text, str):
            return voice_prompt + text + audio_bos
        return [voice_prompt + prompt + audio_bos for prompt in text]

    def test_processor_to_json_string(self):
        # the audio tokenizer is saved as a pointer to its checkpoint, not as a value
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        self.assertEqual(
            obj["audio_tokenizer"],
            {
                "audio_tokenizer_class": "PocketTTSAudioTokenizerModel",
                "audio_tokenizer_name_or_path": processor.audio_tokenizer.name_or_path,
            },
        )

    def test_save_load_pretrained_additional_features(self):
        # `from_pretrained` forwards its kwargs to the audio tokenizer's `from_pretrained` too, whose model rejects the
        # sub-processors' kwargs: loaded without any, as in Dia's processor test
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        self.assertIsInstance(processor.feature_extractor, PocketTTSFeatureExtractor)
        self.assertIsInstance(processor.audio_tokenizer, PocketTTSAudioTokenizerModel)

    def test_half_audio_tokenizer(self):
        # an audio tokenizer loaded without its encoder or decoder: the processor names the missing half
        processor = self.get_processor()
        processor.audio_tokenizer = PocketTTSAudioTokenizerModel.from_pretrained(
            self._audio_tokenizer_dir, decoder_config=None
        )
        with self.assertRaisesRegex(ValueError, "decoder_config=None"):
            processor.decode(torch.randn(2, 32))
        processor.audio_tokenizer = PocketTTSAudioTokenizerModel.from_pretrained(
            self._audio_tokenizer_dir, encoder_config=None
        )
        audio = torch.randn(processor.feature_extractor.sampling_rate).numpy() * 0.1
        with self.assertRaisesRegex(ValueError, "encoder_config=None"):
            processor.apply_chat_template(
                conversation("Hello world", audio), tokenize=True, return_dict=True, add_generation_prompt=True
            )

    def test_audio_tokenizer_placement(self):
        # `device_map` and `dtype` place the audio tokenizer, and are harmless to the other sub-processors
        processor = self.processor_class.from_pretrained(self.tmpdirname, device_map=torch_device, dtype=torch.float16)
        self.assertEqual(processor.audio_tokenizer.device.type, torch.device(torch_device).type)
        self.assertEqual(processor.audio_tokenizer.dtype, torch.float16)
        # inputs follow it: latents go to its device and dtype, waveforms come back in float32 on CPU
        audio = processor.decode(torch.randn(2, 32))
        self.assertEqual((audio.device.type, audio.dtype), ("cpu", torch.float32))

    def _processor_for(self, language):
        processor = self.get_processor()
        processor.chat_template = (FIXTURES_DIR / f"chat_template_{language}.jinja").read_text(encoding="utf-8")
        return processor

    def test_chat_template_prepares_text(self):
        for language in ("english", "french"):
            processor = self._processor_for(language)
            for row in self.fixture["languages"][language]["prompts"]:
                with self.subTest(language=language, text=row["text"]):
                    # without a voice in the conversation (it comes as a cache), the text alone
                    prompt = processor.apply_chat_template(conversation(row["text"]), add_generation_prompt=True)
                    self.assertEqual(prompt, row["prepared_text"] + processor.audio_bos_token)
                    prompt = processor.apply_chat_template(
                        conversation(row["text"], voice=True), add_generation_prompt=True
                    )
                    self.assertEqual(
                        prompt,
                        processor.audio_prompt_bos_token
                        + processor.audio_token
                        + row["prepared_text"]
                        + processor.audio_bos_token,
                    )
                    # the French checkpoint has its own tokenizer, not converted yet
                    if language == "english":
                        inputs = processor.apply_chat_template(
                            conversation(row["text"], voice=True),
                            tokenize=True,
                            return_dict=True,
                            add_generation_prompt=True,
                            processor_kwargs={"audio_prompt_latents": [torch.randn(3, 32)]},
                        )
                        # the original keeps the end-of-speech frame + (its `frames_after_eos` - 1) frames
                        self.assertEqual(inputs["frames_after_eos"].item(), row["frames_after_eos"] - 1)
                        text_ids = inputs["input_ids"][0, 4:-1].tolist()
                        self.assertEqual(text_ids, row["token_ids"])

    def test_chat_template_rejects_other_conversations(self):
        processor = self.get_processor()
        for invalid in (
            [{"role": "system", "content": "You are a TTS."}, *conversation("Hello")],
            [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}],
            conversation("   "),
        ):
            with self.subTest(conversation=invalid), self.assertRaises(Exception):
                processor.apply_chat_template(invalid)

    def test_generation_prompt(self):
        processor = self.get_processor()
        # `add_generation_prompt` appends the BOS latent position, from which the model generates
        prompt = processor.apply_chat_template(conversation("Hello world"), add_generation_prompt=False)
        self.assertFalse(prompt.endswith(processor.audio_bos_token))
        prompt = processor.apply_chat_template(conversation("Hello world"), add_generation_prompt=True)
        self.assertTrue(prompt.endswith(processor.audio_bos_token))

    def test_split_text(self):
        split = self.fixture["languages"]["english"]["split"]
        processor = self.get_processor()
        self.assertListEqual(processor.split_text(split["text"], max_tokens=split["max_tokens"]), split["chunks"])

    def test_call_with_voice_latents(self):
        processor = self.get_processor()
        tokenizer = processor.tokenizer
        texts = ["Hello world. I am Kyutai's Pocket TTS.", "Hello world"]
        voices = [torch.randn(5, 32), torch.randn(3, 32)]
        inputs = processor.apply_chat_template(
            [conversation(text, voice=True) for text in texts],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            processor_kwargs={"audio_prompt_latents": voices},
        )

        self.assertEqual(set(inputs.keys()), set(processor.model_input_names))
        # right-padded to the longest voice, with the mask of the voice frames
        self.assertEqual(inputs["audio_prompt_latents"].shape, (2, 5, 32))
        self.assertListEqual(inputs["audio_prompt_latents_mask"].sum(-1).tolist(), [5, 3])
        torch.testing.assert_close(inputs["audio_prompt_latents"][1, :3], voices[1])
        self.assertListEqual(inputs["frames_after_eos"].tolist(), [2, 4])
        prompt_bos, prompt, audio_bos = (
            tokenizer.convert_tokens_to_ids(token)
            for token in (processor.audio_prompt_bos_token, processor.audio_token, processor.audio_bos_token)
        )
        prepared = processor.apply_chat_template(conversation(texts[0]))
        text_ids = tokenizer(prepared, add_special_tokens=False)["input_ids"]
        # row 0: voice BOS, voice placeholders, the prepared text, the BOS latent position; no padding
        self.assertListEqual(inputs["input_ids"][0].tolist(), [prompt_bos] + [prompt] * 5 + text_ids + [audio_bos])
        # row 1: left-padded, then the same layout
        row = inputs["input_ids"][1][inputs["attention_mask"][1].bool()].tolist()
        self.assertListEqual(row[:4], [prompt_bos, prompt, prompt, prompt])
        self.assertEqual(row[-1], audio_bos)
        self.assertEqual(inputs["input_ids"][1, 0].item(), tokenizer.pad_token_id)

    def test_call_with_voice_audio(self):
        processor = self.get_processor()
        sampling_rate = processor.feature_extractor.sampling_rate
        audio = [torch.randn(sampling_rate // 2).numpy() * 0.1, torch.randn(sampling_rate).numpy() * 0.1]
        inputs = processor.apply_chat_template(
            [conversation("Hello world", audio[0]), conversation("Hello there", audio[1])],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            processor_kwargs={"end_on_pause": False},
        )
        self.assertEqual(set(inputs.keys()), set(processor.model_input_names))
        # the voice is encoded by the audio tokenizer: one latent frame, and one placeholder, per 1920 samples
        num_frames = [-(-len(waveform) // processor.feature_extractor.frame_size) for waveform in audio]
        self.assertListEqual(inputs["audio_prompt_latents_mask"].sum(-1).tolist(), num_frames)
        self.assertListEqual((inputs["input_ids"] == processor.audio_token_id).sum(-1).tolist(), num_frames)
        # the feature extractor pads the waveform to whole frames before the encoder
        waveform = torch.nn.functional.pad(torch.as_tensor(audio[1]), (0, -len(audio[1]) % 1920))
        with torch.no_grad():
            expected = processor.audio_tokenizer.encode(waveform[None, None]).latents[0]
        # batched with the padded shorter waveform: the same latents up to op-order noise
        torch.testing.assert_close(inputs["audio_prompt_latents"][1], expected, atol=1e-4, rtol=1e-4)

    def test_call_takes_one_voice(self):
        processor = self.get_processor()
        prompt = processor.apply_chat_template(conversation("Hello world", voice=True), add_generation_prompt=True)
        # a text-only call leaves the voice placeholder as is
        inputs = processor(text=prompt)
        self.assertNotIn("audio_prompt_latents", inputs)
        self.assertEqual((inputs["input_ids"] == processor.audio_token_id).sum().item(), 1)
        # more waveforms than voices and targets
        with self.assertRaises(ValueError):
            processor(text=prompt, audio=[torch.zeros(1920).numpy()] * 2, audio_prompt_latents=[torch.randn(1, 32)])
        # raw text is not a prompt: it goes through the chat template
        with self.assertRaises(ValueError):
            processor(text="Hello world", audio_prompt_latents=[torch.randn(2, 32)])

    def test_training_conversation(self):
        processor = self.get_processor()
        frame_size = processor.feature_extractor.frame_size
        voice, target = torch.randn(3 * frame_size).numpy() * 0.1, torch.randn(5 * frame_size).numpy() * 0.1
        training_conversation = [
            {"role": "user", "content": [{"type": "audio", "audio": voice}, {"type": "text", "text": "hello world"}]},
            {"role": "assistant", "content": [{"type": "audio", "audio": target}]},
        ]
        # the training text is kept as written, every audio item is an audio placeholder, the target after the first
        # audio position
        prompt = processor.apply_chat_template(training_conversation)
        self.assertEqual(
            prompt,
            processor.audio_prompt_bos_token
            + processor.audio_token
            + "hello world"
            + processor.audio_bos_token
            + processor.audio_token,
        )
        inputs = processor.apply_chat_template(training_conversation, tokenize=True, return_dict=True)
        self.assertEqual(
            set(inputs.keys()),
            {
                "input_ids",
                "attention_mask",
                "audio_prompt_latents",
                "audio_prompt_latents_mask",
                "labels",
                "labels_mask",
            },
        )
        # the target's latents, with the first audio position then one latent position per frame
        self.assertEqual(inputs["labels"].shape, (1, 5, 32))
        audio_bos_id, latent_id = processor.tokenizer.convert_tokens_to_ids(
            [processor.audio_bos_token, processor.audio_latent_token]
        )
        self.assertEqual((inputs["input_ids"] == audio_bos_id).sum().item(), 1)
        self.assertEqual((inputs["input_ids"] == latent_id).sum().item(), 5)
        self.assertEqual((inputs["input_ids"] == processor.audio_token_id).sum().item(), 3)
        # voice and target are encoded in one batch, with no pause appended in training
        with torch.no_grad():
            expected_target = processor.audio_tokenizer.encode(torch.as_tensor(target)[None, None]).latents[0]
            expected_voice = processor.audio_tokenizer.encode(torch.as_tensor(voice)[None, None]).latents[0]
        torch.testing.assert_close(inputs["labels"][0], expected_target, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(inputs["audio_prompt_latents"][0], expected_voice, atol=1e-4, rtol=1e-4)

    def test_training_classifier_free_guidance_rows(self):
        processor = self.get_processor()
        latent_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_latent_token)
        rows = [
            # the voice and the text
            [{"type": "audio"}, {"type": "text", "text": "Hello world."}],
            # the voice dropped
            [{"type": "text", "text": "Hello there."}],
            # the text dropped
            [{"type": "audio"}],
        ]
        conversations = [
            [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "audio"}]}]
            for content in rows
        ]
        # precomputed voices (one per voiced row) and targets
        voices = [torch.randn(2, 32), torch.randn(4, 32)]
        targets = [torch.randn(3, 32), torch.randn(1, 32), torch.randn(2, 32)]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            processor_kwargs={"audio_prompt_latents": voices, "labels": targets},
        )
        # one voice row per prompt with a voice, in order
        self.assertListEqual(inputs["audio_prompt_latents_mask"].sum(-1).tolist(), [2, 4])
        self.assertListEqual((inputs["input_ids"] == processor.audio_token_id).sum(-1).tolist(), [2, 0, 4])
        self.assertListEqual(inputs["labels_mask"].sum(-1).tolist(), [3, 1, 2])
        self.assertListEqual((inputs["input_ids"] == latent_token_id).sum(-1).tolist(), [3, 1, 2])
        torch.testing.assert_close(inputs["audio_prompt_latents"][1, :4], voices[1])
        torch.testing.assert_close(inputs["labels"][1, :1], targets[1])

    def test_decoder_only_audio_tokenizer(self):
        processor = self.get_processor()
        decoder_only = PocketTTSAudioTokenizerModel.from_pretrained(self._audio_tokenizer_dir, encoder_config=None)
        processor = PocketTTSProcessor(
            processor.feature_extractor,
            processor.tokenizer,
            audio_tokenizer=decoder_only,
            chat_template=processor.chat_template,
        )
        # precomputed voices and decoding work, cloning from audio is refused before any feature extraction
        inputs = processor.apply_chat_template(
            conversation("Hello world", voice=True),
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            processor_kwargs={"audio_prompt_latents": [torch.randn(3, 32)]},
        )
        self.assertEqual(inputs["audio_prompt_latents"].shape, (1, 3, 32))
        self.assertEqual(processor.decode(torch.randn(2, 32)).shape[0], 2 * processor.feature_extractor.frame_size)
        with self.assertRaisesRegex(ValueError, "no encoder"):
            processor.apply_chat_template(
                conversation("Hello world", torch.zeros(1920).numpy()), tokenize=True, add_generation_prompt=True
            )

    def test_decode(self):
        processor = self.get_processor()
        frame_size = processor.audio_tokenizer.config.frame_size
        # a padded batch, as `generate` returns it, with the mask of each row's frames
        latents = torch.randn(2, 4, 32)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        audio = processor.decode(latents, audio_latents_mask=mask)
        self.assertListEqual([waveform.shape[0] for waveform in audio], [4 * frame_size, 2 * frame_size])
        # one sample gives one waveform, the same as in the batch (the decoder is causal), faded in
        single = processor.decode(latents[1, :2])
        torch.testing.assert_close(single, audio[1], atol=1e-5, rtol=0)
        self.assertEqual(single[0].item(), 0.0)
