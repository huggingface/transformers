# Copyright 2026 Gradium and The HuggingFace Inc. team. All rights reserved.
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

from transformers import GradiumPhononTTSProcessor, PocketTTSFeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

    from transformers import PocketTTSAudioTokenizerConfig, PocketTTSAudioTokenizerModel


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures/gradium_phonon_tts"
# Gradium's tokenizer is not public: the layout tests run on Pocket TTS's public one, with Phonon's two placeholders.
TOKENIZER_SOURCE = (
    "kyutai/pocket-tts-without-voice-cloning",
    "languages/english/tokenizer.json",
    "00eac05ed3d16bdc3f6b5d598874019c34a89214",
)


def conversation(text, audio=None, voice=False):
    """A generation conversation: the voice as a recording (`audio`), as precomputed latents (`voice=True`, an empty
    `audio` item), or neither."""
    content = [{"type": "text", "text": text}]
    if audio is not None:
        content.insert(0, {"type": "audio", "audio": audio})
    elif voice:
        content.insert(0, {"type": "audio"})
    return [{"role": "user", "content": content}]


@require_torch
class GradiumPhononTTSProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    """
    reproducer: the original Rust runtime's `normalize_text` / `prepare_text_prompt` (`expected_results_processing.json`,
    `gradium-ai/xn-ptts@preview-0.4.0`), against the chat template (`chat_template.jinja`, the converter's).
    """

    processor_class = GradiumPhononTTSProcessor
    audio_input_name = "audio_prompt_latents"

    @classmethod
    def _setup_tokenizer(cls):
        from huggingface_hub import hf_hub_download
        from tokenizers import AddedToken, Tokenizer

        from transformers import TokenizersBackend

        repo_id, filename, revision = TOKENIZER_SOURCE
        tokenizer = Tokenizer.from_file(hf_hub_download(repo_id, filename, revision=revision))
        tokens = ["<|audio_prompt|>", "<|audio_latent|>", "<|audio_bos|>"]
        tokenizer.add_special_tokens([AddedToken(token, special=True, normalized=False) for token in tokens])
        return TokenizersBackend(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            padding=True,
            padding_side="left",
            model_specific_special_tokens=dict(
                zip(["audio_prompt_token", "audio_latent_token", "audio_bos_token"], tokens)
            ),
        )

    @classmethod
    def _setup_feature_extractor(cls):
        return PocketTTSFeatureExtractor()

    @classmethod
    def prepare_processor_dict(cls):
        # a tiny random decoder-only audio tokenizer (the released one has no encoder), saved so the pointer resolves
        if not hasattr(cls, "_audio_tokenizer_dir"):
            cls._audio_tokenizer_dir = tempfile.mkdtemp()
            decoder = {
                "hidden_size": 16,
                "num_filters": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
                "upsample_groups": 16,
                "sliding_window": 251,
                "hidden_act": "gelu",
            }
            config = PocketTTSAudioTokenizerConfig(**decoder, encoder_config=None)
            PocketTTSAudioTokenizerModel(config).save_pretrained(cls._audio_tokenizer_dir)
        return {
            "audio_tokenizer": PocketTTSAudioTokenizerModel.from_pretrained(cls._audio_tokenizer_dir),
            "chat_template": (FIXTURES_DIR / "chat_template.jinja").read_text(encoding="utf-8"),
        }

    def setUp(self):
        super().setUp()
        if self._testMethodName.startswith(
            ("test_apply_chat_template_audio", "test_apply_chat_template_assistant_mask")
        ):
            self.skipTest(
                reason="The common conversations have a system turn or several turns; a Gradium Phonon TTS conversation "
                "is one `user` message."
            )
        if self._testMethodName == "test_chat_template_audio_sampling_rate":
            self.skipTest(
                reason="Gradium Phonon TTS's audio tokenizer has no encoder: a recording in the conversation is refused "
                "(`test_voice_audio_is_refused`)."
            )
        self.fixture = json.load(open(FIXTURES_DIR / "expected_results_processing.json", encoding="utf-8"))

    def prepare_text_inputs(self, batch_size=None, modalities=None):
        # prompts as the chat template renders them: the voice placeholder first (no voice BOS), the first audio position
        # last
        text = super().prepare_text_inputs(batch_size=batch_size)
        if isinstance(text, str):
            return self.audio_token + text + "<|audio_bos|>"
        return [self.audio_token + prompt + "<|audio_bos|>" for prompt in text]

    def prepare_audio_inputs(self, batch_size=None):
        # the released audio tokenizer has no encoder: the common audio inputs are refused, see
        # `test_voice_audio_is_refused`
        self.skipTest(reason="Gradium Phonon TTS's audio tokenizer has no encoder: voices are precomputed latents.")

    def _render(self, processor, text, lang):
        prompt = processor.apply_chat_template(conversation(text, voice=True), lang=lang)
        self.assertTrue(prompt.startswith(processor.audio_token))
        return prompt[len(processor.audio_token) :]

    def test_chat_template_normalizes_text(self):
        processor = self.get_processor()
        for row in self.fixture["normalization"]:
            with self.subTest(lang=row["lang"], text=row["input"]):
                # the fixture holds the runtime's normalisation before `prepare_text_prompt`, which applies to both
                self.assertEqual(
                    self._render(processor, row["input"], row["lang"]),
                    self._render(processor, row["normalized"], "none"),
                )
        for row in self.fixture["prepare_text_prompt"]:
            with self.subTest(text=row["input"]):
                self.assertEqual(self._render(processor, row["input"], "none"), row["prepared"])
                inputs = processor.apply_chat_template(
                    conversation(row["input"], voice=True),
                    lang="none",
                    tokenize=True,
                    return_dict=True,
                    add_generation_prompt=True,
                    processor_kwargs={"audio_prompt_latents": [torch.randn(2, 512)]},
                )
                self.assertEqual(inputs["frames_after_eos"].item(), row["frames_after_eos"])

    def test_chat_template_language(self):
        processor = self.get_processor()
        # English by default; another language picks its spoken forms; an unknown one is an error
        self.assertEqual(self._render(processor, "1+1=2 and 1500", "en"), "1 plus 1 equals 2 and 1 thousand 500.")
        # without a voice item, no voice placeholder
        self.assertEqual(processor.apply_chat_template(conversation("a@b")), "A at b.")
        self.assertEqual(self._render(processor, "1+1=2", "fr"), "1 plus 1 égal 2.")
        with self.assertRaises(Exception):
            processor.apply_chat_template(conversation("Hello"), lang="it")

    def test_call_with_voice_latents(self):
        processor = self.get_processor()
        tokenizer = processor.tokenizer
        voices = [torch.randn(5, 512), torch.randn(3, 512)]
        inputs = processor.apply_chat_template(
            [conversation("Hello world. I am Phonon.", voice=True), conversation("Hello world", voice=True)],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            processor_kwargs={"audio_prompt_latents": voices},
        )
        self.assertEqual(set(inputs.keys()), set(processor.model_input_names))
        self.assertEqual(inputs["audio_prompt_latents"].shape, (2, 5, 512))
        self.assertListEqual(inputs["audio_prompt_latents_mask"].sum(-1).tolist(), [5, 3])
        # Phonon's runtime: 3 frames after the end of speech for texts of at most 4 words, 1 otherwise
        self.assertListEqual(inputs["frames_after_eos"].tolist(), [1, 3])
        prompt, audio_bos = tokenizer.convert_tokens_to_ids([processor.audio_token, processor.audio_bos_token])
        # no voice BOS: the voice placeholders come first, the first audio position last
        row = inputs["input_ids"][1][inputs["attention_mask"][1].bool()].tolist()
        self.assertListEqual(row[:4], [prompt, prompt, prompt, row[3]])
        self.assertNotEqual(row[3], prompt)
        self.assertEqual(row[-1], audio_bos)

    def test_voice_audio_is_refused(self):
        processor = self.get_processor()
        with self.assertRaisesRegex(ValueError, "no encoder"):
            processor.apply_chat_template(
                conversation("Hello world", torch.zeros(1920).numpy()), tokenize=True, add_generation_prompt=True
            )

    def test_decode(self):
        processor = self.get_processor()
        frame_size = processor.audio_tokenizer.config.frame_size
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        waveforms = processor.decode(torch.randn(2, 4, 32), audio_latents_mask=mask)
        self.assertListEqual([waveform.shape[0] for waveform in waveforms], [4 * frame_size, 2 * frame_size])

    def test_processor_to_json_string(self):
        # the audio tokenizer is saved as a pointer to its checkpoint, not as a value
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        self.assertEqual(obj["audio_tokenizer"]["audio_tokenizer_class"], "PocketTTSAudioTokenizerModel")

    def test_save_load_pretrained_additional_features(self):
        # the common test's tokenizer kwargs do not apply to this tokenizer setup: check a feature extractor kwarg
        processor = self.processor_class.from_pretrained(self.tmpdirname, end_on_pause=False)
        self.assertFalse(processor.feature_extractor.end_on_pause)
        self.assertIsNone(processor.audio_tokenizer.encoder)
