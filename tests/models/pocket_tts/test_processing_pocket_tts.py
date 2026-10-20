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

from datasets import Audio, load_dataset
from huggingface_hub import download_bucket_files, hf_hub_download

from transformers import AutoProcessor, AutoTokenizer, PocketTTSFeatureExtractor, PocketTTSProcessor
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch
    from safetensors.torch import load_file

    from transformers import PocketTTSAudioTokenizerConfig, PocketTTSAudioTokenizerModel


def conversation(text, audio=None, voice=False):
    """A generation conversation: the voice as a recording (`audio`), as precomputed latents (`voice=True`, an empty
    `audio` item), or neither (the voice comes as a cache)."""
    content = [{"type": "text", "text": text}]
    if audio is not None:
        content.insert(0, {"type": "audio", "audio": audio})
    elif voice:
        content.insert(0, {"type": "audio"})
    return [{"role": "user", "content": content}]


def training_conversation(voice=None, text="hello world", target=None):
    """A training conversation: the voice (a recording, or `None` for precomputed latents), the text, the target."""
    content = [{"type": "audio", "audio": voice} if voice is not None else {"type": "audio"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    target = {"type": "audio", "audio": target} if target is not None else {"type": "audio"}
    return [{"role": "user", "content": content}, {"role": "assistant", "content": [target]}]


@require_torch
class PocketTTSProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = PocketTTSProcessor
    audio_input_name = "audio_prompt_latents"
    # the converted checkpoint's tokenizer and chat template (`from_pretrained(model_id)` would also forward its
    # kwargs to the audio tokenizer's `from_pretrained`)
    checkpoint_name = "eustlb/pocket-tts"

    @classmethod
    def _setup_tokenizer(cls):
        return AutoTokenizer.from_pretrained(cls.checkpoint_name)

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
            "chat_template": Path(hf_hub_download(cls.checkpoint_name, "chat_template.jinja")).read_text(
                encoding="utf-8"
            ),
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
        self.assertEqual(
            json.loads(processor.to_json_string())["audio_tokenizer"],
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

    def test_audio_tokenizer_halves(self):
        # an audio tokenizer loaded without its encoder or decoder: the processor names the missing half
        processor = self.get_processor()
        audio = torch.randn(processor.feature_extractor.sampling_rate).numpy() * 0.1
        processor.audio_tokenizer = PocketTTSAudioTokenizerModel.from_pretrained(
            self._audio_tokenizer_dir, decoder_config=None
        )
        with self.assertRaisesRegex(ValueError, "decoder_config=None"):
            processor.decode(torch.randn(2, 32))

        # decoder only: precomputed voices and decoding work, cloning from audio is refused
        processor.audio_tokenizer = PocketTTSAudioTokenizerModel.from_pretrained(
            self._audio_tokenizer_dir, encoder_config=None
        )
        self.assertEqual(processor.decode(torch.randn(2, 32)).shape[0], 2 * processor.feature_extractor.frame_size)
        inputs = processor.apply_chat_template(
            conversation("Hello world", voice=True),
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            processor_kwargs={"audio_prompt_latents": [torch.randn(3, 32)]},
        )
        self.assertEqual(inputs["audio_prompt_latents"].shape, (1, 3, 32))
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

    def test_chat_template(self):
        processor = self.get_processor()
        voice_prompt = processor.audio_prompt_bos_token + processor.audio_token
        # the voice prompt first (none when the voice comes as a cache); `add_generation_prompt` appends the first
        # audio position, from which the model generates. The tokenizer prepares the text.
        render = processor.apply_chat_template
        self.assertEqual(render(conversation(" hello world ")), "hello world")
        self.assertEqual(render(conversation("hello world", voice=True)), voice_prompt + "hello world")
        self.assertEqual(
            render(conversation("hello world"), add_generation_prompt=True), "hello world" + processor.audio_bos_token
        )
        # training: the text as written, the target after the first audio position
        self.assertEqual(
            render(training_conversation()),
            voice_prompt + "hello world" + processor.audio_bos_token + processor.audio_token,
        )

    def test_chat_template_rejects_other_conversations(self):
        processor = self.get_processor()
        for invalid in (
            [{"role": "system", "content": "You are a TTS."}, *conversation("Hello")],
            [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}],
            conversation("   "),
        ):
            with self.subTest(conversation=invalid), self.assertRaises(Exception):
                processor.apply_chat_template(invalid)

    def test_call_normalizes_text(self):
        # the tokenizer prepares the text around the placeholders, the transcript of a training sample too
        processor = self.get_processor()
        tokenizer = processor.tokenizer
        prompt = processor.apply_chat_template(conversation("hello world", voice=True), add_generation_prompt=True)
        inputs = processor(text=prompt, audio_prompt_latents=[torch.randn(2, 32)])
        voice_prompt = processor.audio_prompt_bos_token + processor.audio_token * 2
        expected = tokenizer(voice_prompt + "Hello world." + processor.audio_bos_token)["input_ids"]
        self.assertListEqual(inputs["input_ids"][0].tolist(), expected)
        self.assertEqual(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True), "Hello world.")

        inputs = processor.apply_chat_template(
            training_conversation(text="hello world"),
            tokenize=True,
            return_dict=True,
            processor_kwargs={"audio_prompt_latents": [torch.randn(2, 32)], "labels": [torch.randn(1, 32)]},
        )
        self.assertEqual(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True), "Hello world.")

    def test_call_with_voice_latents(self):
        processor = self.get_processor()
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
        # voices right-padded to the longest, with their mask
        self.assertEqual(inputs["audio_prompt_latents"].shape, (2, 5, 32))
        self.assertListEqual(inputs["audio_prompt_latents_mask"].sum(-1).tolist(), [5, 3])
        torch.testing.assert_close(inputs["audio_prompt_latents"][1, :3], voices[1])
        # one placeholder per voice frame; prompts left-padded; short texts get a longer tail after end of speech
        self.assertListEqual((inputs["input_ids"] == processor.audio_token_id).sum(-1).tolist(), [5, 3])
        self.assertListEqual(inputs["attention_mask"][:, 0].tolist(), [1, 0])
        self.assertListEqual(inputs["frames_after_eos"].tolist(), [2, 4])

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
        # one latent frame, and one placeholder, per (started) audio tokenizer frame
        num_frames = [-(-len(waveform) // processor.feature_extractor.frame_size) for waveform in audio]
        self.assertListEqual(inputs["audio_prompt_latents_mask"].sum(-1).tolist(), num_frames)
        self.assertListEqual((inputs["input_ids"] == processor.audio_token_id).sum(-1).tolist(), num_frames)
        # batched with a shorter, padded waveform: the latents of the waveform alone
        waveform = torch.nn.functional.pad(torch.as_tensor(audio[1]), (0, -len(audio[1]) % 1920))
        with torch.no_grad():
            expected = processor.audio_tokenizer.encode(waveform[None, None]).latents[0]
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
        inputs = processor.apply_chat_template(
            training_conversation(voice, target=target), tokenize=True, return_dict=True
        )
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
        # one voice placeholder per voice frame, one latent position per target frame
        latent_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_latent_token)
        self.assertEqual((inputs["input_ids"] == processor.audio_token_id).sum().item(), 3)
        self.assertEqual((inputs["input_ids"] == latent_token_id).sum().item(), 5)
        # voice and target are encoded in one batch, with no pause appended in training
        with torch.no_grad():
            expected_voice = processor.audio_tokenizer.encode(torch.as_tensor(voice)[None, None]).latents[0]
            expected_target = processor.audio_tokenizer.encode(torch.as_tensor(target)[None, None]).latents[0]
        torch.testing.assert_close(inputs["audio_prompt_latents"][0], expected_voice, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(inputs["labels"][0], expected_target, atol=1e-5, rtol=1e-5)

    def test_training_classifier_free_guidance_rows(self):
        processor = self.get_processor()
        latent_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_latent_token)
        voice_dropped = training_conversation(text="Hello there.")
        voice_dropped[0]["content"].pop(0)
        conversations = [training_conversation(text="Hello world."), voice_dropped, training_conversation(text=None)]
        # precomputed voices (one per voiced row) and targets
        voices = [torch.randn(2, 32), torch.randn(4, 32)]
        targets = [torch.randn(3, 32), torch.randn(1, 32), torch.randn(2, 32)]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            processor_kwargs={"audio_prompt_latents": voices, "labels": targets},
        )
        # the voices go to the voiced rows, in order
        self.assertListEqual(inputs["audio_prompt_latents_mask"].sum(-1).tolist(), [2, 4])
        self.assertListEqual((inputs["input_ids"] == processor.audio_token_id).sum(-1).tolist(), [2, 0, 4])
        self.assertListEqual(inputs["labels_mask"].sum(-1).tolist(), [3, 1, 2])
        self.assertListEqual((inputs["input_ids"] == latent_token_id).sum(-1).tolist(), [3, 1, 2])
        torch.testing.assert_close(inputs["audio_prompt_latents"][1, :4], voices[1])
        torch.testing.assert_close(inputs["labels"][1, :1], targets[1])

    def test_decode(self):
        processor = self.get_processor()
        frame_size = processor.audio_tokenizer.config.frame_size
        # a padded batch, as `generate` returns it, with the mask of each row's frames
        latents = torch.randn(2, 4, 32)
        audio = processor.decode(latents, audio_latents_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]))
        self.assertListEqual([waveform.shape[0] for waveform in audio], [4 * frame_size, 2 * frame_size])
        # one sample gives one waveform, the same as in the batch (the decoder is causal), faded in
        single = processor.decode(latents[1, :2])
        torch.testing.assert_close(single, audio[1], atol=1e-5, rtol=0)
        self.assertEqual(single[0].item(), 0.0)


@slow
@require_torch
class PocketTTSProcessingIntegrationTest(unittest.TestCase):
    """
    Check against the original (kyutai-labs/pocket-tts @ b94db17): its text preparation, tokenization, end-of-speech tail
    and sentence splitting, and the voice latents it encodes from a recording.

    reproducers (upload the goldens to ``hf://buckets/hf-internal-testing/pocket-tts-integration-tests/<case>/``):
        https://gist.github.com/eustlb/cf6dd7b0159c162330d435550455d609#file-reproducer-processing-py
        https://gist.github.com/eustlb/cf6dd7b0159c162330d435550455d609#file-reproducer-voice-cloning-py
        https://gist.github.com/eustlb/cf6dd7b0159c162330d435550455d609#file-reproducer-voice-cloning-batch-py
    """

    # the audio tokenizer's voice latents against the original's (on CUDA): up to 1.1e-5 measured
    VOICE_ATOL = 5e-5

    # --- single source of truth, mirrored from the reproducers' common.py ---
    VOICE_CLONING_AUDIO = "en-Alice_woman.wav"
    VOICE_CLONING_BATCH_AUDIO = ["en-Alice_woman.wav", "en-Carter_man.wav"]

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_name = "eustlb/pocket-tts"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_name, device_map="auto")
        cls.bucket = "hf-internal-testing/pocket-tts-integration-tests"

    def _load_expected(self, case, tensors=False):
        files = ["metadata.json"] + (["expected.safetensors"] if tensors else [])
        with tempfile.TemporaryDirectory() as tmp_dir:
            download_bucket_files(
                self.bucket, files=[(f"{case}/{name}", os.path.join(tmp_dir, name)) for name in files]
            )
            with open(os.path.join(tmp_dir, "metadata.json"), encoding="utf-8") as f:
                metadata = json.load(f)
            return (load_file(os.path.join(tmp_dir, "expected.safetensors")), metadata) if tensors else metadata

    def _load_voice_audio(self, filename):
        """A voice recording, decoded and resampled to 24 kHz mono by `datasets`, as the reproducers feed it to the original."""
        dataset = load_dataset("hf-internal-testing/dummy-audio-samples", data_files=[filename], split="train")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=24000, num_channels=1))
        return dataset[0]["audio"]["array"]

    def _apply_chat_template(self, conversations, **processor_kwargs):
        return self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            processor_kwargs=processor_kwargs,
        )

    def test_apply_chat_template_text(self):
        metadata = self._load_expected("processing")
        audio_bos_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.audio_bos_token)
        for text, expected in zip(metadata["inputs"]["texts"], metadata["prompts"]):
            with self.subTest(text=text):
                inputs = self._apply_chat_template(conversation(text))
                prepared_text = self.processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                self.assertEqual(prepared_text, expected["prepared_text"])
                self.assertListEqual(inputs["input_ids"][0].tolist(), expected["token_ids"] + [audio_bos_token_id])
                # the original keeps the end-of-speech frame + (its `frames_after_eos` - 1) frames
                self.assertEqual(inputs["frames_after_eos"].item(), expected["frames_after_eos"] - 1)

    def test_split_text(self):
        metadata = self._load_expected("processing")
        chunks = self.processor.split_text(metadata["inputs"]["long_text"], max_tokens=50)
        self.assertListEqual(chunks, metadata["chunks"])

    def test_apply_chat_template_voice_audio(self):
        expected, metadata = self._load_expected("voice_cloning", tensors=True)
        audio = self._load_voice_audio(self.VOICE_CLONING_AUDIO)
        inputs = self._apply_chat_template(conversation(metadata["inputs"]["text"], audio))
        num_frames = metadata["num_voice_frames"]
        self.assertEqual((inputs["input_ids"] == self.processor.audio_token_id).sum().item(), num_frames)
        torch.testing.assert_close(
            inputs["audio_prompt_latents"][0], expected["audio_prompt_latents"], rtol=0, atol=self.VOICE_ATOL
        )

    def test_apply_chat_template_voice_audio_batch(self):
        # two recordings of different lengths (9.3 s and 27.7 s): padding does not change a row's voice latents
        expected, metadata = self._load_expected("voice_cloning_batch", tensors=True)
        audio = [self._load_voice_audio(filename) for filename in self.VOICE_CLONING_BATCH_AUDIO]
        text = metadata["inputs"]["text"]
        inputs = self._apply_chat_template([conversation(text, audio[0]), conversation(text, audio[1])])
        for row, row_metadata in enumerate(metadata["rows"]):
            num_frames = row_metadata["num_voice_frames"]
            self.assertEqual(inputs["audio_prompt_latents_mask"][row].sum().item(), num_frames)
            torch.testing.assert_close(
                inputs["audio_prompt_latents"][row, :num_frames],
                expected[f"audio_prompt_latents_{row}"],
                rtol=0,
                atol=self.VOICE_ATOL,
            )
