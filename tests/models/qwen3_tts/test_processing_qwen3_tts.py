# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Tests for Qwen3TTSProcessor."""

import copy
import json
import os
import tempfile
import unittest

import numpy as np
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    Qwen2TokenizerFast,
    Qwen3TTSFeatureExtractor,
    Qwen3TTSProcessor,
    is_torch_available,
)
from transformers.testing_utils import require_accelerate, require_torch, slow
from transformers.trainer_utils import set_seed
from transformers.utils import is_soundfile_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Qwen3TTSForConditionalGeneration,
        Qwen3TTSTokenizerConfig,
        Qwen3TTSTokenizerModel,
        Trainer,
        TrainingArguments,
    )

    from .test_modeling_qwen3_tts import Qwen3TTSModelTester


def _build_tiny_audio_tokenizer(num_quantizers=4):
    """Build a tiny Qwen3TTSTokenizerModel for decode/save_audio tests."""
    encoder_config = {
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "num_filters": 8,
        "kernel_size": 7,
        "residual_kernel_size": 3,
        "last_kernel_size": 3,
        "num_residual_layers": 1,
        "upsampling_ratios": [8, 6],
        "codebook_size": 8,
        "codebook_dim": 4,
        "vector_quantization_hidden_dimension": 4,
        "num_quantizers": num_quantizers,
        "num_semantic_quantizers": 1,
        "upsample_groups": 8,
    }
    decoder_config = {
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 32,
        "num_quantizers": num_quantizers,
        "codebook_size": 8,
        "codebook_dim": 8,
        "latent_dim": 16,
        "decoder_dim": 32,
        "upsample_rates": [2, 2],
        "upsampling_ratios": [2, 2],
    }
    config = Qwen3TTSTokenizerConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        encoder_valid_num_quantizers=num_quantizers,
    )
    return Qwen3TTSTokenizerModel(config).eval()


@require_torch
class Qwen3TTSProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen3TTSProcessor
    model_id = None

    @classmethod
    def _setup_tokenizer(cls):
        return Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2-0.5B")

    @classmethod
    def _setup_feature_extractor(cls):
        return Qwen3TTSFeatureExtractor()

    @unittest.skip(reason="Qwen3TTS is a TTS processor; audio chat template tests are not applicable")
    def test_apply_chat_template_audio_0(self):
        pass

    @unittest.skip(reason="Qwen3TTS is a TTS processor; audio chat template tests are not applicable")
    def test_apply_chat_template_audio_1(self):
        pass

    @unittest.skip(reason="Qwen3TTS is a TTS processor; audio chat template tests are not applicable")
    def test_apply_chat_template_audio_2(self):
        pass

    @unittest.skip(reason="Qwen3TTS is a TTS processor; audio chat template tests are not applicable")
    def test_apply_chat_template_audio_3(self):
        pass

    @unittest.skip(reason="Qwen3TTS chat templates format synthesis text and do not process audio inputs")
    def test_chat_template_audio_sampling_rate(self):
        pass

    @unittest.skip(reason="Qwen3TTS chat template returns a list, not a plain string")
    def test_chat_template_jinja_kwargs(self):
        pass

    @unittest.skip(reason="decode/batch_decode are overridden to decode audio codes, not text tokens")
    def test_tokenizer_decode_defaults(self):
        pass

    @unittest.skip(reason="decode/batch_decode are overridden to decode audio codes, not text tokens")
    def test_apply_chat_template_assistant_mask(self):
        pass

    @unittest.skip(reason="Qwen3TTS apply_chat_template is custom and does not use chat_template serialization")
    def test_chat_template_save_loading(self):
        pass

    @unittest.skip(reason="Qwen3TTS __call__ accepts text/audio, not the generic multimodal argument set")
    def test_model_input_names(self):
        pass

    @unittest.skip(reason="Qwen3TTS is a text/audio processor")
    def test_processor_text_has_no_visual(self):
        pass

    @unittest.skip(reason="Qwen3TTS has no multimodal token counting helper")
    def test_get_num_multimodal_tokens_matches_processor_call(self):
        pass

    def test_call_text(self):
        processor = self.get_processor()
        text = "Hello there."

        inputs = processor(text=text, return_tensors="pt")
        tokenizer_inputs = processor.tokenizer([text], padding=False, padding_side="left", return_tensors="pt")

        self.assertEqual(set(inputs.keys()), set(tokenizer_inputs.keys()))
        for key in tokenizer_inputs:
            torch.testing.assert_close(inputs[key], tokenizer_inputs[key])

    def test_call_audio(self):
        processor = self.get_processor()
        audio = np.zeros(2048, dtype=np.float32)

        inputs = processor(audio=audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt")
        feature_inputs = processor.feature_extractor(
            audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt"
        )

        self.assertEqual(set(inputs.keys()), {"input_features"})
        torch.testing.assert_close(inputs["input_features"], feature_inputs["input_features"])

    def test_call_cached_training_inputs(self):
        processor = self.get_processor()
        text = ["Hello there.", "Hello there."]
        audio_codes = torch.randint(0, 8, (2, 3, 4))
        speaker_embeddings = torch.randn(2, 32)
        expected_text = processor.tokenizer(
            [processor._build_synthesis_text(value) for value in text], return_tensors="pt"
        )
        for codes, embeddings in (
            (audio_codes, speaker_embeddings),
            (list(audio_codes), list(speaker_embeddings)),
        ):
            with self.subTest(input_type=type(codes).__name__):
                inputs = processor(text=text, audio_codes=codes, speaker_embeddings=embeddings, return_tensors="pt")
                self.assertEqual(
                    set(inputs), set(expected_text) | {"audio_codes", "audio_attention_mask", "speaker_embeddings"}
                )
                for name, value in expected_text.items():
                    torch.testing.assert_close(inputs[name], value)
                torch.testing.assert_close(inputs.audio_codes, audio_codes)
                torch.testing.assert_close(inputs.audio_attention_mask, torch.ones(2, 3, dtype=torch.long))
                torch.testing.assert_close(inputs.speaker_embeddings, speaker_embeddings)

        single = processor(text=text[0], audio_codes=audio_codes[0], speaker_embeddings=speaker_embeddings[0])
        generation = processor.apply_chat_template([{"role": "user", "content": text[0]}])
        torch.testing.assert_close(single.input_ids, generation.input_ids[0])
        torch.testing.assert_close(single.audio_codes, audio_codes[:1])
        torch.testing.assert_close(single.speaker_embeddings, speaker_embeddings[:1])

    def test_call_cached_training_input_validation(self):
        processor = self.get_processor()
        inputs = {
            "text": "Hello there.",
            "audio_codes": torch.zeros(3, 4, dtype=torch.long),
            "speaker_embeddings": torch.zeros(32),
        }
        for overrides, message in (
            ({"text": None}, "require `text`"),
            ({"audio_codes": None}, "require `text`"),
            ({"speaker_embeddings": None}, "require `text`"),
            ({"audio": np.zeros(2048, dtype=np.float32)}, "not both"),
            ({"return_tensors": "np"}, "return_tensors='pt'"),
            ({"audio_codes": torch.zeros(3, 4)}, "integer token IDs"),
            ({"audio_codes": torch.zeros(0, 4, dtype=torch.long)}, "audio_length"),
            ({"audio_codes": torch.zeros(2, 3, 4, dtype=torch.long)}, "per text prompt"),
            ({"speaker_embeddings": torch.zeros(2, 32)}, "per text prompt"),
            ({"truncation": True}, "cannot be truncated"),
            (
                {
                    "text": ["Hello.", "Hello."],
                    "audio_codes": [torch.zeros(3, 4, dtype=torch.long), torch.zeros(3, 2, dtype=torch.long)],
                },
                "same number of codebooks",
            ),
        ):
            with self.subTest(overrides=list(overrides)):
                with self.assertRaisesRegex(ValueError, message):
                    processor(**{**inputs, **overrides})

    def test_call_cached_training_padding_and_labels(self):
        processor = self.get_processor()
        text = ["Hi.", "This sentence contains more words."]
        audio_codes = [torch.arange(12, dtype=torch.int32).reshape(3, 4) % 8, np.arange(20).reshape(5, 4) % 8]
        speaker_embeddings = torch.randn(2, 32)
        expected_audio_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        for padding_side in ("left", "right"):
            with self.subTest(padding_side=padding_side):
                kwargs = {
                    "text_kwargs": {"padding_side": padding_side, "pad_to_multiple_of": 8, "return_tensors": "pt"}
                }
                inputs = processor(
                    text=text,
                    audio_codes=audio_codes,
                    speaker_embeddings=speaker_embeddings,
                    output_labels=True,
                    **kwargs,
                )
                expected_text = processor.tokenizer(
                    [processor._build_synthesis_text(value) for value in text],
                    padding=True,
                    padding_side=padding_side,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                for name, value in expected_text.items():
                    torch.testing.assert_close(inputs[name], value)
                torch.testing.assert_close(inputs.audio_attention_mask, expected_audio_mask)
                self.assertEqual(inputs.audio_codes.shape, (2, 5, 4))
                self.assertEqual(inputs.labels.dtype, torch.long)
                for index, codes in enumerate(audio_codes):
                    length = len(codes)
                    expected_codes = torch.as_tensor(codes).long()
                    torch.testing.assert_close(inputs.audio_codes[index, :length], expected_codes)
                    torch.testing.assert_close(inputs.labels[index, :length], expected_codes)
                    self.assertTrue((inputs.audio_codes[index, length:] == 0).all())
                    self.assertTrue((inputs.labels[index, length:] == -100).all())
                self.assertEqual(inputs.labels[0, 0, 0].item(), 0)
                inputs.labels[0, 0, 0] = -100
                self.assertEqual(inputs.audio_codes[0, 0, 0].item(), 0)
                self.assertEqual(audio_codes[0][0, 0].item(), 0)

        without_labels = processor(text=text, audio_codes=audio_codes, speaker_embeddings=speaker_embeddings)
        self.assertNotIn("labels", without_labels)
        self.assertIn("audio_attention_mask", processor.model_input_names)
        with self.assertRaisesRegex(ValueError, "requires cached training inputs"):
            processor(text=text, output_labels=True)

    def test_cached_training_batch_forward_backward(self):
        processor = self.get_processor()
        config = Qwen3TTSModelTester(self).get_config()
        config.talker_config.text_vocab_size = len(processor.tokenizer)
        model = Qwen3TTSForConditionalGeneration(config).train()
        inputs = processor(
            text=["Hi.", "This is a longer training example."],
            audio_codes=[torch.randint(0, 64, (1, 2)), torch.randint(0, 64, (3, 2))],
            speaker_embeddings=torch.randn(2, 32),
            output_labels=True,
        )
        output = model(**inputs)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for parameter in (
            model.text_projection.linear_1.weight,
            model.codec_head.weight,
            model.code_predictor.lm_head.weight,
        ):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def prepare_trainer_inputs(self):
        set_seed(42)
        processor = self.get_processor()
        config = Qwen3TTSModelTester(self).get_config()
        config.talker_config.text_vocab_size = len(processor.tokenizer)
        model = Qwen3TTSForConditionalGeneration(config)
        texts = ["Hi.", "A longer sentence.", "Hello.", "This example has the longest audio sequence."]
        examples = [
            {
                "text": text,
                "audio_codes": (torch.arange(length * 2).reshape(length, 2) + index) % 64,
                "speaker_embeddings": torch.linspace(-1, 1, 32) + index / 10,
            }
            for index, (text, length) in enumerate(zip(texts, (1, 2, 4, 8)))
        ]

        def collate(examples):
            return processor(
                text=[example["text"] for example in examples],
                audio_codes=[example["audio_codes"] for example in examples],
                speaker_embeddings=[example["speaker_embeddings"] for example in examples],
                output_labels=True,
            )

        return model, processor, examples, collate

    def get_training_args(self, output_dir, **kwargs):
        return TrainingArguments(
            output_dir=output_dir,
            use_cpu=True,
            max_steps=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=0.05,
            optim="sgd",
            lr_scheduler_type="constant",
            max_grad_norm=0,
            remove_unused_columns=False,
            label_smoothing_factor=0,
            prediction_loss_only=True,
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            disable_tqdm=True,
            **kwargs,
        )

    @require_accelerate
    def test_trainer_gradient_accumulation_matches_manual_microbatches(self):
        for accumulation_steps in (1, 2, 3):
            with self.subTest(gradient_accumulation_steps=accumulation_steps):
                model, processor, examples, collate = self.prepare_trainer_inputs()
                examples = examples[: 2 * accumulation_steps]
                parameters = [
                    model.text_projection.linear_1.weight,
                    model.codec_head.weight,
                    model.code_predictor.lm_head.weight,
                ]
                before = [parameter.detach().clone() for parameter in parameters]
                reference_model = copy.deepcopy(model).train()
                batches = []

                def recording_collator(examples):
                    batch = collate(examples)
                    batches.append({name: value.detach().clone() for name, value in batch.items()})
                    return batch

                with tempfile.TemporaryDirectory() as directory:
                    trainer = Trainer(
                        model=model,
                        args=self.get_training_args(directory, gradient_accumulation_steps=accumulation_steps),
                        train_dataset=examples,
                        data_collator=recording_collator,
                        processing_class=processor,
                    )
                    self.assertFalse(trainer.model_accepts_loss_kwargs)
                    result = trainer.train()

                self.assertEqual(result.global_step, 1)
                self.assertTrue(np.isfinite(result.training_loss))
                self.assertGreater(result.training_loss, 0)
                for original, parameter in zip(before, parameters):
                    self.assertTrue(torch.isfinite(parameter).all())
                    self.assertFalse(torch.equal(original, parameter))
                self.assertEqual(len(batches), 1 if accumulation_steps == 1 else 2)
                if accumulation_steps > 1:
                    self.assertNotEqual(
                        batches[0]["audio_attention_mask"].sum().item(),
                        batches[1]["audio_attention_mask"].sum().item(),
                    )
                optimizer = torch.optim.SGD(reference_model.parameters(), lr=trainer.args.learning_rate)
                losses = []
                # Each stage is independently mean-reduced, so accumulation averages microbatch losses.
                for batch in batches:
                    loss = reference_model(**batch).loss
                    losses.append(loss.detach())
                    (loss / len(batches)).backward()
                optimizer.step()

                self.assertAlmostEqual(result.training_loss, torch.stack(losses).mean().item(), places=5)
                reference_parameters = dict(reference_model.named_parameters())
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, reference_parameters[name], atol=1e-6, rtol=1e-5, msg=name)

    @require_accelerate
    def test_trainer_evaluation_loss_matches_manual_batches(self):
        model, processor, examples, collate = self.prepare_trainer_inputs()
        model.eval()
        with torch.no_grad():
            losses = [model(**collate(examples[index : index + 2])).loss for index in range(0, len(examples), 2)]
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=self.get_training_args(directory),
                eval_dataset=examples,
                data_collator=collate,
                processing_class=processor,
            )
            metrics = trainer.evaluate()
        self.assertTrue(np.isfinite(metrics["eval_loss"]))
        self.assertGreater(metrics["eval_loss"], 0)
        self.assertAlmostEqual(metrics["eval_loss"], torch.stack(losses).mean().item(), places=5)

    @require_accelerate
    def test_trainer_fully_ignored_targets(self):
        model, processor, examples, collate = self.prepare_trainer_inputs()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

        def ignored_collator(examples):
            batch = collate(examples)
            batch["labels"].fill_(-100)
            return batch

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=self.get_training_args(directory),
                train_dataset=examples,
                eval_dataset=examples,
                data_collator=ignored_collator,
                processing_class=processor,
            )
            metrics = trainer.evaluate()
            result = trainer.train()
        self.assertEqual(metrics["eval_loss"], 0)
        self.assertEqual(result.global_step, 1)
        self.assertEqual(result.training_loss, 0)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(parameter).all(), name)
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0, msg=name)

    def test_apply_chat_template_basic(self):
        processor = self.get_processor()
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": "Hello there."}]},
        ]
        inputs = processor.apply_chat_template(conversation)

        self.assertEqual(len(inputs["input_ids"]), 1)
        self.assertEqual(inputs["input_ids"][0].dim(), 2)
        self.assertEqual(inputs["languages"], ["auto"])
        self.assertEqual(inputs["speakers"], [None])
        self.assertNotIn("instruct_ids", inputs)

    def test_apply_chat_template_language_and_speaker(self):
        processor = self.get_processor()
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Welcome."}],
                "language": "English",
                "speaker": "Ryan",
            },
        ]
        inputs = processor.apply_chat_template(conversation)

        self.assertEqual(inputs["languages"], ["English"])
        self.assertEqual(inputs["speakers"], ["Ryan"])

    def test_apply_chat_template_instruct(self):
        processor = self.get_processor()
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "A calm female voice."}]},
            {"role": "user", "content": [{"type": "text", "text": "Good morning."}], "language": "English"},
        ]
        inputs = processor.apply_chat_template(conversation)

        self.assertIn("instruct_ids", inputs)
        self.assertEqual(len(inputs["instruct_ids"]), 1)
        self.assertEqual(inputs["languages"], ["English"])

    def test_apply_chat_template_batch(self):
        processor = self.get_processor()
        conversations = [
            [{"role": "user", "content": [{"type": "text", "text": "First sentence."}]}],
            [{"role": "user", "content": [{"type": "text", "text": "Second sentence."}]}],
        ]
        inputs = processor.apply_chat_template(conversations)

        self.assertEqual(len(inputs["input_ids"]), 2)
        self.assertEqual(inputs["languages"], ["auto", "auto"])

    def test_apply_chat_template_plain_string_content(self):
        processor = self.get_processor()
        conversation = [{"role": "user", "content": "Plain string content."}]
        inputs = processor.apply_chat_template(conversation)

        self.assertEqual(len(inputs["input_ids"]), 1)

    def test_apply_chat_template_requires_user_message(self):
        processor = self.get_processor()
        conversation = [{"role": "system", "content": [{"type": "text", "text": "Only an instruction."}]}]
        with self.assertRaises(ValueError):
            processor.apply_chat_template(conversation)

    def test_decode_batch_and_save_audio(self):
        if not is_soundfile_available():
            self.skipTest("soundfile is required to save audio")
        processor = self.get_processor()
        processor.audio_tokenizer = _build_tiny_audio_tokenizer(num_quantizers=4)

        codes = [torch.randint(0, 8, (6, 4)), torch.randint(0, 8, (5, 4))]
        audios = processor.decode(codes)

        self.assertEqual(len(audios), 2)
        self.assertTrue(all(isinstance(a, torch.Tensor) for a in audios))

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [os.path.join(tmpdir, "out_0.wav"), os.path.join(tmpdir, "out_1.wav")]
            processor.save_audio(audios, paths)
            self.assertTrue(all(os.path.isfile(p) for p in paths))

    def test_decode_single(self):
        processor = self.get_processor()
        processor.audio_tokenizer = _build_tiny_audio_tokenizer(num_quantizers=4)

        codes = torch.randint(0, 8, (6, 4))
        audio = processor.decode(codes)

        self.assertIsInstance(audio, torch.Tensor)

    def test_save_load_audio_tokenizer_reference(self):
        processor = self.get_processor()
        with tempfile.TemporaryDirectory() as directory:
            audio_directory = os.path.join(directory, "audio_tokenizer")
            processor_directory = os.path.join(directory, "processor")
            _build_tiny_audio_tokenizer().save_pretrained(audio_directory)
            processor.audio_tokenizer = Qwen3TTSTokenizerModel.from_pretrained(audio_directory)
            processor.save_pretrained(processor_directory)
            with open(os.path.join(processor_directory, "processor_config.json"), encoding="utf-8") as file:
                metadata = json.load(file)
            self.assertEqual(
                metadata["audio_tokenizer"],
                {
                    "audio_tokenizer_class": "Qwen3TTSTokenizerModel",
                    "audio_tokenizer_name_or_path": audio_directory,
                },
            )
            self.assertFalse(os.path.exists(os.path.join(processor_directory, "model.safetensors")))
            restored = AutoProcessor.from_pretrained(processor_directory)
            self.assertIsInstance(restored, Qwen3TTSProcessor)
            self.assertIsInstance(restored.audio_tokenizer, Qwen3TTSTokenizerModel)
            codes = torch.randint(0, 8, (6, 4))
            torch.testing.assert_close(restored.decode(codes), processor.decode(codes))

    @parameterized.expand(
        ["shahvandit/qwen3-tts-base-hf", "shahvandit/qwen3-tts-customvoice-hf", "shahvandit/qwen3-tts-voicedesign-hf"]
    )
    @slow
    def test_can_load_processor_from_pretrained(self, checkpoint):
        processor = AutoProcessor.from_pretrained(checkpoint)
        self.assertIsInstance(processor, Qwen3TTSProcessor)
        self.assertIsNotNone(processor.tokenizer)
        self.assertIsNotNone(processor.feature_extractor)
        self.assertIsInstance(processor.audio_tokenizer, Qwen3TTSTokenizerModel)
        self.assertEqual(
            processor.to_dict()["audio_tokenizer"],
            {
                "audio_tokenizer_class": "Qwen3TTSTokenizerModel",
                "audio_tokenizer_name_or_path": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            },
        )
        # All variants reference the same codec, so exercise its round-trip only once.
        if checkpoint != "shahvandit/qwen3-tts-base-hf":
            return
        sampling_rate = processor.audio_tokenizer.config.input_sampling_rate
        input_values = torch.sin(torch.arange(sampling_rate) * (2 * torch.pi * 440 / sampling_rate))[None]
        with torch.no_grad():
            codes = processor.audio_tokenizer.encode(input_values).audio_codes[0]
            audio = processor.decode(codes)
        self.assertEqual(codes.shape[-1], processor.audio_tokenizer.config.encoder_config.valid_num_quantizers)
        self.assertEqual(audio.ndim, 1)
        self.assertGreaterEqual(audio.numel(), sampling_rate)
        self.assertTrue(torch.isfinite(audio).all())
