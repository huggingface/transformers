# coding=utf-8
# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

"""Testing suite for the PyTorch AudioFlamingo3 model."""

import tempfile
import unittest
from io import BytesIO
from urllib.request import urlopen

import librosa
import pytest

from transformers import (
    AudioFlamingo3Config,
    AudioFlamingo3ForConditionalGeneration,
    AutoProcessor,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


class AudioFlamingo3ModelTester:
    """
    Builds a tiny AudioFlamingo3 config and synthetic inputs that respect AF3's
    post-pool token accounting: num <sound> tokens per sample == post-pool frame count.
    """

    def __init__(
        self,
        parent,
        audio_token_id=0,
        seq_length=25,
        feat_seq_length=60,
        text_config=None,
        audio_config=None,
        is_training=True,
    ):
        self.parent = parent
        self.audio_token_id = audio_token_id
        self.seq_length = seq_length
        self.feat_seq_length = feat_seq_length
        self.is_training = is_training

        # Small text backbone (Qwen2-ish)
        if text_config is None:
            text_config = {
                "model_type": "qwen2",
                "intermediate_size": 36,
                "initializer_range": 0.02,
                "hidden_size": 32,
                "max_position_embeddings": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "use_mrope": False,
                "vocab_size": 99,
            }
        # Small audio encoder (AF3 Whisper-style)
        if audio_config is None:
            audio_config = {
                "model_type": "audioflamingo3_encoder",
                "d_model": 16,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 16,
                "encoder_layers": 2,
                "num_mel_bins": 80,
                "max_source_positions": 30,
                "init_std": 0.02,
                "avg_pool_kernel_size": 2,
                "avg_pool_stride": 2,
            }

        self.text_config = text_config
        self.audio_config = audio_config

        self.batch_size = 3
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.encoder_seq_length = seq_length

    def get_config(self):
        return AudioFlamingo3Config(
            text_config=self.text_config,
            audio_config=self.audio_config,
            audio_token_id=self.audio_token_id,
        )

    def prepare_config_and_inputs(self):
        # (#windows == batch_size, n_mels, T_mel)
        input_features_values = floats_tensor(
            [self.batch_size, self.audio_config["num_mel_bins"], self.feat_seq_length]
        )
        config = self.get_config()
        # Per-window mel validity (all ones => full length)
        feature_attention_mask = torch.ones([self.batch_size, self.feat_seq_length], dtype=torch.long).to(torch_device)
        return config, input_features_values, feature_attention_mask

    def _post_pool_tokens_per_window(self, T_mel):
        # Mirror AF3 processor math:
        # pre = (L_mel - 1)//2 + 1, post = (pre - 2)//2 + 1
        pre = (T_mel - 1) // 2 + 1
        post = (pre - 2) // 2 + 1
        return post

    def prepare_config_and_inputs_for_common(self):
        config, input_features_values, feature_attention_mask = self.prepare_config_and_inputs()
        # Every window has same T_mel here
        num_audio_tokens_per_sample = self._post_pool_tokens_per_window(input_features_values.shape[-1])

        # Build token ids with left padding sentinel and K <sound> tokens
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=torch_device)
        attention_mask[:, :1] = 0  # left padding sentinel

        # Fill first K positions (after padding) with the audio token id, for each sample
        input_ids[:, 1 : 1 + num_audio_tokens_per_sample] = config.audio_token_id

        inputs_dict = {
            "input_features": input_features_values,
            "feature_attention_mask": feature_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class AudioFlamingo3ForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `AudioFlamingo3ForConditionalGeneration`.
    """

    all_model_classes = (AudioFlamingo3ForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = AudioFlamingo3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AudioFlamingo3Config, has_text_modality=False)

    @unittest.skip(reason="Compile not yet supported for AudioFlamingo3 models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported for AudioFlamingo3 models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="AudioFlamingo3 tests avoid right-padding equivalence; fusion is in-place.")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    def test_sdpa_can_dispatch_composite_models(self):
        # AF3 is audio+text composite; verify SDPA toggles propagate to submodules.
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # SDPA (default)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                text_attn = "sdpa" if model.language_model._supports_sdpa else "eager"
                audio_attn = "sdpa" if model.audio_tower._supports_sdpa else "eager"

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(model.language_model.config._attn_implementation == text_attn)
                self.assertTrue(model.audio_tower.config._attn_implementation == audio_attn)

                # Eager
                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.audio_tower.config._attn_implementation == "eager")

                for _, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")


@require_torch
class AudioFlamingo3ForConditionalGenerationIntegrationTest(unittest.TestCase):
    """
    Slow tests against the public checkpoint to validate processor-model alignment and in-place fusion.
    """

    def setUp(self):
        # Public AF3 checkpoint
        self.checkpoint = "lashahub/audio-flamingo-3"
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_integration_single_sample_alignment_and_mismatch(self):
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(self.checkpoint).to(torch_device).eval()

        # Use a small public audio clip
        url = "https://audioflamingo3.github.io/static/emergent/Dogs%20barking%20in%20sync%20with%20the%20music.wav"
        sr = getattr(self.processor.feature_extractor, "sampling_rate", 16000)
        raw_audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=sr)

        # Build a ChatML prompt with a single <sound> placeholder (AF3 supports 0 or 1 per sample)
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<sound>What's that sound?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = self.processor(text=prompt, audio=[raw_audio], return_tensors="pt", padding=True).to(torch_device)

        # Generate a few tokens (content not asserted strictly; we care about alignment)
        _ = model.generate(**inputs, max_new_tokens=16)

        # --- Alignment check: #<sound> tokens == sum of post-pool frame counts ---
        feat_mask = inputs["feature_attention_mask"]  # (#windows, T_mel)
        L_mel = feat_mask.sum(-1).to(dtype=torch.long)  # per-window lengths

        pre = (L_mel - 1) // 2 + 1
        post = (pre - 2) // 2 + 1
        total_post_frames = int(post.sum().item())

        sound_id = int(model.config.audio_token_id)
        num_sound_tokens = int((inputs["input_ids"] == sound_id).sum().item())

        self.assertEqual(
            num_sound_tokens,
            total_post_frames,
            msg=f"Expanded <sound> token count ({num_sound_tokens}) must match post-pool frames ({total_post_frames}).",
        )

        # --- Mismatch path: artificially increase the number of <sound> tokens and expect a hard error ---
        bad_inputs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        iid = bad_inputs["input_ids"]

        # Flip a few non-sound tokens to <sound> to force mismatch
        non_sound_positions = torch.nonzero(iid != sound_id, as_tuple=False)
        # take first 3 positions from the first sequence (be robust to short prompts)
        to_flip = non_sound_positions[:3]
        for pos in to_flip:
            b, t = int(pos[0]), int(pos[1])
            iid[b, t] = sound_id

        with self.assertRaisesRegex(ValueError, r"Audio tokens and features mismatch: tokens=\d+, frames=\d+"):
            model.generate(**bad_inputs, max_new_tokens=8)

    @slow
    def test_integration_batch_three_samples_and_global_alignment(self):
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(self.checkpoint).to(torch_device).eval()

        sr = getattr(self.processor.feature_extractor, "sampling_rate", 16000)
        urls = [
            "https://audioflamingo3.github.io/static/emergent/Dogs%20barking%20in%20sync%20with%20the%20music.wav",
            "https://audioflamingo3.github.io/static/emergent/Aside%20from%20_Interstellar%20Spaces_,%20Saxophones%20do%20not%20bark%20nor%20meow.wav",
            "https://audioflamingo3.github.io/static/emergent/Unlikely%20existence%20of%20banjo%20and%20rain%20sounds%20at%20the%20same%20time%20in%20the%20training%20data.wav",
        ]
        audios = [librosa.load(BytesIO(urlopen(u).read()), sr=sr)[0] for u in urls]

        texts = [
            (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<sound>What's happening in the audio?<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<sound>Describe the sound.<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<sound>What is this sound?<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        ]

        # AF3 processor requires 1:1 text:audio
        batch = self.processor(text=texts, audio=audios, return_tensors="pt", padding=True)
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(torch_device)

        gen = model.generate(**batch, max_new_tokens=16)
        # Basic sanity on decode; don't pin to specific strings
        new_tokens = gen[:, batch["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        self.assertEqual(len(decoded), 3)
        self.assertTrue(all(isinstance(s, str) for s in decoded))

        # Global alignment: total <sound> tokens == total post-pool frames
        feat_mask = batch["feature_attention_mask"]  # (sum_windows, T_mel)
        L_mel = feat_mask.sum(-1).to(dtype=torch.long)
        pre = (L_mel - 1) // 2 + 1
        post = (pre - 2) // 2 + 1
        total_post_frames = int(post.sum().item())
        sound_id = int(model.config.audio_token_id)
        num_sound_tokens = int((batch["input_ids"] == sound_id).sum().item())
        self.assertEqual(num_sound_tokens, total_post_frames)

    @slow
    def test_integration_batch_three_samples_processor_apply_chat_template(self):
        import tempfile
        from urllib.request import urlopen

        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(self.checkpoint).to(torch_device).eval()
        sr = getattr(self.processor.feature_extractor, "sampling_rate", 16000)

        urls = [
            "https://audioflamingo3.github.io/static/emergent/Dogs%20barking%20in%20sync%20with%20the%20music.wav",
            "https://audioflamingo3.github.io/static/emergent/Aside%20from%20_Interstellar%20Spaces_,%20Saxophones%20do%20not%20bark%20nor%20meow.wav",
            "https://audioflamingo3.github.io/static/emergent/Unlikely%20existence%20of%20banjo%20and%20rain%20sounds%20at%20the%20same%20time%20in%20the%20training%20data.wav",
        ]

        # Create temp files and fill them directly
        tmp_audios = []
        for u in urls:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(urlopen(u).read())
                tmp_audios.append(f.name)

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's happening in the audio?"},
                        {"type": "audio", "path": tmp_audios[0]},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the sound."},
                        {"type": "audio", "path": tmp_audios[1]},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this sound?"},
                        {"type": "audio", "path": tmp_audios[2]},
                    ],
                }
            ],
        ]

        batch = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            sampling_rate=sr,
            return_dict=True,
            return_attention_mask=True,
        ).to(torch_device)

        gen = model.generate(**batch, max_new_tokens=16)

        # Basic sanity on decode
        inp_len = batch["input_ids"].shape[1]
        new_tokens = gen[:, inp_len:]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertEqual(len(decoded), 3)
        self.assertTrue(all(isinstance(s, str) for s in decoded))

        # Global alignment
        feat_mask = batch["feature_attention_mask"]
        L_mel = feat_mask.sum(-1).to(dtype=torch.long)
        pre = (L_mel - 1) // 2 + 1
        post = (pre - 2) // 2 + 1
        total_post_frames = int(post.sum().item())
        sound_id = int(model.config.audio_token_id)
        num_sound_tokens = int((batch["input_ids"] == sound_id).sum().item())
        self.assertEqual(num_sound_tokens, total_post_frames)
