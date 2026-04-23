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
import unittest
from pathlib import Path

import torch

from transformers import (
    AutoProcessor,
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRForForcedAlignment,
    Qwen3ASRModel,
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


class Qwen3ASRModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 3
        self.seq_length = 25
        self.num_mel_bins = 20
        self.feat_seq_length = 100  # mel frames per sample
        self.audio_token_id = 0
        self.is_training = False

        text_config = {
            "model_type": "qwen3",
            "vocab_size": 99,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 52,
            "bos_token_id": 0,
            "pad_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": False,
        }
        audio_config = {
            "model_type": "qwen3_asr_audio_encoder",
            "num_mel_bins": self.num_mel_bins,
            "d_model": 8,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 16,
            "output_dim": text_config["hidden_size"],
            "downsample_hidden_size": 4,
        }

        self.text_config = text_config
        self.audio_config = audio_config
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.hidden_size = text_config["hidden_size"]
        self.encoder_seq_length = self.seq_length

    def get_config(self):
        return Qwen3ASRConfig(
            audio_config=self.audio_config,
            text_config=self.text_config,
            audio_token_id=self.audio_token_id,
        )

    def _num_audio_tokens(self, config):
        """Compute how many tokens the audio encoder produces for feat_seq_length frames."""
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

        return int(
            _get_feat_extract_output_lengths(
                torch.tensor(self.feat_seq_length),
                config.audio_config.n_window,
            ).item()
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        num_audio_tokens = self._num_audio_tokens(config)

        # Batched audio features (batch, mel, time) + mask (batch, time)
        input_features = floats_tensor([self.batch_size, self.num_mel_bins, self.feat_seq_length])
        input_features_mask = torch.ones([self.batch_size, self.feat_seq_length], dtype=torch.long).to(torch_device)

        # Text with audio token placeholders
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)
        attention_mask[:, :1] = 0
        input_ids[:, 1 : 1 + num_audio_tokens] = config.audio_token_id

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "input_features_mask": input_features_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()


@require_torch
class Qwen3ASRForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Qwen3ASRForConditionalGeneration, Qwen3ASRModel) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "audio-text-to-text": Qwen3ASRForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )

    # Similar to Qwen3OmniMoe,
    skip_test_audio_features_output_shape = True  # as the audio encoder merges batch_size and output_lengths in dim 0
    _is_composite = True
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def setUp(self):
        self.model_tester = Qwen3ASRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3ASRConfig)

    @unittest.skip(reason="Same as Qwen3OmniMoe.")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(
        reason="Like other audio LMs (Audio Flamingo, Voxtral) inputs_embeds corresponding to audio tokens are replaced when input features are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip("Does not has no attribute `hf_device_map`")
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="See test_model_parallelism")
    def test_model_parallel_beam_search(self):
        pass


@require_torch
class Qwen3ASRForConditionalGenerationIntegrationTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cleanup(torch_device, gc_collect=True)
        cls.checkpoint = "bezzam/Qwen3-ASR-0.6B"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_fixture_single_matches(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/3e0551708631784aeb684e0e838299f3#file-reproducer-py
        """
        path = Path(__file__).parent.parent.parent / "fixtures/qwen3_asr/expected_results_single.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        exp_ids = torch.tensor(raw["token_ids"])
        exp_txt = raw["transcriptions"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
                    },
                ],
            }
        ]

        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map="auto", dtype=torch.bfloat16
        ).eval()

        batch = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=model.dtype)
        seq = model.generate(**batch, max_new_tokens=32)

        inp_len = batch["input_ids"].shape[1]
        gen_ids = seq[:, inp_len:] if seq.shape[1] >= inp_len else seq
        torch.testing.assert_close(gen_ids.cpu(), exp_ids)
        txt = self.processor.decode(seq, skip_special_tokens=True)
        self.assertListEqual(txt, exp_txt)

    @slow
    def test_fixture_batch_matches(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/3e0551708631784aeb684e0e838299f3#file-reproducer-py
        """
        path = Path(__file__).parent.parent.parent / "fixtures/qwen3_asr/expected_results_batched.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        exp_ids = torch.tensor(raw["token_ids"])
        exp_txt = raw["transcriptions"]

        conversation = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
                        },
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
                        },
                    ],
                }
            ],
        ]

        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map="auto", dtype=torch.bfloat16
        ).eval()
        batch = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device, dtype=model.dtype)

        seq = model.generate(**batch, max_new_tokens=32)

        inp_len = batch["input_ids"].shape[1]
        gen_ids = seq[:, inp_len:] if seq.shape[1] >= inp_len else seq
        torch.testing.assert_close(gen_ids.cpu(), exp_ids)
        txt = self.processor.decode(seq, skip_special_tokens=True)
        self.assertListEqual(txt, exp_txt)


@require_torch
class Qwen3ForcedAlignerIntegrationTest(unittest.TestCase):
    """
    reproducer scripts (create JSON fixtures directly in repo): https://gist.github.com/ebezzam/3e0551708631784aeb684e0e838299f3#file-reproducer_timestamps-py
    """

    @classmethod
    def setUp(cls):
        cleanup(torch_device, gc_collect=True)
        cls.aligner_checkpoint = "bezzam/Qwen3-ForcedAligner-0.6B"
        cls.aligner_processor = AutoProcessor.from_pretrained(cls.aligner_checkpoint)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_aligner(self):
        return Qwen3ASRForForcedAlignment.from_pretrained(
            self.aligner_checkpoint,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

    def _run_alignment(self, model, audio, transcript, language):
        """Run forced alignment and return list of timestamp dicts."""
        aligner_inputs, word_lists = self.aligner_processor.prepare_forced_aligner_inputs(
            audio=audio,
            transcript=transcript,
            language=language,
        )
        aligner_inputs = aligner_inputs.to(model.device, model.dtype)

        with torch.inference_mode():
            outputs = model(**aligner_inputs)

        return self.aligner_processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=model.config.timestamp_token_id,
        )

    @slow
    def test_fixture_timestamps_single(self):
        path = Path(__file__).parent.parent.parent / "fixtures/qwen3_asr/expected_timestamps_single.json"
        with open(path, "r", encoding="utf-8") as f:
            expected = json.load(f)

        model = self._load_aligner()
        audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"

        timestamps = self._run_alignment(
            model,
            audio=audio_url,
            transcript=expected["text"],
            language=expected["language"],
        )[0]

        self.assertEqual(len(timestamps), len(expected["time_stamps"]))
        for pred, exp in zip(timestamps, expected["time_stamps"]):
            self.assertAlmostEqual(pred["start_time"], exp["start_time"], places=2)
            self.assertAlmostEqual(pred["end_time"], exp["end_time"], places=2)

    @slow
    def test_fixture_timestamps_batched(self):
        path = Path(__file__).parent.parent.parent / "fixtures/qwen3_asr/expected_timestamps_batched.json"
        with open(path, "r", encoding="utf-8") as f:
            expected_batch = json.load(f)

        model = self._load_aligner()
        audio_urls = [
            "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
        ]

        batch_timestamps = self._run_alignment(
            model,
            audio=audio_urls,
            transcript=[e["text"] for e in expected_batch],
            language=[e["language"] for e in expected_batch],
        )

        self.assertEqual(len(batch_timestamps), len(expected_batch))
        for sample_idx, (pred_ts, exp) in enumerate(zip(batch_timestamps, expected_batch)):
            self.assertEqual(
                len(pred_ts),
                len(exp["time_stamps"]),
                f"Sample {sample_idx}: expected {len(exp['time_stamps'])} timestamps, got {len(pred_ts)}",
            )
            for pred, exp_ts in zip(pred_ts, exp["time_stamps"]):
                self.assertAlmostEqual(pred["start_time"], exp_ts["start_time"])
                self.assertAlmostEqual(pred["end_time"], exp_ts["end_time"])
