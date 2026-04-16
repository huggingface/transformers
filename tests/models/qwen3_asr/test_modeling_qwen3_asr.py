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
            "model_type": "qwen3_audio_encoder",
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
    all_model_classes = (Qwen3ASRForConditionalGeneration,) if is_torch_available() else ()
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
    test_torch_exportable = False  # Audio encoder has data-dependent ops incompatible with torch.export

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
