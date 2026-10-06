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
from unittest.mock import patch

from transformers import (
    AutoProcessor,
    OmniASRAudioConfig,
    OmniASRCTCConfig,
    OmniASRCTCForCTC,
    is_datasets_available,
    is_torch_available,
)
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, random_attention_mask


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch


class OmniASRCTCForCTCModelTester:
    def __init__(self, parent, batch_size=3, num_samples=80, vocab_size=32, pad_token_id=0, is_training=False):
        self.parent = parent
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.is_training = is_training

        self.conv_dim = [16, 16]
        self.conv_kernel = [5, 3]
        self.conv_stride = [3, 2]
        self.output_seq_length = 12
        self.seq_length = self.output_seq_length
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 2

    def get_config(self):
        return OmniASRCTCConfig(
            audio_config=OmniASRAudioConfig(
                hidden_size=self.hidden_size,
                conv_dim=self.conv_dim,
                conv_kernel=self.conv_kernel,
                conv_stride=self.conv_stride,
                num_attention_heads=self.num_attention_heads,
                num_hidden_layers=self.num_hidden_layers,
                intermediate_size=32,
                num_conv_pos_embeddings=8,
                num_conv_pos_embedding_groups=2,
                layerdrop=0.0,
            ),
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.num_samples])
        padding_mask = random_attention_mask([self.batch_size, self.num_samples])
        return self.get_config(), input_values, padding_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_values, padding_mask = self.prepare_config_and_inputs()
        return config, {"input_values": input_values, "padding_mask": padding_mask}

    def create_and_check_model(self, config, input_values, padding_mask):
        model = OmniASRCTCForCTC(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_values, padding_mask=padding_mask)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.output_seq_length, self.vocab_size))


@require_torch
class OmniASRCTCForCTCModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (OmniASRCTCForCTC,) if is_torch_available() else ()
    all_generative_model_classes = ()  # OmniASRCTCForCTC has a custom `generate`
    _is_composite = True
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = OmniASRCTCForCTCModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OmniASRCTCConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    @unittest.skip(reason="OmniASRCTCForCTC is an encoder with a CTC head: it takes the waveform, not inputs_embeds.")
    def test_model_get_set_embeddings(self):
        pass

    def test_sdpa_can_dispatch_on_flash(self):
        # Need to account for `padding_mask`
        prepare_config_and_inputs = self.model_tester.prepare_config_and_inputs

        def prepare_with_full_ones_mask():
            config, input_values, padding_mask = prepare_config_and_inputs()
            return config, input_values, torch.ones_like(padding_mask)

        with patch.object(self.model_tester, "prepare_config_and_inputs", new=prepare_with_full_ones_mask):
            super().test_sdpa_can_dispatch_on_flash()


@require_torch
class OmniASRCTCForCTCIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "bezzam/omniasr-ctc-300m-v2"
        cls.dtype = torch.float32
        cls.fixtures_path = Path(__file__).parent.parent.parent / "fixtures/omniasr_ctc"
        cls.processor = AutoProcessor.from_pretrained("bezzam/omniasr-ctc-300m-v2")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
        if cls._dataset is None:
            cls._dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            cls._dataset = cls._dataset.cast_column(
                "audio", Audio(sampling_rate=cls.processor.feature_extractor.sampling_rate)
            )

    def _load_datasamples(self, num_samples):
        self._load_dataset()
        ds = self._dataset
        speech_samples = ds.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    @slow
    def test_ctc_300m_v2_model_integration(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_ctc-py
        """
        with open(self.fixtures_path / "expected_results_single.json", encoding="utf-8") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["pred_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(1)
        model = OmniASRCTCForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map="auto")

        inputs = self.processor(samples)
        inputs.to(model.device, dtype=self.dtype)
        predicted_ids = model.generate(**inputs)
        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_ctc_300m_v2_model_integration_batched(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_ctc_batch-py
        """
        with open(self.fixtures_path / "expected_results_batch.json", encoding="utf-8") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["pred_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(3)
        model = OmniASRCTCForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map="auto")

        inputs = self.processor(samples)
        inputs.to(model.device, dtype=self.dtype)
        encoder_lengths = model._get_subsampling_output_length(inputs["padding_mask"].sum(-1))

        predicted_ids = model.generate(**inputs)
        for idx, length in enumerate(encoder_lengths.tolist()):
            torch.testing.assert_close(predicted_ids[idx, :length].cpu(), EXPECTED_TOKEN_IDS[idx, :length])
        predicted_transcripts = self.processor.decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)
