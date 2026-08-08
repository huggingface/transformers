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

from transformers import is_datasets_available, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        OmniASRForConditionalGeneration,
        OmniASRForCTC,
    )


@require_torch
class OmniASRForCTCIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "bezzam/omniasr-ctc-300m-v2"
        cls.dtype = torch.float32
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
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/omniasr/expected_results_single.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["pred_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(1)
        model = OmniASRForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map="auto")

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
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/omniasr/expected_results_batch.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["pred_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(3)
        model = OmniASRForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map="auto")

        inputs = self.processor(samples)
        inputs.to(model.device, dtype=self.dtype)
        predicted_ids = model.generate(**inputs)
        encoder_lengths = model._get_feat_extract_output_lengths(inputs["attention_mask"].sum(-1))
        for idx, length in enumerate(encoder_lengths.tolist()):
            torch.testing.assert_close(predicted_ids[idx, :length].cpu(), EXPECTED_TOKEN_IDS[idx, :length])
        predicted_transcripts = self.processor.decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)


@require_torch
class OmniASRForConditionalGenerationIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "bezzam/omniasr-llm-300m-v2"
        cls.dtype = torch.float32
        cls.processor = AutoProcessor.from_pretrained("bezzam/omniasr-llm-300m-v2")

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
    def test_llm_300m_v2_model_integration(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_llm-py
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/omniasr/expected_results_single_llm.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["pred_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(1)
        model = OmniASRForConditionalGeneration.from_pretrained(
            self.checkpoint_name, torch_dtype=self.dtype, device_map="auto"
        )

        inputs = self.processor(
            samples,
            return_tensors="pt",
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            language=["eng_Latn"],
        )
        inputs.to(model.device, dtype=self.dtype)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
            )

        torch.testing.assert_close(generated_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_llm_300m_v2_model_integration_batched(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_llm_batch-py
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/omniasr/expected_results_batch_llm.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = raw_data["pred_ids"]
        EXPECTED_HYPOTHESIS_LENS = raw_data["hypothesis_lens"]
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(3)
        model = OmniASRForConditionalGeneration.from_pretrained(
            self.checkpoint_name, torch_dtype=self.dtype, device_map="auto"
        )

        inputs = self.processor(
            samples,
            return_tensors="pt",
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            padding=True,
            language=["eng_Latn"] * len(samples),
        )
        inputs.to(model.device, dtype=self.dtype)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
            )

        # The audio context is left-padded, so each sample decodes exactly as it would on its own. Compare each
        # hypothesis over its own length; what follows it is padding.
        for idx, length in enumerate(EXPECTED_HYPOTHESIS_LENS):
            torch.testing.assert_close(
                generated_ids[idx, :length].cpu(), torch.tensor(EXPECTED_TOKEN_IDS[idx][:length])
            )

        predicted_transcripts = self.processor.decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_llm_300m_v2_generate_is_padding_invariant(self):
        """
        Batching must not change what a sample transcribes to. The shortest and longest clips of the dataset are
        paired so that the short one carries ~6x its own length in padding: without the audio `attention_mask`
        reaching the encoder, and without the audio being left-padded, it decodes to a shorter, degraded hypothesis.
        """
        samples = self._load_datasamples(6)
        shortest, longest = min(samples, key=len), max(samples, key=len)
        model = OmniASRForConditionalGeneration.from_pretrained(
            self.checkpoint_name, torch_dtype=self.dtype, device_map="auto"
        )

        def generate(batch):
            inputs = self.processor(
                batch,
                sampling_rate=self.processor.feature_extractor.sampling_rate,
                language=["eng_Latn"] * len(batch),
                padding=True,
            )
            inputs.to(model.device, dtype=self.dtype)
            with torch.no_grad():
                return model.generate(**inputs, max_new_tokens=200)

        alone = generate([shortest])[0].cpu()
        batched = generate([shortest, longest])[0].cpu()
        torch.testing.assert_close(batched[: alone.shape[0]], alone)
        # everything past the hypothesis is padding
        self.assertTrue(bool((batched[alone.shape[0] :] == model.config.pad_token_id).all()))
