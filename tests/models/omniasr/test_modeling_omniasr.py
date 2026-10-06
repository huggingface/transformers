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

from transformers import (
    AutoProcessor,
    LlamaConfig,
    OmniASRAudioConfig,
    OmniASRConfig,
    OmniASRForConditionalGeneration,
    OmniASRModel,
    is_datasets_available,
    is_torch_available,
)
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...alm_tester import ALMModelTest, ALMModelTester
from ...test_modeling_common import floats_tensor


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch


class OmniASRModelTester(ALMModelTester):
    config_class = OmniASRConfig
    base_model_class = OmniASRModel
    conditional_generation_class = OmniASRForConditionalGeneration
    text_config_class = LlamaConfig
    audio_config_class = OmniASRAudioConfig
    audio_mask_key = "padding_mask"

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("seq_length", 20)
        kwargs.setdefault("feat_seq_length", 80)
        kwargs.setdefault("conv_dim", [16, 16])
        kwargs.setdefault("conv_kernel", [5, 3])
        kwargs.setdefault("conv_stride", [3, 2])
        kwargs.setdefault("num_conv_pos_embeddings", 8)
        kwargs.setdefault("num_conv_pos_embedding_groups", 2)
        kwargs.setdefault("audio_token_id", 0)
        kwargs.setdefault("layerdrop", 0.0)
        kwargs.setdefault("head_dim", 8)
        super().__init__(parent, **kwargs)

    def create_audio_features(self):
        # OmniASR is fed the raw waveform, not mel features.
        return floats_tensor([self.batch_size, self.feat_seq_length])

    def get_audio_feature_key(self):
        return "input_values"

    def get_audio_embeds_mask(self, audio_mask):
        # Mirrors `OmniASRPreTrainedModel._get_subsampling_output_length`.
        lengths = audio_mask.sum(-1)
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        positions = torch.arange(int(lengths.max()), device=audio_mask.device)[None, :]
        return (positions < lengths[:, None]).long()


@require_torch
class OmniASRForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    model_tester_class = OmniASRModelTester

    @unittest.skip(
        reason="Like other audio LMs (Voxtral, Qwen3 ASR) inputs_embeds corresponding to audio tokens are replaced when input values are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass


@require_torch
class OmniASRForConditionalGenerationIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "bezzam/omniasr-llm-300m-v2"
        cls.dtype = torch.float32
        cls.fixtures_path = Path(__file__).parent.parent.parent / "fixtures/omniasr"
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
        with open(self.fixtures_path / "expected_results_single.json", encoding="utf-8") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["pred_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(1)
        model = OmniASRForConditionalGeneration.from_pretrained(
            self.checkpoint_name, torch_dtype=self.dtype, device_map="auto"
        )

        inputs = self.processor.apply_transcription_request(samples, language="eng_Latn")
        inputs.to(model.device, dtype=self.dtype)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
            )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]

        torch.testing.assert_close(generated_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_llm_300m_v2_model_integration_batched(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_llm_batch-py
        """
        with open(self.fixtures_path / "expected_results_batch.json", encoding="utf-8") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = raw_data["pred_ids"]
        EXPECTED_HYPOTHESIS_LENS = raw_data["hypothesis_lens"]
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(3)
        model = OmniASRForConditionalGeneration.from_pretrained(
            self.checkpoint_name, torch_dtype=self.dtype, device_map="auto"
        )

        inputs = self.processor.apply_transcription_request(samples, language=["eng_Latn"] * len(samples))
        inputs.to(model.device, dtype=self.dtype)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
            )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
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
        paired so that the short one carries ~6x its own length in padding: without the audio `padding_mask`
        reaching the encoder, and without the audio being left-padded, it decodes to a shorter, degraded hypothesis.
        """
        samples = self._load_datasamples(6)
        shortest, longest = min(samples, key=len), max(samples, key=len)
        model = OmniASRForConditionalGeneration.from_pretrained(
            self.checkpoint_name, torch_dtype=self.dtype, device_map="auto"
        )

        def generate(batch):
            inputs = self.processor.apply_transcription_request(batch, language=["eng_Latn"] * len(batch))
            inputs.to(model.device, dtype=self.dtype)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=200)
            return generated_ids[:, inputs["input_ids"].shape[1] :]

        alone = generate([shortest])[0].cpu()
        batched = generate([shortest, longest])[0].cpu()
        torch.testing.assert_close(batched[: alone.shape[0]], alone)
        # everything past the hypothesis is padding
        self.assertTrue(bool((batched[alone.shape[0] :] == model.config.pad_token_id).all()))
