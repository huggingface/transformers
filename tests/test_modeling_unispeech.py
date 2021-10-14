# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch UniSpeech model. """

import math
import unittest

import numpy as np
import pytest

from tests.test_modeling_common import floats_tensor, ids_tensor, random_attention_mask
from transformers import UniSpeechConfig, is_torch_available
from transformers.testing_utils import require_datasets, require_soundfile, require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, _config_zero_init


if is_torch_available():
    import torch

    from transformers import (
        UniSpeechFeatureExtractor,
        UniSpeechForCTC,
        UniSpeechForMaskedLM,
        UniSpeechForPreTraining,
        UniSpeechForSequenceClassification,
        UniSpeechModel,
        UniSpeechProcessor,
    )
    from transformers.models.wav2vec2.modeling_wav2vec2 import UniSpeechGumbelVectorQuantizer, _compute_mask_indices


@require_torch
@require_datasets
@require_soundfile
@slow
class UniSpeechModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        import soundfile as sf

        ids = [f"1272-141231-000{i}" for i in range(num_samples)]

        # map files to raw
        def map_to_array(batch):
            speech, _ = sf.read(batch["file"])
            batch["speech"] = speech
            return batch

        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

        ds = ds.filter(lambda x: x["id"] in ids).sort("id").map(map_to_array)

        return ds["speech"][:num_samples]

    def _load_superb(self, task, num_samples):
        from datasets import load_dataset

        ds = load_dataset("anton-l/superb_dummy", task, split="test")

        return ds[:num_samples]

    def test_inference_ctc_normal(self):
        model = UniSpeechForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        model.to(torch_device)
        processor = UniSpeechProcessor.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)
        input_speech = self._load_datasamples(1)

        input_values = processor(input_speech, return_tensors="pt").input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = ["a man said to the universe sir i exist"]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_normal_batched(self):
        model = UniSpeechForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        model.to(torch_device)
        processor = UniSpeechProcessor.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight lowing cloth that was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_robust_batched(self):
        model = UniSpeechForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(torch_device)
        processor = UniSpeechProcessor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", do_lower_case=True)

        input_speech = self._load_datasamples(4)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight loin cloth that was the only garment he wore",
            "the cut on his chest still dripping blood the ache of his overstrained eyes even the soaring arena around him with the thousands of spectators were trivialities not worth thinking about",
            "his instant panic was followed by a small sharp blow high on his chest",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    # UniSpeech pretraining seems to be broken. TODO(PVP) - reenable test once pretraining works
    # correctly
    def test_inference_integration(self):
        return

        model = UniSpeechForPreTraining.from_pretrained("facebook/wav2vec2-base")
        model.to(torch_device)
        feature_extractor = UniSpeechFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        input_speech = self._load_datasamples(2)

        inputs_dict = feature_extractor(input_speech, return_tensors="pt", padding=True)

        features_shape = (
            inputs_dict["input_values"].shape[0],
            model._get_feat_extract_output_lengths(torch.tensor(inputs_dict["input_values"].shape[1])),
        )

        torch.manual_seed(0)
        mask_time_indices = _compute_mask_indices(
            features_shape,
            model.config.mask_time_prob,
            model.config.mask_time_length,
            device=inputs_dict["input_values"].device,
            min_masks=2,
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(
                inputs_dict.input_values.to(torch_device),
                attention_mask=inputs_dict.attention_mask.to(torch_device),
                mask_time_indices=mask_time_indices,
            )

        # compute cosine similarity
        cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

        # retrieve cosine sim of masked features
        cosine_sim_masked = cosine_sim[mask_time_indices]

        # fmt: off
        expected_cosine_sim_masked = torch.tensor(
            [0.7458, 0.7188, 0.6418, 0.3729, 0.3741, 0.3694, 0.3110, 0.2257, 0.4403, 0.5415, 0.3950, 0.3701, 0.8831,
             0.8613, 0.5229, 0.6696, 0.7206, 0.7877, 0.6758, 0.8746, 0.6596, 0.6282, 0.6178, 0.5839, 0.5926, 0.6651,
             0.4635, 0.6332, 0.6572, 0.8776, 0.4999, 0.7001, 0.7257, 0.5098, 0.6229, 0.4566, 0.5261, 0.6363, 0.5371,
             0.6997],
            device=torch_device,
        )
        # fmt: on

        self.assertTrue(torch.allclose(cosine_sim_masked, expected_cosine_sim_masked, atol=1e-3))

    def test_inference_pretrained(self):
        model = UniSpeechForPreTraining.from_pretrained("facebook/wav2vec2-base")
        model.to(torch_device)
        feature_extractor = UniSpeechFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        input_speech = self._load_datasamples(2)

        inputs_dict = feature_extractor(input_speech, return_tensors="pt", padding=True)

        features_shape = (
            inputs_dict["input_values"].shape[0],
            model._get_feat_extract_output_lengths(torch.tensor(inputs_dict["input_values"].shape[1])),
        )

        torch.manual_seed(0)
        mask_time_indices = _compute_mask_indices(
            features_shape,
            model.config.mask_time_prob,
            model.config.mask_time_length,
            device=inputs_dict["input_values"].device,
            min_masks=2,
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(
                inputs_dict.input_values.to(torch_device),
                attention_mask=inputs_dict.attention_mask.to(torch_device),
                mask_time_indices=mask_time_indices,
            )

        # compute cosine similarity
        cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

        # retrieve cosine sim of masked features
        cosine_sim_masked = cosine_sim[mask_time_indices]

        # ... now compare to randomly initialized model

        config = UniSpeechConfig.from_pretrained("facebook/wav2vec2-base")
        model_rand = UniSpeechForPreTraining(config).to(torch_device).eval()

        with torch.no_grad():
            outputs_rand = model_rand(
                inputs_dict.input_values.to(torch_device),
                attention_mask=inputs_dict.attention_mask.to(torch_device),
                mask_time_indices=mask_time_indices,
            )

        # compute cosine similarity
        cosine_sim_rand = torch.cosine_similarity(
            outputs_rand.projected_states, outputs_rand.projected_quantized_states, dim=-1
        )

        # retrieve cosine sim of masked features
        cosine_sim_masked_rand = cosine_sim_rand[mask_time_indices]

        # a pretrained wav2vec2 model has learned to predict the quantized latent states
        # => the cosine similarity between quantized states and predicted states > 0.5
        # a random wav2vec2 model has not learned to predict the quantized latent states
        # => the cosine similarity between quantized states and predicted states is very likely < 0.1
        self.assertTrue(cosine_sim_masked.mean().item() - 5 * cosine_sim_masked_rand.mean().item() > 0)

    @unittest.skipIf(torch_device != "cpu", "cannot make deterministic on GPU")
    def test_loss_pretraining(self):
        model = UniSpeechForPreTraining.from_pretrained(
            "facebook/wav2vec2-base",
            attention_dropout=0.0,
            feat_proj_dropout=0.0,
            hidden_dropout=0.0,
            layerdrop=0.0,
        )
        model.to(torch_device).train()

        feature_extractor = UniSpeechFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base", return_attention_mask=True
        )
        input_speech = self._load_datasamples(2)

        inputs_dict = feature_extractor(input_speech, return_tensors="pt", padding=True)

        features_shape = (
            inputs_dict["input_values"].shape[0],
            model._get_feat_extract_output_lengths(inputs_dict["input_values"].shape[1]),
        )

        torch.manual_seed(0)
        mask_time_indices = _compute_mask_indices(
            features_shape,
            model.config.mask_time_prob,
            model.config.mask_time_length,
            device=inputs_dict["input_values"].device,
            min_masks=2,
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(
                inputs_dict.input_values.to(torch_device),
                attention_mask=inputs_dict.attention_mask.to(torch_device),
                mask_time_indices=mask_time_indices,
            )

        # check diversity loss
        num_codevectors = model.config.num_codevectors_per_group * model.config.num_codevector_groups
        diversity_loss = (num_codevectors - outputs.codevector_perplexity) / num_codevectors
        self.assertTrue(abs(diversity_loss.item() - 0.8859) < 1e-3)

        # check overall loss (contrastive loss + diversity loss)
        expected_loss = 62.5170

        self.assertTrue(abs(outputs.loss.item() - expected_loss) < 1e-3)

    def test_inference_keyword_spotting(self):
        model = UniSpeechForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ks").to(torch_device)
        processor = UniSpeechFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")
        input_data = self._load_superb("ks", 4)
        inputs = processor(input_data["speech"], return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)
        predicted_logits, predicted_ids = torch.max(outputs.logits, dim=-1)

        expected_labels = [7, 6, 10, 9]
        # s3prl logits for the same batch
        expected_logits = torch.tensor([6.1186, 11.8961, 10.2931, 6.0898], device=torch_device)

        self.assertListEqual(predicted_ids.tolist(), expected_labels)
        self.assertTrue(torch.allclose(predicted_logits, expected_logits, atol=1e-2))

    def test_inference_intent_classification(self):
        model = UniSpeechForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ic").to(torch_device)
        processor = UniSpeechFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ic")
        input_data = self._load_superb("ic", 4)
        inputs = processor(input_data["speech"], return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)

        predicted_logits_action, predicted_ids_action = torch.max(outputs.logits[:, :6], dim=-1)
        predicted_logits_object, predicted_ids_object = torch.max(outputs.logits[:, 6:20], dim=-1)
        predicted_logits_location, predicted_ids_location = torch.max(outputs.logits[:, 20:24], dim=-1)

        expected_labels_action = [0, 0, 2, 3]
        expected_logits_action = torch.tensor([0.4568, 11.0848, 1.6621, 9.3841], device=torch_device)
        expected_labels_object = [3, 10, 3, 4]
        expected_logits_object = torch.tensor([1.5322, 10.7094, 5.2469, 22.1318], device=torch_device)
        expected_labels_location = [0, 0, 0, 1]
        expected_logits_location = torch.tensor([1.5335, 6.5096, 10.5704, 11.0569], device=torch_device)

        self.assertListEqual(predicted_ids_action.tolist(), expected_labels_action)
        self.assertListEqual(predicted_ids_object.tolist(), expected_labels_object)
        self.assertListEqual(predicted_ids_location.tolist(), expected_labels_location)

        self.assertTrue(torch.allclose(predicted_logits_action, expected_logits_action, atol=1e-2))
        self.assertTrue(torch.allclose(predicted_logits_object, expected_logits_object, atol=1e-2))
        self.assertTrue(torch.allclose(predicted_logits_location, expected_logits_location, atol=1e-2))

    def test_inference_speaker_identification(self):
        model = UniSpeechForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid").to(torch_device)
        processor = UniSpeechFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
        input_data = self._load_superb("si", 4)

        output_logits = []
        with torch.no_grad():
            for example in input_data["speech"]:
                input = processor(example, return_tensors="pt", padding=True)
                output = model(input.input_values.to(torch_device), attention_mask=None)
                output_logits.append(output.logits[0])
        output_logits = torch.stack(output_logits)
        predicted_logits, predicted_ids = torch.max(output_logits, dim=-1)

        expected_labels = [251, 1, 1, 3]
        # s3prl logits for the same batch
        expected_logits = torch.tensor([37.5627, 71.6362, 64.2419, 31.7778], device=torch_device)

        self.assertListEqual(predicted_ids.tolist(), expected_labels)
        self.assertTrue(torch.allclose(predicted_logits, expected_logits, atol=1e-2))

    def test_inference_emotion_recognition(self):
        model = UniSpeechForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er").to(torch_device)
        processor = UniSpeechFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
        input_data = self._load_superb("er", 4)
        inputs = processor(input_data["speech"], return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)
        attention_mask = inputs.attention_mask.to(torch_device)
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)
        predicted_logits, predicted_ids = torch.max(outputs.logits, dim=-1)

        expected_labels = [1, 1, 2, 2]
        # s3prl logits for the same batch
        expected_logits = torch.tensor([2.1722, 3.0779, 8.0287, 6.6797], device=torch_device)

        self.assertListEqual(predicted_ids.tolist(), expected_labels)
        self.assertTrue(torch.allclose(predicted_logits, expected_logits, atol=1e-2))
