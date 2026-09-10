# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch GraniteSpeech5 model."""

import json
import unittest
from pathlib import Path

from transformers import is_datasets_available, is_torch_available
from transformers.testing_utils import cleanup, require_torch, require_torchaudio, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        GraniteSpeech5CTCConfig,
        GraniteSpeech5Encoder,
        GraniteSpeech5EncoderConfig,
        GraniteSpeech5ForCTC,
    )


FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures/granite_speech5"


class GraniteSpeech5EncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=50,
        is_training=True,
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        head_dim=16,
        num_mel_bins=6,
        vocab_size=30,
        context_size=13,
        max_position_embeddings=64,
        subsample_layers=[0],
        dropout=0.0,  # so gradient checkpointing doesn't fail
    ):
        # testing suite parameters
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training

        # config parameters
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_mel_bins = num_mel_bins
        # the front-end concatenates deltas and stacks frame pairs, so the encoder input is 4x wider
        self.feature_size = 4 * num_mel_bins
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.max_position_embeddings = max_position_embeddings
        self.subsample_layers = subsample_layers
        self.dropout = dropout

        # Calculate output sequence length after the subsampling conformer blocks
        self.output_seq_length = seq_length
        for _ in subsample_layers:
            self.output_seq_length = self.output_seq_length // 2
        # the first recorded hidden state (the subsampling module's output) is at the input frame rate
        self.encoder_seq_length = self.seq_length
        self.key_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.feature_size])
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()

        return config, input_features, attention_mask

    def get_config(self):
        return GraniteSpeech5EncoderConfig(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            num_mel_bins=self.num_mel_bins,
            vocab_size=self.vocab_size,
            context_size=self.context_size,
            max_position_embeddings=self.max_position_embeddings,
            subsample_layers=self.subsample_layers,
            attention_dropout=self.dropout,
            activation_dropout=self.dropout,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = GraniteSpeech5Encoder(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features, attention_mask=attention_mask)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, config.hidden_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class GraniteSpeech5EncoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (GraniteSpeech5Encoder,) if is_torch_available() else ()

    has_attentions = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = GraniteSpeech5EncoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraniteSpeech5EncoderConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="GraniteSpeech5Encoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass


class GraniteSpeech5ForCTCModelTester:
    def __init__(self, parent, encoder_kwargs=None, is_training=True, vocab_size=30, pad_token_id=0):
        if encoder_kwargs is None:
            encoder_kwargs = {}

        self.parent = parent
        self.encoder_model_tester = GraniteSpeech5EncoderModelTester(parent, vocab_size=vocab_size, **encoder_kwargs)
        self.is_training = is_training

        self.batch_size = self.encoder_model_tester.batch_size
        self.seq_length = self.encoder_model_tester.seq_length
        self.encoder_seq_length = self.encoder_model_tester.encoder_seq_length
        self.output_seq_length = self.encoder_model_tester.output_seq_length
        self.num_hidden_layers = self.encoder_model_tester.num_hidden_layers
        self.hidden_size = self.encoder_model_tester.hidden_size

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def prepare_config_and_inputs(self):
        _, input_features, attention_mask = self.encoder_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_features, attention_mask

    def get_config(self):
        return GraniteSpeech5CTCConfig(
            encoder_config=self.encoder_model_tester.get_config(),
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = GraniteSpeech5ForCTC(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features, attention_mask=attention_mask)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.output_seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class GraniteSpeech5ForCTCModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (GraniteSpeech5ForCTC,) if is_torch_available() else ()
    all_generative_model_classes = ()  # GraniteSpeech5ForCTC has a custom generate method
    pipeline_model_mapping = (
        {
            "feature-extraction": GraniteSpeech5Encoder,
            "automatic-speech-recognition": GraniteSpeech5ForCTC,
        }
        if is_torch_available()
        else {}
    )

    has_attentions = False
    test_resize_embeddings = False
    _is_composite = True

    def setUp(self):
        self.model_tester = GraniteSpeech5ForCTCModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraniteSpeech5CTCConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_ctc_loss(self):
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = GraniteSpeech5ForCTC(config=config)
        model.to(torch_device)
        model.eval()

        max_label_length = int(model._get_subsampling_output_length(attention_mask.sum(-1)).min()) - 1
        labels = ids_tensor((input_features.shape[0], max_label_length), config.vocab_size - 1) + 1

        model.config.ctc_loss_reduction = "sum"
        sum_loss = model(input_features, attention_mask=attention_mask, labels=labels).loss.item()

        model.config.ctc_loss_reduction = "mean"
        mean_loss = model(input_features, attention_mask=attention_mask, labels=labels).loss.item()

        self.assertTrue(isinstance(sum_loss, float))
        self.assertTrue(isinstance(mean_loss, float))

    def test_generate(self):
        config, input_features, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = GraniteSpeech5ForCTC(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            sequences = model.generate(input_features, attention_mask=attention_mask)
        self.assertEqual(sequences.shape, (self.model_tester.batch_size, self.model_tester.output_seq_length))
        # frames beyond each sample's subsampled length are filled with the CTC blank (the pad token)
        output_mask = model._get_output_attention_mask(attention_mask, target_length=sequences.shape[1])
        self.assertTrue((sequences[~output_mask] == config.pad_token_id).all())

    @unittest.skip(reason="GraniteSpeech5ForCTC does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(
        reason="`ctc_head` is tied to `encoder.out` across top-level submodules, which accelerate's disk offload does not support"
    )
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(
        reason="`ctc_head` is tied to `encoder.out` across top-level submodules, which accelerate's disk offload does not support"
    )
    def test_disk_offload_safetensors(self):
        pass


@require_torch
@require_torchaudio
class GraniteSpeech5ForCTCIntegrationTest(unittest.TestCase):
    """
    fixtures reproducer: https://gist.github.com/eustlb/16b67666c78536b3a8ec7d7b99e7eedf
    """

    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "ibm-granite/granite-speech-5.0-470m-turboctc"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_name)

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
    def test_model_integration_single(self):
        RESULTS_PATH = FIXTURES_DIR / "expected_results_single.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["token_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(1)
        model = GraniteSpeech5ForCTC.from_pretrained(self.checkpoint_name, device_map="auto")

        inputs = self.processor(samples, sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs.to(model.device, dtype=model.dtype)
        predicted_ids = model.generate(**inputs)
        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_model_integration_batch(self):
        RESULTS_PATH = FIXTURES_DIR / "expected_results_batch.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["token_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(4)
        model = GraniteSpeech5ForCTC.from_pretrained(self.checkpoint_name, device_map="auto")

        inputs = self.processor(samples, sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs.to(model.device, dtype=model.dtype)
        predicted_ids = model.generate(**inputs)
        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)
