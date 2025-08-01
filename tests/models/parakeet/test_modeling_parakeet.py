# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch ParakeetCTC model."""

import unittest

from transformers import is_datasets_available, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, random_attention_mask


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        ParakeetConfig,
        ParakeetEncoder,
        ParakeetEncoderConfig,
        ParakeetForCTC,
    )


class ParakeetEncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,
        is_training=True,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act="silu",
        dropout_positions=0.1,
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=32,
        use_bias=True,
        num_mel_bins=80,
        scale_input=True,
    ):
        # testing suite parameters
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_mel_bins = num_mel_bins
        self.is_training = is_training

        # config parameters
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout_positions = dropout_positions
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.use_bias = use_bias
        self.num_mel_bins = num_mel_bins
        self.scale_input = scale_input

        # Calculate output sequence length after subsampling
        self.output_seq_length = seq_length // subsampling_factor
        self.encoder_seq_length = self.output_seq_length
        self.key_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins])
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()

        return config, input_features, attention_mask

    def get_config(self):
        return ParakeetEncoderConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            dropout_positions=self.dropout_positions,
            conv_kernel_size=self.conv_kernel_size,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            use_bias=self.use_bias,
            num_mel_bins=self.num_mel_bins,
            scale_input=self.scale_input,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = ParakeetEncoder(config=config)
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
class ParakeetEncoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ParakeetEncoder,) if is_torch_available() else ()

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = ParakeetEncoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ParakeetEncoderConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="ParakeetEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass


class ParakeetForCTCModelTester:
    def __init__(self, parent, encoder_kwargs=None, is_training=True, vocab_size=128, blank_token_id=0):
        if encoder_kwargs is None:
            encoder_kwargs = {}

        self.parent = parent
        self.encoder_model_tester = ParakeetEncoderModelTester(parent, **encoder_kwargs)
        self.is_training = is_training

        self.batch_size = self.encoder_model_tester.batch_size
        self.output_seq_length = self.encoder_model_tester.output_seq_length

        self.vocab_size = vocab_size
        self.blank_token_id = blank_token_id

    def prepare_config_and_inputs(self):
        _, input_features, attention_mask = self.encoder_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_features, attention_mask

    def get_config(self):
        return ParakeetConfig.from_encoder_config(
            encoder_config=self.encoder_model_tester.get_config(),
            vocab_size=self.vocab_size,
            blank_token_id=self.blank_token_id,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = ParakeetForCTC(config=config)
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
class ParakeetForCTCModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ParakeetForCTC,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": ParakeetEncoder,
            "automatic-speech-recognition": ParakeetForCTC,
        }
        if is_torch_available()
        else {}
    )

    test_attention_outputs = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    _is_composite = True

    def setUp(self):
        self.model_tester = ParakeetForCTCModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ParakeetConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="ParakeetEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass


@require_torch
class ParakeetForCTCIntegrationTest(unittest.TestCase):
    _dataset = None

    def setUp(self):
        # TODO: update with the correct checkpoint
        self.checkpoint_name = "eustlb/parakeet-ctc-1.1b"
        self.dtype = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_name)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
        # Lazy loading of the dataset. Because it is a class method, it will only be loaded once per pytest process.
        if cls._dataset is None:
            cls._dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            # using 16000 here for simplicity, should rather be processor.feature_extractor.sampling_rate
            cls._dataset = cls._dataset.cast_column("audio", Audio(sampling_rate=16000))

    def _load_datasamples(self, num_samples):
        self._load_dataset()
        ds = self._dataset
        speech_samples = ds.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    @slow
    def test_1b_model_integration(self):
        samples = self._load_datasamples(1)

        model = ParakeetForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map=torch_device)

        inputs = self.processor(samples, sampling_rate=16000)
        inputs.to(torch_device, dtype=self.dtype)

        predicted_ids = model.generate(**inputs)
        predicted_transcripts = self.processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTS = [
            "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"
        ]

        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTS)

    @slow
    def test_1b_model_integration_batched(self):
        samples = self._load_datasamples(5)

        model = ParakeetForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map=torch_device)

        inputs = self.processor(samples, sampling_rate=16000)
        inputs.to(torch_device, dtype=self.dtype)

        predicted_ids = model.generate(**inputs)
        predicted_transcripts = self.processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTS = [
            "mr quilter is the apostle of the middle classes and we are glad to welcome his gospel",
            "nor is mr quilter's manner less interesting than his matter",
            "he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind",
            "he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca",
            "linnell's pictures are a sort of up guards and adam paintings and mason's exquisite idylls are as national as a jingo poem mr burket foster's landscapes smile at one much in the same way that mr carker used to flash his teeth and mr john collier gives his sitter a cheerful slap on the back before he says like a shampooer in a<unk>urkish bath next man",
        ]

        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTS)
