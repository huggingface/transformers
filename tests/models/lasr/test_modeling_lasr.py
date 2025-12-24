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
"""Testing suite for the PyTorch Lasr model."""

import tempfile
import unittest

from transformers import is_datasets_available, is_torch_available, pipeline
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        LasrCTCConfig,
        LasrEncoder,
        LasrEncoderConfig,
        LasrForCTC,
    )


class LasrEncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,
        is_training=True,
        hidden_size=64,
        num_hidden_layers=2,
        num_mel_bins=80,
        num_attention_heads=4,
        intermediate_size=256,
        conv_kernel_size=8,
        subsampling_conv_channels=32,
        subsampling_conv_kernel_size=5,
        subsampling_conv_stride=2,
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
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_conv_channels = subsampling_conv_channels
        self.subsampling_conv_kernel_size = subsampling_conv_kernel_size
        self.subsampling_conv_stride = subsampling_conv_stride

        self.num_mel_bins = num_mel_bins

        # output sequence length after subsampling
        self.output_seq_length = self._get_output_seq_length(self.seq_length)
        self.encoder_seq_length = self.output_seq_length
        self.key_length = self.output_seq_length

    def _get_output_seq_length(self, seq_length):
        kernel_size = self.subsampling_conv_kernel_size
        stride = self.subsampling_conv_stride
        num_layers = 2

        input_length = seq_length
        for _ in range(num_layers):
            input_length = (input_length - kernel_size) // stride + 1

        return input_length

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins])
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()

        return config, input_features, attention_mask

    def get_config(self):
        return LasrEncoderConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            conv_kernel_size=self.conv_kernel_size,
            subsampling_conv_channels=self.subsampling_conv_channels,
            subsampling_conv_kernel_size=self.subsampling_conv_kernel_size,
            subsampling_conv_stride=self.subsampling_conv_stride,
            num_mel_bins=self.num_mel_bins,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = LasrEncoder(config=config)
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

    def check_ctc_loss(self, config, input_values, *args):
        model = LasrForCTC(config=config)
        model.to(torch_device)

        # make sure that dropout is disabled
        model.eval()

        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.long)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], min(max_length_labels) - 1), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        model.config.ctc_loss_reduction = "sum"
        sum_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()

        model.config.ctc_loss_reduction = "mean"
        mean_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(sum_loss, float))
        self.parent.assertTrue(isinstance(mean_loss, float))


@require_torch
class LasrEncoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (LasrEncoder,) if is_torch_available() else ()

    test_resize_embeddings = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = LasrEncoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LasrEncoderConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="LasrEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass


class LasrForCTCModelTester:
    def __init__(self, parent, encoder_kwargs=None, is_training=True, vocab_size=128, pad_token_id=0):
        if encoder_kwargs is None:
            encoder_kwargs = {}

        self.parent = parent
        self.encoder_model_tester = LasrEncoderModelTester(parent, **encoder_kwargs)
        self.is_training = is_training

        self.batch_size = self.encoder_model_tester.batch_size
        self.output_seq_length = self.encoder_model_tester.output_seq_length
        self.num_hidden_layers = self.encoder_model_tester.num_hidden_layers
        self.seq_length = vocab_size
        self.hidden_size = self.encoder_model_tester.hidden_size

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.encoder_seq_length = self.encoder_model_tester.encoder_seq_length

    def prepare_config_and_inputs(self):
        _, input_features, attention_mask = self.encoder_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_features, attention_mask

    def get_config(self):
        return LasrCTCConfig.from_encoder_config(
            encoder_config=self.encoder_model_tester.get_config(),
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = LasrForCTC(config=config)
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

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.encoder_model_tester.check_ctc_loss(*config_and_inputs)


@require_torch
class LasrForCTCModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (LasrForCTC,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": LasrEncoder,
            "automatic-speech-recognition": LasrForCTC,
        }
        if is_torch_available()
        else {}
    )

    test_attention_outputs = False

    test_resize_embeddings = False
    test_torch_exportable = True

    _is_composite = True

    def setUp(self):
        self.model_tester = LasrForCTCModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LasrCTCConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="LasrEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    # Original function assumes vision+text model, so overwrite since Lasr is audio+text
    # Below is modified from `tests/models/granite_speech/test_modeling_granite_speech.py`
    def test_sdpa_can_dispatch_composite_models(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")


class LasrForCTCIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "google/medasr"
        cls.dtype = torch.bfloat16
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_name)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
        # Lazy loading of the dataset. Because it is a class method, it will only be loaded once per pytest process.
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
    def test_model_integration(self):
        # fmt: off
        EXPECTED_TOKENS = torch.tensor([
            [0,0,0,0,0,0,0,0,0,0,0,0,315,315,0,9,0,0,0,0,398,0,121,121,0,0,65,0,0,0,57,57,0,0,7,0,0,14,0,0,0,27,13,0,0,35,0,46,0,0,0,0,16,0,0,7,0,0,192,15,0,15,46,46,0,0,54,100,5,5,0,5,5,71,71,0,0,0,0,0,0,19,19,0,0,0,150,0,142,142,0,0,106,100,100,0,15,0,0,0,0,18,0,0,0,50,121,121,0,30,279,279,0,0,0,63,63,0,0,0,0,188,0,0,0,35,0,0,51,29,0,42,0,9,9,0,0,0,0,0,0,0,2]
        ])
        # fmt: on

        # fmt: off
        EXPECTED_TRANSCRIPTIONS = [
            'Mr. Chelter is the apostle of the middle classes and we are glad to welcome his gost ball.'
        ]
        # fmt: on

        samples = self._load_datasamples(1)
        model = LasrForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map=torch_device)
        model.eval()
        model.to(torch_device)

        # -- apply
        inputs = self.processor(samples)
        inputs.to(torch_device, dtype=self.dtype)
        predicted_ids = model.generate(**inputs)
        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKENS)
        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_model_integration_batched(self):
        # fmt: off
        EXPECTED_TOKENS = torch.tensor([
            [0,0,0,0,0,0,0,0,0,0,0,0,315,315,0,9,0,0,0,0,398,0,121,121,0,0,65,0,0,0,57,57,0,0,7,0,0,14,0,0,0,27,13,0,0,35,0,46,0,0,0,0,16,0,0,7,0,0,192,15,0,15,46,46,0,0,54,100,5,5,0,5,5,71,71,0,0,0,0,0,0,19,19,0,0,0,150,0,142,142,0,0,106,100,100,0,15,0,0,0,0,18,0,0,0,50,121,121,0,30,279,279,0,0,0,63,63,0,0,0,0,188,0,0,0,35,0,0,51,29,0,42,0,9,9,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,117,25,25,0,0,0,57,0,0,0,0,0,0,315,0,9,9,0,0,0,398,0,0,0,0,65,0,34,34,5,0,0,0,179,0,0,17,31,31,0,0,0,0,4,343,0,0,0,0,0,0,24,24,0,0,65,65,0,0,228,0,0,22,0,0,0,0,0,304,304,0,0,0,0,63,63,0,0,0,0,0,0,0,0,113,0,8,0,65,0,0,0,9,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,144,0,0,0,450,450,0,0,5,5,0,294,294,0,0,0,0,0,0,0,48,0,0,0,0,0,0,102,0,0,0,0,149,0,0,0,0,0,0,234,0,0,35,0,0,198,0,0,0,0,0,136,136,0,11,5,5,56,56,0,0,0,16,16,0,0,7,0,0,0,286,286,26,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,64,0,0,0,0,0,398,68,68,35,35,21,21,11,11,5,5,0,0,19,0,0,0,4,4,74,0,0,86,0,0,0,0,49,0,0,10,39,0,0,0,0,305,0,13,21,0,22,0,0,0,0,0,0,0,360,360,0,0,0,0,80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,5,0,178,0,95,0,41,0,0,57,0,0,0,0,290,11,62,62,17,0,0,137,0,0,0,0,0,89,0,99,99,22,22,0,0,0,0,19,0,0,53,0,5,0,0,58,58,5,5,147,147,8,8,5,5,0,0,4,4,13,30,0,0,30,61,61,0,0,0,0,110,0,0,35,0,0,0,58,58,101,101,23,23,41,41,0,0,0,18,0,0,0,7,0,0,192,192,0,82,82,0,0,9,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,144,0,0,0,299,0,0,0,0,0,391,0,0,91,0,0,0,0,0,104,0,0,0,5,5,0,0,0,0,0,50,222,222,0,130,0,0,0,0,0,0,0,126,0,25,0,39,39,0,12,12,68,0,43,0,0,0,138,0,0,0,252,0,30,30,0,0,0,0,424,0,0,0,0,0,0,57,57,0,0,0,0,0,0,58,29,29,41,41,0,0,0,0,0,0,0,0,106,68,0,10,52,0,0,0,0,0,0,0,0,0,351,0,0,0,0,134,134,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,19,0,0,0,0,265,0,0,0,212,212,0,0,207,207,0,112,0,0,0,0,24,0,0,0,53,0,0,0,0,0,127,0,0,0,0,0,317,0,0,0,0,0,0,0,16,16,0,0,0,0,0,0,0,0,0,4,74,74,0,0,153,0,20,0,0,0,0,89,0,0,60,0,84,84,0,11,0,0,0,9,9,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,163,0,36,0,0,121,0,34,34,5,5,0,0,0,0,59,0,84,84,0,314,0,5,0,0,0,0,0,0,0,142,142,0,0,0,0,14,0,0,97,97,25,8,8,16,16,0,0,0,0,0,0,0,0,0,0,0,0,0,206,206,0,0,0,0,0,43,0,248,248,0,5,5,0,19,19,0,0,14,15,0,0,15,28,21,0,0,0,0,0,0,0,0,0,0,208,36,36,8,8,22,0,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,19,0,0,0,0,113,0,0,5,0,56,0,5,0,0,0,0,0,0,0,139,139,0,0,324,0,0,5,0,73,10,10,0,0,0,53,0,0,0,46,5,5,0,0,0,0,0,142,142,0,0,0,0,80,0,0,0,0,0,0,0,0,0,0,4,4,17,0,0,123,0,0,0,29,29,0,0,0,0,80,80,0,0,14,0,0,0,106,0,22,22,0,0,13,0,0,0,0,0,0,0,59,0,0,21,21,10,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,315,0,0,9,0,0,0,0,141,61,0,197,0,161,0,0,0,47,13,13,0,5,0,65,65,0,5,5,0,0,4,4,397,397,0,0,5,5,5,30,11,11,242,242,5,0,0,0,0,0,0,0,0,4,4,5,5,21,21,0,23,46,46,0,0,0,102,0,0,0,0,0,171,0,0,0,0,0,0,0,390,390,0,0,0,0,24,24,0,7,7,0,0,0,458,0,0,0,0,0,0,380,0,0,0,5,0,48,48,0,0,315,315,0,9,9,0,0,0,0,132,0,26,0,0,52,0,31,0,0,0,0,0,0,0,0,0,0,0,0,0,294,0,12,12,0,18,0,0,0,0,47,100,100,5,5,70,0,0,63,0,0,0,0,4,4,88,88,10,10,60,60,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,0,0,0,0,315,0,9,9,0,0,4,0,260,0,56,17,17,0,0,0,132,0,29,0,0,0,31,31,0,0,0,0,0,0,0,0,0,0,0,0,413,0,0,5,5,0,0,63,0,0,4,5,5,73,0,65,0,0,0,0,0,14,0,0,0,0,0,0,54,222,0,31,31,0,0,269,269,0,0,0,0,0,0,4,5,5,100,100,0,27,0,0,0,0,24,0,0,7,0,0,0,383,383,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,360,0,0,0,0,37,0,0,0,0,336,0,0,0,5,0,0,0,0,0,288,288,0,0,0,14,0,0,0,0,0,0,0,155,155,11,11,233,233,233,13,13,0,13,13,0,0,0,0,0,24,0,0,14,0,4,4,8,61,61,0,52,235,235,0,0,0,51,11,11,0,60,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,449,0,0,0,0,0,0,0,179,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2]
        ])
        # fmt: on

        # fmt: off
        EXPECTED_TRANSCRIPTIONS = [
            "Mr. Chelter is the apostle of the middle classes and we are glad to welcome his gost ball.",
            "nor is Mr. Chter's manner less interesting than his matter.",
            "He tells us that at this fistive season of the year with Christmas and rose beef looming before as similly is drawn from eating and its results occur most readily to the mind.",
            "He has grave dos whether surfedrig latenc work is really griek after all and can discover in it but little of rocky ethica.",
            "Linel's pictures are a sort of upgards and addum paintings and masons exquisite itles are as national as a gingo pme. Mr. Burkeet fosters landscapes smile at one much in the same ways that Mr. Carker used to flash his teeth and Mr. Jonn Caler gives his sitter a cheerful slap in the back before he says like a shampoo in a turkish bath. next man",
        ]
        # fmt: on

        samples = self._load_datasamples(5)
        model = LasrForCTC.from_pretrained(
            self.checkpoint_name,
            torch_dtype=self.dtype,
            device_map=torch_device,
        )
        model.eval()
        model.to(torch_device)

        # -- apply
        inputs = self.processor(samples)
        inputs.to(torch_device, dtype=self.dtype)
        predicted_ids = model.generate(**inputs)
        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKENS)
        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_model_integration_pipe_with_chunk(self):
        EXPECTED_TRANSCRIPTIONS = [
            {"text": ". is the apsle of the middle classes and we are glad to welcome his goible.</s>"}
        ]

        samples = self._load_datasamples(1)
        pipe = pipeline(
            task="automatic-speech-recognition", model=self.checkpoint_name, dtype=self.dtype, device_map=torch_device
        )
        self.assertEqual(pipe.type, "ctc")
        predicted_transcripts = pipe(samples, chunk_length_s=3, stride_length_s=1)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)
