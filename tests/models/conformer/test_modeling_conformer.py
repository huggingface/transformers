# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import math
import tempfile
import unittest
from pathlib import Path

from transformers import is_datasets_available, is_torch_available, pipeline
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, random_attention_mask


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        ConformerCTCConfig,
        ConformerEncoder,
        ConformerEncoderConfig,
        ConformerForCTC,
    )
    from transformers.models.conformer.convert_nemo_to_hf import convert_checkpoint


class ConformerEncoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=256,
        is_training=True,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act="silu",
        conv_kernel_size=9,
        subsampling_factor=4,
        subsampling_conv_channels=64,
        num_mel_bins=80,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.0,
        layerdrop=0.0,
        scale_input=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.num_mel_bins = num_mel_bins
        self.subsampling_conv_kernel_size = subsampling_conv_kernel_size
        self.subsampling_conv_stride = subsampling_conv_stride
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.scale_input = scale_input

        self.output_seq_length = self._get_output_seq_length(seq_length)
        self.encoder_seq_length = self.output_seq_length
        self.key_length = self.output_seq_length

    def _get_output_seq_length(self, seq_length):
        kernel_size = self.subsampling_conv_kernel_size
        padding = (self.subsampling_conv_kernel_size - 1) // 2
        stride = self.subsampling_conv_stride

        length = seq_length
        for _ in range(int(math.log2(self.subsampling_factor))):
            length = ((length + (2 * padding) - (kernel_size - 1) - 1) // stride) + 1

        return length

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.num_mel_bins])
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()

        return config, input_features, attention_mask

    def get_config(self):
        return ConformerEncoderConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            conv_kernel_size=self.conv_kernel_size,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            num_mel_bins=self.num_mel_bins,
            subsampling_conv_kernel_size=self.subsampling_conv_kernel_size,
            subsampling_conv_stride=self.subsampling_conv_stride,
            dropout=self.dropout,
            dropout_positions=self.dropout,
            layerdrop=self.layerdrop,
            activation_dropout=self.dropout,
            attention_dropout=self.dropout,
            scale_input=self.scale_input,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = ConformerEncoder(config=config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            result = model(input_features, attention_mask=attention_mask)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, config.hidden_size)
        )
        self.parent.assertEqual(result.attention_mask.shape, (self.batch_size, self.output_seq_length))

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class ConformerEncoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ConformerEncoder,) if is_torch_available() else ()

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = ConformerEncoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ConformerEncoderConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="ConformerEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass


class ConformerForCTCModelTester:
    def __init__(self, parent, encoder_kwargs=None, is_training=True, vocab_size=32, pad_token_id=0):
        if encoder_kwargs is None:
            encoder_kwargs = {}

        self.parent = parent
        self.encoder_model_tester = ConformerEncoderModelTester(parent, **encoder_kwargs)
        self.is_training = is_training

        self.batch_size = self.encoder_model_tester.batch_size
        self.seq_length = self.encoder_model_tester.seq_length
        self.output_seq_length = self.encoder_model_tester.output_seq_length
        self.num_hidden_layers = self.encoder_model_tester.num_hidden_layers
        self.hidden_size = self.encoder_model_tester.hidden_size

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.encoder_seq_length = self.encoder_model_tester.encoder_seq_length
        self.key_length = self.encoder_model_tester.key_length

    def prepare_config_and_inputs(self):
        _, input_features, attention_mask = self.encoder_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_features, attention_mask

    def get_config(self):
        return ConformerCTCConfig(
            encoder_config=self.encoder_model_tester.get_config(),
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = ConformerForCTC(config=config)
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
class ConformerForCTCModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ConformerForCTC,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": ConformerEncoder,
            "automatic-speech-recognition": ConformerForCTC,
        }
        if is_torch_available()
        else {}
    )

    test_attention_outputs = False
    test_resize_embeddings = False

    _is_composite = True

    def setUp(self):
        self.model_tester = ConformerForCTCModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ConformerCTCConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="ConformerEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass


class ConformerForCTCIntegrationTest(unittest.TestCase):
    _temporary_directory = None
    _dataset = None

    @classmethod
    def setUp(cls):
        cls._temporary_directory = tempfile.TemporaryDirectory()

        cls.checkpoint_name = "nvidia/stt_en_conformer_ctc_small"
        cls.checkpoint_path = cls._convert_checkpoint()
        cls.dtype = torch.bfloat16
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_path)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def tearDownClass(cls):
        if cls._temporary_directory is not None:
            cls._temporary_directory.cleanup()
            cls._temporary_directory = None
            cls.checkpoint_path = None

    @classmethod
    def _convert_checkpoint(cls):
        path = Path(cls._temporary_directory.name)
        convert_checkpoint(cls.checkpoint_name, path)
        return path

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
    def test_model_integration(self):
        # fmt: off
        EXPECTED_TOKEN_IDS = torch.tensor([
            [
                1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 596,
                1024, 1024, 1024, 1024, 1024, 1024, 1024, 672, 1024, 1024, 1024, 66, 1024, 1024,
                1024, 97, 1024, 1024, 1024, 24, 1024, 1024, 2, 1024, 1024, 3, 1024, 1024, 330,
                1024, 1024, 1, 1024, 1024, 4, 1024, 1024, 56, 1024, 1024, 8, 1024, 1024, 2, 1024,
                245, 12, 1024, 1024, 12, 12, 56, 1024, 1024, 36, 98, 1024, 1, 1024, 1024, 1, 1,
                1024, 67, 1024, 1024, 1024, 1024, 1024, 6, 1024, 1024, 1024, 32, 1024, 1024, 58,
                1024, 1024, 1024, 120, 98, 1024, 1024, 12, 1024, 1024, 1024, 5, 1024, 1024, 1024,
                32, 1024, 66, 1024, 1024, 742, 1024, 1024, 15, 15, 1024, 64, 1024, 1024, 1024,
                1024, 1024, 113, 1024, 1024, 1, 1024, 1024, 29, 1024, 1024, 1024, 209, 1024,
                1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
            ]
        ])
        # fmt: on
        EXPECTED_TRANSCRIPTIONS = [
            "mister quilter is the apostle of the midle clases and we are glad to welcome his gospel"
        ]

        samples = self._load_datasamples(1)
        model = ConformerForCTC.from_pretrained(
            self.checkpoint_path,
            torch_dtype=self.dtype,
            device_map=torch_device,
        )
        model.eval()
        model.to(torch_device)

        inputs = self.processor(samples, sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs.to(torch_device, dtype=self.dtype)
        predicted_ids = model.generate(**inputs)
        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_model_integration_batched(self):
        EXPECTED_TRANSCRIPTIONS = [
            "mister quilter is the apostle of the midle clases and we are glad to welcome his gospel",
            "nor is mister quilter's manner less interesting than his matter",
            "he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results ocur most readily to the mind",
            "he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky itca",
            "linll's pictures are a sort of upgards and adam paintings and mason's exquisite idols are as national as a jingo poem mr birket foster's landscapes smile at one much in the same way that mr carker used to flash his teeth and mr john collier gives his sitter a cheerful slap on the back before he says like a shampoer in a turkish bath next man",
        ]

        samples = self._load_datasamples(5)
        model = ConformerForCTC.from_pretrained(self.checkpoint_path, torch_dtype=self.dtype, device_map=torch_device)
        model.eval()
        model.to(torch_device)

        inputs = self.processor(samples, sampling_rate=self.processor.feature_extractor.sampling_rate)
        inputs.to(torch_device, dtype=self.dtype)
        predicted_ids = model.generate(**inputs)
        self.assertEqual(predicted_ids.shape, (5, 736))
        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_model_integration_pipe_with_chunk(self):
        EXPECTED_TRANSCRIPTIONS = [{"text": "mister quilter is the apostle of the middle class"}]

        samples = self._load_datasamples(1)
        pipe = pipeline(
            task="automatic-speech-recognition", model=self.checkpoint_path, dtype=self.dtype, device_map=torch_device
        )
        self.assertEqual(pipe.type, "ctc")
        predicted_transcripts = pipe(samples, chunk_length_s=3, stride_length_s=1)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)
