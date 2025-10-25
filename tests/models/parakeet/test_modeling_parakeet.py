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
"""Testing suite for the PyTorch Parakeet model."""

import json
import copy
import tempfile
import unittest
from pathlib import Path

from transformers import is_datasets_available, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        ParakeetCTCConfig,
        ParakeetTDTConfig,
        ParakeetEncoder,
        ParakeetEncoderConfig,
        ParakeetTDTDecoder,
        ParakeetTDTDecoderConfig,
        ParakeetTDTJoint,
        ParakeetTDTJointConfig,
        ParakeetForCTC,
        ParakeetForTDT,
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
        dropout=0,  # so gradient checkpointing doesn't fail
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
        self.dropout = dropout
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
            dropout=self.dropout,
            dropout_positions=self.dropout,
            layerdrop=self.dropout,
            activation_dropout=self.dropout,
            attention_dropout=self.dropout,
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

    def check_ctc_loss(self, config, input_values, *args):
        model = ParakeetForCTC(config=config)
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
class ParakeetEncoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ParakeetEncoder,) if is_torch_available() else ()

    test_resize_embeddings = False
    test_torch_exportable = True

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


class ParakeetTDTDecoderModelTester:
    def __init__(
        self,
        parent,
        batch_size=16,
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        seq_length=32,
        is_training=True,
        dropout=0,  # so gradient checkpointing doesn't fail
    ):
        # testing suite parameters
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training

        # config parameters
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.seq_length = seq_length
        self.output_seq_length = seq_length
        self.vocab_size = vocab_size

    def prepare_config_and_inputs(self):
        input_token = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        config = self.get_config()

        return config, input_token

    def get_config(self):
        return ParakeetTDTDecoderConfig(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
        )

    def create_and_check_model(self, config, input_token):
        pass
        model = ParakeetTDTDecoder(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_token)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, config.hidden_size, self.output_seq_length)
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_token = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_token": input_token,
        }
        return config, inputs_dict




@require_torch
class ParakeetTDTDecoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ParakeetTDTDecoder,) if is_torch_available() else ()

    test_resize_embeddings = False
    test_torch_exportable = True
    has_attentions = False
    is_encoder_decoder = False

    def setUp(self):
        self.model_tester = ParakeetTDTDecoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ParakeetTDTDecoderConfig, has_text_modality=False, common_properties=['hidden_size','num_hidden_layers'])

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(hidden_states.shape[1], expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length


        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            for k in config.sub_configs:
                if getattr(config, k) is not None:
                    getattr(config, k).output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="this class only returns the last hidden state not prior ones, and there is no gradient on last hidden state w.r.t output.")
    def test_retain_grad_hidden_states_attentions(self):
        pass


class ParakeetTDTJointModelTester:
    def __init__(
        self,
        parent,
        batch_size=16,
        vocab_size=128,
        hidden_size=64,
        pred_hidden_size=64,
        enc_hidden_size=64,
        num_hidden_layers=2,
        durations=[0,1,2,3,4],
        is_training=True,
        dropout=0.1,  # so gradient checkpointing doesn't fail
    ):
        # testing suite parameters
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training

        # config parameters
        self.hidden_size = hidden_size
        self.pred_hidden_size = pred_hidden_size
        self.enc_hidden_size = enc_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.t_length = 1  # so far only support 1
        self.u_length = 1  # so far only support 1
        self.output_seq_length = -1
        self.vocab_size = vocab_size
        self.durations = durations

    def prepare_config_and_inputs(self):
        enc = floats_tensor([self.batch_size, self.t_length, self.enc_hidden_size])
        pred = floats_tensor([self.batch_size, self.u_length, self.pred_hidden_size])
        config = self.get_config()

        return config, enc, pred

    def get_config(self):
        return ParakeetTDTJointConfig(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            pred_hidden_size=self.enc_hidden_size,
            enc_hidden_size=self.enc_hidden_size,
            vocab_size=self.vocab_size,
            durations=self.durations,
        )

    def create_and_check_model(self, config, enc, pred):
        model = ParakeetTDTJoint(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(enc, pred)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, config.vocab_size + 1 + len(config.durations))
        )

    def prepare_config_and_inputs_for_common(self):
        config, enc, pred = self.prepare_config_and_inputs()
        inputs_dict = {
            "enc": enc,
            "pred": pred,
        }
        return config, inputs_dict




@require_torch
class ParakeetTDTJointModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ParakeetTDTJoint,) if is_torch_available() else ()

    test_resize_embeddings = False
    test_torch_exportable = True
    has_attentions = False
    is_encoder_decoder = False

    def setUp(self):
        self.model_tester = ParakeetTDTJointModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ParakeetTDTJointConfig, has_text_modality=False, common_properties=['hidden_size'])

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="this class doesn't have hidden states.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="this class doesn't have hidden states.")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="ParakeetJoint does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass



class ParakeetForCTCModelTester:
    def __init__(self, parent, encoder_kwargs=None, is_training=True, vocab_size=128, pad_token_id=0):
        if encoder_kwargs is None:
            encoder_kwargs = {}

        self.parent = parent
        self.encoder_model_tester = ParakeetEncoderModelTester(parent, **encoder_kwargs)
        self.is_training = is_training

        self.batch_size = self.encoder_model_tester.batch_size
        self.output_seq_length = self.encoder_model_tester.output_seq_length
        self.num_hidden_layers = self.encoder_model_tester.num_hidden_layers
        self.seq_length = vocab_size
        self.hidden_size = self.encoder_model_tester.hidden_size

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def prepare_config_and_inputs(self):
        _, input_features, attention_mask = self.encoder_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_features, attention_mask

    def get_config(self):
        return ParakeetCTCConfig.from_encoder_config(
            encoder_config=self.encoder_model_tester.get_config(),
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
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

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.encoder_model_tester.check_ctc_loss(*config_and_inputs)


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

    test_resize_embeddings = False
    test_torch_exportable = True

    _is_composite = True

    def setUp(self):
        self.model_tester = ParakeetForCTCModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ParakeetCTCConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="ParakeetEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    # Original function assumes vision+text model, so overwrite since Parakeet is audio+text
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


@require_torch
class ParakeetForCTCIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "nvidia/parakeet-ctc-1.1b"
        cls.dtype = torch.bfloat16
        cls.processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")

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
    def test_1b_model_integration(self):
        """
        bezzam reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/6382bdabfc64bb2541ca9f77deb7678d#file-reproducer_single-py
        eustlb reproducer: https://gist.github.com/eustlb/6e9e3aa85de3f7c340ec3c36e65f2fe6
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/parakeet/expected_results_single.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["token_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(1)
        model = ParakeetForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map=torch_device)
        model.eval()
        model.to(torch_device)

        # -- apply
        inputs = self.processor(samples)
        inputs.to(torch_device, dtype=self.dtype)
        predicted_ids = model.generate(**inputs)
        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_1b_model_integration_batched(self):
        """
        bezzam reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/6382bdabfc64bb2541ca9f77deb7678d#file-reproducer_batched-py
        eustlb reproducer: https://gist.github.com/eustlb/575b5da58de34a70116a1955b1183596
        """

        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/parakeet/expected_results_batch.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["token_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(5)
        model = ParakeetForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map=torch_device)
        model.eval()
        model.to(torch_device)

        # -- apply
        inputs = self.processor(samples)
        inputs.to(torch_device, dtype=self.dtype)
        predicted_ids = model.generate(**inputs)
        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)



class ParakeetForTDTModelTester:
    def __init__(self,
                 parent,
                 encoder_kwargs=None,
                 decoder_kwargs=None,
                 joint_kwargs=None,
                 is_training=True,
                 vocab_size=128,
                 durations=[0,1,2,3,4],
                 pad_token_id=0
                 ):
        if encoder_kwargs is None:
            encoder_kwargs = {}
        if decoder_kwargs is None:
            decoder_kwargs = {}
        if joint_kwargs is None:
            joint_kwargs = {}

        self.parent = parent
        self.encoder_model_tester = ParakeetEncoderModelTester(parent, **encoder_kwargs)
        self.decoder_model_tester = ParakeetTDTDecoderModelTester(parent, **decoder_kwargs)
        self.joint_model_tester = ParakeetTDTJointModelTester(parent, **joint_kwargs)
        self.is_training = is_training

        self.batch_size = self.encoder_model_tester.batch_size
        self.output_seq_length = self.encoder_model_tester.output_seq_length
        self.num_hidden_layers = self.encoder_model_tester.num_hidden_layers
        self.seq_length = vocab_size
        self.enc_hidden_size = self.encoder_model_tester.hidden_size
        self.hidden_size = self.encoder_model_tester.hidden_size  # this field is needed for test class
        self.pred_hidden_size = self.decoder_model_tester.hidden_size
        self.joint_hidden_size = self.joint_model_tester.hidden_size

        self.durations = durations

        self.vocab_size = vocab_size + len(self.durations) + 1
        self.pad_token_id = pad_token_id


    def prepare_config_and_inputs(self):
        _, input_features, attention_mask = self.encoder_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_features, attention_mask

    def get_config(self):
        return ParakeetTDTConfig.from_configs(
            encoder_config=self.encoder_model_tester.get_config(),
            decoder_config=self.decoder_model_tester.get_config(),
            joint_config=self.joint_model_tester.get_config(),
            vocab_size=self.vocab_size,
            durations=self.durations,
        )

    def create_and_check_model(self, config, input_features, attention_mask):
        model = ParakeetForTDT(config=config)
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
class ParakeetForTDTModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ParakeetForTDT,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": ParakeetEncoder,
            "automatic-speech-recognition": ParakeetForTDT,
        }
        if is_torch_available()
        else {}
    )

    test_attention_outputs = False

    test_resize_embeddings = False
    test_torch_exportable = True

    _is_composite = True

    def setUp(self):
        self.model_tester = ParakeetForTDTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ParakeetTDTConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="ParakeetEncoder does not use inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="batching not supported")
    def test_batching_equivalence(self):
        pass

    # Original function assumes vision+text model, so overwrite since Parakeet is audio+text
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


@require_torch
class ParakeetForTDTIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "hainanx/parakeet-tdt-0.6b-v3"
        cls.dtype = torch.bfloat16
        cls.processor = AutoProcessor.from_pretrained("hainanx/parakeet-tdt-0.6b-v3")

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
    def test_1b_model_integration(self):
        """
        bezzam reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/6382bdabfc64bb2541ca9f77deb7678d#file-reproducer_single-py
        eustlb reproducer: https://gist.github.com/eustlb/6e9e3aa85de3f7c340ec3c36e65f2fe6
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/parakeet/expected_results_single.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["token_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(1)
        model = ParakeetForTDT.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map=torch_device)
        model.eval()
        model.to(torch_device)

        # -- apply
        inputs = self.processor(samples)
        inputs.to(torch_device, dtype=self.dtype)
        predicted_ids = model.generate(**inputs)
        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)
