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
import unittest

from transformers import PeAudioConfig, PeAudioEncoderConfig
from transformers.audio_utils import load_audio
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


if is_torch_available():
    import torch

    from transformers import (
        ModernBertConfig,
        PeAudioEncoder,
        PeAudioFrameLevelModel,
        PeAudioModel,
    )


class PeAudioEncoderTester:
    def __init__(
        self,
        parent,
        config_kwargs={
            "dac_config": {
                "encoder_hidden_size": 16,
                "downsampling_ratios": [2, 4, 4],
                "decoder_hidden_size": 16,
                "n_codebooks": 6,
                "codebook_size": 512,
                "codebook_dim": 32,
                "quantizer_dropout": 0.0,
                "commitment_loss_weight": 0.25,
                "codebook_loss_weight": 1.0,
            },
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-5,
            "use_cache": True,
            "rope_theta": 20000,
            "rope_scaling": None,
            "attention_bias": False,
            "max_window_layers": 28,
            "attention_dropout": 0.0,
        },
        batch_size=12,
        num_channels=1,
        audio_seq_length=160,
        is_training=True,
    ):
        self.parent = parent

        self.config_kwargs = config_kwargs
        for key, value in config_kwargs.items():
            setattr(self, key, value)

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.audio_seq_length = audio_seq_length
        self.is_training = is_training

    @property
    def seq_length(self):
        config = self.get_config()
        # seq_length is what gets feeded to the transformer
        # we first have to divide by hop_length to get the number of frames
        # then we add 1 because we add the class token
        return self.audio_seq_length // config.dac_config.hop_length + 1

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.num_channels, self.audio_seq_length])
        # Generate valid_lengths in range [1, self.audio_seq_length] to ensure at least one valid frame
        valid_lengths = ids_tensor([self.batch_size], self.audio_seq_length - 1) + 1
        padding_mask = torch.arange(self.audio_seq_length, device=torch_device)[None, :] < valid_lengths[:, None]
        padding_mask = padding_mask.int()
        config = self.get_config()

        return config, input_values, padding_mask

    def get_config(self):
        if not hasattr(self, "_config"):
            self._config = PeAudioEncoderConfig(**self.config_kwargs)
        return self._config

    def create_and_check_model(self, config, input_values, padding_mask):
        model = PeAudioEncoder(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_values, padding_mask=padding_mask)
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_values, padding_mask = config_and_inputs
        inputs_dict = {"input_values": input_values, "padding_mask": padding_mask}
        return config, inputs_dict


@require_torch
class PeAudioEncoderTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (PeAudioEncoder,)
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = PeAudioEncoderTester(self)
        self.config_tester = ConfigTester(
            self, config_class=PeAudioEncoderConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="PeAudioEncoder does not have usual input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("PeAudioEncoder does not support feed forward chunking")
    def test_feed_forward_chunking(self):
        pass


class PeAudioTextModelTester:
    """
    Only a ModelTester and no PeAudioTextModelTest since text model is ModernBertModel that is already tested.
    """

    def __init__(
        self,
        parent,
        config_kwargs={
            "vocab_size": 99,
            "pad_token_id": 0,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_activation": "gelu",
            "mlp_dropout": 0.0,
            "attention_dropout": 0.0,
            "embedding_dropout": 0.0,
            "classifier_dropout": 0.0,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "is_decoder": False,
            "initializer_range": 0.02,
        },
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,  # TODO: to check
    ):
        self.parent = parent

        self.config_kwargs = config_kwargs
        for key, value in config_kwargs.items():
            setattr(self, key, value)

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return ModernBertConfig(**self.config_kwargs)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class PeAudioModelTester:
    def __init__(self, parent, text_kwargs=None, audio_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if audio_kwargs is None:
            audio_kwargs = {}

        self.parent = parent
        self.text_model_tester = PeAudioTextModelTester(parent, **text_kwargs)
        self.audio_model_tester = PeAudioEncoderTester(parent, **audio_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        _, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        _, input_values, padding_mask = self.audio_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, input_values, padding_mask

    def get_config(self):
        text_config = self.text_model_tester.get_config()
        audio_config = self.audio_model_tester.get_config()
        return PeAudioConfig(
            text_config=text_config.to_dict(),
            audio_config=audio_config.to_dict(),
            projection_dim=32,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, input_values, padding_mask):
        model = PeAudioModel(config).to(torch_device).eval()
        with torch.no_grad():
            _ = model(input_ids, input_values, attention_mask, padding_mask)

        # TODO: there is no logits per audio for now
        # self.parent.assertEqual(result.logits_per_audio.shape, (self.audio_model_tester.batch_size, self.text_model_tester.batch_size))
        # self.parent.assertEqual(result.logits_per_text.shape, (self.text_model_tester.batch_size, self.audio_model_tester.batch_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, input_values, padding_mask = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_values": input_values,
            "padding_mask": padding_mask,
        }
        return config, inputs_dict


@require_torch
class PeAudioModelTest(ModelTesterMixin, unittest.TestCase):
    # TODO: add PipelineTesterMixin
    all_model_classes = (PeAudioModel,)
    additional_model_inputs = ["input_values", "padding_mask"]
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False
    _is_composite = True

    def setUp(self):
        self.model_tester = PeAudioModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=PeAudioConfig, has_text_modality=False, common_properties=[], hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="PeAudioModel does not have usual input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="PeAudioModel does not support feed forward chunking yet")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="PeAudioModel uses some timm stuff not compatible")
    def test_save_load(self):
        pass

    @unittest.skip(reason="@eustlb this is not really expected")
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="@eustlb this is not really expected")
    def test_can_init_all_missing_weights(self):
        pass


@require_torch
class PeAudioIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint_name = "/raid/eustache/sam-audio/pe-a-frame-small"
        self.dtype = torch.float32

    @slow
    @unittest.skip(reason="TODO when released")
    def test_inference(self):
        checkpoint_name = "/raid/eustache/sam-audio/pe-av-small"
        descriptions = ["glass breaking", "somebody speaking"]
        audio_file = "https://huggingface.co/datasets/eustlb/dummy-audio-samples-higgs/resolve/main/glass_breaking.mp3"

        # processor = PeAudioProcessor.from_pretrained(checkpoint_name)
        model = PeAudioModel.from_pretrained(checkpoint_name, dtype=self.dtype, device_map=torch_device)

        inputs = self.processor(
            text=descriptions,
            audio=[load_audio(audio_file, self.processor.feature_extractor.sampling_rate)],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(torch_device, dtype=self.dtype)
        model(**inputs)

    @slow
    @unittest.skip(reason="TODO when released")
    def test_inference_frame_level(self):
        checkpoint_name = "/raid/eustache/sam-audio/pe-a-frame-small"
        descriptions = ["glass breaking", "somebody speaking"]
        audio_file = "https://huggingface.co/datasets/eustlb/dummy-audio-samples-higgs/resolve/main/glass_breaking.mp3"

        # processor = PeAudioProcessor.from_pretrained(checkpoint_name)
        model = PeAudioFrameLevelModel.from_pretrained(checkpoint_name, dtype=self.dtype, device_map=torch_device)

        inputs = self.processor(
            text=descriptions,
            audio=[load_audio(audio_file, self.processor.feature_extractor.sampling_rate)],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(torch_device, dtype=self.dtype)

        outputs = model(**inputs)
        #
        # TODO: this should be incorporated into the `forward` pass itself
        threshold = 0.3
        logits_per_audio = outputs.logits_per_audio
        probs_per_audio = logits_per_audio.sigmoid()
        preds = probs_per_audio > threshold

        # fmt: off
        EXPECTED = torch.tensor([
            [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
            [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True]
        ])
        # fmt: on
        torch.testing.assert_close(preds, EXPECTED)
