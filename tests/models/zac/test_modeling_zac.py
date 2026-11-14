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
"""Testing suite for the PyTorch ZAC-1 model."""

import inspect
import json
import os
import unittest
from pathlib import Path

from huggingface_hub.errors import StrictDataclassClassValidationError

from tests.test_configuration_common import ConfigTester
from tests.test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from transformers import ZacConfig
from transformers.testing_utils import require_torch, require_torchaudio, slow, torch_device
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers import ZacModel


# TODO: move to final Zyphra repo
ZAC_CHECKPOINT = os.environ.get("ZAC_TEST_DIR", "Zyphra/ZAC-1-hf")


class ZacModelTester:
    def __init__(self, parent, batch_size=2, num_frames=8, lane_frames=4, lane_context_frames=2, is_training=False):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        # The real sampling rates keep the sinc resamplers valid; the hop is shrunk to 735 samples (60 Hz frames).
        self.hop_length = 735
        self.num_frames = num_frames
        self.num_samples = num_frames * self.hop_length
        self.lane_frames = lane_frames
        self.lane_context_frames = lane_context_frames
        # lanes of 4 frames re-reading 2: frames [0, 4), then 2 new frames per lane
        self.num_lanes = 1 + (num_frames - lane_frames + lane_context_frames - 1) // (
            lane_frames - lane_context_frames
        )
        # teacher features of one lane: 4 * 735 samples -> 1067 at 16 kHz -> 5 fbank frames -> 3 stacked frames
        self.num_feature_frames = 3

    def get_config(self):
        return ZacConfig(
            hop_length=self.hop_length,
            internal_hop_length=736,
            encoder_hidden_size=4,
            encoder_strides=[2, 4],
            encoder_patch_sizes=[23, 4],
            encoder_input_sizes=[368, 64],
            encoder_hidden_sizes=[32, 32],
            encoder_output_sizes=[16, 32],
            encoder_num_attention_heads=[2, 2],
            encoder_num_hidden_layers=[1, 1],
            encoder_intermediate_sizes=[64, 64],
            encoder_sliding_windows=[8, 8],
            hidden_size=32,
            semantic_hidden_size=16,
            semantic_intermediate_size=32,
            semantic_num_layers=1,
            decoder_input_sizes=[32, 16],
            decoder_hidden_sizes=[32, 32],
            decoder_output_sizes=[32, None],
            decoder_num_attention_heads=[2, 2],
            decoder_num_hidden_layers=[1, 1],
            decoder_intermediate_sizes=[64, 64],
            decoder_sliding_windows=[8, 8],
            head_dim=16,
            head_num_layers=1,
            head_intermediate_size=64,
            head_hop_length=368,
            head_n_ffts=[736, 1472],
            codebook_size=16,
            lane_frames=self.lane_frames,
            lane_context_frames=self.lane_context_frames,
            semantic_model_config={
                "model_type": "wav2vec2-bert",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 64,
                "feature_projection_input_dim": 160,
            },
        )

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, 1, self.num_samples], scale=0.5)
        input_features = floats_tensor([self.batch_size, self.num_lanes, self.num_feature_frames, 160], scale=1.0)
        inputs_dict = {"input_values": input_values, "input_features": input_features}
        return self.get_config(), inputs_dict

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()

    def create_and_check_model_forward(self, config, inputs_dict):
        model = ZacModel(config=config).to(torch_device).eval()
        result = model(**inputs_dict)
        self.parent.assertEqual(result.audio_values.shape, (self.batch_size, 1, self.num_samples))
        self.parent.assertEqual(result.audio_codes.shape, (self.batch_size, config.num_quantizers, self.num_frames))
        self.parent.assertTrue(bool((result.audio_codes < config.codebook_size).all()))

    def create_and_check_lanes_match_single_pass(self, config, inputs_dict):
        """With lanes long enough for the whole clip, the codes are those of one lane: the lane schedule only splits
        longer audio."""
        model = ZacModel(config=config).to(torch_device).eval()
        input_values = inputs_dict["input_values"][:1]
        codes = model.encode(input_values, inputs_dict["input_features"][:1]).audio_codes
        self.parent.assertEqual(codes.shape[-1], self.num_frames)
        # the first lane covers the first `lane_frames` frames without context: same codes as a clip of that length
        short = input_values[..., : self.lane_frames * self.hop_length]
        short_codes = model.encode(short, inputs_dict["input_features"][:1, :1]).audio_codes
        torch.testing.assert_close(codes[..., : self.lane_frames], short_codes)


def load_reference_clip(seconds: float = 5.0) -> "torch.Tensor":
    """`hf-internal-testing/librispeech_asr_dummy` validation sample 0 (16 kHz FLAC, decoded losslessly), resampled to
    44.1 kHz with `torchaudio.functional.resample` and cut to `seconds`, as in the reproducer."""
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    import torchaudio
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet", repo_type="dataset"
    )
    audio, _ = sf.read(io.BytesIO(pq.read_table(path).to_pylist()[0]["audio"]["bytes"]), dtype="float32")
    audio = torchaudio.functional.resample(torch.from_numpy(audio)[None], 16000, 44100)[0]
    return audio[: int(seconds * 44100)]


@require_torch
class ZacModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ZacModel,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_resize_embeddings = False
    test_torch_exportable = False
    additional_model_inputs = ["input_features", "padding_mask", "input_features_mask"]

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        inputs_dict.pop("output_attentions", None)
        inputs_dict.pop("output_hidden_states", None)
        return inputs_dict

    def setUp(self):
        self.model_tester = ZacModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=ZacConfig, hidden_size=32, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        self.model_tester.create_and_check_model_forward(*self.model_tester.prepare_config_and_inputs())

    def test_lanes_match_single_pass(self):
        self.model_tester.create_and_check_lanes_match_single_pass(*self.model_tester.prepare_config_and_inputs())

    def test_config_hop_validation(self):
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "internal_hop_length"):
            ZacConfig(encoder_strides=[2, 2])

    def test_decode_rejects_out_of_range_codes(self):
        config = self.model_tester.get_config()
        model = ZacModel(config).to(torch_device).eval()
        codes = ids_tensor([1, config.num_quantizers, 2], config.codebook_size)
        codes[0, 3, 1] = config.codebook_size
        with self.assertRaises(ValueError):
            model.decode(codes)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = ZacModel(config)
        arg_names = [*inspect.signature(model.forward).parameters.keys()]
        self.assertListEqual(arg_names[:4], ["input_values", "input_features", "padding_mask", "input_features_mask"])

    @unittest.skip(reason="ZacModel has no input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="ZacModel does not return attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="ZacModel does not return hidden states")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="ZacModel does not return hidden states or attentions")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(
        reason="The Wav2Vec2-BERT teacher uses relative position embeddings that produce a dense attention bias "
        "incompatible with Flash Attention"
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="The codes are discrete: the quantizers' nearest-neighbour search is not differentiable")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="The codes are discrete: the quantizers' nearest-neighbour search is not differentiable")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="The codes are discrete: the quantizers' nearest-neighbour search is not differentiable")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skip(
        reason="ZAC-1 computes its RoPE frequencies as `exp(-2 i log(theta) / dim)`; only the default rope type is used"
    )
    def test_model_rope_scaling_frequencies(self):
        pass


@slow
@require_torch
@require_torchaudio
class ZacIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.fixtures_path = Path(__file__).parent.parent.parent / "fixtures/zac"

    def test_decode(self):
        """
        Reproducer: zonos3_repro/reproducer.py `decode` (ZAC-1 `zac.load_decoder("keep2_plain")` on CPU in float32).
        """
        with open(self.fixtures_path / "expected_decode.json", encoding="utf-8") as f:
            expected = json.load(f)
        model = ZacModel.from_pretrained(ZAC_CHECKPOINT, dtype=torch.float32).to(torch_device).eval()
        audio_codes = torch.tensor(expected["audio_codes"], device=torch_device)[None]
        with torch.no_grad():
            audio_values = model.decode(audio_codes).audio_values
        self.assertEqual(audio_values.shape[-1], expected["num_samples"])
        torch.testing.assert_close(
            audio_values[0, 0, :4000].cpu(), torch.tensor(expected["audio_values"]), atol=1e-4, rtol=1e-4
        )

    def test_encode(self):
        """
        Reproducer: zonos3_repro/reproducer.py `encode` (ZAC-1 encoder with its lane schedule on CPU in float32).
        """
        from transformers import ZacFeatureExtractor

        with open(self.fixtures_path / "expected_encode.json", encoding="utf-8") as f:
            expected = json.load(f)
        feature_extractor = ZacFeatureExtractor.from_pretrained(ZAC_CHECKPOINT)
        model = ZacModel.from_pretrained(ZAC_CHECKPOINT, dtype=torch.float32).to(torch_device).eval()
        audio = load_reference_clip()

        inputs = feature_extractor(audio.numpy(), sampling_rate=44100, return_tensors="pt").to(torch_device)
        self.assertEqual(inputs["input_values"].shape[-1], expected["input_values_len"])
        with torch.no_grad():
            audio_codes = model.encode(**inputs).audio_codes
        torch.testing.assert_close(audio_codes[0].cpu(), torch.tensor(expected["audio_codes"]), atol=0, rtol=0)
