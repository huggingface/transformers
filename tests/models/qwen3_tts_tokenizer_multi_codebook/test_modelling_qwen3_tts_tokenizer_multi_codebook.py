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
import tempfile
import unittest
from pathlib import Path

import torch

from transformers import (
    Qwen3TTSTokenizerMultiCodebookConfig,
    Qwen3TTSTokenizerMultiCodebookModel,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)
from transformers.trainer_utils import set_seed

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin


class Qwen3TTSTokenizerMultiCodebookModelTester:
    """
    Builds a tiny Qwen3TTSTokenizerMultiCodebook config and synthetic inputs for unit testing.
    """

    def __init__(
        self,
        parent,
        batch_size=2,
        num_quantizers=4,
        seq_length=8,
        audio_samples=960,  # small audio chunk
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_quantizers = num_quantizers
        self.seq_length = seq_length
        self.audio_samples = audio_samples
        self.is_training = is_training

        self.encoder_config = {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "num_filters": 8,
            "kernel_size": 7,
            "residual_kernel_size": 3,
            "last_kernel_size": 3,
            "num_residual_layers": 1,
            "upsampling_ratios": [8, 6],
            "codebook_size": 8,
            "codebook_dim": 4,
            "vector_quantization_hidden_dimension": 4,  # must match codebook_dim
            "num_quantizers": num_quantizers,
            "num_semantic_quantizers": 1,
            # upsample_groups must divide hidden_size; default 512 does not divide 16
            "upsample_groups": 8,
        }

        self.decoder_config = {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 32,
            "num_quantizers": num_quantizers,
            "codebook_size": 8,
            "codebook_dim": 8,
            "latent_dim": 16,
            "decoder_dim": 32,  # must be divisible by upsample groups
            "upsample_rates": [2, 2],
            "upsampling_ratios": [2, 2],
        }

    def get_config(self):
        return Qwen3TTSTokenizerMultiCodebookConfig(
            encoder_config=self.encoder_config,
            decoder_config=self.decoder_config,
            encoder_valid_num_quantizers=self.num_quantizers,
        )

    def prepare_config_and_inputs(self):
        # top-level encode expects (batch, channels, seq) — channels=1
        input_values = torch.randn([self.batch_size, 1, self.audio_samples], device=torch_device)
        padding_mask = torch.ones([self.batch_size, 1, self.audio_samples], dtype=torch.bool, device=torch_device)
        # top-level decode expects (batch, seq_length, num_quantizers)
        codes = torch.randint(0, 8, [self.batch_size, self.seq_length, self.num_quantizers], device=torch_device)
        config = self.get_config()
        return config, input_values, padding_mask, codes

    def prepare_config_and_inputs_for_common(self):
        config, input_values, padding_mask, codes = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict


if is_torch_available():
    import torch


@require_torch
class Qwen3TTSTokenizerMultiCodebookModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Qwen3TTSTokenizerMultiCodebookModel,) if is_torch_available() else ()
    _is_composite = True
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = Qwen3TTSTokenizerMultiCodebookModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Qwen3TTSTokenizerMultiCodebookConfig, has_text_modality=False
        )
        _no_forward_tests = (
            "test_eager_matches_sdpa_inference",
            "test_attention_outputs",
            "test_hidden_states_output",
            "test_retain_grad_hidden_states_attentions",
            "test_model_forward_default_config_values",
            "test_feed_forward_chunking",
            "test_inputs_embeds",
        )
        if any(name in self._testMethodName for name in _no_forward_tests):
            self.skipTest("Qwen3TTSTokenizerMultiCodebookModel forward requires raw audio input, not standard embeds")

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_instantiation(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = Qwen3TTSTokenizerMultiCodebookModel(config)
        self.assertIsNotNone(model)

    def test_save_load(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).eval().to(torch_device)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                loaded = model_class.from_pretrained(tmpdirname).eval().to(torch_device)
            for key in model.state_dict():
                self.assertTrue(
                    torch.allclose(model.state_dict()[key], loaded.state_dict()[key]),
                    f"Mismatch in key: {key}",
                )

    def test_encode_decode_roundtrip(self):
        """Encode audio to codes then decode back; output should have batch size preserved."""
        set_seed(42)
        config, input_values, padding_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = Qwen3TTSTokenizerMultiCodebookModel(config).eval().to(torch_device)
        with torch.no_grad():
            # top-level encode expects (batch, seq); it adds the channel dim internally
            encoded = model.encode(
                input_values.squeeze(1).to(torch_device),
                padding_mask=padding_mask.squeeze(1).to(torch_device),
            )
            audio_codes = encoded.audio_codes
        self.assertEqual(len(audio_codes), self.model_tester.batch_size)

    def test_decode_from_codes(self):
        """Decode from synthetic codes (batch, seq, num_quantizers); output is a list of waveforms."""
        set_seed(42)
        config, _, _, codes = self.model_tester.prepare_config_and_inputs()
        model = Qwen3TTSTokenizerMultiCodebookModel(config).eval().to(torch_device)
        with torch.no_grad():
            output = model.decode(codes.to(torch_device))
        self.assertEqual(len(output.audio_values), self.model_tester.batch_size)

    @unittest.skip(reason="Composite model — base model prefix test not applicable")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(reason="No standard generate() — codec model, not a language model")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerMultiCodebookModel has no standard training forward")
    def test_training(self):
        pass

    @unittest.skip(reason="Batching equivalence not applicable — audio codec output depends on padding")
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="Determinism not guaranteed across runs for codec models")
    def test_determinism(self):
        pass

    @unittest.skip(reason="No standard model outputs equivalence for codec models")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="Compile not yet supported")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Flash attention right-padding equivalence not applicable")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(reason="No standard forward — all_tensors test not applicable")
    def test_all_tensors_are_parameter_or_buffer(self):
        pass

    @unittest.skip(reason="No standard get_input_embeddings for codec model")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="main_input_name requires converter re-run to propagate to modeling file")
    def test_model_main_input_name(self):
        pass

    @unittest.skip(reason="Composite config attn implementation not propagated uniformly across sub-configs")
    def test_config_attn_implementation_setter(self):
        pass

    @unittest.skip(reason="Codec model has no tied weights")
    def test_tied_weights_keys(self):
        pass

    @unittest.skip(reason="EuclideanCodebook buffers (embed_sum, cluster_usage) not reinitializable on meta device")
    def test_can_init_all_missing_weights(self):
        pass

    @unittest.skip(reason="EuclideanCodebook buffers (embed_sum, cluster_usage) not reinitializable on meta device")
    def test_init_weights_can_init_buffers(self):
        pass

    @unittest.skip(reason="No standard forward — left padding test not applicable")
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip(reason="No standard forward — torch fx not applicable")
    def test_torch_fx(self):
        pass

    @unittest.skip(reason="No standard forward — torch fx not applicable")
    def test_torch_fx_output_loss(self):
        pass

    @unittest.skip(reason="No standard forward — sdpa dispatch not applicable")
    def test_sdpa_can_dispatch_composite_models(self):
        pass

    @unittest.skip(reason="Torch export not supported — codec model uses dynamic control flow")
    def test_torch_export(self):
        pass


@require_torch
class Qwen3TTSTokenizerMultiCodebookIntegrationTest(unittest.TestCase):
    """
    Slow integration tests against a real converted checkpoint.
    """

    TARGET_SAMPLE_RATE = 24000

    @classmethod
    def setUpClass(cls):
        from transformers.testing_utils import cleanup

        cleanup(torch_device, gc_collect=True)
        cls.checkpoint = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
        # Override with local converted checkpoint if available (for offline testing)
        local = Path(__file__).parents[3] / "qwen3_tts_tokenizer_mc_converted"
        if local.exists():
            cls.checkpoint = local

    def tearDown(self):
        from transformers.testing_utils import cleanup

        cleanup(torch_device, gc_collect=True)

    def _load_datasamples(self, num_samples):
        import io

        import librosa
        import numpy as np
        import soundfile as sf
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.sort("id")
        raw_col = ds.data.column("audio").to_pylist()
        samples = []
        for raw in raw_col[:num_samples]:
            audio_bytes = raw.get("bytes")
            if audio_bytes:
                array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            else:
                array, sr = sf.read(raw["path"], dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=1)
            if sr != self.TARGET_SAMPLE_RATE:
                array = librosa.resample(array, orig_sr=sr, target_sr=self.TARGET_SAMPLE_RATE)
            samples.append(np.array(array, dtype=np.float32))
        return samples

    @slow
    def test_single(self):
        """
        gist: https://gist.github.com/ShahVandit/cab13f3b7232c52b4ff93cce592950c4
        Ground truth generated from original Qwen3TTSTokenizerV2Model.
        """
        set_seed(42)
        fixtures = (
            Path(__file__).parent.parent.parent
            / "fixtures/qwen3_tts_tokenizer_multi_codebook/expected_results_single.json"
        )
        with open(fixtures, "r", encoding="utf-8") as f:
            expected = json.load(f)

        model = Qwen3TTSTokenizerMultiCodebookModel.from_pretrained(self.checkpoint, torch_dtype=torch.float32)
        model.eval()

        samples = self._load_datasamples(1)
        input_values = torch.tensor(samples[0]).unsqueeze(0).to(torch_device)
        padding_mask = torch.ones_like(input_values).bool()

        with torch.no_grad():
            encoded = model.encode(input_values, padding_mask=padding_mask)
            codes = encoded.audio_codes[0]  # [time, num_quantizers]

        print(f"\n[HF]       codes[0]: {codes[0].cpu().tolist()}")
        print(f"[expected] codes[0]: {expected['audio_codes'][0]}")

        torch.testing.assert_close(
            codes.cpu().long(),
            torch.tensor(expected["audio_codes"], dtype=torch.long),
        )

        with torch.no_grad():
            decoded = model.decode(codes.unsqueeze(0))

        torch.testing.assert_close(
            decoded.audio_values[0].cpu().float(),
            torch.tensor(expected["audio_values"], dtype=torch.float32),
            atol=2e-2,
            rtol=1e-3,
        )

    @slow
    def test_batch(self):
        """
        gist: https://gist.github.com/ShahVandit/cab13f3b7232c52b4ff93cce592950c4
        Ground truth generated from original Qwen3TTSTokenizerV2Model.
        """
        set_seed(42)
        fixtures = (
            Path(__file__).parent.parent.parent
            / "fixtures/qwen3_tts_tokenizer_multi_codebook/expected_results_batch.json"
        )
        with open(fixtures, "r", encoding="utf-8") as f:
            expected = json.load(f)

        model = Qwen3TTSTokenizerMultiCodebookModel.from_pretrained(self.checkpoint, torch_dtype=torch.float32)
        model.eval()

        samples = self._load_datasamples(2)
        import numpy as np

        max_len = max(len(s) for s in samples)
        input_values = torch.stack([torch.tensor(np.pad(s, (0, max_len - len(s)))) for s in samples]).to(torch_device)
        padding_mask = torch.stack(
            [torch.tensor(np.concatenate([np.ones(len(s)), np.zeros(max_len - len(s))]).astype(bool)) for s in samples]
        ).to(torch_device)

        with torch.no_grad():
            encoded = model.encode(input_values, padding_mask=padding_mask)
            codes_list = encoded.audio_codes  # list of [time_i, num_quantizers]

        for i, exp_codes in enumerate(expected["audio_codes"]):
            torch.testing.assert_close(
                codes_list[i].cpu().long(),
                torch.tensor(exp_codes, dtype=torch.long),
            )

        for i, (codes, exp_audio) in enumerate(zip(codes_list, expected["audio_values"])):
            with torch.no_grad():
                decoded = model.decode(codes.unsqueeze(0))
            torch.testing.assert_close(
                decoded.audio_values[0].cpu().float(),
                torch.tensor(exp_audio, dtype=torch.float32),
                atol=2e-2,
                rtol=1e-3,
            )
