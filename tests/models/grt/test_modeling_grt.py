# Copyright 2026 MatterWave AI and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GRT model."""

import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import AutoTokenizer, GRTConfig, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import GRTForCausalLM, GRTModel
    from transformers.models.grt.modeling_grt import GRTPearlNorm, GRTRadialGELU, _build_butterfly_perms


class GRTModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = GRTModel

    def __init__(self, parent, **kwargs):
        # Every projection rotates pairs of dimensions through butterfly permutations, so the widths must be even
        # and split into whole butterfly blocks.
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("intermediate_size", 64)
        kwargs.setdefault("ffn_n_rotations", 3)
        kwargs.setdefault("n_qkv_rotations", 2)
        kwargs.setdefault("n_mixer_rotations", 1)
        super().__init__(parent, **kwargs)


@require_torch
class GRTModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = GRTModelTester
    # The embedding is a table of phase angles plus log magnitudes, not an `nn.Embedding` weight.
    test_resize_embeddings = False

    # The phase-angle tables of the embedding and the cosine head keep a large gradient after the loss has converged:
    # on the fixed batch the loss falls ~98% but the total grad norm only ~78%.
    training_grad_norm_reduction_threshold = 0.7

    @unittest.skip(reason="`GRTEmbedding` stores phase angles and log magnitudes, not an `nn.Embedding`.")
    def test_model_get_set_embeddings(self):
        pass

    def test_config_validates_widths(self):
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "divisible by `num_attention_heads`"):
            GRTConfig(hidden_size=32, num_attention_heads=5)
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "`head_dim` \\(3\\) must be even"):
            GRTConfig(hidden_size=12, num_attention_heads=4)
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "at least `hidden_size`"):
            GRTConfig(hidden_size=64, num_attention_heads=2, intermediate_size=32)
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "`intermediate_size` \\(80\\) must be a"):
            GRTConfig(hidden_size=64, num_attention_heads=2, intermediate_size=80, ffn_n_rotations=5)

    def test_embedding_has_constant_norm(self):
        """With zero log magnitude every token has norm `sqrt(hidden_size / 2)`: each `(cos, sin)` pair adds 1."""
        config = self.model_tester.get_config()
        model = GRTModel(config).to(torch_device).eval()
        with torch.no_grad():
            model.embedding.log_magnitude.zero_()
            norms = model.embedding(torch.arange(config.vocab_size, device=torch_device)).norm(dim=-1)

        torch.testing.assert_close(norms, torch.full_like(norms, (config.hidden_size / 2) ** 0.5))

    def test_cosine_head_matches_explicit_cosine(self):
        """The head's closed form must equal the cosine against the full embedding table, magnitudes included."""
        config = self.model_tester.get_config()
        config.tie_word_embeddings = True
        model = GRTForCausalLM(config).to(torch_device).eval()
        self.assertIs(model.lm_head.phase_angles, model.model.embedding.phase_angles)
        hidden = torch.randn(2, 5, config.hidden_size, device=torch_device)
        with torch.no_grad():
            model.model.embedding.log_magnitude.uniform_(-2.0, 2.0)
            table = model.model.embedding(torch.arange(config.vocab_size, device=torch_device))
            expected = torch.nn.functional.cosine_similarity(hidden[..., None, :], table, dim=-1, eps=1e-12)
            expected = expected * torch.nn.functional.softplus(model.lm_head.output_scale)
            logits = model.lm_head(hidden)

        torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)

    def test_pearl_norm_preserves_pair_phase(self):
        norm = GRTPearlNorm(32).to(torch_device)
        torch.nn.init.ones_(norm.scale)
        x = torch.randn(2, 5, 32, device=torch_device)
        y = norm(x)

        x_pairs, y_pairs = x.view(2, 5, 16, 2), y.view(2, 5, 16, 2)
        torch.testing.assert_close(
            torch.atan2(y_pairs[..., 1], y_pairs[..., 0]), torch.atan2(x_pairs[..., 1], x_pairs[..., 0])
        )

    def test_radial_gelu_preserves_pair_phase(self):
        x = torch.randn(2, 5, 32, device=torch_device)
        y = GRTRadialGELU()(x)

        cos = torch.nn.functional.cosine_similarity(x.view(2, 5, 16, 2), y.view(2, 5, 16, 2), dim=-1)
        torch.testing.assert_close(cos, torch.ones_like(cos), rtol=1e-3, atol=1e-3)

    def test_qkv_projection_is_position_independent(self):
        """A position's Q/K/V depends on that position only, which is what makes the KV cache exact."""
        config = self.model_tester.get_config()
        attention = GRTModel(config).to(torch_device, torch.float64).eval().blocks[0].attention

        heads = torch.randn(6, attention.num_heads, attention.head_dim, device=torch_device, dtype=torch.float64)
        with torch.no_grad():
            full = attention._project(heads, attention.q_angles, attention.q_scales, attention.q_coupling)
            last = attention._project(heads[-1:], attention.q_angles, attention.q_scales, attention.q_coupling)
        torch.testing.assert_close(full[-1:], last, rtol=1e-10, atol=1e-10)

    def test_butterfly_permutation_is_a_bijection(self):
        perms, inv_perms = _build_butterfly_perms(6, 64)
        for perm, inv_perm in zip(perms, inv_perms):
            self.assertEqual(sorted(perm.tolist()), list(range(64)))
            torch.testing.assert_close(perm[inv_perm], torch.arange(64))

    def test_half_precision_positions_past_256(self):
        """Position angles are formed in fp32: a bf16 position past 256 is not even an exact integer."""
        config = self.model_tester.get_config()
        model = GRTModel(config).eval()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model_bf16 = GRTModel.from_pretrained(tmp_dir, dtype=torch.bfloat16).to(torch_device).eval()
            model_fp16 = GRTModel.from_pretrained(tmp_dir, dtype=torch.float16).to(torch_device).eval()
        self.assertEqual(model_bf16.pos_encoding.log_freqs.dtype, torch.float32)
        self.assertEqual(model_fp16.pos_encoding.log_freqs.dtype, torch.float32)

        hidden = torch.randn(1, 4, config.hidden_size).to(torch.bfloat16)
        position_ids = torch.tensor([[3, 257, 515, 1001]])
        input_ids = torch.tensor([[1, 2, 3, 4]], device=torch_device)
        with torch.no_grad():
            expected = model.pos_encoding(hidden.float(), position_ids)
            rotated = model_bf16.pos_encoding(hidden.to(torch_device), position_ids.to(torch_device))
            out_bf16 = model_bf16(input_ids, position_ids=position_ids.to(torch_device)).last_hidden_state
            out_fp16 = model_fp16(input_ids, position_ids=position_ids.to(torch_device)).last_hidden_state
        torch.testing.assert_close(rotated.float().cpu(), expected, rtol=1e-2, atol=1e-2)
        self.assertEqual(out_bf16.dtype, torch.bfloat16)
        self.assertEqual(out_fp16.dtype, torch.float16)

    def test_weight_quantization_8bit_and_4bit(self):
        """8-bit and 4-bit weight quantization reduces weight memory while keeping rotation angles in float."""
        config = self.model_tester.get_config()
        config.vocab_size = 256
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_attention_heads = 4
        input_ids = torch.tensor([[1, 7, 19, 42, 100]], device=torch_device)

        def _storage_bytes(m):
            seen = set()
            total = 0
            for t in (*m.parameters(), *m.buffers()):
                if t.data_ptr() not in seen:
                    seen.add(t.data_ptr())
                    total += t.numel() * t.element_size()
            return total

        torch.manual_seed(0)
        ref_model = GRTForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            ref_logits = ref_model(input_ids).logits
            fp32_bytes = _storage_bytes(ref_model)

        for bits, atol in ((8, 0.25), (4, 0.85)):
            torch.manual_seed(0)
            q_model = GRTForCausalLM(config).to(torch_device).eval()
            q_model.quantize_weights(bits=bits)
            q_bytes = _storage_bytes(q_model)
            self.assertLess(q_bytes, fp32_bytes)

            expected_q_dtype = torch.int8 if bits == 8 else torch.uint8
            self.assertEqual(q_model.model.embedding._table_q.dtype, expected_q_dtype)
            self.assertEqual(q_model.lm_head._unit_table_q.dtype, expected_q_dtype)
            for block in q_model.model.blocks:
                for name in ("q", "k", "v", "o"):
                    self.assertEqual(getattr(block.attention, f"{name}_angles").dtype, torch.float32)
                    self.assertEqual(getattr(block.attention, f"{name}_scales").dtype, torch.float32)
                    self.assertEqual(getattr(block.attention, f"{name}_coupling").dtype, expected_q_dtype)
                if block.mixer is not None:
                    self.assertEqual(block.mixer.angles.dtype, torch.float32)
                    self.assertEqual(block.mixer.scales.dtype, torch.float32)
                    self.assertEqual(block.mixer.coupling.dtype, expected_q_dtype)

            with torch.no_grad():
                q_logits = q_model(input_ids).logits
            torch.testing.assert_close(q_logits, ref_logits, atol=atol, rtol=0.1)


@slow
@require_torch
class GRTIntegrationTest(unittest.TestCase):
    model_id = "matterwaveai/grt-30m"
    prompt = "Once upon a time"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_logits(self):
        model = GRTForCausalLM.from_pretrained(self.model_id, dtype=torch.float32, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        input_ids = tokenizer(self.prompt, return_tensors="pt").input_ids.to(torch_device)
        self.assertEqual(input_ids.tolist(), [[12522, 5193, 264, 882]])

        with torch.no_grad():
            logits = model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor([[-15.9131, -14.4139, -12.3241, -16.8289]])
        EXPECTED_SLICE = torch.tensor(
            [
                -4.1509, -8.5352, -11.1400, -12.0285, -13.3995, -13.3531, -6.6003, -8.0093, -7.1135, -7.9299, -9.5767, 0.9591,
                -5.5828, -1.4786, -5.5291, -10.5926, -8.1160, -8.6207, -11.0523, -10.5673, -11.1221, -10.1382, -9.1870, -7.9439,
            ]
        )  # fmt: skip
        torch.testing.assert_close(logits.mean(-1), EXPECTED_MEAN, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(logits[0, -1, :24], EXPECTED_SLICE, rtol=1e-3, atol=1e-3)

    def test_model_generation(self):
        model = GRTForCausalLM.from_pretrained(self.model_id, dtype=torch.float32, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.prompt, return_tensors="pt").to(torch_device)

        generated = model.generate(**inputs, max_new_tokens=12, do_sample=False)

        EXPECTED_TOKENS = [11, 498, 2578, 387, 20293, 1128, 6896, 374, 279, 1616, 311, 653]
        EXPECTED_TEXT = "Once upon a time, you might be wondering what exactly is the way to do"
        self.assertEqual(generated[0, inputs.input_ids.shape[1] :].tolist(), EXPECTED_TOKENS)
        self.assertEqual(tokenizer.decode(generated[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_cache_matches_no_cache(self):
        """Decoding one token against the cache matches a full forward pass."""
        model = GRTForCausalLM.from_pretrained(self.model_id, dtype=torch.float32, device_map=torch_device)
        input_ids = torch.tensor([[12522, 5193, 264, 882, 11, 498, 2578, 387]], device=torch_device)

        with torch.no_grad():
            full = model(input_ids, use_cache=False).logits[:, -1]
            prefix = model(input_ids[:, :-1], use_cache=True)
            step = model(input_ids[:, -1:], past_key_values=prefix.past_key_values).logits[:, -1]

        torch.testing.assert_close(step, full, rtol=1e-4, atol=1e-4)
