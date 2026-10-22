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

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow


if is_torch_available():
    import torch
    from safetensors.torch import save_file
    from torch.nn import functional as F

    from transformers import AutoModelForMaskedLM, AutoTokenizer, BertTokenizer, EuroBertForMaskedLM
    from transformers.models.eurobert.convert_neobert_to_eurobert import (
        convert_config,
        convert_neobert_checkpoint,
        convert_state_dict,
    )


@require_torch
class NeoBertConversionTest(unittest.TestCase):
    def setUp(self):
        self.original_config = {
            "architectures": ["NeoBERTLMHead"],
            "auto_map": {"AutoModelForMaskedLM": "model.NeoBERTLMHead"},
            "model_type": "neobert",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "max_length": 64,
            "norm_eps": 1e-5,
            "pad_token_id": 0,
        }
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"word{i}" for i in range(27)]
        self.tokenizer = BertTokenizer(vocab={token: index for index, token in enumerate(vocab)})
        self.config = convert_config(self.original_config, 32, self.tokenizer)
        generator = torch.Generator().manual_seed(0)

        def weight(*shape):
            return torch.randn(*shape, generator=generator) * 0.1

        self.state_dict = {
            "model.encoder.weight": weight(32, 16),
            "model.layer_norm.weight": 1 + weight(16),
            "decoder.weight": weight(32, 16),
            "decoder.bias": weight(32),
        }
        for layer_idx in range(2):
            prefix = f"model.transformer_encoder.{layer_idx}"
            self.state_dict.update(
                {
                    f"{prefix}.qkv.weight": weight(48, 16),
                    f"{prefix}.wo.weight": weight(16, 16),
                    f"{prefix}.ffn.w12.weight": weight(64, 16),
                    f"{prefix}.ffn.w3.weight": weight(16, 32),
                    f"{prefix}.attention_norm.weight": 1 + weight(16),
                    f"{prefix}.ffn_norm.weight": 1 + weight(16),
                }
            )

    def original_logits(self, input_ids, attention_mask, position_ids):
        # Evaluate the source layout directly: fused per-head QKV, adjacent complex RoPE pairs, and fused SwiGLU.
        state = self.state_dict
        hidden = F.embedding(input_ids, state["model.encoder.weight"])
        frequencies = 10000.0 ** (-torch.arange(0, 8, 2).float() / 8)
        angles = position_ids[..., None] * frequencies
        rotation = torch.polar(torch.ones_like(angles), angles).unsqueeze(2)
        mask = attention_mask[:, None, None, :].bool()
        for layer_idx in range(2):
            prefix = f"model.transformer_encoder.{layer_idx}"
            normed = F.rms_norm(hidden, (16,), state[f"{prefix}.attention_norm.weight"], 1e-5)
            qkv = F.linear(normed, state[f"{prefix}.qkv.weight"]).reshape(*input_ids.shape, 2, 24)
            query, key, value = qkv.chunk(3, dim=-1)

            def rotate(tensor):
                pairs = torch.view_as_complex(tensor.float().reshape(*tensor.shape[:-1], 4, 2))
                return torch.view_as_real(pairs * rotation).flatten(-2)

            attention = F.scaled_dot_product_attention(
                rotate(query).transpose(1, 2), rotate(key).transpose(1, 2), value.transpose(1, 2), attn_mask=mask
            )
            hidden = hidden + F.linear(attention.transpose(1, 2).reshape(*hidden.shape), state[f"{prefix}.wo.weight"])
            normed = F.rms_norm(hidden, (16,), state[f"{prefix}.ffn_norm.weight"], 1e-5)
            gate, up = F.linear(normed, state[f"{prefix}.ffn.w12.weight"]).chunk(2, dim=-1)
            hidden = hidden + F.linear(F.silu(gate) * up, state[f"{prefix}.ffn.w3.weight"])
        hidden = F.rms_norm(hidden, (16,), state["model.layer_norm.weight"], 1e-5)
        return F.linear(hidden, state["decoder.weight"], state["decoder.bias"])

    @parameterized.expand([("eager", False), ("eager", True), ("sdpa", False), ("sdpa", True)])
    def test_logits_match_original(self, backend, padding):
        self.config._attn_implementation = backend
        model = EuroBertForMaskedLM(self.config).eval()
        model.load_state_dict(convert_state_dict(self.state_dict, self.config), strict=True)
        input_ids = torch.tensor([[2, 5, 4, 6, 3, 0], [2, 7, 8, 9, 10, 3]])
        attention_mask = input_ids.ne(0) if padding else torch.ones_like(input_ids)
        # Nonzero, different positions per batch exercise RoPE beyond the first token.
        position_ids = torch.arange(6).unsqueeze(0) + torch.tensor([[3], [11]])
        with torch.no_grad():
            expected = self.original_logits(input_ids, attention_mask, position_ids)
            actual = model(input_ids, attention_mask=attention_mask, position_ids=position_ids).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_conversion_and_auto_loading(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as output_dir:
            source = Path(source_dir)
            (source / "config.json").write_text(json.dumps(self.original_config), encoding="utf-8")
            save_file(self.state_dict, source / "model.safetensors")
            self.tokenizer.save_pretrained(source_dir)
            convert_neobert_checkpoint(source_dir, output_dir)
            model, loading_info = AutoModelForMaskedLM.from_pretrained(
                output_dir, trust_remote_code=False, output_loading_info=True
            )
            tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=False)
            self.assertIsInstance(model, EuroBertForMaskedLM)
            self.assertFalse(loading_info["missing_keys"])
            self.assertFalse(loading_info["unexpected_keys"])
            self.assertEqual(model.config.model_type, "eurobert")
            self.assertEqual(model.config.intermediate_size, 32)
            self.assertEqual(model.config.max_position_embeddings, 64)
            self.assertEqual(model.config.mask_token_id, tokenizer.mask_token_id)
            self.assertEqual(model.config.bos_token_id, tokenizer.cls_token_id)
            self.assertEqual(model.config.eos_token_id, tokenizer.sep_token_id)
            self.assertFalse(hasattr(model.config, "auto_map"))
            self.assertEqual(tokenizer.model_max_length, 64)
            inputs = tokenizer(["word0 [MASK]", "word1 word2 word3"], padding=True, return_tensors="pt")
            self.assertNotIn("token_type_ids", inputs)
            with torch.no_grad():
                actual = model(**inputs).logits
                expected = self.original_logits(
                    inputs.input_ids, inputs.attention_mask, torch.arange(inputs.input_ids.shape[1]).unsqueeze(0)
                )
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            self.assertNotEqual(model.lm_head.weight.data_ptr(), model.model.embed_tokens.weight.data_ptr())

    def test_unexpected_weights_are_rejected(self):
        state_dict = {**self.state_dict, "classifier.weight": torch.zeros(2, 16)}
        with self.assertRaisesRegex(ValueError, "classifier.weight"):
            convert_state_dict(state_dict, self.config)

    def test_classification_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as output_dir:
            config = {**self.original_config, "architectures": ["NeoBERTForSequenceClassification"]}
            (Path(source_dir) / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "NeoBERTLMHead"):
                convert_neobert_checkpoint(source_dir, output_dir)


@require_torch
class NeoBertConversionIntegrationTest(unittest.TestCase):
    @slow
    def test_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as output_dir:
            convert_neobert_checkpoint(
                "chandar-lab/NeoBERT", output_dir, revision="5424c8efeea6491b151d62dee55a752165407430"
            )
            model = AutoModelForMaskedLM.from_pretrained(output_dir, trust_remote_code=False)
            tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=False)
            inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
            mask_index = inputs.input_ids[0].tolist().index(tokenizer.mask_token_id)
            # Source NeoBERTLMHead in float32, using unfused SwiGLU and its original complex RoPE buffer.
            expected = torch.tensor([9.3852119, -2.2100248, 1.5240648])
            for backend in ("eager", "sdpa"):
                with self.subTest(backend=backend):
                    model.set_attn_implementation(backend)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    self.assertTrue(torch.isfinite(logits).all())
                    torch.testing.assert_close(logits[0, mask_index, 3000:3003], expected, atol=2e-4, rtol=1e-5)
                    self.assertEqual(tokenizer.decode(logits[0, mask_index].argmax()), "paris")
