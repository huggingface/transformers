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

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    from safetensors.torch import load_file, save_file

    from transformers import KlmForCausalLM
    from transformers.core_model_loading import Stack, Unbind
    from transformers.models.klm.convert_klm_weights_to_hf import convert as convert_checkpoint
    from transformers.models.klm.convert_klm_weights_to_hf import convert_config, convert_state_dict


@require_torch
class KlmCheckpointConversionTest(unittest.TestCase):
    def test_default_checkpoint_reloads_directly_without_replacing_parameters(self):
        # Unsloth DecisionTrainer uses encoder.load_state_dict(), not from_pretrained(),
        # when restoring its best checkpoint or resuming a full fine-tune.
        with tempfile.TemporaryDirectory() as source:
            self._original_checkpoint(source)
            for auto_class in (AutoModel, AutoModelForCausalLM):
                with self.subTest(auto_class=auto_class.__name__):
                    model = auto_class.from_pretrained(source)
                    optimizer = torch.optim.AdamW(model.parameters())
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.add_(0.125)
                    expected = {key: value.clone() for key, value in model.state_dict().items()}
                    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
                    with tempfile.TemporaryDirectory() as saved:
                        model.save_pretrained(saved)
                        state = load_file(str(Path(saved) / "model.safetensors"))
                        self.assertTrue(any("attn_res_query" in key for key in state))
                        keys = list(state)
                        with torch.no_grad():
                            for parameter in model.parameters():
                                parameter.zero_()
                        missing, unexpected = model.load_state_dict(state, strict=False)
                        self.assertEqual(missing, [])
                        self.assertEqual(unexpected, [])
                        self.assertEqual(list(state), keys)
                        model.load_state_dict(state, strict=True)
                    self.assertEqual(
                        {name: id(parameter) for name, parameter in model.named_parameters()}, parameter_ids
                    )
                    self.assertEqual(
                        {id(p) for group in optimizer.param_groups for p in group["params"]},
                        set(parameter_ids.values()),
                    )
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_partial_original_projection_and_assign(self):
        with tempfile.TemporaryDirectory() as source:
            model = self._original_checkpoint(source)
        parameter = model.model.layers[0].linear_attn.b_proj.weight
        value = torch.randn_like(parameter, dtype=torch.float64)
        state = {"model.blocks.0.mixer.b_proj.weight": value}
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        self.assertNotIn("model.layers.0.linear_attn.b_proj.weight", missing)
        self.assertEqual(unexpected, [])
        self.assertEqual(model.model.layers[0].linear_attn.b_proj.weight.dtype, torch.float64)
        torch.testing.assert_close(model.model.layers[0].linear_attn.b_proj.weight, value, rtol=0, atol=0)
        self.assertEqual(list(state), ["model.blocks.0.mixer.b_proj.weight"])

    def test_incomplete_original_convolution_does_not_partially_load(self):
        with tempfile.TemporaryDirectory() as source:
            model = self._original_checkpoint(source)
            original = load_file(str(Path(source) / "model.safetensors"))
        before = model.model.norm.weight.detach().clone()
        partial = {
            "model.blocks.0.mixer.q_conv1d.weight": original["model.blocks.0.mixer.q_conv1d.weight"],
            "model.norm.weight": torch.zeros_like(before),
        }
        with self.assertRaisesRegex(ValueError, "convolution"):
            model.load_state_dict(partial, strict=False)
        torch.testing.assert_close(model.model.norm.weight, before, rtol=0, atol=0)

    def test_mixed_original_and_native_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as source:
            model = self._original_checkpoint(source)
            original = load_file(str(Path(source) / "model.safetensors"))
        original["model.layers.0.input_residual.query"] = model.model.layers[0].input_residual.query.detach().clone()
        with self.assertRaisesRegex(ValueError, "original and native"):
            model.load_state_dict(original, strict=False)

    def test_native_partial_state_keeps_pytorch_strictness(self):
        with tempfile.TemporaryDirectory() as source:
            model = self._original_checkpoint(source)
        state = {"model.norm.weight": model.model.norm.weight.detach().clone(), "unknown.weight": torch.zeros(1)}
        missing, unexpected = model.load_state_dict(state, strict=False)
        self.assertTrue(missing)
        self.assertEqual(unexpected, ["unknown.weight"])
        with self.assertRaises(RuntimeError):
            model.load_state_dict(state)

    def test_untied_output_weight_is_not_inferred(self):
        with tempfile.TemporaryDirectory() as source:
            model = self._original_checkpoint(source)
            original = load_file(str(Path(source) / "model.safetensors"))
        model.config.tie_word_embeddings = False
        model.lm_head = torch.nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False)
        expected = model.lm_head.weight.detach().clone()
        missing, unexpected = model.load_state_dict(original, strict=False)
        self.assertEqual(missing, ["lm_head.weight"])
        self.assertEqual(unexpected, [])
        torch.testing.assert_close(model.lm_head.weight, expected, rtol=0, atol=0)

    def test_partial_and_malformed_original_residual_bank(self):
        with tempfile.TemporaryDirectory() as source:
            model = self._original_checkpoint(source)
            original = load_file(str(Path(source) / "model.safetensors"))
        rows = original["model.attn_res_query"]
        missing, unexpected = model.load_state_dict({"model.attn_res_query": rows}, strict=False)
        self.assertEqual(unexpected, [])
        self.assertNotIn("model.layers.0.input_residual.query", missing)
        torch.testing.assert_close(model.model.layers[0].input_residual.query, rows[0], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "residual bank"):
            model.load_state_dict({"model.attn_res_query": rows[:-1]}, strict=False)

    def test_converter_writes_native_checkpoint(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            self._original_checkpoint(source)
            # Tokenizer copying is independent of the weight layout under test.
            with patch("transformers.models.klm.convert_klm_weights_to_hf.AutoTokenizer.from_pretrained"):
                convert_checkpoint(source, target)
            state = load_file(str(Path(target) / "model.safetensors"))
            self.assertIn("model.layers.0.linear_attn.conv1d.weight", state)
            self.assertNotIn("model.attn_res_query", state)
            restored = AutoModelForCausalLM.from_pretrained(target)
            missing, unexpected = restored.load_state_dict(state, strict=False)
            self.assertLessEqual(set(missing), {"lm_head.weight"})
            self.assertEqual(unexpected, [])

    def test_residual_row_conversion(self):
        matrix = torch.arange(12).reshape(3, 4)
        split = Unbind(dim=0)
        rows = split.convert({"bank": [matrix]}, ["bank"], ["a", "b", "c"])
        self.assertEqual(rows["a"].shape, (4,))
        # Collection order must not change the original row order when saving.
        restored = split.reverse_op.convert({key: [rows[key]] for key in ("c", "a", "b")}, ["a", "b", "c"], ["bank"])
        torch.testing.assert_close(restored["bank"], matrix, rtol=0, atol=0)
        for inputs, targets in (
            ({}, ["a", "b", "c"]),
            ({"bank": [matrix, matrix]}, ["a", "b", "c"]),
            ({"bank": matrix, "other": matrix}, ["a", "b", "c"]),
            ({"bank": matrix}, ["a", "b"]),
            ({"bank": matrix}, ["a", "a", "c"]),
        ):
            with self.assertRaises(ValueError):
                split.convert(inputs, ["bank"], targets)
        for inputs, sources in (
            ({"a": [matrix, matrix]}, ["a"]),
            ({"a": matrix, "b": matrix}, ["a"]),
            ({"a": matrix}, ["a", "b"]),
            ({"a": matrix}, ["a", "a"]),
            ({}, []),
        ):
            with self.assertRaises(ValueError):
                Stack().convert(inputs, sources, ["bank"])

    def _original_checkpoint(self, directory):
        config = {
            "model_type": "klm",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "attention_every": 2,
            "gdn_heads": 2,
            "gdn_head_dim": 8,
            "gdn_norm_eps": 2e-5,
            "decay_floor": -12.0,
            "n_bigrams": 7,
            "n_trigrams": 5,
            "ngram_dim": 4,
            "rope_theta": 12345.0,
            "tie_word_embeddings": True,
        }
        model = KlmForCausalLM(convert_config(config)).eval()
        original = {}
        residuals = {"query": [], "weight": []}
        for name, tensor in model.state_dict().items():
            if name == "lm_head.weight":
                continue
            if name.endswith(("_residual.query", "_residual.weight")):
                residuals[name.rsplit(".", 1)[1]].append(torch.randn_like(tensor))
                continue
            name = name.replace("model.embed_tokens.", "model.embed.")
            name = name.replace("model.embed_ngrams.", "model.memory.")
            name = name.replace("bigram_embeddings.", "bigram.").replace("trigram_embeddings.", "trigram.")
            name = name.replace("model.layers.", "model.blocks.")
            name = name.replace(".input_layernorm.", ".mixer_norm.")
            name = name.replace(".post_attention_layernorm.", ".ffn_norm.")
            name = re.sub(r"\.mlp\.(gate|up|down)_proj\.", r".ffn.\1.", name)
            name = name.replace(".self_attn.", ".mixer.").replace(".linear_attn.", ".mixer.")
            name = re.sub(r"\.(f|g)_a_proj\.", r".\1_proj.0.", name)
            name = re.sub(r"\.(f|g)_b_proj\.", r".\1_proj.1.", name)
            if name.endswith(".mixer.conv1d.weight"):
                for component, weight in zip("qkv", tensor.chunk(3, dim=0)):
                    original[name.replace("conv1d", f"{component}_conv1d")] = weight.clone()
            else:
                original[name] = tensor.clone()
        original["model.attn_res_query"] = torch.stack(residuals["query"])
        original["model.attn_res_norm"] = torch.stack(residuals["weight"])
        Path(directory, "config.json").write_text(json.dumps(config), encoding="utf-8")
        save_file(original, Path(directory, "model.safetensors"))
        model.load_state_dict(convert_state_dict(original, model.config), strict=False)
        return model

    def test_native_loading_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = self._original_checkpoint(directory)
            config = AutoConfig.from_pretrained(directory)
            self.assertEqual(config.linear_num_heads, 2)
            self.assertEqual(config.linear_head_dim, 8)
            self.assertEqual(config.num_bigrams, 7)
            self.assertEqual(config.layer_types, ["linear_attention", "full_attention"] * 2)
            self.assertEqual(config.rope_parameters["rope_theta"], 12345.0)
            input_ids = torch.tensor([[1, 4, 9, 2, 6, 7]])
            for auto_class, expected in ((AutoModel, reference.model), (AutoModelForCausalLM, reference)):
                with self.subTest(auto_class=auto_class.__name__):
                    loaded, info = auto_class.from_pretrained(directory, output_loading_info=True)
                    self.assertFalse(info["missing_keys"])
                    self.assertFalse(info["unexpected_keys"])
                    for key, tensor in expected.state_dict().items():
                        torch.testing.assert_close(loaded.state_dict()[key], tensor, rtol=0, atol=0)
                    with torch.no_grad():
                        torch.testing.assert_close(loaded(input_ids)[0], expected(input_ids)[0])
                    for save_original_format in (True, False):
                        with tempfile.TemporaryDirectory() as saved:
                            loaded.save_pretrained(saved, save_original_format=save_original_format)
                            restored, info = auto_class.from_pretrained(saved, output_loading_info=True)
                            self.assertFalse(info["missing_keys"])
                            self.assertFalse(info["unexpected_keys"])
                            for key, tensor in loaded.state_dict().items():
                                torch.testing.assert_close(restored.state_dict()[key], tensor, rtol=0, atol=0)
