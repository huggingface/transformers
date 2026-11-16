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

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    from safetensors.torch import save_file

    from transformers import KlmForCausalLM
    from transformers.core_model_loading import Stack, Unbind
    from transformers.models.klm.convert_klm_weights_to_hf import convert_config, convert_state_dict


@require_torch
class KlmCheckpointConversionTest(unittest.TestCase):
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
