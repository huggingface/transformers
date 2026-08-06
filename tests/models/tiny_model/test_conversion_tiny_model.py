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
import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from huggingface_hub import hf_hub_download

from transformers import TinyModelForCausalLM
from transformers.models.tiny_model.convert_tiny_model_weights_to_hf import (
    _convert_state_dict,
    convert_tiny_model_checkpoint,
    main,
)
from transformers.testing_utils import slow


def make_original_state_dict(num_hidden_layers=2):
    def tensor(shape, offset):
        numel = torch.Size(shape).numel()
        return (torch.arange(numel, dtype=torch.float32).reshape(shape) + offset).to(torch.bfloat16)

    state_dict = {
        "embed.weight": tensor((10_000, 16), 1),
        "pos_embed": tensor((1, 8, 16), 2),
        "lm_head.weight": tensor((10_000, 16), 3),
        "lm_head.bias": tensor((10_000,), 4),
    }
    for layer_idx in range(num_hidden_layers):
        prefix = f"torso.{layer_idx}"
        state_dict.update(
            {
                f"{prefix}.attn.Q.weight": tensor((16, 16), 10 + layer_idx),
                f"{prefix}.attn.K.weight": tensor((16, 16), 20 + layer_idx),
                f"{prefix}.attn.V.weight": tensor((16, 16), 30 + layer_idx),
                f"{prefix}.attn.O.weight": tensor((16, 16), 40 + layer_idx),
                f"{prefix}.attn.O.bias": tensor((16,), 50 + layer_idx),
                f"{prefix}.mlp.read_in.weight": tensor((64, 16), 60 + layer_idx),
                f"{prefix}.mlp.read_in.bias": tensor((64,), 70 + layer_idx),
                f"{prefix}.mlp.write_out.weight": tensor((16, 64), 80 + layer_idx),
                f"{prefix}.mlp.write_out.bias": tensor((16,), 90 + layer_idx),
            }
        )
    return state_dict


class TinyModelStateDictConversionTest(unittest.TestCase):
    def test_infers_two_and_four_layer_configs(self):
        for num_hidden_layers in (2, 4):
            with self.subTest(num_hidden_layers=num_hidden_layers):
                config, converted = _convert_state_dict(
                    make_original_state_dict(num_hidden_layers),
                    expected_num_hidden_layers=num_hidden_layers,
                )

                self.assertEqual(config.vocab_size, 10_000)
                self.assertEqual(config.hidden_size, 16)
                self.assertEqual(config.intermediate_size, 64)
                self.assertEqual(config.num_hidden_layers, num_hidden_layers)
                self.assertEqual(config.num_attention_heads, 16)
                self.assertEqual(config.max_position_embeddings, 8)
                self.assertEqual(len(converted), 4 + 9 * num_hidden_layers)

    def test_maps_every_tensor_without_transposing(self):
        original = make_original_state_dict()
        _, converted = _convert_state_dict(original)

        torch.testing.assert_close(
            converted["model.embed_positions.weight"], original["pos_embed"].squeeze(0), rtol=0, atol=0
        )
        torch.testing.assert_close(
            converted["model.layers.0.self_attn.q_proj.weight"],
            original["torso.0.attn.Q.weight"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            converted["model.layers.1.mlp.fc2.weight"],
            original["torso.1.mlp.write_out.weight"],
            rtol=0,
            atol=0,
        )

    def test_rejects_missing_and_unexpected_keys(self):
        original = make_original_state_dict()
        del original["torso.1.attn.O.bias"]
        original["torso.0.layer_norm.weight"] = torch.zeros(16, dtype=torch.bfloat16)

        with self.assertRaisesRegex(ValueError, "torso.1.attn.O.bias.*torso.0.layer_norm.weight"):
            _convert_state_dict(original)

    def test_rejects_noncontiguous_layers(self):
        original = make_original_state_dict()
        original = {key.replace("torso.1.", "torso.2."): value for key, value in original.items()}

        with self.assertRaisesRegex(ValueError, "contiguous from zero"):
            _convert_state_dict(original)

    def test_rejects_expected_layer_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Expected 4 decoder layers.*contains 2"):
            _convert_state_dict(make_original_state_dict(), expected_num_hidden_layers=4)

    def test_rejects_non_string_keys_and_non_tensor_values(self):
        original = make_original_state_dict()
        original[0] = torch.zeros(1, dtype=torch.bfloat16)
        with self.assertRaisesRegex(TypeError, "keys must be strings"):
            _convert_state_dict(original)

        original = make_original_state_dict()
        original["metadata"] = "not a tensor"
        with self.assertRaisesRegex(TypeError, "values must be tensors.*metadata"):
            _convert_state_dict(original)

    def test_rejects_wrong_dtype_and_shape(self):
        original = make_original_state_dict()
        original["lm_head.bias"] = original["lm_head.bias"].float()
        with self.assertRaisesRegex(ValueError, "bfloat16.*lm_head.bias"):
            _convert_state_dict(original)

        original = make_original_state_dict()
        original["torso.0.attn.Q.weight"] = torch.zeros((16, 15), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, r"torso\.0\.attn\.Q\.weight.*\(16, 16\).*\(16, 15\)"):
            _convert_state_dict(original)

        original = make_original_state_dict()
        original["pos_embed"] = torch.zeros((2, 8, 16), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, r"pos_embed.*\(1, max_position_embeddings, 16\).*"):
            _convert_state_dict(original)

        original = make_original_state_dict()
        original["torso.0.mlp.read_in.weight"] = torch.zeros((63, 16), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, r"intermediate size.*4 \* hidden_size"):
            _convert_state_dict(original)


class TinyModelCheckpointConversionTest(unittest.TestCase):
    def test_saves_and_reloads_exact_bfloat16_checkpoint(self):
        original = make_original_state_dict()
        _, expected = _convert_state_dict(original)

        with TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            checkpoint_path = temporary_directory / "tiny_model.pt"
            output_dir = temporary_directory / "converted"
            torch.save(original, checkpoint_path)

            model = convert_tiny_model_checkpoint(
                checkpoint_path,
                output_dir,
                expected_num_hidden_layers=2,
            )

            self.assertTrue((output_dir / "config.json").is_file())
            self.assertTrue((output_dir / "model.safetensors").is_file())
            self.assertEqual(model.config.num_attention_heads, 16)
            self.assertEqual(model.config.dtype, torch.bfloat16)
            self.assertFalse(model.config.attention_bias)
            self.assertTrue(model.config.attention_output_bias)
            self.assertTrue(model.config.mlp_bias)
            self.assertTrue(model.config.lm_head_bias)
            self.assertFalse(model.config.tie_word_embeddings)
            self.assertEqual(set(model.state_dict()), set(expected))
            for key, tensor in model.state_dict().items():
                self.assertEqual(tensor.dtype, torch.bfloat16)
                torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
            self.assertNotEqual(model.model.embed_tokens.weight.data_ptr(), model.lm_head.weight.data_ptr())

    @slow
    def test_released_checkpoints(self):
        revision = "502a1f2453f61260c937f7807a1270a167faba07"
        input_ids = torch.tensor([[9_996, 51, 56, 4, 36]])
        # Computed directly from the pinned raw checkpoints with the source equations in FP32 SDPA.
        cases = [
            (
                "tiny_model.pt",
                4,
                "dec406b1ad94cb345b2606d7f8cffa7c1114fcb60850e949eb17274cec30a8c3",
                [
                    13.61935043334961,
                    27.038311004638672,
                    14.778780937194824,
                    15.210494995117188,
                    19.164371490478516,
                    13.294827461242676,
                    15.938150405883789,
                    9.0501070022583,
                    14.516134262084961,
                    8.900970458984375,
                ],
                3.20711088180542,
            ),
            (
                "tiny_model_2L_1E.pt",
                2,
                "04e8df0cd677a7060558e5c9eb3aaa30dbfe84e4ecc92bf17ef0e405dcf33baf",
                [
                    14.836343765258789,
                    27.606307983398438,
                    16.4638671875,
                    16.65711784362793,
                    19.727964401245117,
                    15.745035171508789,
                    15.660707473754883,
                    8.073421478271484,
                    15.538188934326172,
                    12.202945709228516,
                ],
                3.2003679275512695,
            ),
            (
                "tiny_model_2L_3E.pt",
                2,
                "26dfc06da85d0e5d4de51a2e90108f9d585a81677bf4bac0ac079e780fda31f4",
                [
                    25.153167724609375,
                    39.79271697998047,
                    28.825111389160156,
                    27.843313217163086,
                    31.72383689880371,
                    27.25177764892578,
                    27.752153396606445,
                    19.26030158996582,
                    27.339975357055664,
                    23.36639404296875,
                ],
                15.778340339660645,
            ),
        ]

        for filename, num_hidden_layers, expected_sha256, expected_logits, expected_mean in cases:
            with self.subTest(filename=filename), TemporaryDirectory() as output_dir:
                checkpoint_path = hf_hub_download(
                    repo_id="noanabeshima/tiny_model",
                    filename=filename,
                    revision=revision,
                )
                checkpoint_hash = hashlib.sha256()
                with open(checkpoint_path, "rb") as checkpoint_file:
                    for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                        checkpoint_hash.update(chunk)
                actual_sha256 = checkpoint_hash.hexdigest()
                self.assertEqual(actual_sha256, expected_sha256)

                convert_tiny_model_checkpoint(
                    checkpoint_path,
                    output_dir,
                    expected_num_hidden_layers=num_hidden_layers,
                )
                model = TinyModelForCausalLM.from_pretrained(
                    output_dir,
                    dtype=torch.float32,
                ).eval()
                self.assertEqual(model.config._attn_implementation, "sdpa")
                with torch.no_grad():
                    logits = model(input_ids).logits
                    generated_ids = model.generate(input_ids, max_new_tokens=3, do_sample=False)

                torch.testing.assert_close(
                    logits[0, -1, :10],
                    torch.tensor(expected_logits),
                    rtol=1e-5,
                    atol=1e-5,
                )
                torch.testing.assert_close(logits[0, -1].mean(), torch.tensor(expected_mean), rtol=1e-5, atol=1e-5)
                self.assertEqual(generated_ids.tolist(), [[9_996, 51, 56, 4, 36, 1, 38, 6]])

    def test_cli_converts_local_checkpoint(self):
        with TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            checkpoint_path = temporary_directory / "tiny_model_2L_3E.pt"
            output_dir = temporary_directory / "converted"
            torch.save(make_original_state_dict(), checkpoint_path)

            main(
                [
                    "--checkpoint_path",
                    str(checkpoint_path),
                    "--output_dir",
                    str(output_dir),
                    "--expected_num_hidden_layers",
                    "2",
                ]
            )

            self.assertTrue((output_dir / "config.json").is_file())
            self.assertTrue((output_dir / "model.safetensors").is_file())
