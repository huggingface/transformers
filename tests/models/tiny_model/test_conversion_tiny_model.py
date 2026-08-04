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
        "embed.weight": tensor((10_000, 8), 1),
        "pos_embed": tensor((1, 8, 8), 2),
        "lm_head.weight": tensor((10_000, 8), 3),
        "lm_head.bias": tensor((10_000,), 4),
    }
    for layer_idx in range(num_hidden_layers):
        prefix = f"torso.{layer_idx}"
        state_dict.update(
            {
                f"{prefix}.attn.Q.weight": tensor((8, 8), 10 + layer_idx),
                f"{prefix}.attn.K.weight": tensor((8, 8), 20 + layer_idx),
                f"{prefix}.attn.V.weight": tensor((8, 8), 30 + layer_idx),
                f"{prefix}.attn.O.weight": tensor((8, 8), 40 + layer_idx),
                f"{prefix}.attn.O.bias": tensor((8,), 50 + layer_idx),
                f"{prefix}.mlp.read_in.weight": tensor((32, 8), 60 + layer_idx),
                f"{prefix}.mlp.read_in.bias": tensor((32,), 70 + layer_idx),
                f"{prefix}.mlp.write_out.weight": tensor((8, 32), 80 + layer_idx),
                f"{prefix}.mlp.write_out.bias": tensor((8,), 90 + layer_idx),
            }
        )
    return state_dict


class TinyModelStateDictConversionTest(unittest.TestCase):
    def test_infers_two_and_four_layer_configs(self):
        for num_hidden_layers in (2, 4):
            with self.subTest(num_hidden_layers=num_hidden_layers):
                config, converted = _convert_state_dict(
                    make_original_state_dict(num_hidden_layers),
                    num_attention_heads=2,
                    expected_num_hidden_layers=num_hidden_layers,
                )

                self.assertEqual(config.vocab_size, 10_000)
                self.assertEqual(config.hidden_size, 8)
                self.assertEqual(config.intermediate_size, 32)
                self.assertEqual(config.num_hidden_layers, num_hidden_layers)
                self.assertEqual(config.num_attention_heads, 2)
                self.assertEqual(config.max_position_embeddings, 8)
                self.assertEqual(len(converted), 4 + 9 * num_hidden_layers)

    def test_maps_every_tensor_without_transposing(self):
        original = make_original_state_dict()
        _, converted = _convert_state_dict(original, num_attention_heads=2)

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
        original["torso.0.layer_norm.weight"] = torch.zeros(8, dtype=torch.bfloat16)

        with self.assertRaisesRegex(ValueError, "torso.1.attn.O.bias.*torso.0.layer_norm.weight"):
            _convert_state_dict(original, num_attention_heads=2)

    def test_rejects_noncontiguous_layers(self):
        original = make_original_state_dict()
        original = {key.replace("torso.1.", "torso.2."): value for key, value in original.items()}

        with self.assertRaisesRegex(ValueError, "contiguous from zero"):
            _convert_state_dict(original, num_attention_heads=2)

    def test_rejects_wrong_dtype_and_shape(self):
        original = make_original_state_dict()
        original["lm_head.bias"] = original["lm_head.bias"].float()
        with self.assertRaisesRegex(ValueError, "bfloat16.*lm_head.bias"):
            _convert_state_dict(original, num_attention_heads=2)

        original = make_original_state_dict()
        original["torso.0.attn.Q.weight"] = torch.zeros((8, 7), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, r"torso\.0\.attn\.Q\.weight.*\(8, 8\).*\(8, 7\)"):
            _convert_state_dict(original, num_attention_heads=2)


class TinyModelCheckpointConversionTest(unittest.TestCase):
    def test_saves_and_reloads_exact_bfloat16_checkpoint(self):
        original = make_original_state_dict()
        _, expected = _convert_state_dict(original, num_attention_heads=2)

        with TemporaryDirectory() as temporary_directory:
            temporary_directory = Path(temporary_directory)
            checkpoint_path = temporary_directory / "tiny_model.pt"
            output_dir = temporary_directory / "converted"
            torch.save(original, checkpoint_path)

            model = convert_tiny_model_checkpoint(
                checkpoint_path,
                output_dir,
                num_attention_heads=2,
                expected_num_hidden_layers=2,
            )

            self.assertTrue((output_dir / "config.json").is_file())
            self.assertTrue((output_dir / "model.safetensors").is_file())
            self.assertEqual(set(model.state_dict()), set(expected))
            for key, tensor in model.state_dict().items():
                self.assertEqual(tensor.dtype, torch.bfloat16)
                torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
            self.assertNotEqual(model.model.embed_tokens.weight.data_ptr(), model.lm_head.weight.data_ptr())

    @slow
    def test_released_checkpoints(self):
        revision = "502a1f2453f61260c937f7807a1270a167faba07"
        input_ids = torch.tensor([[9_996, 51, 56, 4, 36]])
        cases = [
            (
                "tiny_model.pt",
                4,
                "dec406b1ad94cb345b2606d7f8cffa7c1114fcb60850e949eb17274cec30a8c3",
                [
                    13.619353294372559,
                    27.03830909729004,
                    14.778779983520508,
                    15.210500717163086,
                    19.164371490478516,
                    13.294825553894043,
                    15.938150405883789,
                    9.050104141235352,
                    14.516134262084961,
                    8.900968551635742,
                ],
                3.20711088180542,
            ),
            (
                "tiny_model_2L_1E.pt",
                2,
                "04e8df0cd677a7060558e5c9eb3aaa30dbfe84e4ecc92bf17ef0e405dcf33baf",
                [
                    14.836339950561523,
                    27.60630989074707,
                    16.463863372802734,
                    16.657115936279297,
                    19.72796630859375,
                    15.745035171508789,
                    15.660703659057617,
                    8.073421478271484,
                    15.538191795349121,
                    12.202946662902832,
                ],
                3.2003674507141113,
            ),
            (
                "tiny_model_2L_3E.pt",
                2,
                "26dfc06da85d0e5d4de51a2e90108f9d585a81677bf4bac0ac079e780fda31f4",
                [
                    25.153165817260742,
                    39.7927131652832,
                    28.825101852416992,
                    27.843303680419922,
                    31.723831176757812,
                    27.25177764892578,
                    27.752153396606445,
                    19.260297775268555,
                    27.339969635009766,
                    23.366392135620117,
                ],
                15.778337478637695,
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
                    attn_implementation="eager",
                ).eval()
                with torch.no_grad():
                    logits = model(input_ids).logits
                    generated_ids = model.generate(input_ids, max_new_tokens=3, do_sample=False)

                torch.testing.assert_close(
                    logits[0, -1, :10],
                    torch.tensor(expected_logits),
                    rtol=1e-5,
                    atol=1e-5,
                )
                self.assertAlmostEqual(logits[0, -1].mean().item(), expected_mean, places=5)
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
                    "--num_attention_heads",
                    "2",
                    "--expected_num_hidden_layers",
                    "2",
                ]
            )

            self.assertTrue((output_dir / "config.json").is_file())
            self.assertTrue((output_dir / "model.safetensors").is_file())
