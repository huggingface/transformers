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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from transformers import is_torch_available
from transformers.testing_utils import require_torch
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


if is_torch_available():
    import torch
    from safetensors.torch import save_file

    from transformers import Apertus1p5VisionTokenizerConfig, Apertus1p5VisionTokenizerModel
    from transformers.models.apertus1p5 import convert_apertus1p5_vision_tokenizer_to_hf as vision_conversion
    from transformers.models.apertus1p5 import convert_apertus1p5_weights_to_hf as conversion
    from transformers.models.apertus1p5.convert_apertus1p5_vision_tokenizer_to_hf import _comparable


@require_torch
class Apertus1p5ConversionTest(unittest.TestCase):
    def test_valid_logits_layout(self):
        tail_min = torch.finfo(torch.float32).min
        pruned_logits = torch.tensor([[[1.0, -2.0, tail_min, tail_min]]])
        self.assertTrue(conversion._has_valid_logits_layout(pruned_logits, output_vocab_size=2, vocab_size=4))

        physical_only_logits = pruned_logits[..., :2]
        self.assertFalse(conversion._has_valid_logits_layout(physical_only_logits, output_vocab_size=2, vocab_size=4))

        oversized_logits = torch.cat((pruned_logits, pruned_logits[..., -1:]), dim=-1)
        self.assertFalse(conversion._has_valid_logits_layout(oversized_logits, output_vocab_size=2, vocab_size=4))

        finite_tail = pruned_logits.clone()
        finite_tail[..., -1] = 0
        self.assertFalse(conversion._has_valid_logits_layout(finite_tail, output_vocab_size=2, vocab_size=4))

        # a -inf tail is the stale pre-finfo.min layout the checker exists to catch
        neginf_tail = pruned_logits.clone()
        neginf_tail[..., 2:] = -torch.inf
        self.assertFalse(conversion._has_valid_logits_layout(neginf_tail, output_vocab_size=2, vocab_size=4))

        nonfinite_prefix = pruned_logits.clone()
        nonfinite_prefix[..., 0] = torch.inf
        self.assertFalse(conversion._has_valid_logits_layout(nonfinite_prefix, output_vocab_size=2, vocab_size=4))

        unpruned_logits = torch.tensor([[[1.0, -2.0, 3.0, 4.0]]])
        self.assertTrue(conversion._has_valid_logits_layout(unpruned_logits, output_vocab_size=4, vocab_size=4))

    def test_fp32_tokenizer_source_check(self):
        # fp32 floats and integer tensors (e.g. codebook indices) pass
        conversion._check_fp32_tokenizer_source("vision tokenizer", {"w": torch.ones(2), "idx": torch.arange(2)})
        with self.assertRaisesRegex(ValueError, "float32"):
            conversion._check_fp32_tokenizer_source("audio tokenizer", {"w": torch.ones(2, dtype=torch.bfloat16)})

    def test_build_config_stamps_architectures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            for source in ("apertus", "vision", "audio"):
                (tmp / source).mkdir()
            (tmp / "apertus" / "config.json").write_text(
                json.dumps({"model_type": "apertus", "architectures": ["ApertusForCausalLM"]})
            )
            (tmp / "vision" / "config.json").write_text(json.dumps({"model_type": "apertus1p5_vision_tokenizer"}))
            (tmp / "audio" / "config.json").write_text(json.dumps({"model_type": "wavtokenizer"}))

            config = conversion.build_config(str(tmp / "apertus"), str(tmp / "vision"), str(tmp / "audio"))

        self.assertEqual(config.architectures, ["Apertus1p5ForConditionalGeneration"])
        # the backbone's own entrypoint must not leak into the text sub-config
        self.assertIsNone(getattr(config.text_config, "architectures", None))

    def test_convert_rejects_output_dir_equal_to_source(self):
        # writing the composite into a source directory would overwrite its config and delete its weights
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "same directory"):
                conversion.convert("apertus", "vision", tmp_dir, tmp_dir)
            with self.assertRaisesRegex(ValueError, "same directory"):
                conversion.write_processor(tmp_dir, "audio", tmp_dir)

    def test_build_config_rejects_unconverted_tokenizer_sources(self):
        """An original-format source must be refused, not absorbed into a composite that only fails on load."""
        raw_emu35 = {"model_type": "Emu3p5VisionVQ", "ch": 256, "ch_mult": [1, 1, 2, 2, 4], "z_channels": 256}
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            for source in ("apertus", "vision", "audio"):
                (tmp / source).mkdir()
            (tmp / "apertus" / "config.json").write_text(json.dumps({"model_type": "apertus"}))
            (tmp / "audio" / "config.json").write_text(json.dumps({"model_type": "wavtokenizer"}))

            (tmp / "vision" / "config.json").write_text(json.dumps(raw_emu35))
            with self.assertRaisesRegex(ValueError, "convert_apertus1p5_vision_tokenizer_to_hf.py"):
                conversion.build_config(str(tmp / "apertus"), str(tmp / "vision"), str(tmp / "audio"))

            # the same guard on the audio side
            (tmp / "vision" / "config.json").write_text(json.dumps({"model_type": "apertus1p5_vision_tokenizer"}))
            (tmp / "audio" / "config.json").write_text(json.dumps({"model_type": "something_else"}))
            with self.assertRaisesRegex(ValueError, "convert_wavtokenizer_checkpoint.py"):
                conversion.build_config(str(tmp / "apertus"), str(tmp / "vision"), str(tmp / "audio"))

    def test_resolve_checkpoint_dir_rejects_a_file(self):
        """A path to an original-format `.ckpt` must not be forwarded to the Hub as a repo id."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "wavtokenizer_large_unify_600_24k.ckpt"
            checkpoint.write_bytes(b"original format")
            with self.assertRaisesRegex(ValueError, "is a file"):
                conversion.resolve_checkpoint_dir(str(checkpoint))

    def test_convert_removes_stale_canonical_weight_files(self):
        config = Mock(tie_word_embeddings=False)
        converted_weights = {"lm_head.weight": torch.ones(2, 2)}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            for filename in (SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME):
                (output_dir / filename).write_bytes(b"stale")
            old_shard = output_dir / "model-old-00001-of-00001.safetensors"
            old_shard.write_bytes(b"unreferenced")

            with (
                patch.object(conversion, "build_config", return_value=config),
                patch.object(
                    conversion,
                    "remapped_sources",
                    return_value=[("apertus", "model.safetensors", converted_weights)],
                ),
                patch.object(conversion, "write_processor"),
            ):
                conversion.convert("apertus", "vision", "audio", str(output_dir))

            self.assertFalse((output_dir / SAFE_WEIGHTS_NAME).exists())
            self.assertFalse((output_dir / WEIGHTS_NAME).exists())
            self.assertFalse((output_dir / WEIGHTS_INDEX_NAME).exists())
            self.assertTrue(old_shard.exists())

            with (output_dir / SAFE_WEIGHTS_INDEX_NAME).open() as f:
                index = json.load(f)
            self.assertEqual(index["weight_map"], {"lm_head.weight": "model-apertus-model.safetensors"})
            self.assertTrue((output_dir / "model-apertus-model.safetensors").exists())


@require_torch
class Apertus1p5VisionTokenizerConversionTest(unittest.TestCase):
    """Conversion of the original EMU3.5 vision tokenizer, exercised on a tiny synthetic source (no downloads)."""

    # An EMU3.5 `config.json` in the ORIGINAL field names, shrunk to a two-stage encoder. Every GroupNorm in
    # the port hard-codes `num_groups=32`, so each stage width `ch * ch_mult[i]` must be a multiple of 32.
    # `resolution=8` with `attn_resolutions=[4]` still places one attention block. The spatial factor is 2
    # rather than the released 16, which keeps `verify` honest about reading it from the config.
    ORIGINAL_CONFIG = {
        "architectures": ["Emu3p5VisionVQModel"],
        "model_type": "Emu3p5VisionVQ",
        "auto_map": {"AutoModel": "modeling_emu3p5visionvq.Emu3p5VisionVQModel"},
        "codebook_size": 16,
        "embed_dim": 8,
        "z_channels": 8,
        "in_channels": 3,
        "ch": 32,
        "ch_mult": [1, 2],
        "num_res_blocks": 1,
        "attn_resolutions": [4],
        "resolution": 8,
        "dropout": 0.0,
        "out_ch": 3,
        "double_z": False,
        "torch_dtype": "float32",
    }

    def _write_original(self, directory, config=None, dtype=torch.float32):
        """Write a synthetic original checkpoint (`config.json` + `model.safetensors`); return the kept half."""
        config = self.ORIGINAL_CONFIG if config is None else config
        directory = Path(directory)
        # the kept half comes from the real model class, so its key names and shapes are correct by construction
        kept = Apertus1p5VisionTokenizerModel(vision_conversion.convert_config(config)).state_dict()
        dropped = {
            "post_quant_conv.weight": torch.zeros(8, 8, 1, 1),
            "post_quant_conv.bias": torch.zeros(8),
            "decoder.conv_in.weight": torch.zeros(4, 8, 3, 3),
            "decoder.up.0.block.0.norm1.weight": torch.zeros(4),
            "decoder.conv_out.bias": torch.zeros(3),
        }
        tensors = {key: value.to(dtype).contiguous() for key, value in {**kept, **dropped}.items()}
        save_file(tensors, str(directory / SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
        (directory / "config.json").write_text(json.dumps(config))
        return kept

    def test_convert_config_maps_original_fields(self):
        config = vision_conversion.convert_config(self.ORIGINAL_CONFIG)
        self.assertEqual(config.codebook_size, 16)
        self.assertEqual(config.embed_dim, 8)
        self.assertEqual(config.latent_channels, 8)  # <- z_channels
        self.assertEqual(config.in_channels, 3)
        self.assertEqual(config.base_channels, 32)  # <- ch
        self.assertEqual(list(config.channel_multiplier), [1, 2])  # <- ch_mult
        self.assertEqual(config.num_res_blocks, 1)
        self.assertEqual(list(config.attn_resolutions), [4])
        self.assertEqual(config.resolution, 8)
        self.assertEqual(config.dropout, 0.0)
        self.assertEqual(config.spatial_scale_factor, 2)
        # decoder-only fields and the original entrypoint must not leak into our config
        for leaked in ("out_ch", "double_z", "auto_map"):
            self.assertIsNone(getattr(config, leaked, None))

    def test_convert_config_reads_every_field_from_the_source(self):
        """Each mapped field must come from the source, not from the config class default.

        `ORIGINAL_CONFIG` keeps `in_channels` and `dropout` at their released values, which happen to equal
        the defaults of `Apertus1p5VisionTokenizerConfig`. On that fixture alone, dropping either mapping
        would go unnoticed, so this test gives every field a value that differs from the default.
        """
        defaults = Apertus1p5VisionTokenizerConfig()
        distinct = {**self.ORIGINAL_CONFIG, "in_channels": 1, "dropout": 0.25}
        config = vision_conversion.convert_config(distinct)

        for target, source in vision_conversion.ORIGINAL_CONFIG_FIELDS.items():
            with self.subTest(field=target):
                self.assertEqual(_comparable(getattr(config, target)), _comparable(distinct[source]))
                self.assertNotEqual(
                    _comparable(getattr(config, target)),
                    _comparable(getattr(defaults, target)),
                    f"`{target}` equals the class default, so this fixture cannot detect a lost mapping",
                )
        self.assertEqual(config.dropout, 0.25)
        self.assertNotEqual(config.dropout, defaults.dropout)

    def test_convert_config_rejects_incomplete_config(self):
        incomplete = {key: value for key, value in self.ORIGINAL_CONFIG.items() if key != "z_channels"}
        with self.assertRaisesRegex(ValueError, "z_channels"):
            vision_conversion.convert_config(incomplete)

    def test_convert_config_defaults_dropout(self):
        without_dropout = {key: value for key, value in self.ORIGINAL_CONFIG.items() if key != "dropout"}
        self.assertEqual(vision_conversion.convert_config(without_dropout).dropout, 0.0)

    def test_convert_checkpoint_rejects_output_dir_equal_to_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "same directory"):
                vision_conversion.convert_checkpoint(tmp_dir, tmp_dir)

    def test_verify_rejects_a_half_precision_checkpoint(self):
        """`_keep_in_fp32_modules_strict` upcasts on load, so `verify` must read the stored dtypes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source, output = Path(tmp_dir) / "original", Path(tmp_dir) / "converted"
            source.mkdir()
            self._write_original(source)
            vision_conversion.convert_checkpoint(str(source), str(output))
            # re-save the converted checkpoint in half precision behind the converter's back
            Apertus1p5VisionTokenizerModel.from_pretrained(output).to(torch.bfloat16).save_pretrained(output)
            with self.assertRaisesRegex(RuntimeError, "stored dtype"):
                vision_conversion.verify(str(output), str(source))

    def test_convert_state_dict_drops_decoder_branch(self):
        original = {
            "encoder.conv_in.weight": torch.ones(2, 2),
            "quantize.embedding.weight": torch.ones(2, 2),
            "quant_conv.weight": torch.ones(2, 2),
            "decoder.conv_in.weight": torch.ones(2, 2),
            "post_quant_conv.weight": torch.ones(2, 2),
        }
        converted = vision_conversion.convert_state_dict(original)
        self.assertEqual(set(converted), {"encoder.conv_in.weight", "quantize.embedding.weight", "quant_conv.weight"})
        # kept tensors are passed through untouched
        self.assertIs(converted["encoder.conv_in.weight"], original["encoder.conv_in.weight"])

    def test_converts_and_reloads_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source, output = Path(tmp_dir) / "original", Path(tmp_dir) / "converted"
            source.mkdir()
            kept = self._write_original(source)

            vision_conversion.convert_checkpoint(str(source), str(output))
            reloaded = Apertus1p5VisionTokenizerModel.from_pretrained(output)

            state_dict = reloaded.state_dict()
            self.assertEqual(set(state_dict), set(kept))
            for key, value in kept.items():
                torch.testing.assert_close(state_dict[key], value, rtol=0, atol=0)

            with (output / "config.json").open() as f:
                saved_config = json.load(f)
            self.assertEqual(saved_config["model_type"], "apertus1p5_vision_tokenizer")
            self.assertEqual(saved_config["latent_channels"], 8)
            self.assertEqual(saved_config["base_channels"], 32)
            self.assertNotIn("z_channels", saved_config)

    def test_verify_passes_on_a_converted_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source, output = Path(tmp_dir) / "original", Path(tmp_dir) / "converted"
            source.mkdir()
            self._write_original(source)
            vision_conversion.convert_checkpoint(str(source), str(output))
            # runs all six checks, including the odd-sided (non-multiple-of-the-factor) size
            vision_conversion.verify(str(output), str(source))

    def test_rejects_half_precision_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source, output = Path(tmp_dir) / "original", Path(tmp_dir) / "converted"
            source.mkdir()
            self._write_original(source, dtype=torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "float32"):
                vision_conversion.convert_checkpoint(str(source), str(output))

    def test_rejects_unexpected_source_tensor(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source, output = Path(tmp_dir) / "original", Path(tmp_dir) / "converted"
            source.mkdir()
            kept = self._write_original(source)
            tensors = {**kept, "encoder.bogus.weight": torch.zeros(2)}
            save_file(
                {key: value.contiguous() for key, value in tensors.items()},
                str(source / SAFE_WEIGHTS_NAME),
                metadata={"format": "pt"},
            )
            # the strict load is the correctness gate for the tensor set
            with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                vision_conversion.convert_checkpoint(str(source), str(output))


if __name__ == "__main__":
    unittest.main()
