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
    from safetensors.torch import load_file, save_file

    from transformers import (
        Apertus1p5ForConditionalGeneration,
        Apertus1p5TextConfig,
        Apertus1p5TextForCausalLM,
        Apertus1p5VisionTokenizerConfig,
        Apertus1p5VisionTokenizerModel,
        WavTokenizerConfig,
        WavTokenizerModel,
    )
    from transformers.models.apertus1p5 import convert_apertus1p5_weights_to_hf as conversion


@require_torch
class Apertus1p5ConversionTest(unittest.TestCase):
    def test_valid_logits_layout(self):
        pruned_logits = torch.tensor([[[1.0, -2.0]]])
        self.assertTrue(conversion._has_valid_logits_layout(pruned_logits, output_vocab_size=2))
        self.assertFalse(conversion._has_valid_logits_layout(pruned_logits[..., :1], output_vocab_size=2))

        padded_logits = torch.nn.functional.pad(pruned_logits, (0, 2), value=torch.finfo(torch.float32).min)
        self.assertFalse(conversion._has_valid_logits_layout(padded_logits, output_vocab_size=2))

        for value in (torch.nan, torch.inf, -torch.inf):
            with self.subTest(value=value):
                nonfinite_logits = pruned_logits.clone()
                nonfinite_logits[..., 0] = value
                self.assertFalse(conversion._has_valid_logits_layout(nonfinite_logits, output_vocab_size=2))

        unpruned_logits = torch.tensor([[[1.0, -2.0, 3.0, 4.0]]])
        self.assertTrue(conversion._has_valid_logits_layout(unpruned_logits, output_vocab_size=4))

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
            (tmp / "vision" / "config.json").write_text(
                json.dumps({**Apertus1p5VisionTokenizerConversionTest.ORIGINAL_CONFIG, "codebook_size": 131072})
            )
            (tmp / "audio" / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "wavtokenizer",
                        "architectures": ["WavTokenizerModel"],
                        "transformers_version": "5.0.0",
                    }
                )
            )

            config = conversion.build_config(str(tmp / "apertus"), str(tmp / "vision"), str(tmp / "audio"))

        self.assertEqual(config.architectures, ["Apertus1p5ForConditionalGeneration"])
        self.assertIsInstance(config.text_config, Apertus1p5TextConfig)
        self.assertIsInstance(config.audio_config, WavTokenizerConfig)
        self.assertEqual(config.text_config.model_type, "apertus1p5_text")
        # source entrypoints must not leak into nested sub-configs
        self.assertIsNone(getattr(config.text_config, "architectures", None))
        self.assertIsNone(getattr(config.audio_config, "architectures", None))

    def test_remapped_sources_prunes_audio_decoder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            for source in ("apertus", "vision", "audio"):
                (tmp / source).mkdir()
            save_file({"model.layer.weight": torch.ones(1)}, tmp / "apertus" / SAFE_WEIGHTS_NAME)
            Apertus1p5VisionTokenizerConversionTest._write_original(
                tmp / "vision",
                config={**Apertus1p5VisionTokenizerConversionTest.ORIGINAL_CONFIG, "codebook_size": 131072},
            )
            save_file(
                {
                    "encoder_model.encoder.weight": torch.ones(1),
                    "encoder_model.quantizer.codebook.embed": torch.ones(1),
                    "backbone.weight": torch.ones(1),
                    "head.linear.weight": torch.ones(1),
                },
                tmp / "audio" / SAFE_WEIGHTS_NAME,
            )

            remapped = list(
                conversion.remapped_sources(*(str(tmp / source) for source in ("apertus", "vision", "audio")))
            )

        audio_state_dict = next(state_dict for source, _, state_dict in remapped if source == "wavtokenizer")
        self.assertEqual(
            set(audio_state_dict),
            {
                "model.audio_tokenizer.encoder.weight",
                "model.audio_tokenizer.quantizer.codebook.embed",
            },
        )

    def test_remapped_sources_rejects_unknown_audio_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            for source in ("apertus", "vision", "audio"):
                (tmp / source).mkdir()
            save_file({"model.layer.weight": torch.ones(1)}, tmp / "apertus" / SAFE_WEIGHTS_NAME)
            Apertus1p5VisionTokenizerConversionTest._write_original(
                tmp / "vision",
                config={**Apertus1p5VisionTokenizerConversionTest.ORIGINAL_CONFIG, "codebook_size": 131072},
            )
            save_file({"unexpected.weight": torch.ones(1)}, tmp / "audio" / SAFE_WEIGHTS_NAME)

            with self.assertRaisesRegex(ValueError, "Unexpected key in the WavTokenizer checkpoint"):
                list(conversion.remapped_sources(*(str(tmp / source) for source in ("apertus", "vision", "audio"))))

    def test_convert_rejects_audio_source_without_encoder_or_quantizer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            for source in ("apertus", "vision", "audio"):
                (tmp / source).mkdir()
            (tmp / "apertus" / "config.json").write_text(
                json.dumps({"model_type": "apertus", "tie_word_embeddings": True})
            )
            (tmp / "vision" / "config.json").write_text(
                json.dumps({**Apertus1p5VisionTokenizerConversionTest.ORIGINAL_CONFIG, "codebook_size": 131072})
            )
            (tmp / "audio" / "config.json").write_text(json.dumps({"model_type": "wavtokenizer"}))
            save_file({"model.layer.weight": torch.ones(1)}, tmp / "apertus" / SAFE_WEIGHTS_NAME)
            Apertus1p5VisionTokenizerConversionTest._write_original(
                tmp / "vision",
                config={**Apertus1p5VisionTokenizerConversionTest.ORIGINAL_CONFIG, "codebook_size": 131072},
            )
            save_file(
                {"backbone.weight": torch.ones(1), "head.linear.weight": torch.ones(1)},
                tmp / "audio" / SAFE_WEIGHTS_NAME,
            )

            with (
                patch.object(conversion, "write_processor"),
                self.assertRaisesRegex(ValueError, "required audio encoder, quantizer weights"),
            ):
                conversion.convert(str(tmp / "apertus"), str(tmp / "vision"), str(tmp / "audio"), str(tmp / "output"))

    def test_build_config_rejects_unrelated_text_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            for source in ("text", "vision", "audio"):
                (tmp / source).mkdir()
            (tmp / "text" / "config.json").write_text(json.dumps({"model_type": "llama"}))
            (tmp / "vision" / "config.json").write_text(
                json.dumps({**Apertus1p5VisionTokenizerConversionTest.ORIGINAL_CONFIG, "codebook_size": 131072})
            )
            (tmp / "audio" / "config.json").write_text(json.dumps({"model_type": "wavtokenizer"}))

            with self.assertRaisesRegex(ValueError, "not an Apertus text checkpoint"):
                conversion.build_config(str(tmp / "text"), str(tmp / "vision"), str(tmp / "audio"))

    def test_convert_rejects_output_dir_equal_to_source(self):
        # writing the composite into a source directory would overwrite its config and delete its weights
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "same directory"):
                conversion.convert("apertus", "vision", tmp_dir, tmp_dir)
            with self.assertRaisesRegex(ValueError, "same directory"):
                conversion.write_processor(tmp_dir, "audio", tmp_dir)
            alias = Path(tmp_dir) / "alias"
            alias.symlink_to(tmp_dir, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "same directory"):
                conversion.convert("apertus", tmp_dir, "audio", str(alias))
            self.assertEqual(list(Path(tmp_dir).iterdir()), [alias])

    def test_build_config_rejects_invalid_tokenizer_sources(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            for source in ("text", "vision", "audio"):
                (tmp / source).mkdir()
            (tmp / "text" / "config.json").write_text(json.dumps({"model_type": "apertus"}))
            (tmp / "audio" / "config.json").write_text(json.dumps({"model_type": "wavtokenizer"}))
            for model_type in ("apertus1p5_vision_tokenizer", "Emu3VisionVQ", None):
                with self.subTest(model_type=model_type):
                    (tmp / "vision" / "config.json").write_text(json.dumps({"model_type": model_type}))
                    with self.assertRaisesRegex(ValueError, "original EMU3.5"):
                        conversion.build_config(str(tmp / "text"), str(tmp / "vision"), str(tmp / "audio"))
            (tmp / "vision" / "config.json").write_text(
                json.dumps({**Apertus1p5VisionTokenizerConversionTest.ORIGINAL_CONFIG, "codebook_size": 131072})
            )
            (tmp / "audio" / "config.json").write_text(json.dumps({"model_type": "original_audio"}))
            with self.assertRaisesRegex(ValueError, "convert_wavtokenizer_checkpoint.py"):
                conversion.build_config(str(tmp / "text"), str(tmp / "vision"), str(tmp / "audio"))

    def test_cli_resolves_only_required_sources(self):
        with (
            patch.object(conversion, "resolve_checkpoint_dir", side_effect=lambda source, **kwargs: source) as resolve,
            patch.object(conversion, "convert") as convert,
            patch.object(conversion, "write_processor") as processor,
            patch.object(conversion, "verify_composite") as verify,
        ):
            conversion.main(["--output_dir", "out", "--skip_convert", "--verify"])
            resolve.assert_not_called()
            convert.assert_not_called()
            verify.assert_called_once_with("out")
            conversion.main(
                [
                    "--output_dir",
                    "out",
                    "--processor_only",
                    "--apertus_checkpoint",
                    "text",
                    "--audio_tokenizer_checkpoint",
                    "audio",
                ]
            )
            self.assertEqual([call.args[0] for call in resolve.call_args_list], ["text", "audio"])
            processor.assert_called_once_with("text", "audio", "out")
            convert.assert_not_called()
            resolve.reset_mock()
            conversion.main(
                [
                    "--output_dir",
                    "out",
                    "--apertus_checkpoint",
                    "text",
                    "--vision_tokenizer_checkpoint",
                    "BAAI/Emu3.5-VisionTokenizer@revision",
                    "--audio_tokenizer_checkpoint",
                    "audio",
                ]
            )
            self.assertEqual(resolve.call_args_list[1].kwargs, {"allow_patterns": ["config.json", SAFE_WEIGHTS_NAME]})
            convert.assert_called_once_with("text", "BAAI/Emu3.5-VisionTokenizer@revision", "audio", "out")

    def test_vision_download_selects_only_config_and_weights(self):
        with patch.object(conversion, "snapshot_download", return_value="snapshot") as download:
            conversion.resolve_checkpoint_dir(
                "BAAI/Emu3.5-VisionTokenizer@revision", allow_patterns=["config.json", SAFE_WEIGHTS_NAME]
            )
        download.assert_called_once_with(
            "BAAI/Emu3.5-VisionTokenizer", revision="revision", allow_patterns=["config.json", SAFE_WEIGHTS_NAME]
        )

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
        audio_weights = {
            "model.audio_tokenizer.encoder.weight": torch.ones(1),
            "model.audio_tokenizer.quantizer.weight": torch.ones(1),
        }

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
                    return_value=[
                        ("apertus", "model.safetensors", converted_weights),
                        ("wavtokenizer", "model.safetensors", audio_weights),
                    ],
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
            self.assertEqual(
                index["weight_map"],
                {
                    "lm_head.weight": "model-apertus-model.safetensors",
                    "model.audio_tokenizer.encoder.weight": "model-wavtokenizer-model.safetensors",
                    "model.audio_tokenizer.quantizer.weight": "model-wavtokenizer-model.safetensors",
                },
            )
            self.assertTrue((output_dir / "model-apertus-model.safetensors").exists())
            self.assertTrue((output_dir / "model-wavtokenizer-model.safetensors").exists())


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

    @staticmethod
    def _to_original_key(key: str) -> str:
        """Undo the grouped encoder layout to synthesize an original EMU3.5 checkpoint."""
        parts = key.split(".")
        if len(parts) < 4 or parts[:2] != ["encoder", "stages"]:
            return key

        stage_idx, module_name = parts[2:4]
        if module_name == "layers" and len(parts) >= 7:
            layer_idx, component = parts[4:6]
            original_component = "block" if component == "resnet" else "attn"
            return ".".join(("encoder", "down", stage_idx, original_component, layer_idx, *parts[6:]))
        if module_name == "downsample":
            return ".".join(("encoder", "down", stage_idx, "downsample", *parts[4:]))
        return key

    @classmethod
    def _write_original(cls, directory, config=None, dtype=torch.float32):
        """Write a synthetic original checkpoint (`config.json` + `model.safetensors`); return the kept half."""
        config = cls.ORIGINAL_CONFIG if config is None else config
        directory = Path(directory)
        # The kept half comes from the real model class, so its shapes and converted names are correct by construction.
        kept = Apertus1p5VisionTokenizerModel(conversion.convert_vision_config(config)).state_dict()
        original_kept = {cls._to_original_key(key): value for key, value in kept.items()}
        dropped = {
            "post_quant_conv.weight": torch.zeros(8, 8, 1, 1),
            "post_quant_conv.bias": torch.zeros(8),
            "decoder.conv_in.weight": torch.zeros(4, 8, 3, 3),
            "decoder.up.0.block.0.norm1.weight": torch.zeros(4),
            "decoder.conv_out.bias": torch.zeros(3),
        }
        tensors = {key: value.to(dtype).contiguous() for key, value in {**original_kept, **dropped}.items()}
        save_file(tensors, str(directory / SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
        (directory / "config.json").write_text(json.dumps(config))
        return kept

    def test_convert_config_maps_original_fields(self):
        config = conversion.convert_vision_config(self.ORIGINAL_CONFIG)
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
        config = conversion.convert_vision_config(distinct)

        for target, source in conversion.VISION_CONFIG_FIELDS.items():
            with self.subTest(field=target):
                actual, default = getattr(config, target), getattr(defaults, target)
                if isinstance(actual, (list, tuple)):
                    actual, default = list(actual), list(default)
                self.assertEqual(actual, distinct[source])
                self.assertNotEqual(actual, default, f"`{target}` must differ from its default in this fixture")
        self.assertEqual(config.dropout, 0.25)
        self.assertNotEqual(config.dropout, defaults.dropout)

    def test_convert_config_rejects_incomplete_config(self):
        incomplete = {key: value for key, value in self.ORIGINAL_CONFIG.items() if key != "z_channels"}
        with self.assertRaisesRegex(ValueError, "z_channels"):
            conversion.convert_vision_config(incomplete)

    def test_convert_config_defaults_dropout(self):
        without_dropout = {key: value for key, value in self.ORIGINAL_CONFIG.items() if key != "dropout"}
        self.assertEqual(conversion.convert_vision_config(without_dropout).dropout, 0.0)

    def test_convert_state_dict_drops_decoder_branch(self):
        original = {
            "encoder.conv_in.weight": torch.ones(2, 2),
            "encoder.down.1.block.2.conv1.weight": torch.ones(2, 2),
            "encoder.down.1.attn.2.q.weight": torch.ones(2, 2),
            "encoder.down.1.downsample.conv.weight": torch.ones(2, 2),
            "quantize.embedding.weight": torch.ones(2, 2),
            "quant_conv.weight": torch.ones(2, 2),
            "decoder.conv_in.weight": torch.ones(2, 2),
            "post_quant_conv.weight": torch.ones(2, 2),
        }
        converted = conversion.convert_vision_state_dict(original)
        self.assertEqual(
            set(converted),
            {
                "encoder.conv_in.weight",
                "encoder.stages.1.layers.2.resnet.conv1.weight",
                "encoder.stages.1.layers.2.attention.q.weight",
                "encoder.stages.1.downsample.conv.weight",
                "quantize.embedding.weight",
                "quant_conv.weight",
            },
        )
        # kept tensors are passed through untouched
        self.assertIs(converted["encoder.conv_in.weight"], original["encoder.conv_in.weight"])
        self.assertIs(
            converted["encoder.stages.1.layers.2.resnet.conv1.weight"],
            original["encoder.down.1.block.2.conv1.weight"],
        )

    def test_original_vision_assembles_into_separate_composite_shard(self):
        text_config = Apertus1p5TextConfig(
            vocab_size=266752,
            output_vocab_size=40,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_act="gelu",
        )
        audio_config = WavTokenizerConfig(
            num_filters=8,
            upsampling_ratios=[2, 2],
            hidden_size=32,
            codebook_dim=32,
            codebook_size=12,
            decoder_hidden_size=32,
            decoder_intermediate_size=64,
            decoder_num_layers=2,
        )
        text = Apertus1p5TextForCausalLM(text_config)
        audio = WavTokenizerModel(audio_config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            text.save_pretrained(tmp / "text", max_shard_size="10KB")
            audio.save_pretrained(tmp / "audio")
            (tmp / "vision").mkdir()
            vision_weights = self._write_original(
                tmp / "vision", config={**self.ORIGINAL_CONFIG, "codebook_size": 131072}
            )
            with patch.object(conversion, "write_processor"):
                conversion.convert(*(str(tmp / part) for part in ("text", "vision", "audio", "output")))
            reloaded, info = Apertus1p5ForConditionalGeneration.from_pretrained(
                tmp / "output", output_loading_info=True
            )
            self.assertFalse({key: value for key, value in info.items() if value})
            self.assertEqual(reloaded.config.vision_config.latent_channels, self.ORIGINAL_CONFIG["z_channels"])
            self.assertEqual(reloaded.config.vision_config.channel_multiplier, self.ORIGINAL_CONFIG["ch_mult"])
            expected = {
                "model.language_model." + k.removeprefix("model.") if k != "lm_head.weight" else k: v
                for k, v in text.state_dict().items()
            }
            expected.update({"model.vision_tokenizer." + k: v for k, v in vision_weights.items()})
            expected.update({"model.audio_tokenizer." + k: v for k, v in audio.encoder_model.state_dict().items()})
            actual = reloaded.state_dict()
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            index = json.loads((tmp / "output" / SAFE_WEIGHTS_INDEX_NAME).read_text())["weight_map"]
            vision_shards = {value for key, value in index.items() if key.startswith("model.vision_tokenizer.")}
            self.assertEqual(vision_shards, {"model-vision_tokenizer-model.safetensors"})
            for key, shard in index.items():
                prefix = (
                    "vision_tokenizer"
                    if key.startswith("model.vision_tokenizer.")
                    else ("wavtokenizer" if key.startswith("model.audio_tokenizer.") else "apertus")
                )
                self.assertTrue(shard.startswith(f"model-{prefix}-"))
            self.assertFalse(any("decoder." in key or "post_quant_conv." in key for key in index))

    def test_rejects_mismatched_vision_weight_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir)
            self._write_original(source)
            weights = load_file(source / SAFE_WEIGHTS_NAME)
            weights["quantize.embedding.weight"] = weights["quantize.embedding.weight"][:1]
            save_file(weights, str(source / SAFE_WEIGHTS_NAME))
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                conversion.load_vision_tokenizer(str(source))

    def test_rejects_half_precision_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "original"
            source.mkdir()
            self._write_original(source, dtype=torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "float32"):
                conversion.load_vision_tokenizer(str(source))

    def test_rejects_unexpected_source_tensor(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "original"
            source.mkdir()
            self._write_original(source)
            tensors = {**load_file(source / SAFE_WEIGHTS_NAME), "encoder.bogus.weight": torch.zeros(2)}
            save_file(
                {key: value.contiguous() for key, value in tensors.items()},
                str(source / SAFE_WEIGHTS_NAME),
                metadata={"format": "pt"},
            )
            # the strict load is the correctness gate for the tensor set
            with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                conversion.load_vision_tokenizer(str(source))


if __name__ == "__main__":
    unittest.main()
