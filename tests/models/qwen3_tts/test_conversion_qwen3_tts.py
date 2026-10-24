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

"""Offline regression tests using checkpoints in the original Qwen3-TTS schema."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from transformers import Qwen2TokenizerFast, Qwen3TTSConfig, Qwen3TTSFeatureExtractor, is_torch_available
from transformers.testing_utils import require_librosa, require_soundfile, require_tokenizers, require_torch


if is_torch_available():
    import torch
    from safetensors.torch import save_file

    from transformers import AutoProcessor, GenerationConfig, Qwen3TTSForConditionalGeneration
    from transformers.models.qwen3_tts.convert_qwen3_tts_to_hf import (
        convert_checkpoint,
        create_config_from_checkpoint,
    )

    from .test_modeling_qwen3_tts import Qwen3TTSModelTester
    from .test_processing_qwen3_tts import _build_tiny_audio_tokenizer


@require_torch
class Qwen3TTSConversionTest(unittest.TestCase):
    def _convert_config(self, directory, original, audio_config):
        path = Path(directory)
        (path / "config.json").write_text(json.dumps(original), encoding="utf-8")
        config = create_config_from_checkpoint(path, audio_config)
        config.save_pretrained(path / "converted")
        return Qwen3TTSConfig.from_pretrained(path / "converted", local_files_only=True)

    def test_rotary_parameters_control_effective_frequencies(self):
        base = Qwen3TTSModelTester(self).get_config().to_dict()
        audio_config = _build_tiny_audio_tokenizer().config
        for case in ("theta_only", "legacy_scaling", "canonical_precedence"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                original = copy.deepcopy(base)
                talker = original["talker_config"]
                predictor = talker["code_predictor_config"]
                for component in (talker, predictor):
                    component.pop("rope_parameters", None)
                    component["rope_theta"] = 12345.0
                    if case != "theta_only":
                        component["rope_scaling"] = {"rope_type": "default"}
                    if case == "canonical_precedence":
                        component["rope_parameters"] = {"rope_type": "default", "rope_theta": 23456.0}
                if case != "theta_only":
                    talker["rope_scaling"].update(mrope_section=[4, 2, 2], interleaved=True)
                if case == "canonical_precedence":
                    talker["rope_parameters"].update(mrope_section=[4, 2, 2], interleaved=True)
                config = self._convert_config(directory, original, audio_config)
                model = Qwen3TTSForConditionalGeneration(config).eval()
                theta = 23456.0 if case == "canonical_precedence" else 12345.0
                for backbone in (model.model, model.code_predictor.model):
                    self.assertEqual(backbone.config.rope_parameters["rope_theta"], theta)
                    frequencies = backbone.rotary_emb.inv_freq
                    dimension = frequencies.numel() * 2
                    expected = 1.0 / theta ** (torch.arange(0, dimension, 2).float() / dimension)
                    torch.testing.assert_close(frequencies, expected)
                self.assertNotIn("interleaved", config.talker_config.rope_parameters)

    def test_attention_layout_and_window_boundary_control_masks(self):
        base = Qwen3TTSModelTester(self).get_config().to_dict()
        audio_config = _build_tiny_audio_tokenizer().config
        for explicit_layout in (False, True):
            with self.subTest(explicit_layout=explicit_layout), tempfile.TemporaryDirectory() as directory:
                original = copy.deepcopy(base)
                talker = original["talker_config"]
                # Explicit order deliberately differs from the max_window_layers-derived order.
                expected_types = (
                    ["sliding_attention", "full_attention"]
                    if explicit_layout
                    else ["full_attention", "sliding_attention"]
                )
                for component in (talker, talker["code_predictor_config"]):
                    component.update(use_sliding_window=True, sliding_window=2, max_window_layers=1)
                    component.pop("layer_types", None)
                    if explicit_layout:
                        component["layer_types"] = expected_types
                config = self._convert_config(directory, original, audio_config)
                model = Qwen3TTSForConditionalGeneration(config).eval()
                model.set_attn_implementation("eager")
                for backbone in (model.model, model.code_predictor.model):
                    self.assertEqual(backbone.config.max_window_layers, 1)
                    self.assertEqual(backbone.config.layer_types, expected_types)
                    with torch.no_grad():
                        output = backbone(
                            inputs_embeds=torch.ones(1, 6, backbone.config.hidden_size),
                            use_cache=False,
                            output_attentions=True,
                        )
                    for layer, attention, layer_type in zip(backbone.layers, output.attentions, expected_types):
                        self.assertEqual(
                            layer.self_attn.sliding_window, 2 if layer_type == "sliding_attention" else None
                        )
                        distant_attention = attention[0, :, -1, 0]
                        if layer_type == "sliding_attention":
                            self.assertTrue((distant_attention == 0).all())
                        else:
                            self.assertTrue((distant_attention > 0).all())

    def _save_original_weights(self, model, source, sharded):
        original = {}
        for name, tensor in model.state_dict().items():
            if name == "code_predictor.lm_head.weight":
                heads = tensor.chunk(model.config.talker_config.num_code_groups - 1)
                # Reverse insertion order so conversion must recover the head indices, not file order.
                for index in reversed(range(len(heads))):
                    original[f"talker.code_predictor.lm_head.{index}.weight"] = heads[index].contiguous()
                continue
            name = name.replace("text_projection.linear_1", "text_projection.linear_fc1").replace(
                "text_projection.linear_2", "text_projection.linear_fc2"
            )
            if name.startswith("model.embed_tokens"):
                name = name.replace("model.embed_tokens", "model.codec_embedding", 1)
            if not name.startswith("speaker_encoder."):
                name = "talker." + name
            original[name] = tensor.contiguous()
        if not sharded:
            save_file(original, str(source / "model.safetensors"))
            return
        keys = list(original)
        weight_map = {}
        for index in range(2):
            filename = f"model-{index + 1:05d}-of-00002.safetensors"
            shard = {name: original[name] for name in keys[index::2]}
            save_file(shard, str(source / filename))
            weight_map.update(dict.fromkeys(shard, filename))
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")

    @require_tokenizers
    @require_librosa
    @require_soundfile
    def test_conversion_reload_reference_audio_and_generation_defaults(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        for sharded, bfloat16, with_generation_config in ((False, False, True), (True, True, False)):
            with self.subTest(sharded=sharded, bfloat16=bfloat16), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(7)
                root = Path(directory)
                source, output, codec_path = root / "source", root / "output", root / "codec"
                source.mkdir()
                # Generation reserves the top 1024 talker IDs; real codec IDs occupy the lower 2048.
                codec = _build_tiny_audio_tokenizer(num_quantizers=4)
                codec.config.quantizer_config.codebook_size = 2048
                codec.config.quantizer_config.semantic_quantizer_config.codebook_size = 2048
                codec = type(codec)(codec.config).eval()
                codec.save_pretrained(codec_path)
                tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|audio|>", "assistant", "Hello", "."]
                backend = Tokenizer(
                    WordLevel({token: index for index, token in enumerate(tokens)}, unk_token=tokens[0])
                )
                backend.pre_tokenizer = Whitespace()
                tokenizer = Qwen2TokenizerFast(
                    tokenizer_object=backend,
                    unk_token=tokens[0],
                    eos_token=tokens[0],
                    pad_token=tokens[0],
                    additional_special_tokens=tokens[1:4],
                )
                tokenizer.chat_template = '{{ messages[-1]["content"] }}'
                tokenizer.save_pretrained(source)
                feature_extractor = Qwen3TTSFeatureExtractor(feature_size=8, n_fft=64, hop_length=16, win_length=64)
                feature_extractor.save_pretrained(source)
                config = Qwen3TTSModelTester(self).get_config(with_speaker_encoder=True)
                config.audio_config = codec.config
                config.im_start_token_id, config.im_end_token_id = 1, 2
                config.talker_config.text_vocab_size = len(tokenizer)
                config.talker_config.vocab_size = 3072
                config.talker_config.codec_eos_token_id = 2150
                config.talker_config.num_code_groups = 4
                config.talker_config.code_predictor_config.num_code_groups = 4
                config.talker_config.code_predictor_config.vocab_size = 2048
                reference = Qwen3TTSForConditionalGeneration(config).eval()
                original_config = config.to_dict()
                original_config.pop("audio_config")
                talker = original_config["talker_config"]
                rope = talker.pop("rope_parameters")
                talker["rope_theta"] = rope.pop("rope_theta")
                talker["rope_scaling"] = {**rope, "interleaved": True}
                predictor = talker["code_predictor_config"]
                predictor["rope_theta"] = predictor.pop("rope_parameters")["rope_theta"]
                (source / "config.json").write_text(json.dumps(original_config), encoding="utf-8")
                settings = {
                    "max_new_tokens": 4,
                    "do_sample": True,
                    "top_k": 1,
                    "top_p": 0.8,
                    "temperature": 0.7,
                    "subtalker_dosample": True,
                    "subtalker_top_k": 1,
                    "subtalker_top_p": 0.8,
                    "subtalker_temperature": 0.6,
                }
                if with_generation_config:
                    GenerationConfig(**settings).save_pretrained(source)
                self._save_original_weights(reference, source, sharded)
                convert_checkpoint(str(source), str(output), None, bfloat16, "20MB", str(codec_path))
                model = Qwen3TTSForConditionalGeneration.from_pretrained(output, local_files_only=True).eval()
                processor = AutoProcessor.from_pretrained(output, local_files_only=True)
                dtype = torch.bfloat16 if bfloat16 else torch.float32
                self.assertEqual(model.dtype, dtype)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, reference.state_dict()[name].to(dtype), rtol=0, atol=0, msg=name)
                waveform = np.sin(np.arange(2048, dtype=np.float32) * 0.03) * 0.1
                conversation = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello."}, {"type": "audio", "audio": waveform}],
                    }
                ]
                inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={"load_audio_backend": "librosa"},
                )
                expected = feature_extractor(
                    waveform, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
                )
                torch.testing.assert_close(inputs.input_features, expected.input_features)
                self.assertTrue(inputs.feature_attention_mask.all())
                inputs = inputs.to(dtype=dtype)
                if with_generation_config:
                    for name, value in settings.items():
                        self.assertEqual(getattr(model.generation_config, name), value)
                    # Top-k=1 on both generators must reproduce greedy codes for any RNG state.
                    with torch.no_grad():
                        expected_codes = reference.generate(
                            **inputs, max_new_tokens=4, do_sample=False, subtalker_dosample=False
                        ).sequences[0]
                        for seed in (11, 29):
                            torch.manual_seed(seed)
                            codes = model.generate(**inputs).sequences[0]
                            torch.testing.assert_close(codes, expected_codes)
                else:
                    expected_config, _ = reference._prepare_generation_config(None)
                    actual_config, _ = model._prepare_generation_config(None)
                    for name in ("max_new_tokens", "do_sample", "subtalker_dosample", "subtalker_top_k"):
                        self.assertEqual(getattr(actual_config, name), getattr(expected_config, name))
                    with torch.no_grad():
                        codes = model.generate(
                            **inputs, max_new_tokens=4, do_sample=False, subtalker_dosample=False
                        ).sequences[0]
                self.assertEqual(tuple(codes.shape), (3, 4))
                with torch.no_grad():
                    audio = processor.decode([codes])[0]
                self.assertTrue(torch.isfinite(audio).all())
                self.assertGreater(torch.count_nonzero(audio).item(), 0)
