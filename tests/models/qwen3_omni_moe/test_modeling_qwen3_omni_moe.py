# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
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
"""Testing suite for the PyTorch Qwen2.5-Omni model."""

import tempfile
import unittest
from io import BytesIO
from urllib.request import urlopen

import librosa
import pytest
import requests

from transformers import (
    AutoProcessor,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class Qwen3OmniMoeThinkerForConditionalGenerationTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        feat_seq_length=30,
        num_channels=3,
        image_size=16,
        seq_length=39,
        audio_token_id=1,
        image_token_id=2,
        video_token_id=3,
        position_id_per_seconds=13,
        seconds_per_chunk=2,
        audio_start_token_id=4,
        audio_end_token_id=5,
        user_token_id=6,
        vision_start_token_id=7,
        vision_end_token_id=8,
        initializer_range=0.02,
    ):
        self.parent = parent
        self.vision_config = {
            "depth": 2,
            "embed_dim": 32,
            "hidden_act": "quick_gelu",
            "hidden_size": 32,
            "out_hidden_size": 32,
            "intermediate_size": 24,
            "mlp_ratio": 4,
            "num_heads": 4,
            "patch_size": 16,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "initializer_range": 0.02,
            "deepstack_visual_indexes": [1],
        }
        self.audio_config = {
            "model_type": "qwen_omni_thinker_audio_encoder",
            "d_model": 32,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 32,
            "encoder_layers": 2,
            "num_mel_bins": 20,
            "max_source_positions": 1500,
            "initializer_range": 0.02,
            "n_window": 50,
            "output_dim": 32,
            "n_window_infer": 100,
        }
        self.text_config = {
            "rope_parameters": {
                "mrope_section": [1, 1, 2],
                "rope_type": "default",
                "type": "default",
                "interleaved": True,
            },
            "vocab_size": 99,
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-06,
            "use_cache": True,
            "tie_word_embeddings": False,
            "rope_theta": 1000000.0,
            "use_sliding_window": False,
            "sliding_window": 50,
            "max_window_layers": 3,
            "attention_dropout": 0.0,
            "pad_token_id": 0,
            "initializer_range": 0.02,
            "moe_intermediate_size": 32,
            "num_experts_per_tok": 2,
            "num_experts": 8,
            "decoder_sparse_step": 1,
        }
        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.seconds_per_chunk = seconds_per_chunk
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.user_token_id = user_token_id
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.feat_seq_length = feat_seq_length
        self.num_channels = num_channels
        self.image_size = image_size
        self.seq_length = seq_length
        self.is_training = False

        # Used from `self.model_tester` by common model tests
        self.num_hidden_layers = self.text_config["num_hidden_layers"]
        self.hidden_size = self.text_config["hidden_size"]
        self.num_attention_heads = self.text_config["num_attention_heads"]
        self.vocab_size = self.text_config["vocab_size"]

    def get_config(self):
        return Qwen3OmniMoeThinkerConfig(
            audio_config=self.audio_config,
            vision_config=self.vision_config,
            text_config=self.text_config,
            audio_token_id=self.audio_token_id,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            position_id_per_seconds=self.position_id_per_seconds,
            seconds_per_chunk=self.seconds_per_chunk,
            audio_start_token_id=self.audio_start_token_id,
            audio_end_token_id=self.audio_end_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            user_token_id=self.user_token_id,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )
        pixel_grid_thw = torch.LongTensor(
            [[1, self.image_size / patch_size, self.image_size / patch_size]] * self.batch_size
        ).to(pixel_values.device)
        input_features_values = floats_tensor(
            [self.batch_size, self.audio_config["num_mel_bins"], self.feat_seq_length]
        )
        feature_attention_mask = torch.ones([self.batch_size, self.feat_seq_length], dtype=torch.long).to(torch_device)
        return config, pixel_values, pixel_grid_thw, input_features_values, feature_attention_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, pixel_grid_thw, input_features_values, feature_attention_mask = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.get_text_config().vocab_size - 3) + 3
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)

        # Make sure no other tokens are set to special, to prevetn flakiness
        tokens_to_replace = torch.tensor(
            [
                config.image_token_id,
                config.audio_token_id,
                config.audio_start_token_id,
                config.audio_end_token_id,
                config.vision_start_token_id,
                config.vision_end_token_id,
            ],
            device=input_ids.device,
        )
        input_ids[torch.isin(input_ids, tokens_to_replace)] = config.text_config.pad_token_id

        attention_mask[:, :1] = 0

        # Audio token placeholders should be wrapped in start and end token ids
        audio_feat_length = (((self.feat_seq_length - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1
        input_ids[:, 1] = config.audio_start_token_id
        input_ids[:, 2 : (2 + audio_feat_length)] = config.audio_token_id
        input_ids[:, 2 + audio_feat_length] = config.audio_end_token_id

        # Image token placeholders should be wrapped in start and end token ids
        input_ids[:, -4:-1] = torch.tensor(
            [config.vision_start_token_id, config.image_token_id, config.vision_end_token_id]
        )
        inputs_dict = {
            "input_features": input_features_values,
            "feature_attention_mask": feature_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": pixel_grid_thw,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict

    def create_and_check_qwenomnithinker_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = Qwen3OmniMoeThinkerForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type=torch_device, dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class Qwen3OmniMoeThinkerForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Qwen3OmniMoeThinkerForConditionalGeneration`.
    """

    all_model_classes = (Qwen3OmniMoeThinkerForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (Qwen3OmniMoeThinkerForConditionalGeneration,) if is_torch_available() else ()
    skip_test_audio_features_output_shape = True  # Qwen3OmniMoe merges batch_size and audio_output_lengths in index 0
    _is_composite = True
    model_split_percents = [0.5, 0.9]

    def setUp(self):
        self.model_tester = Qwen3OmniMoeThinkerForConditionalGenerationTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3OmniMoeThinkerConfig, has_text_modality=False)

    @unittest.skip(reason="Cpu not yet supported because in QwenOmniThinker models")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Disk offload bin not yet supported because in QwenOmniThinker models")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Disk offload safetensors not yet supported because in QwenOmniThinker models")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(reason="Correct missing keys not yet supported because in QwenOmniThinker models")
    def test_correct_missing_keys(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in QwenOmniThinker models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Sdpa dispatch not yet supported because in QwenOmniThinker models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="QwenOmniThinker does not support output_hidden_states test")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="Don't have time to investigate at time of merge")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    def test_sdpa_can_dispatch_composite_models(self):
        # overwrite because Qwen2 is audio+text model (not vision+text)
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                text_attn = "sdpa" if model.model._supports_sdpa else "eager"
                audio_attn = "sdpa" if model.audio_tower._supports_sdpa else "eager"
                vision_attn = "sdpa" if model.visual._supports_sdpa else "eager"
                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(model.model.config._attn_implementation == text_attn)
                self.assertTrue(model.audio_tower.config._attn_implementation == audio_attn)
                self.assertTrue(model.visual.config._attn_implementation == vision_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.audio_tower.config._attn_implementation == "eager")
                self.assertTrue(model_eager.visual.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")

    def attention_mask_padding_matches_padding_free_with_position_ids(
        self, attn_implementation: str, fa_kwargs: bool = False
    ):
        max_new_tokens = 30
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.float16]:
                dummy_input = dummy_input.to(torch.bfloat16)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                if 0 in inputs_dict["attention_mask"][:, -1]:
                    inputs_dict["attention_mask"] = inputs_dict["attention_mask"].flip(1)
                dummy_attention_mask = inputs_dict["attention_mask"]
                inputs_dict["input_ids"][~dummy_attention_mask.bool()] = config.get_text_config().pad_token_id

                model = (
                    model_class.from_pretrained(
                        tmpdirname,
                        dtype=torch.bfloat16,
                        attn_implementation=attn_implementation,
                    )
                    .to(torch_device)
                    .eval()
                )

                # flatten
                padfree_inputs_dict = {
                    "input_features": inputs_dict["input_features"],
                    "feature_attention_mask": inputs_dict["feature_attention_mask"],
                    "pixel_values": inputs_dict["pixel_values"],
                    "image_grid_thw": inputs_dict["image_grid_thw"],
                    "input_ids": inputs_dict["input_ids"][dummy_attention_mask.bool()].unsqueeze(0),
                }

                # add position_ids
                vision_position_ids, deltas = model.get_rope_index(
                    input_ids=inputs_dict["input_ids"],
                    image_grid_thw=inputs_dict["image_grid_thw"],
                    attention_mask=inputs_dict["attention_mask"],
                    audio_seqlens=torch.sum(inputs_dict["feature_attention_mask"], dim=1),
                )  # [3, bs, padded-seq-len]
                vision_padfree_positions = vision_position_ids[:, dummy_attention_mask.bool()].view(
                    3, -1
                )  # [3, bs*padfree-len]
                text_padfree_positions = torch.cat(
                    [torch.arange(length) for length in dummy_attention_mask.sum(1).tolist()]
                )  # [1, bs*padfree-len]
                text_padfree_positions = text_padfree_positions.long().unsqueeze(0).to(torch_device)
                padfree_inputs_dict["position_ids"] = torch.cat([text_padfree_positions, vision_padfree_positions])[
                    :, None, :
                ]

                if fa_kwargs:
                    cu_seq_lens = [0] + dummy_attention_mask.sum(1).tolist()
                    cu_seq_lens = torch.tensor(cu_seq_lens, device=torch_device)
                    max_length = cu_seq_lens.diff().max().item()
                    padfree_inputs_dict.update(
                        {
                            "cu_seq_lens_q": cu_seq_lens.cumsum(-1).to(dtype=torch.int32),
                            "cu_seq_lens_k": cu_seq_lens.cumsum(-1).to(dtype=torch.int32),
                            "max_length_q": max_length,
                            "max_length_k": max_length,
                        }
                    )

                res_padded = model(**inputs_dict, use_cache=False)
                res_padfree = model(**padfree_inputs_dict, use_cache=False)

                logits_padded = res_padded.logits[inputs_dict["attention_mask"].bool()]
                logits_padfree = res_padfree.logits[0]

                # acceptable numerical instability
                tol = torch.finfo(torch.bfloat16).eps
                torch.testing.assert_close(logits_padded, logits_padfree, rtol=tol, atol=tol)

    @unittest.skip("Cannot do contrastive generation, has custom `generate()`")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("Cannot do contrastive generation, has custom `generate()`")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Cannot do contrastive generation, has custom `generate()`")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("Cannot generate from inputs embeds")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    # TODO (joao, raushan): there are multiple standardization issues in this model that prevent this test from
    # passing, fix me
    @unittest.skip("Cannot handle 4D attention mask")
    @pytest.mark.torch_compile_test
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip(
        "There seems to be something wrong with the config, that does not play well with this test. TODO fix me"
    )
    def test_save_load(self):
        pass

    @unittest.skip("Cannot handle 4D attention mask")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("In a rush to merge, cannot investigate now")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Cannot handle 4D attention mask")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("Cannot handle 4D attention mask")
    def test_custom_4d_attention_mask(self):
        pass

    @unittest.skip("We don't really care about this one, test is not that slow")
    def test_model_is_small(self):
        pass

    @unittest.skip("Qwen3Omni has no base model, model architecture is special")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip("FIXME this is important, but in a rush to merge, cannot investigate now")
    def test_get_rope_index_video_with_audio(self):
        image_grid_thw = torch.empty((0, 3), dtype=torch.long)

        # 3 * 2 * 2 = 12 video tokens
        video_grid_thw = torch.tensor([[3, 2, 2]], dtype=torch.long)

        # num_audio_tokens = ((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1
        # i.e.: 300 audio_seqlen -> 75 audio tokens
        audio_seqlens = torch.tensor([300], dtype=torch.long)

        second_per_grids = torch.tensor([1.0], dtype=torch.float)

        use_audio_in_video = True

        # fmt: off
        expected_position_ids = torch.tensor([
            [[
                 0,  1, # text
                 2,  2, # vision_bos + audio_bos

                # video chunk
                  3,  3,  3,  3,
                 28, 28, 28, 28,

                # audio chunk
                 3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                45, 46, 47, 48, 49, 50, 51, 52,

                # video chunk
                53, 53, 53, 53,

                # audio chunk
                53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,

                78, 78, # audio_eos + vision_eos
                79, 80, # text
            ]],
            [[
                 0,  1, # text
                 2,  2, # vision_bos + audio_bos

                # video chunk
                 3,  3,  4,  4,
                 3,  3,  4,  4,

                # audio chunk
                 3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                45, 46, 47, 48, 49, 50, 51, 52,

                # video chunk
                 3,  3,  4,  4,

                # audio chunk
                53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,

                78, 78, # audio_eos + vision_eos
                79, 80, # text
            ]],
            [[
                 0,  1, # text
                 2,  2, # vision_bos + audio_bos

                # video chunk
                 3,  4,  3,  4,
                 3,  4,  3,  4,

                # audio chunk
                 3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                45, 46, 47, 48, 49, 50, 51, 52,

                # video chunk
                3,  4,  3,  4,

                # audio chunk
                53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,

                78, 78, # audio_eos + vision_eos
                79, 80, # text
            ]],
        ], dtype=torch.long)
        # fmt: on

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            input_ids = torch.tensor(
                [
                    [
                        100,
                        101,
                    ]
                    + [
                        config.vision_start_token_id,
                        config.audio_start_token_id,
                    ]
                    # 1st chunk: 8 video tokens, 50 audio tokens
                    + [config.video_token_id] * 2 * 2 * 2
                    + [config.audio_token_id] * 50
                    +
                    # 2nd chunk: 4 video tokens, 25 audio tokens
                    [config.video_token_id] * 1 * 2 * 2
                    + [config.audio_token_id] * 25
                    + [
                        config.audio_end_token_id,
                        config.vision_end_token_id,
                    ]
                    + [
                        102,
                        103,
                    ]
                ],
                dtype=torch.long,
            )

            model = model_class(config)

            position_ids, mrope_position_deltas = model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=None,
                use_audio_in_video=use_audio_in_video,
                audio_seqlens=audio_seqlens,
                second_per_grids=second_per_grids,
            )

            self.assertTrue(torch.equal(position_ids, expected_position_ids))

    def _image_features_get_expected_num_attentions(self, model_tester=None):
        if model_tester is None:
            model_tester = self.model_tester
        return model_tester.vision_config["depth"]

    def _image_features_get_expected_num_hidden_states(self, model_tester=None):
        if model_tester is None:
            model_tester = self.model_tester
        return model_tester.vision_config["depth"] + 1

    def _audio_features_get_expected_num_attentions(self, model_tester=None):
        if model_tester is None:
            model_tester = self.model_tester
        return model_tester.audio_config["encoder_layers"]

    def _audio_features_get_expected_num_hidden_states(self, model_tester=None):
        if model_tester is None:
            model_tester = self.model_tester
        return model_tester.audio_config["encoder_layers"] + 1

    def _video_features_get_expected_num_attentions(self, model_tester=None):
        if model_tester is None:
            model_tester = self.model_tester
        return model_tester.vision_config["depth"]

    def _video_features_get_expected_num_hidden_states(self, model_tester=None):
        if model_tester is None:
            model_tester = self.model_tester
        return model_tester.vision_config["depth"] + 1


@require_torch
class Qwen3OmniModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", min_pixels=28 * 28, max_pixels=56 * 56
        )
        self.audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        self.audio_url_additional = (
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
        )
        self.image_url = "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg"
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": self.audio_url},
                    {"type": "image", "image_url": self.image_url},
                    {"type": "text", "text": "What's that sound and what kind of dog is this?"},
                ],
            }
        ]

        self.raw_audio, _ = librosa.load(
            BytesIO(urlopen(self.audio_url).read()), sr=self.processor.feature_extractor.sampling_rate
        )
        self.raw_audio_additional, _ = librosa.load(
            BytesIO(urlopen(self.audio_url_additional).read()), sr=self.processor.feature_extractor.sampling_rate
        )
        self.raw_image = Image.open(requests.get(self.image_url, stream=True).raw)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_integration_test(self):
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype=torch.bfloat16, device_map="auto"
        )

        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=text, audio=[self.raw_audio], images=[self.raw_image], return_tensors="pt", padding=True
        ).to(torch.bfloat16)

        expected_input_ids = torch.tensor(
            [
                151644,
                872,
                198,
                151669,
                151675,
                151675,
                151675,
                151675,
                151675,
                151675,
                151675,
                151675,
                151675,
                151675,
                151675,
                151675,
                151675,
            ]
        )
        torch.allclose(expected_input_ids, inputs.input_ids[0][:17], atol=3e-3)

        expected_pixel_slice = torch.tensor(
            [
                [0.5234, 0.6016, 0.6562],
                [0.9297, 0.9375, 0.9453],
                [0.4902, 0.5078, 0.4902],
                [0.8438, 0.8438, 0.8359],
                [0.9688, 0.9688, 0.9688],
                [0.9609, 0.9531, 0.9531],
            ],
            dtype=torch.bfloat16,
            device="cpu",
        )
        assert torch.allclose(expected_pixel_slice, inputs.pixel_values[:6, :3], atol=3e-3)

        # verify generation
        inputs = inputs.to(torch_device)

        output = model.generate(
            **inputs, thinker_temperature=0, thinker_do_sample=False, return_audio=False, thinker_max_new_tokens=20
        )

        EXPECTED_DECODED_TEXT = Expectations({
            ("cuda", (8, 6)): "user\nWhat's that sound and what kind of dog is this?\nassistant\nBased on the audio and visual information, here is a breakdown of what you're hearing and seeing:-",
            ("rocm", (9, 4)): "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog is a Labrador Retriever.",
        }).get_expectation()  # fmt: skip

        decoded_text = self.processor.decode(output[0], skip_special_tokens=True)
        self.assertEqual(decoded_text, EXPECTED_DECODED_TEXT)

    @slow
    def test_small_model_integration_test_batch(self):
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype=torch.bfloat16, device_map="auto"
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text] * 2,
            audio=[self.raw_audio, self.raw_audio],
            images=[self.raw_image, self.raw_image],
            return_tensors="pt",
            padding=True,
        ).to(torch_device, dtype=torch.bfloat16)

        output = model.generate(
            **inputs, thinker_temperature=0, thinker_do_sample=False, return_audio=False, thinker_max_new_tokens=20
        )

        EXPECTED_DECODED_TEXTS = Expectations(
            {
                ("cuda", 7) : [
                    "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is of glass shattering, and the dog in the picture is a Labrador Retriever",
                    "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is of glass shattering, and the dog in the picture is a Labrador Retriever",
                ],
                ("cuda", 8): [
                    "user\nWhat's that sound and what kind of dog is this?\nassistant\nBased on the audio and visual information, here is a breakdown of what you're hearing and seeing:\n\n",
                    "user\nWhat's that sound and what kind of dog is this?\nassistant\nBased on the audio and visual information, here is a breakdown of what you're hearing and seeing:\n\n"
                ],
                ("rocm", (9, 4)): [
                    "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog is a Labrador Retriever.",
                    "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog is a Labrador Retriever.",
                ],
            }
        ).get_expectation()  # fmt: skip

        decoded_texts = self.processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(decoded_texts, EXPECTED_DECODED_TEXTS)

    @slow
    def test_small_model_integration_test_multiturn(self):
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype=torch.bfloat16, device_map="auto"
        )

        messages = [
            self.messages[0],
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The sound is glass shattering, and the dog appears to be a Labrador Retriever.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": self.audio_url_additional},
                    {"type": "text", "text": "How about this one?"},
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=text,
            audio=[self.raw_audio, self.raw_audio_additional],
            images=[self.raw_image],
            return_tensors="pt",
            padding=True,
        ).to(torch_device, dtype=torch.bfloat16)

        output = model.generate(
            **inputs, thinker_temperature=0, thinker_do_sample=False, return_audio=False, thinker_max_new_tokens=20
        )

        EXPECTED_DECODED_TEXT = "user\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog appears to be a Labrador Retriever.\nuser\nHow about this one?\nassistant\nThe sound is a person coughing."

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_w_audio(self):
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype=torch.bfloat16, device_map="auto"
        )
        audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio_url}],
            },
        ]
        audio, _ = librosa.load(BytesIO(urlopen(audio_url).read()), sr=self.processor.feature_extractor.sampling_rate)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, audio=[audio], return_tensors="pt", padding=True).to(
            torch_device, dtype=torch.bfloat16
        )

        output = model.generate(
            **inputs,
            thinker_temperature=0,
            thinker_do_sample=False,
            thinker_max_new_tokens=20,
            talker_max_new_tokens=10,
        )

        EXPECTED_DECODED_TEXTS = Expectations(
            {
                ("cuda", 7): "system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.\nuser\n\nassistant\nWell, I can try. But it's not always that accurate. I might be able to make",
                ("cuda", 8): "'system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.\nuser\n\nassistant\nYes, I can analyze audio inputs to understand spoken content, and I can also make inferences about'",
            }
        )  # fmt: skip
        EXPECTED_DECODED_TEXT = EXPECTED_DECODED_TEXTS.get_expectation()

        self.assertEqual(
            self.processor.decode(output[0][0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
        self.assertFalse(torch.isnan(output[1]).any().item())

    @slow
    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    def test_small_model_integration_test_batch_flashatt2(self):
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text, text],
            audio=[self.raw_audio, self.raw_audio],
            images=[self.raw_image, self.raw_image],
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        output = model.generate(**inputs, thinker_temperature=0, thinker_do_sample=False, return_audio=False)

        EXPECTED_DECODED_TEXT = Expectations({
            ("cuda", None): "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog appears to be a Labrador Retriever.",
            ("cuda", (8, 6)): "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog is a Labrador Retriever.",
            ("rocm", (9, 4)): "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog is a Labrador Retriever.",
        }).get_expectation()  # fmt: skip

        decoded_texts = self.processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(decoded_texts[0], EXPECTED_DECODED_TEXT)
        self.assertEqual(decoded_texts[1], EXPECTED_DECODED_TEXT)
