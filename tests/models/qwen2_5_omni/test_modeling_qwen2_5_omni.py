# coding=utf-8
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

import inspect
import tempfile
import unittest
from io import BytesIO
from urllib.request import urlopen

import librosa
import numpy as np
import requests
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    Qwen2_5OmniModel,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_deepspeed,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    require_torch_sdpa,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    set_model_tester_for_less_flaky_test,
    slow,
    torch_device,
)
from transformers.utils import (
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    _deepspeed_zero3,
    floats_tensor,
    ids_tensor,
    sdpa_kernel,
)


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class Qwen2_5OmniThinkerForConditionalGenerationTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        feat_seq_length=200,
        num_channels=3,
        image_size=14,
        seq_length=39,
        vision_config={
            "depth": 2,
            "embed_dim": 32,
            "hidden_act": "quick_gelu",
            "hidden_size": 32,
            "mlp_ratio": 4,
            "num_heads": 4,
            "patch_size": 14,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "initializer_range": 0.02,
        },
        audio_config={
            "model_type": "qwen_omni_thinker_audio_encoder",
            "d_model": 32,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 32,
            "encoder_layers": 2,
            "num_mel_bins": 20,
            "max_source_positions": 1500,
            "initializer_range": 0.02,
            "n_window": 100,
            "output_dim": 32,
        },
        rope_scaling={"mrope_section": [1, 1, 2], "rope_type": "default", "type": "default"},
        audio_token_index=1,
        image_token_index=2,
        video_token_index=3,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=1024,
        rms_norm_eps=1e-06,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=50,
        max_window_layers=3,
        attention_dropout=0.0,
        position_id_per_seconds=25,
        seconds_per_chunk=2,
        audio_start_token_id=4,
        audio_end_token_id=5,
        user_token_id=6,
        init_std=0.02,
    ):
        self.parent = parent
        self.audio_config = audio_config
        self.vision_config = vision_config
        self.audio_token_index = audio_token_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.position_id_per_seconds = position_id_per_seconds
        self.seconds_per_chunk = seconds_per_chunk
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.user_token_id = user_token_id
        self.init_std = init_std
        self.batch_size = batch_size
        self.feat_seq_length = feat_seq_length
        self.num_channels = num_channels
        self.image_size = image_size
        self.seq_length = seq_length
        self.is_training = False
        self.num_hidden_states_types = 0

    def get_config(self):
        return Qwen2_5OmniThinkerConfig(
            audio_config=self.audio_config,
            vision_config=self.vision_config,
            audio_token_index=self.audio_token_index,
            image_token_index=self.image_token_index,
            video_token_index=self.video_token_index,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=self.use_cache,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=self.rope_theta,
            use_sliding_window=self.use_sliding_window,
            sliding_window=self.sliding_window,
            max_window_layers=self.max_window_layers,
            attention_dropout=self.attention_dropout,
            rope_scaling=self.rope_scaling,
            position_id_per_seconds=self.position_id_per_seconds,
            seconds_per_chunk=self.seconds_per_chunk,
            audio_start_token_id=self.audio_start_token_id,
            audio_end_token_id=self.audio_end_token_id,
            user_token_id=self.user_token_id,
            init_std=self.init_std,
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
            [
                self.audio_config["num_mel_bins"],
                self.feat_seq_length * self.batch_size,
            ]
        )
        feature_attention_mask = torch.ones([self.batch_size, self.feat_seq_length], dtype=torch.long).to(torch_device)
        return config, pixel_values, pixel_grid_thw, input_features_values, feature_attention_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, pixel_grid_thw, input_features_values, feature_attention_mask = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size - 3) + 3
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)

        attention_mask[:, :1] = 0
        audio_feat_length = ((self.feat_seq_length - 1) // 2 + 1 - 2) // 2 + 1
        input_ids[:, 1 : (1 + audio_feat_length)] = config.audio_token_index
        input_ids[:, -2] = config.image_token_index
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
        model = Qwen2_5OmniThinkerForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class Qwen2_5OmniThinkerForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `Qwen2_5OmniThinkerForConditionalGeneration`.
    """

    all_model_classes = (Qwen2_5OmniThinkerForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    _is_composite = True
    # Doesn't run generation tests. We have a custom `generate` with partial feature
    all_generative_model_classes = ()

    def setUp(self):
        self.model_tester = Qwen2_5OmniThinkerForConditionalGenerationTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen2_5OmniThinkerConfig, has_text_modality=False)

    @unittest.skip(reason="Compile not yet supported because in QwenOmniThinker models")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in QwenOmniThinker models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="QwenOmniThinker does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="QwenOmniThinker does not support output_hidden_states test")
    def test_model_outputs_equivalence(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[-1][0]
        hidden_states.retain_grad()

        if self.has_attentions:
            attentions = outputs.attentions[0]
            attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)

        if self.has_attentions:
            self.assertIsNotNone(attentions.grad)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states[1]

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @require_torch_sdpa
    def test_eager_matches_sdpa_inference(
        self, name, torch_dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        # TODO: we shouldn't need to do this skip, i.e. the test would be composable from the model tester. CLIP-like
        # models have a custom mixin, which we detect to skip this test.
        if not any(".ModelTesterMixin" in str(base) for base in self.__class__.__bases__):
            self.skipTest(reason="CLIP-like models have a different `test_eager_matches_sdpa_inference`")

        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        # convert shorthand name to torch.dtype
        if torch_dtype == "fp16":
            torch_dtype = torch.float16
        elif torch_dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "fp32":
            torch_dtype = torch.float32

        if not is_torch_fp16_available_on_device(torch_device) and torch_dtype == torch.float16:
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        if not is_torch_bf16_available_on_device(torch_device) and torch_dtype == torch.bfloat16:
            self.skipTest(
                f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
            )

        # Dictionary of tolerances for eager <> sdpa tests. Key = (device, sdpa_kernels_enabled, dtype)
        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
        rtols = {
            ("cpu", False, torch.float32): 1e-4,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-4,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-4,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-4,
            ("cuda", True, torch.bfloat16): 3e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        set_model_tester_for_less_flaky_test(self)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            set_config_for_less_flaky_test(config)
            model = model_class(config)
            # TODO: standardize the interfaces for musicgen models, see other todo in this test
            if model.__class__.__name__ == "MusicgenMelodyForConditionalGeneration":
                is_encoder_decoder = True
            else:
                is_encoder_decoder = model.config.is_encoder_decoder

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_from_pretrained_kwargs = {
                    "pretrained_model_name_or_path": tmpdirname,
                    "torch_dtype": torch_dtype,
                }

                if (
                    hasattr(config, "use_mask_token")
                    or "use_mask_token" in inspect.signature(model.__init__).parameters
                ):
                    model_from_pretrained_kwargs["use_mask_token"] = True

                # TODO: remove this try/except, models should have a shared API
                try:
                    model_sdpa = model_class.from_pretrained(
                        **model_from_pretrained_kwargs, attn_implementation="sdpa"
                    )
                except ValueError:
                    model_sdpa = model_class.from_pretrained(**model_from_pretrained_kwargs)
                model_sdpa = model_sdpa.eval().to(torch_device, dtype=torch_dtype)

                model_eager = model_class.from_pretrained(**model_from_pretrained_kwargs, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device, dtype=torch_dtype)

                set_model_for_less_flaky_test(model_eager)
                set_model_for_less_flaky_test(model_sdpa)

                can_output_attn = "output_attentions" in inspect.signature(model_sdpa.forward).parameters
                if not (self.has_attentions and can_output_attn) and output_attentions:
                    self.skipTest(reason="Model does not support output_attentions")

                # TODO: if we can also check with `batch_size=1` without being flaky?
                for batch_size in [7]:
                    # musicgen decoder models; TODO: find better abstraction
                    if hasattr(self.model_tester, "num_codebooks") and not hasattr(model_eager, "text_encoder"):
                        input_data_batch_size = batch_size * self.model_tester.num_codebooks
                    else:
                        input_data_batch_size = batch_size

                    dummy_input = inputs_dict[model.main_input_name]

                    if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                        dummy_input = dummy_input.to(torch_dtype)

                    dummy_input = dummy_input[:input_data_batch_size]
                    if dummy_input.shape[0] != input_data_batch_size:
                        if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                            extension = torch.rand(
                                input_data_batch_size - dummy_input.shape[0],
                                *dummy_input.shape[1:],
                                dtype=torch_dtype,
                                device=torch_device,
                            )
                            dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)
                        else:
                            extension = torch.randint(
                                high=5,
                                size=(input_data_batch_size - dummy_input.shape[0], *dummy_input.shape[1:]),
                                dtype=dummy_input.dtype,
                                device=torch_device,
                            )
                            dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)

                    if not use_attention_mask:
                        dummy_attention_mask = None
                    else:
                        dummy_attention_mask = inputs_dict.get("attention_mask", None)
                        if dummy_attention_mask is None:
                            if is_encoder_decoder:
                                seqlen = inputs_dict.get("decoder_input_ids", dummy_input).shape[-1]
                            else:
                                seqlen = dummy_input.shape[-1]
                            dummy_attention_mask = torch.ones(batch_size, seqlen).to(torch.int64).to(torch_device)

                        dummy_attention_mask = dummy_attention_mask[:batch_size]
                        if dummy_attention_mask.shape[0] != batch_size:
                            extension = torch.ones(
                                batch_size - dummy_attention_mask.shape[0],
                                *dummy_attention_mask.shape[1:],
                                dtype=dummy_attention_mask.dtype,
                                device=torch_device,
                            )
                            dummy_attention_mask = torch.cat((dummy_attention_mask, extension), dim=0)
                            dummy_attention_mask = dummy_attention_mask.to(torch_device)

                        dummy_attention_mask[:] = 1
                        if padding_side == "left":
                            dummy_attention_mask[-1, :2] = 0
                            dummy_attention_mask[-1, 2:] = 1
                        elif padding_side == "right":
                            dummy_attention_mask[-1, -2:] = 0
                            dummy_attention_mask[-1, :-2] = 1

                    if is_encoder_decoder:
                        # musicgen encoder-decoder models; TODO: find better abstraction
                        if hasattr(self.model_tester, "num_codebooks"):
                            input_data_batch_size = batch_size * self.model_tester.num_codebooks
                        else:
                            input_data_batch_size = batch_size

                        decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)[:input_data_batch_size]
                        if decoder_input_ids.shape[0] != input_data_batch_size:
                            extension = torch.ones(
                                input_data_batch_size - decoder_input_ids.shape[0],
                                *decoder_input_ids.shape[1:],
                                dtype=decoder_input_ids.dtype,
                                device=torch_device,
                            )
                            decoder_input_ids = torch.cat((decoder_input_ids, extension), dim=0)
                            decoder_input_ids = decoder_input_ids.to(torch_device)

                        # TODO: never an `attention_mask` arg here?
                        processed_inputs = {
                            model.main_input_name: dummy_input,
                            "decoder_input_ids": decoder_input_ids,
                            "decoder_attention_mask": dummy_attention_mask,
                            "output_hidden_states": True,
                        }
                    else:
                        processed_inputs = {
                            model.main_input_name: dummy_input,
                            "output_hidden_states": True,
                        }

                        # Otherwise fails for e.g. WhisperEncoderModel
                        if "attention_mask" in inspect.signature(model_eager.forward).parameters:
                            processed_inputs["attention_mask"] = dummy_attention_mask

                        if (
                            self.has_attentions
                            and "output_attentions" in inspect.signature(model_sdpa.forward).parameters
                        ):
                            processed_inputs["output_attentions"] = output_attentions
                    if "bool_masked_pos" in inspect.signature(model_eager.forward).parameters:
                        dummy_mask = torch.ones((self.model_tester.num_masks,))

                        # In case of additional token (like class) we define a custom `mask_length`
                        if hasattr(self.model_tester, "mask_length"):
                            mask_length = self.model_tester.mask_length - dummy_mask.size(0)
                        else:
                            mask_length = self.model_tester.seq_length - dummy_mask.size(0)
                        dummy_mask = torch.cat([dummy_mask, torch.zeros(mask_length)])
                        dummy_bool_masked_pos = dummy_mask.expand(batch_size, -1).bool()
                        processed_inputs["bool_masked_pos"] = dummy_bool_masked_pos.to(torch_device)

                    if "noise" in inspect.signature(model_eager.forward).parameters:
                        np.random.seed(2)
                        num_patches = int((self.model_tester.image_size // self.model_tester.patch_size) ** 2)
                        noise = np.random.uniform(size=(batch_size, num_patches))
                        processed_inputs["noise"] = torch.from_numpy(noise)

                    # TODO: test gradients as well (& for FA2 as well!)
                    with torch.no_grad():
                        with sdpa_kernel(
                            enable_flash=enable_kernels,
                            enable_math=True,
                            enable_mem_efficient=enable_kernels,
                        ):
                            prepared_inputs = self._prepare_for_class(processed_inputs, model_class)
                            outputs_eager = model_eager(**prepared_inputs)
                            outputs_sdpa = model_sdpa(**prepared_inputs)

                    # TODO: rename logits -> hidden_states
                    if hasattr(outputs_eager, "vision_hidden_states"):
                        logits_eager = outputs_eager.vision_hidden_states[-1]
                        logits_sdpa = outputs_sdpa.vision_hidden_states[-1]
                    elif hasattr(outputs_eager, "audio_values"):
                        logits_eager = outputs_eager.audio_values
                        logits_sdpa = outputs_sdpa.audio_values
                    else:
                        logits_eager = (
                            outputs_eager.decoder_hidden_states[-1]
                            if hasattr(outputs_eager, "decoder_hidden_states")
                            else outputs_eager.hidden_states[1][-1]
                        )
                        logits_sdpa = (
                            outputs_sdpa.decoder_hidden_states[-1]
                            if hasattr(outputs_sdpa, "decoder_hidden_states")
                            else outputs_sdpa.hidden_states[1][-1]
                        )

                    if torch_device in ["cpu", "cuda"]:
                        atol = atols[torch_device, enable_kernels, torch_dtype]
                        rtol = rtols[torch_device, enable_kernels, torch_dtype]
                    elif torch_device == "xpu":
                        # As of PyTorch 2.5 XPU backend supports only torch.nn.attention.SDPBackend.MATH
                        # which is implemented on PyTorch level using aten operators and is
                        # device agnostic with respect to implementation of each aten operator.
                        atol = atols["cuda", False, torch_dtype]
                        rtol = rtols["cuda", False, torch_dtype]
                    else:
                        atol = 1e-7
                        rtol = 1e-4

                    # Masked tokens output slightly deviates - we don't mind that.
                    if use_attention_mask:
                        _logits_sdpa = torch.zeros_like(input=logits_sdpa)
                        _logits_eager = torch.zeros_like(input=logits_eager)

                        _logits_sdpa[:-1] = logits_sdpa[:-1]
                        _logits_eager[:-1] = logits_eager[:-1]

                        if padding_side == "left":
                            _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, 2:]
                            _logits_eager[-1:, 2:] = logits_eager[-1:, 2:]

                        elif padding_side == "right":
                            _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, :-2]
                            _logits_eager[-1:, 2:] = logits_eager[-1:, :-2]

                        logits_sdpa = _logits_sdpa
                        logits_eager = _logits_eager

                    results = [
                        torch.allclose(_logits_sdpa, _logits_eager, atol=atol, rtol=rtol)
                        for (_logits_sdpa, _logits_eager) in zip(logits_sdpa, logits_eager)
                    ]
                    # If 80% batch elements have matched results, it's fine
                    if np.mean(results) < 0.8:
                        mean_relative_diff = ((logits_sdpa - logits_eager).abs() / (logits_eager.abs() + 1e-12)).mean()
                        raise ValueError(
                            f"mean relative difference: {mean_relative_diff:.3e}, torch atol = {atol}, torch rtol = "
                            f"{rtol}"
                        )

    @require_torch_sdpa
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

    @require_deepspeed
    @require_torch_gpu
    def test_resize_tokens_embeddings_with_deepspeed(self):
        ds_config = {
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu", "pin_memory": True},
            },
            "train_batch_size": 3,
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_tokens_embeddings()

    @require_deepspeed
    @require_torch_gpu
    def test_resize_embeddings_untied_with_deepspeed(self):
        ds_config = {
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu", "pin_memory": True},
            },
            "train_batch_size": 3,
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_embeddings_untied()


@require_torch
class Qwen2_5OmniModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
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
        model = Qwen2_5OmniModel.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.float32, device_map="auto")

        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], audios=[self.raw_audio], images=[self.raw_image], return_tensors="pt", padding=True
        )

        expected_input_ids = torch.tensor(
            [
                151644,
                8948,
                198,
                2610,
                525,
                264,
                10950,
                17847,
                13,
                151645,
                198,
                151644,
                872,
                198,
                151647,
                151646,
                151648,
            ]
        )
        assert torch.allclose(expected_input_ids, inputs.input_ids[0][:17], atol=3e-3)

        expected_pixel_slice = torch.tensor(
            [
                [0.8792, 0.8792, 0.9084],
                [1.1858, 1.1858, 1.2296],
                [1.2004, 1.2004, 1.2150],
                [1.4340, 1.4340, 1.4194],
                [1.3902, 1.4048, 1.4194],
                [1.5216, 1.5362, 1.5362],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        assert torch.allclose(expected_pixel_slice, inputs.pixel_values[:6, :3], atol=3e-3)

        # verify generation
        inputs = inputs.to(torch_device)

        output = model.generate(**inputs, thinker_temperature=0, thinker_do_sample=False, return_audio=False)

        EXPECTED_DECODED_TEXT = "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog appears to be a Labrador Retriever."

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch(self):
        model = Qwen2_5OmniModel.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.float32, device_map="auto")
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text, text],
            audios=[self.raw_audio, self.raw_audio],
            images=[self.raw_image, self.raw_image],
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        output = model.generate(**inputs, thinker_temperature=0, thinker_do_sample=False, return_audio=False)

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog appears to be a Labrador Retriever.",
            "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog appears to be a Labrador Retriever.",
        ]

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_multiturn(self):
        model = Qwen2_5OmniModel.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.float32, device_map="auto")

        messages = [
            self.messages[0],
            {
                "role": "assistant",
                "content": "The sound is glass shattering, and the dog appears to be a Labrador Retriever.",
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
            text=[text],
            audios=[self.raw_audio, self.raw_audio_additional],
            images=[self.raw_image],
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        output = model.generate(**inputs, thinker_temperature=0, thinker_do_sample=False, return_audio=False)

        EXPECTED_DECODED_TEXT = "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog appears to be a Labrador Retriever.\nuser\nHow about this one?\nassistant\nThe sound is a cough."

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_w_audio(self):
        model = Qwen2_5OmniModel.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.float32, device_map="auto")
        audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"

        messages = [
            {
                "role": "system",
                "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            },
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio_url}],
            },
        ]
        audio, _ = librosa.load(BytesIO(urlopen(audio_url).read()), sr=self.processor.feature_extractor.sampling_rate)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], audios=[audio], return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, thinker_temperature=0, thinker_do_sample=False)

        EXPECTED_DECODED_TEXT = "system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.\nuser\n\nassistant\nWell, I can't really guess your age and gender just from your voice. There are so many factors that can affect how a voice sounds, like the environment you're in, how you're feeling at the moment, and even the microphone you're using. But if you want to share more about your voice, like if it's high - pitched or low - pitched, that might give me a bit of an idea. So, what can you tell me about your voice?"

        self.assertEqual(
            self.processor.decode(output[0][0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
        self.assertFalse(torch.isnan(output[1]).any().item())

    @slow
    @require_flash_attn
    @require_torch_gpu
    def test_small_model_integration_test_batch_flashatt2(self):
        model = Qwen2_5OmniModel.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text, text],
            audios=[self.raw_audio, self.raw_audio],
            images=[self.raw_image, self.raw_image],
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        output = model.generate(**inputs, thinker_temperature=0, thinker_do_sample=False, return_audio=False)

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog appears to be a Labrador Retriever.",
            "system\nYou are a helpful assistant.\nuser\nWhat's that sound and what kind of dog is this?\nassistant\nThe sound is glass shattering, and the dog appears to be a Labrador Retriever.",
        ]

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True)[0],
            self.processor.batch_decode(output, skip_special_tokens=True)[1],
        )
