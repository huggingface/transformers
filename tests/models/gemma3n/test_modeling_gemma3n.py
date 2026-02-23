# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Gemma3n model."""

import copy
import inspect
import unittest

import numpy as np
import pytest
from datasets import load_dataset
from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Gemma3nAudioConfig,
    Gemma3nAudioFeatureExtractor,
    Gemma3nConfig,
    StaticCache,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_deterministic_for_xpu,
    require_torch,
    require_torch_accelerator,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin, has_similar_generate_outputs
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    _test_eager_matches_sdpa_inference,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch

    from transformers import (
        Gemma3nAudioEncoder,
        Gemma3nForCausalLM,
        Gemma3nForConditionalGeneration,
        Gemma3nModel,
        Gemma3nTextModel,
    )


class Gemma3nAudioModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=32,  # feature_size / input_feat_size
        sampling_rate=16_000,
        raw_audio_length=8_000,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.raw_audio_length = raw_audio_length
        self.is_training = is_training

    def get_feature_extractor_config(self):
        return {
            "feature_size": self.num_channels,
            "sampling_rate": self.sampling_rate,
            "padding_value": 0.0,
            "return_attention_mask": True,
            "frame_length_ms": 32.0,
            "hop_length_ms": 10.0,
            "dither": 0.0,  # Important for determinism
        }

    def get_audio_encoder_config(self):
        return Gemma3nAudioConfig(
            input_feat_size=self.num_channels,
            hidden_size=32,
            conf_num_attention_heads=4,
            conf_num_hidden_layers=2,
            sscp_conv_channel_size=(16, 8),
            conf_conv_kernel_size=3,
            conf_attention_chunk_size=4,
            conf_attention_context_left=5,
        )

    def prepare_config_and_inputs_for_common(self):
        # Prepare inputs for the audio encoder
        feature_extractor_config = self.get_feature_extractor_config()
        audio_encoder_config = self.get_audio_encoder_config()

        np.random.seed(0)
        raw_speech_1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, self.raw_audio_length)).astype(np.float32)
        raw_speech_2 = np.random.randn(self.raw_audio_length // 2).astype(np.float32)
        raw_speech = [raw_speech_1, raw_speech_2]

        feature_extractor = Gemma3nAudioFeatureExtractor(**feature_extractor_config)
        audio_inputs = feature_extractor(raw_speech, return_tensors="pt")

        input_features = audio_inputs["input_features"]
        # The encoder expects a padding mask (True for padding), while the feature extractor
        # returns an attention mask (True for valid tokens). We must invert it.
        input_features_mask = ~audio_inputs["input_features_mask"].to(torch.bool)

        inputs_dict = {
            "audio_mel": input_features,
            "audio_mel_mask": input_features_mask,
        }
        return audio_encoder_config, inputs_dict


@unittest.skip("Skipped for now!")
@require_torch
class Gemma3nAudioModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Gemma3nAudioEncoder,) if is_torch_available() else ()

    test_missing_keys = False
    is_generative = False
    _is_stateful = True
    main_input_name = "audio_mel"

    def setUp(self):
        self.model_tester = Gemma3nAudioModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma3nAudioConfig, hidden_size=37)
        torch.manual_seed(0)

        # The following values are golden outputs from a deterministic run of the components.
        # They are used to ensure that changes to the code do not alter the numerical output.
        # Generated with seeds np.random.seed(0) and torch.manual_seed(0).
        self.expected_input_features_shape = (2, 48, 32)
        self.expected_input_features_slice = np.array([-5.733152, -5.337127, -4.916284, -4.378989, -3.7622747])
        self.expected_input_features_mask_shape = (2, 48)
        self.expected_input_features_mask_slice = np.array([True, True, True, True, False])

        self.expected_encoder_output_shape = (2, 3, 32)
        self.expected_encoder_output_slice = torch.tensor([-0.4159, 0.6459, 0.6305, 2.2902, 0.9683])
        self.expected_encoder_mask_shape = (2, 3)
        self.expected_encoder_mask_slice = torch.tensor([False, False, True])

        # Prepare a shared feature extractor and raw audio for the tests
        self.feature_extractor = Gemma3nAudioFeatureExtractor(**self.model_tester.get_feature_extractor_config())
        np.random.seed(0)
        raw_speech_1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, self.model_tester.raw_audio_length)).astype(
            np.float32
        )
        raw_speech_2 = np.random.randn(self.model_tester.raw_audio_length // 2).astype(np.float32)
        self.raw_speech = [raw_speech_1, raw_speech_2]

    @unittest.skip("Audio encoder does not support attention output")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Audio encoder does not support hidden state output")
    def test_hidden_states_output(self):
        pass

    @unittest.skip("Audio encoder returns a tuple, not a ModelOutput object, skipping equivalence test.")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip("Audio encoder does not support retaining gradients on hidden states/attentions.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("Audio encoder does not have a concept of token embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("Audio encoder does not have a concept of token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip("This model has a complex downsampling scheme that is hard to test with the generic batching test.")
    def test_batching_equivalence(self):
        pass

    def test_feature_extractor(self):
        """
        Tests the feature extractor's output against pre-computed golden values.
        This ensures the NumPy-based audio preprocessing is correct and consistent.
        """
        audio_inputs = self.feature_extractor(
            self.raw_speech, padding="longest", pad_to_multiple_of=128, return_tensors="np"
        )

        input_features = audio_inputs["input_features"]
        self.assertEqual(input_features.shape, self.expected_input_features_shape)
        np.testing.assert_allclose(input_features[0, 0, :5], self.expected_input_features_slice, rtol=1e-5, atol=1e-5)

        input_features_mask = audio_inputs["input_features_mask"]
        self.assertEqual(input_features_mask.shape, self.expected_input_features_mask_shape)
        # The second audio sample is shorter (22 frames vs 48), so its mask should become False at index 22
        np.testing.assert_array_equal(input_features_mask[1, 21:26], self.expected_input_features_mask_slice)

    def test_audio_encoder(self):
        """
        Tests the audio encoder's forward pass against pre-computed golden values.
        This ensures the PyTorch-based audio encoding model is correct and consistent.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Gemma3nAudioEncoder(config).to(torch_device).eval()

        with torch.no_grad():
            encoder_output, encoder_mask = model(**inputs_dict)

        # Check output encodings
        self.assertEqual(encoder_output.shape, self.expected_encoder_output_shape)
        torch.testing.assert_close(
            encoder_output[0, 0, :5], self.expected_encoder_output_slice.to(torch_device), rtol=1e-4, atol=1e-4
        )

        # Check output mask (True means padded)
        # Second sample has 22 feature frames. After downsampling by 4 (conv) -> 5 frames. After downsampling by 4 (reduction) -> 1 frame.
        # So the mask should be [False, True, True]
        self.assertEqual(encoder_mask.shape, self.expected_encoder_mask_shape)
        torch.testing.assert_close(encoder_mask[1, :], self.expected_encoder_mask_slice.to(torch_device))


class Gemma3nTextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Gemma3nTextModel
        causal_lm_class = Gemma3nForCausalLM

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        vocab_size_per_layer_input=99,
        hidden_size=16,
        hidden_size_per_layer_input=16,
        num_hidden_layers=4,  # override to correctly test sharing cache pattern
        num_kv_shared_layers=2,  # important to override
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],  # similarly we want to test sharing on both types
        num_attention_heads=2,
        num_key_value_heads=2,
        altup_num_inputs=2,
        intermediate_size=21,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        is_decoder=False,
    ):
        self._verify_and_infer_model_attributes()
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.hidden_size = hidden_size
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.num_hidden_layers = num_hidden_layers
        self.num_kv_shared_layers = num_kv_shared_layers
        self.layer_types = layer_types
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.altup_num_inputs = altup_num_inputs
        self.intermediate_size = intermediate_size
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.is_decoder = is_decoder


@require_torch
class Gemma3nTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Gemma3nTextModelTester
    _is_stateful = True
    model_split_percents = [0.5, 0.6]

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        "Gemma3n has special hidden states shape with 1 additional dim (which is then reduced with projections)"

        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (output_length - prompt_length))

        # When `output_hidden_states=True`, each iteration of generate appends the hidden states corresponding to the
        # new token(s)
        for generated_length, iter_hidden_states in enumerate(hidden_states):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length
            expected_shape = (config.altup_num_inputs, batch_size, model_input_length, config.hidden_size)
            # check hidden size
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [expected_shape] * len(iter_hidden_states),
            )

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
    ):
        "We need to relax a bit the `atols` and `rtols` for fp32 here due to the altup projections"
        atols = {
            ("cpu", False, torch.float32): 5e-2,  # this was relaxed
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 5e-2,  # this was relaxed
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 5e-2,  # this was relaxed
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 5e-2,  # this was relaxed
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        rtols = {
            ("cpu", False, torch.float32): 1e-2,  # this was relaxed
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-2,  # this was relaxed
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-2,  # this was relaxed
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-2,  # this was relaxed
            ("cuda", True, torch.bfloat16): 3e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        _test_eager_matches_sdpa_inference(
            self,
            name,
            dtype,
            padding_side,
            use_attention_mask,
            output_attentions,
            enable_kernels,
            atols=atols,
            rtols=rtols,
        )

    @pytest.mark.generate
    @unittest.skip("Gemma3n does not support QuantizedCache as it performs cache manipulation in the forward pass")
    def test_generate_with_quant_cache(self):
        pass

    @unittest.skip("Gemma3n applies key/query norm which doesn't work with packing")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Gemma3n applies key/query norm which doesn't work with packing")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Gemma3n only support fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @pytest.mark.generate
    def test_generate_from_inputs_embeds_with_static_cache(self):
        """
        Test that StaticCache can generate from inputs_embeds and calculates max_cache_length
        correctly in `generate()`. We force the model to not stop generation until max-length is reached
        to verify that the cache length is indeed set correctly and we don't run out of index when slicing the cache.
        """
        for model_class in self.all_generative_model_classes:
            # Here, we should ideally not skip any model, and test them all. However, some old models cannot correctly
            # use a static cache because they don't create the causal masks correctly.
            # TODO: cyril -> relax this by adding a `_support_static_cache` attribute
            if not model_class._can_compile_fullgraph:
                self.skipTest(reason="This model does not support the static cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            if config.is_encoder_decoder:
                self.skipTest(reason="This model is encoder-decoder and has Encoder-Decoder Cache")

            model = model_class(config).to(torch_device).eval()
            if "inputs_embeds" not in inspect.signature(model.prepare_inputs_for_generation).parameters:
                self.skipTest(reason="This model does not support `inputs_embeds` in generation")

            input_ids = inputs_dict.pop("input_ids")

            model.config.use_cache = True
            model.config.is_decoder = True
            batch_size = input_ids.shape[0]
            max_new_tokens = 10

            # here we force to not stop at eos and go until max-length
            model.generation_config.eos_token_id = model.config.get_text_config().eos_token_id = -1
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "cache_implementation": "static",
                "return_dict_in_generate": True,  # Required to return `past_key_values`
            }

            text_config = model.config.get_text_config()
            head_dim = (
                getattr(text_config, "head_dim", None) or text_config.hidden_size // text_config.num_attention_heads
            )
            num_key_value_heads = (
                text_config.num_attention_heads
                if getattr(text_config, "num_key_value_heads", None) is None
                else text_config.num_key_value_heads
            )
            num_hidden_layers = text_config.num_hidden_layers

            inputs_embeds = model.get_input_embeddings()(input_ids)
            outputs = model.generate(inputs_embeds=inputs_embeds, **generation_kwargs, **inputs_dict)

            # we should get `max_length - 1` in shape, not `max_length - embeds_length`.
            # -1 because the last generated token isn't yet in the cache.
            max_length = max_new_tokens + inputs_embeds.shape[1] - 1
            cache_shape = [batch_size, num_key_value_heads, max_length, head_dim]
            self.assertIsInstance(outputs.past_key_values, StaticCache)
            self.assertEqual(len(outputs.past_key_values), num_hidden_layers - text_config.num_kv_shared_layers)
            self.assertListEqual(list(outputs.past_key_values.layers[0].keys.shape), cache_shape)

    @pytest.mark.generate
    def test_generate_with_static_cache(self):
        """
        Tests that generating with static cache give almost same results as with dynamic cache, and the output cache
        has the expected shapes
        """
        for model_class in self.all_generative_model_classes:
            # Here, we should ideally not skip any model, and test them all. However, some old models cannot correctly
            # use a static cache because they don't create the causal masks correctly.
            # TODO: cyril -> relax this by adding a `_support_static_cache` attribute
            if not model_class._can_compile_fullgraph:
                self.skipTest(reason="This model does not support the static cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            set_config_for_less_flaky_test(config)
            main_input = inputs_dict[model_class.main_input_name]

            if config.is_encoder_decoder:
                self.skipTest(reason="This model is encoder-decoder and has Encoder-Decoder Cache")

            config.is_decoder = True
            batch_size = main_input.shape[0]
            seq_length = self.model_tester.seq_length
            max_new_tokens = 20

            for dtype in (torch.float32, torch.bfloat16):
                model = model_class(copy.deepcopy(config)).to(torch_device).to(dtype).eval()
                inputs_dict = {
                    k: v.to(dtype) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v
                    for k, v in inputs_dict.items()
                }
                set_model_for_less_flaky_test(model)

                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "return_dict_in_generate": True,  # Required to return `past_key_values`
                    "output_scores": True,
                    "use_cache": True,
                }

                static_cache_generation = model.generate(
                    **generation_kwargs, **inputs_dict, cache_implementation="static"
                )

                # Check 1: The cache shapes must match the expected shapes
                max_cache_len = seq_length + max_new_tokens - 1  # cache len = gen len - 1, the last token has no cache
                text_config = config.text_config if hasattr(config, "text_config") else config
                head_dim = (
                    getattr(text_config, "head_dim", None)
                    or text_config.hidden_size // text_config.num_attention_heads
                )
                num_key_value_heads = (
                    text_config.num_attention_heads
                    if getattr(text_config, "num_key_value_heads", None) is None
                    else text_config.num_key_value_heads
                )
                num_hidden_layers = text_config.num_hidden_layers
                cache_shape = (batch_size, num_key_value_heads, max_cache_len, head_dim)
                self.assertTrue(isinstance(static_cache_generation.past_key_values, StaticCache))
                self.assertTrue(
                    len(static_cache_generation.past_key_values)
                    == num_hidden_layers - text_config.num_kv_shared_layers
                )
                self.assertTrue(static_cache_generation.past_key_values.layers[0].keys.shape == cache_shape)

                # Check 2: The outputs must be similar to the case with dynamic cache
                dynamic_cache_generation = model.generate(**generation_kwargs, **inputs_dict)
                self.assertTrue(has_similar_generate_outputs(dynamic_cache_generation, static_cache_generation))

    def test_model_rope_scaling_frequencies(self):
        """Tests the frequency properties of the different RoPE scaling types on the model RoPE layer."""
        # Gemma3n has different RoPE configs per layer type
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Retrieves the RoPE layer class from the base model class. Uses `.named_modules()` to avoid hardcoding the
        # named location of the RoPE layer class.
        base_model = self.model_tester.base_model_class(config)
        possible_rope_attributes = [
            "pos_emb",
            "rotary_emb",  # most common case
            "global_rotary_emb",
            "local_rotary_emb",
        ]
        for name, module in base_model.named_modules():
            if any(potential_name in name for potential_name in possible_rope_attributes):
                rope_class = type(module)
                break

        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(
            1, dtype=torch.float32, device=torch_device
        )  # used exclusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        rope_params = {"rope_type": "default", "rope_theta": 10_000.0}
        config.rope_parameters = {"sliding_attention": rope_params, "full_attention": rope_params}
        original_rope = rope_class(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short, layer_type="sliding_attention")
        original_cos_long, original_sin_long = original_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        rope_params = {"rope_type": "linear", "factor": scaling_factor, "rope_theta": 10_000.0}
        config.rope_parameters = {"sliding_attention": rope_params, "full_attention": rope_params}
        linear_scaling_rope = rope_class(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        rope_params = {"rope_type": "dynamic", "factor": scaling_factor, "rope_theta": 10_000.0}
        config.rope_parameters = {"sliding_attention": rope_params, "full_attention": rope_params}
        ntk_scaling_rope = rope_class(config=config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue(
            (ntk_scaling_rope.sliding_attention_inv_freq <= original_rope.sliding_attention_inv_freq).all()
        )

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        rope_params = {"rope_type": "yarn", "factor": scaling_factor, "rope_theta": 10_000.0}
        config.rope_parameters = {"sliding_attention": rope_params, "full_attention": rope_params}
        yarn_scaling_rope = rope_class(config=config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(yarn_cos_short, yarn_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(yarn_sin_short, yarn_sin_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_short, original_cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_long, original_sin_long)


class Gemma3nVision2TextModelTester:
    text_config = {"activation_sparsity_pattern": None}
    forced_config_args = ["text_config"]

    def __init__(
        self,
        parent,
        mm_tokens_per_image=2,
        image_token_id=3,
        boi_token_id=4,
        eoi_token_id=5,
        boa_token_id=6,
        eoa_token_id=7,
        audio_token_id=8,
        seq_length=25,
        is_training=True,
        vision_config=None,
        use_cache=False,
        vision_soft_tokens_per_image=4,
        audio_soft_tokens_per_image=4,
    ):
        self.parent = parent
        # `image_token_id` is set to 0 to pass "resize_embeddings" test, do not modify
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_id = image_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_id = eoa_token_id
        self.audio_token_id = audio_token_id
        self.llm_tester = Gemma3nTextModelTester(self.parent)
        self.text_config = self.llm_tester.get_config()
        self.audio_tester = Gemma3nAudioModelTester(self.parent)
        self.audio_config = self.audio_tester.get_audio_encoder_config()
        # NOTE: gemma3n uses mobilenet backbone but timm doens't let us
        # create a tiny MobileNet. So we use a random ViT backbone for testing!
        if vision_config is None:
            vision_config = {
                "architecture": "vit_pe_core_large_patch14_336",
                "use_labels": True,
                "image_size": 20,
                "patch_size": 5,
                "num_channels": 3,
                "is_training": True,
                "hidden_size": 32,
                "num_key_value_heads": 1,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 37,
                "model_args": {
                    "embed_dim": 64,
                    "img_size": (20, 20),
                    "depth": 2,
                    "global_pool": "",
                    "use_post_transformer_norm": False,
                    "init_values": 0.1,
                    "ref_feat_shape": (1, 1),
                },
            }
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.pad_token_id = self.text_config.pad_token_id
        self.vision_soft_tokens_per_image = vision_soft_tokens_per_image
        self.audio_soft_tokens_per_image = audio_soft_tokens_per_image

        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.encoder_seq_length = seq_length
        self.use_cache = use_cache

    def get_config(self):
        return Gemma3nConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            audio_config=self.audio_config,
            image_token_id=self.image_token_id,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
            boa_token_id=self.boa_token_id,
            eoa_token_id=self.eoa_token_id,
            audio_token_id=self.audio_token_id,
            mm_tokens_per_image=self.mm_tokens_per_image,
            vision_soft_tokens_per_image=self.vision_soft_tokens_per_image,
            audio_soft_tokens_per_image=self.audio_soft_tokens_per_image,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # set the 3 first tokens to be image, and ensure that no other tokens are image tokens
        # do not change this unless you modified image size or patch size
        input_ids[input_ids == config.image_token_id] = self.pad_token_id
        input_ids[:, : self.vision_soft_tokens_per_image] = config.image_token_id

        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[input_ids == config.image_token_id] = 1

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        return config, inputs_dict


@require_torch
class Gemma3nVision2TextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Gemma3nModel, Gemma3nForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (Gemma3nForConditionalGeneration,) if is_torch_available() else ()

    test_missing_keys = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]

    # MP works but offload doesn't work when the SigLIP MultiheadAttention is offloaded
    # TODO: One potential solution would be to add to set preload_module_classes = ["SiglipMultiheadAttentionPoolingHead"]
    # in the dispatch_model function
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def setUp(self):
        self.model_tester = Gemma3nVision2TextModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Gemma3nConfig,
            hidden_size=37,
            text_config={"activation_sparsity_pattern": None},
        )

    @unittest.skip(
        reason="Siglip has no FLEX attention, and we don't have a proper way to set/test attn in VLMs. TODO @raushan"
    )
    def test_flex_attention_with_grads(self):
        pass

    @unittest.skip("Gemma3n applies key/query norm which doesn't work with packing")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Gemma3n applies key/query norm which doesn't work with packing")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Cannot set `output_attentions` for timm models.")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Cannot set `output_attentions` for timm models.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("Cannot set `output_attentions` on timm models.")
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip("timm model has no gradient")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip("timm model has no gradient")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skip("timm model has no gradient")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    def _image_features_get_expected_num_hidden_states(self, model_tester=None):
        return 2

    @parameterized.expand([True, False, None])
    @unittest.skip("Audio modality is not tested here")
    def test_get_audio_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("Audio modality is not tested here")
    def test_get_audio_features_hidden_states(self, return_dict: bool | None):
        pass

    @unittest.skip("Audio modality is not tested here")
    def test_get_audio_features_attentions(self, return_dict: bool | None):
        pass

    @pytest.mark.generate
    @unittest.skip("Gemma3n does not support QuantizedCache as it performs cache manipulation in the forward pass")
    def test_generate_with_quant_cache(self):
        pass

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        """
        NOTE: Gemma3n has special hidden states shape with 1 additional dim (which is
        then reduced with projections)
        """

        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (output_length - prompt_length))

        # When `output_hidden_states=True`, each iteration of generate appends the hidden states corresponding to the
        # new token(s)
        for generated_length, iter_hidden_states in enumerate(hidden_states):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length
            expected_shape = (config.altup_num_inputs, batch_size, model_input_length, config.hidden_size)
            # check hidden size
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [expected_shape] * len(iter_hidden_states),
            )

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
    ):
        "We need to relax a bit the `atols` and `rtols` for fp32 here due to the altup projections"
        atols = {
            ("cpu", False, torch.float32): 5e-2,  # this was relaxed
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 5e-2,  # this was relaxed
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 5e-2,  # this was relaxed
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 5e-2,  # this was relaxed
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        rtols = {
            ("cpu", False, torch.float32): 1e-2,  # this was relaxed
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-2,  # this was relaxed
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-2,  # this was relaxed
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-2,  # this was relaxed
            ("cuda", True, torch.bfloat16): 3e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        _test_eager_matches_sdpa_inference(
            self,
            name,
            dtype,
            padding_side,
            use_attention_mask,
            output_attentions,
            enable_kernels,
            atols=atols,
            rtols=rtols,
        )


@slow
@require_torch_accelerator
class Gemma3nIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("Google/gemma-3n-E4B-it", padding_side="left")

        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        audio_ds = load_dataset(
            "etechgrid/28.5k_wavfiles_dataset", "default", data_files="wav_dataset/103-1240-0000.wav"
        )
        self.audio_file_path = audio_ds["train"][0]["audio"].metadata.path
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_4b_bf16(self):
        model_id = "Google/gemma-3n-E4B-it"

        model = Gemma3nForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)
        EXPECTED_TEXTS = Expectations({
            ("cuda", None): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach next to a clear blue ocean. The cow is facing the viewer with its head slightly'],
            ("rocm", (9, 4)): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach next to a turquoise ocean. The sky is blue with a few white clouds. The'],
        }).get_expectation()  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_with_audio(self):
        """
        Tests the full model pipeline with batched audio inputs provided as file paths.
        This ensures the processor correctly loads and processes audio files.
        """

        model_id = "Google/gemma-3n-E4B-it"

        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=torch_device
        )

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe the following speech segment in English:"},
                        {"type": "audio", "audio": str(self.audio_file_path)},
                    ],
                }
            ],
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        ).to(torch_device, dtype=model.dtype)

        input_len = inputs["input_ids"].shape[-1]

        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        output = output[:, input_len:]
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = ["Chapter 1. Mrs. Rachel Lind is surprised.\n\nMrs. Rachel Lind"]
        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_4b_batch(self):
        model_id = "Google/gemma-3n-E4B-it"

        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=torch_device
        )

        messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                    },
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg",
                    },
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            [self.messages, messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(torch_device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)
        EXPECTED_TEXTS = Expectations({
            ("cuda", None): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach next to a clear blue ocean. The cow is facing the viewer with its head slightly', "user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, the images are not identical. \n\nHere's a breakdown of the differences:\n\n* **Subject:** The first image features a cow"],
            ("rocm", (9, 4)): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach next to a clear blue ocean. The cow is facing the viewer with its head slightly', "user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, the images are not identical. \n\nHere's a breakdown of the differences:\n\n* **Subject Matter:** The first image shows a"],
            ("xpu", None): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach next to a turquoise ocean. The cow is facing the viewer with its head slightly turned', "user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, the images are not identical. \n\nHere's a breakdown of the differences:\n\n* **Subject:** The first image features a cow"],
        }).get_expectation()  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_4b_image(self):
        model_id = "Google/gemma-3n-E4B-it"

        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=torch_device
        )

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_NUM_IMAGES = 1  # Gemma3n does not support crops
        EXPECTED_TEXTS = Expectations({
            ("cuda", None): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach next to a clear blue ocean. The cow is facing the viewer with its head slightly'],
            ("xpu", None): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach next to a clear blue ocean. The cow is facing the viewer with its head slightly'],
            ("rocm", (9, 4)): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach next to a turquoise ocean. The sky is blue with a few white clouds. The'],
        }).get_expectation()  # fmt: skip
        self.assertEqual(len(inputs["pixel_values"]), EXPECTED_NUM_IMAGES)
        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_deterministic_for_xpu
    def test_model_4b_multiimage(self):
        model_id = "Google/gemma-3n-E4B-it"

        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=torch_device
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg",
                    },
                    {"type": "text", "text": "What do you see here?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(torch_device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations({
            ("cuda", None): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat do you see here?\nmodel\nIn the image, I see a street scene in what appears to be a Chinatown district. Here are some of the key elements:\n\n* **A'],
            ("xpu", None): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat do you see here?\nmodel\nIn the image, I see a street scene in what appears to be a Chinatown district. Here are the key elements:\n\n* **A prominent red'],
            ("rocm", (9, 4)): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat do you see here?\nmodel\nIn the image, I see a street scene in what appears to be a Chinatown district. \n\nHere are some key elements:\n\n* **A'],
        }).get_expectation()  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)

    @unittest.skip("For now, using a gemma model with the 3n class is not supported")
    def test_model_1b_text_only(self):
        model_id = "google/gemma-3-1b-it"

        model = Gemma3nForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        inputs = tokenizer("Write a poem about Machine Learning.", return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = ['Write a poem about Machine Learning.\n\n---\n\nThe data flows, a river deep,\nWith patterns hidden, secrets sleep.\nA neural net, a watchful eye,\nLearning']  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_generation_beyond_sliding_window(self):
        """Test that we can correctly generate beyond the sliding window. This is non trivial as
        we need to correctly slice the attention mask in all cases (because we use a hybrid cache).
        Outputs for every attention functions should be coherent and identical.
        """
        model_id = "google/gemma-3n-E2B-it"

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", dtype=torch.bfloat16, device_map=torch_device
        )

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.get_text_config().sliding_window)

        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)[:, input_size:]
        output_text = tokenizer.batch_decode(out)

        EXPECTED_COMPLETIONS = [" and the people are so friendly. I'm so glad I came here. I'm so", ", green, yellow, orange, purple, pink, brown, black, white.\n\nHere'"]  # fmt: skip
        self.assertEqual(output_text, EXPECTED_COMPLETIONS)

    @require_deterministic_for_xpu
    def test_generation_beyond_sliding_window_with_generation_config(self):
        """Same as `test_generation_beyond_sliding_window`, but passing a GenerationConfig. Regression test for #36684 --
        ensures `cache_implementation='hybrid'` is correctly inherited from the base `model.generation_config`.
        """

        model_id = "google/gemma-3n-E2B-it"

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=torch_device)

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.get_text_config().sliding_window)

        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)[:, input_size:]
        output_text = tokenizer.batch_decode(out)

        EXPECTED_COMPLETIONS = Expectations({
            # FIXME: This test is VERY flaky on ROCm
            ("cuda", None): [" and I'm glad I came here. This is a nice place. This is a nice place", ", green, yellow, orange, purple, pink, brown, black, white.\n\nHere'"],
            ("rocm", (9, 4)): [' and I think it makes this place special. This is a nice place. This is a nice place', ', green, yellow, purple, orange, pink, brown, black, white.\n\nHere are'],
            ("xpu", None): [" and I think it's a nice place to visit. This is a nice place. This is", ", green, yellow, orange, purple, pink, brown, black, white.\n\nHere'"],
        }).get_expectation()  # fmt: skip
        self.assertEqual(output_text, EXPECTED_COMPLETIONS)
