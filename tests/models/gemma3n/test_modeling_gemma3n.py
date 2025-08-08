# coding=utf-8
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

import tempfile
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
    Gemma3nTextConfig,
    GenerationConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    _test_eager_matches_sdpa_inference,
    floats_tensor,
    ids_tensor,
)
from ..gemma.test_modeling_gemma import GemmaModelTester


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
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    is_generative = False
    _is_stateful = True
    main_input_name = "audio_mel"
    test_initialization = False
    test_can_init_all_missing_weights = False

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

        print(input_features[0, 0, :5])

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

        print(encoder_output[0, 0, :5])

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


class Gemma3nTextModelTester(GemmaModelTester):
    activation_sparsity_pattern = None
    forced_config_args = ["activation_sparsity_pattern"]

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
        self._verify_model_attributes()
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

    if is_torch_available():
        config_class = Gemma3nTextConfig
        model_class = Gemma3nTextModel
        for_causal_lm_class = Gemma3nForCausalLM


@require_torch
class Gemma3nTextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Gemma3nTextModel, Gemma3nForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (Gemma3nForCausalLM,) if is_torch_available() else ()
    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]

    def setUp(self):
        self.model_tester = Gemma3nTextModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Gemma3nConfig,
            hidden_size=37,
            text_config={"activation_sparsity_pattern": None},
        )

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
        # NOTE: `HybridCache` may have different lengths on different layers, if this test starts failing add more
        # elaborate checks
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
    @require_torch_sdpa
    def test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
    ):
        "We need to relax a bit the `atols` for fp32 here due to the altup projections"
        atols = {
            ("cpu", False, torch.float32): 1e-3,  # this was relaxed
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-3,  # this was relaxed
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-3,  # this was relaxed
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-3,  # this was relaxed
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
        _test_eager_matches_sdpa_inference(
            self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels, atols=atols
        )

    @pytest.mark.generate
    @unittest.skip(
        "Gemma3n has a special shape for hidden states (due to per-layer projs) which is not compatible with contrastive decoding"
    )
    def test_contrastive_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(
        "Gemma3n has a special shape for hidden states (due to per-layer projs) which is not compatible with contrastive decoding"
    )
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @unittest.skip(
        "Gemma3n has a special shape for hidden states (due to per-layer projs) which is not compatible with contrastive decoding"
    )
    def test_contrastive_generate_low_memory(self):
        pass

    @pytest.mark.generate
    @unittest.skip(
        "Gemma3n has a special shape for hidden states (due to per-layer projs) which is not compatible with dola decoding"
    )
    def test_dola_decoding_sample(self):
        pass

    @pytest.mark.generate
    @unittest.skip("Gemma3n does not support QuantizedCache as it performs cache manipulation in the forward pass")
    def test_generate_with_quant_cache(self):
        pass


class Gemma3nVision2TextModelTester:
    text_config = {"activation_sparsity_pattern": None}
    forced_config_args = ["text_config"]

    def __init__(
        self,
        parent,
        mm_tokens_per_image=2,
        image_token_index=1,
        boi_token_index=2,
        eoi_token_index=3,
        seq_length=25,
        is_training=True,
        vision_config={
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
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        use_cache=False,
    ):
        self.parent = parent
        # `image_token_index` is set to 0 to pass "resize_embeddings" test, do not modify
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_index = image_token_index
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.llm_tester = Gemma3nTextModelTester(self.parent)
        self.text_config = self.llm_tester.get_config()
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.pad_token_id = self.text_config.pad_token_id

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
            image_token_index=self.image_token_index,
            boi_token_index=self.boi_token_index,
            eoi_token_index=self.eoi_token_index,
            mm_tokens_per_image=self.mm_tokens_per_image,
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
        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[:, :1] = config.image_token_index

        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[input_ids == config.image_token_index] = 1

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        return config, inputs_dict


@unittest.skip("Skipped for now!")
@require_torch
class Gemma3nVision2TextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Gemma3nModel, Gemma3nForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (Gemma3nForConditionalGeneration,) if is_torch_available() else ()
    test_headmasking = False
    test_pruning = False
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

    @unittest.skip(reason="SiglipVisionModel (vision backbone) does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SiglipVisionModel (vision backbone) does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SiglipVisionModel (vision backbone) does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        reason="HybridCache can't be gathered because it is not iterable. Adding a simple iter and dumping `distributed_iterator`"
        " as in Dynamic Cache doesnt work. NOTE: @gante all cache objects would need better compatibility with multi gpu setting"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip("Gemma3n has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Gemma3n has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip("Gemma3n has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Gemma3n has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("Gemma3n has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("Gemma3n has HybridCache and doesn't support low_memory generation")
    def test_beam_search_low_memory(self):
        pass

    @unittest.skip("Gemma3n has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("Gemma3n has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Gemma3n has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("Gemma3n has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("Gemma3n has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip(
        reason="Siglip (vision backbone) uses the same initialization scheme as the Flax original implementation"
    )
    def test_initialization(self):
        pass

    @unittest.skip(
        reason="Siglip has no FLEX attention, and we don't have a proper way to set/test attn in VLMs. TODO @raushan"
    )
    def test_flex_attention_with_grads(self):
        pass

    def test_automodelforcausallm(self):
        """
        Regression test for #36741 -- make sure `AutoModelForCausalLM` works with a Gemma3n config, i.e. that
        `AutoModelForCausalLM.from_pretrained` pulls the text config before loading the model
        """
        config = self.model_tester.get_config()
        model = Gemma3nForConditionalGeneration(config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            for_causal_lm = AutoModelForCausalLM.from_pretrained(tmp_dir)
            self.assertIsInstance(for_causal_lm, Gemma3nForCausalLM)


@unittest.skip("Skipped for now!")
@slow
@require_torch_gpu
@require_read_token
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
        self.audio_file_path = audio_ds["train"][0]["audio"]["path"]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_4b_bf16(self):
        model_id = "Google/gemma-3n-E4B-it"

        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, low_cpu_mem_usage=True, dtype=torch.bfloat16
        ).to(torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown cow standing on a sandy beach with clear blue water and a blue sky in the background. It looks like']  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_with_audio(self):
        """
        Tests the full model pipeline with batched audio inputs provided as file paths.
        This ensures the processor correctly loads and processes audio files.
        """

        model_id = "Google/gemma-3n-E4B-it"

        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, low_cpu_mem_usage=True, dtype=torch.bfloat16
        ).to(torch_device)

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
            model_id, low_cpu_mem_usage=False, dtype=torch.bfloat16
        ).to(torch_device)

        messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                    },
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
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
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = [
            'user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown cow standing on a sandy beach with clear turquoise water and a blue sky in the background. It looks like',
            "user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, these images are not identical. \n\nHere's a breakdown of the differences:\n\n*   **Image 1:** Shows a cow"
        ]  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_4b_crops(self):
        model_id = "Google/gemma-3n-E4B-it"

        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, low_cpu_mem_usage=True, dtype=torch.bfloat16
        ).to(torch_device)

        crop_config = {
            "images_kwargs": {
                "do_pan_and_scan": True,
                "pan_and_scan_max_num_crops": 448,
                "pan_and_scan_min_crop_size": 32,
                "pan_and_scan_min_ratio_to_activate": 0.3,
            }
        }

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **crop_config,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_NUM_IMAGES = 3  # one for the origin image and two crops of images
        EXPECTED_TEXTS = ['user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a beach with a turquoise ocean and blue sky in the background.']  # fmt: skip
        self.assertEqual(len(inputs["pixel_values"]), EXPECTED_NUM_IMAGES)
        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_4b_multiimage(self):
        model_id = "Google/gemma-3n-E4B-it"

        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, low_cpu_mem_usage=True, dtype=torch.bfloat16
        ).to(torch_device)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
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
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = ["user\nYou are a helpful assistant.\n\n\n\n\n\nWhat do you see here?\nmodel\nOkay, let's break down what I see in this image:\n\n**Overall Scene:**\n\nIt looks like a street scene in a vibrant,"]  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_1b_text_only(self):
        model_id = "google/gemma-3-1b-it"

        model = Gemma3nForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, dtype=torch.bfloat16).to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        inputs = tokenizer("Write a poem about Machine Learning.", return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = ['Write a poem about Machine Learning.\n\n---\n\nThe data flows, a river deep,\nWith patterns hidden, secrets sleep.\nA neural net, a watchful eye,\nLearning']  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)

    # TODO: raushan FA2 generates gibberish for no reason, check later
    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    def test_model_4b_flash_attn(self):
        model_id = "Google/gemma-3n-E4B-it"

        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown and white cow standing on a sandy beach next to a turquoise ocean. It looks like a very sunny and']  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)

    @parameterized.expand([("flash_attention_2",), ("sdpa",), ("eager",)])
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. This is non trivial as
        we need to correctly slice the attention mask in all cases (because we use a HybridCache).
        Outputs for every attention functions should be coherent and identical.
        """
        model_id = "google/gemma-3-1b-it"

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=attn_implementation, dtype=torch.float16
        ).to(torch_device)

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.sliding_window)

        out = model.generate(**inputs, max_new_tokens=20)[:, input_size:]
        output_text = tokenizer.batch_decode(out)

        EXPECTED_COMPLETIONS = [" and I'm going to take a walk.\n\nI really enjoy the scenery, and I'", ", green, yellow, orange, purple, brown, black, white, gray.\n\nI'"]  # fmt: skip
        self.assertEqual(output_text, EXPECTED_COMPLETIONS)

    def test_generation_beyond_sliding_window_with_generation_config(self):
        """
        Same as `test_generation_beyond_sliding_window`, but passing a GenerationConfig. Regression test for #36684 --
        ensures `cache_implementation='hybrid'` is correctly inherited from the base `model.generation_config`.
        """
        model_id = "google/gemma-3-1b-it"
        attn_implementation = "sdpa"

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=attn_implementation, dtype=torch.float16
        ).to(torch_device)

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.sliding_window)

        generation_config = GenerationConfig(max_new_tokens=20)

        out = model.generate(**inputs, generation_config=generation_config)[:, input_size:]
        output_text = tokenizer.batch_decode(out)

        EXPECTED_COMPLETIONS = [" and I'm going to take a walk.\n\nI really enjoy the scenery, and I'", ", green, yellow, orange, purple, brown, black, white, gray.\n\nI'"]  # fmt: skip
        self.assertEqual(output_text, EXPECTED_COMPLETIONS)
