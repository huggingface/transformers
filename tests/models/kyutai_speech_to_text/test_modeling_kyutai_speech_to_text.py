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
"""Testing suite for the PyTorch Moshi ASR model."""

import gc
import inspect
import tempfile
import unittest

import datasets
import pytest
from parameterized import parameterized

from transformers import (
    KyutaiSpeechToTextConfig,
    KyutaiSpeechToTextForConditionalGeneration,
    KyutaiSpeechToTextProcessor,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_accelerate,
    require_torch,
    require_torch_accelerator,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin, has_similar_generate_outputs
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        KyutaiSpeechToTextForConditionalGeneration,
        KyutaiSpeechToTextModel,
    )


class KyutaiSpeechToTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        text_seq_length=1,
        input_values_length=192,  # gives 3 audio tokens, corresponding to the default in GenerationTesterMixin
        is_training=False,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        codebook_vocab_size=2049,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=None,
        max_position_embeddings=512,
        rope_theta=10000.0,
        hidden_act="silu",
        head_dim=None,
        initializer_range=0.02,
        use_cache=True,
        sliding_window=512,
        attention_dropout=0.1,
        ffn_dim=38,
        rms_norm_eps=1e-6,
        num_codebooks=8,
        frame_size=64,
        delay_in_tokens=5,
        audio_bos_token_id=2048,
        audio_pad_token_id=2048,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        codec_config={
            "model_type": "mimi",
            "num_quantizers": 8,
            "audio_channels": 1,
            "chunk_in_sec": None,
            "hidden_size": 16,
            "num_filters": 8,
            "num_residual_layers": 1,
            "upsampling_ratios": [8, 4],
            "codebook_size": 16,
            "vector_quantization_hidden_dimension": 16,
            "upsample_groups": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "sliding_window": 4,
            "codebook_dim": 16,
            "use_cache": False,
        },
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.text_seq_length = text_seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.codebook_vocab_size = codebook_vocab_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.head_dim = head_dim
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.ffn_dim = ffn_dim
        self.rms_norm_eps = rms_norm_eps
        self.num_codebooks = num_codebooks
        self.frame_size = frame_size
        self.delay_in_tokens = delay_in_tokens
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.codec_config = codec_config
        self.scope = scope
        self.input_values_length = input_values_length

    def get_config(self):
        return KyutaiSpeechToTextConfig(
            codebook_vocab_size=self.codebook_vocab_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            hidden_act=self.hidden_act,
            head_dim=self.head_dim,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            sliding_window=self.sliding_window,
            attention_dropout=self.attention_dropout,
            ffn_dim=self.ffn_dim,
            rms_norm_eps=self.rms_norm_eps,
            num_codebooks=self.num_codebooks,
            frame_size=self.frame_size,
            delay_in_tokens=self.delay_in_tokens,
            audio_bos_token_id=self.audio_bos_token_id,
            audio_pad_token_id=self.audio_pad_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            codec_config=self.codec_config,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = KyutaiSpeechToTextModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs(self):
        config = self.get_config()

        text_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1) + 1
        codebook_input_ids = (
            ids_tensor([self.batch_size, self.seq_length, self.num_codebooks], self.codebook_vocab_size - 1) + 1
        )

        input_ids = torch.cat([text_input_ids.unsqueeze(2), codebook_input_ids], dim=2)
        attention_mask = text_input_ids.ne(1).to(torch_device)

        return config, input_ids, attention_mask

    def prepare_config_and_inputs_generate(self):
        config = self.get_config()

        input_ids = torch.ones([self.batch_size, 1], dtype=torch.long, device=torch_device)
        input_values = floats_tensor([self.batch_size, 1, self.input_values_length])
        padding_mask = torch.ones_like(input_values, dtype=torch.int32, device=torch_device)

        return config, input_ids, input_values, padding_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common_generate(self):
        config_and_inputs = self.prepare_config_and_inputs_generate()
        (
            config,
            input_ids,
            input_values,
            padding_mask,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "input_values": input_values,
            "padding_mask": padding_mask,
        }
        return config, inputs_dict


@require_torch
class KyutaiSpeechToTextModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            KyutaiSpeechToTextModel,
            KyutaiSpeechToTextForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": KyutaiSpeechToTextModel,
            "automatic-speech-recognition": KyutaiSpeechToTextForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    def setUp(self):
        self.model_tester = KyutaiSpeechToTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=KyutaiSpeechToTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels)

        return inputs_dict

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        # monkey patch prepare_config_and_inputs_for_common

        prepare_config_and_inputs_for_common = self.model_tester.prepare_config_and_inputs_for_common
        original_batch_size = self.model_tester.batch_size

        self.model_tester.prepare_config_and_inputs_for_common = (
            self.model_tester.prepare_config_and_inputs_for_common_generate
        )
        self.model_tester.batch_size = batch_size

        config, filtered_inputs_dict = super().prepare_config_and_inputs_for_generate()
        self.model_tester.prepare_config_and_inputs_for_common = prepare_config_and_inputs_for_common

        self.model_tester.batch_size = original_batch_size
        return config, filtered_inputs_dict

    @pytest.mark.skip(reason="Moshi ASR has custom embedding approach (text and audio embeddings).")
    def test_model_get_set_embeddings(self):
        pass

    @pytest.mark.skip(reason="Moshi ASR has custom embedding approach (text and audio embeddings).")
    def test_tie_model_weights(self):
        pass

    @pytest.mark.skip(reason="Moshi ASR has custom embedding approach (text and audio embeddings).")
    def test_resize_embeddings_untied(self):
        pass

    @pytest.mark.skip(reason="Moshi ASR has custom embedding approach (text and audio embeddings).")
    def test_resize_tokens_embeddings(self):
        pass

    @pytest.mark.skip(reason="Moshi ASR has custom embedding approach (text and audio embeddings).")
    def test_tied_weights_keys(self):
        pass

    @pytest.mark.skip(reason="Does not apply to Moshi ASR that requires input_values.")
    def test_generate_without_input_ids(self):
        pass

    def test_initialization(self):
        """
        Overrides [ModelTesterMixin.test_initialization] because of specificities of Mimi codec model.
        See https://github.com/huggingface/transformers/blob/1077603410cd73ba71d64a522033574d66d64b55/tests/models/mimi/test_modeling_mimi.py#L384-L397
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = ["conv", "input_proj", "output_proj"]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @require_torch_sdpa
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        if use_attention_mask or (not use_attention_mask and dtype == "fp32" and not output_attentions):
            self.skipTest("Test is failing, fix me :) ")
        parent_parameterized_test = getattr(ModelTesterMixin, self._testMethodName)
        parent_parameterized_test(self)

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_safetensors(self):
        pass

    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        # NOTE: left-padding results in small numerical differences. This is expected.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

        # First, filter out models that don't support left padding
        # - The model must have generative capabilities
        if len(self.all_generative_model_classes) == 0:
            self.skipTest(reason="No generative architecture available for this model.")

        # - The model must support padding
        if not self.has_attentions:
            self.skipTest(reason="This model doesn't support padding.")

        # - The model must be a decoder-only architecture (encoder-based architectures use right-padding)
        decoder_only_classes = []
        for model_class in self.all_generative_model_classes:
            config, _ = self.prepare_config_and_inputs_for_generate()
            if config.is_encoder_decoder:
                continue
            else:
                decoder_only_classes.append(model_class)
        if len(decoder_only_classes) == 0:
            self.skipTest(reason="No decoder-only architecture available for this model.")

        # - Decoder-only architectures derived from encoder-decoder models could support it in theory, but we haven't
        #   added support for it yet. We skip these models for now.
        has_encoder_attributes = any(
            attr_name
            for attr_name in config.to_dict()
            if attr_name.startswith("encoder") and attr_name != "encoder_no_repeat_ngram_size"
        )
        if has_encoder_attributes:
            self.skipTest(
                reason="The decoder-only derived from encoder-decoder models are not expected to support left-padding."
            )

        # Then, test left-padding
        def _prepare_model_kwargs(input_ids, attention_mask, signature):
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "position_ids" in signature:
                position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                model_kwargs["position_ids"] = position_ids
            if "cache_position" in signature:
                cache_position = torch.arange(input_ids.shape[1], device=torch_device)
                model_kwargs["cache_position"] = cache_position
            return model_kwargs

        for model_class in decoder_only_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            input_ids = inputs_dict["input_ids"]
            attention_mask = inputs_dict.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # no cache as some models require special cache classes to be init outside forward
            model.generation_config.use_cache = False

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, signature)
            next_logits_wo_padding = model(**model_kwargs).logits[:, -1, :]

            # With left-padding (length 32)
            # can hardcode pad_token to be 0 as we'll do attn masking anyway
            pad_token_id = (
                config.get_text_config().pad_token_id if config.get_text_config().pad_token_id is not None else 0
            )
            pad_size = (input_ids.shape[0], 32, *input_ids.shape[2:])
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat(
                (torch.zeros(pad_size[:2], dtype=input_ids.dtype, device=torch_device), attention_mask), dim=1
            )
            model_kwargs = _prepare_model_kwargs(padded_input_ids, padded_attention_mask, signature)
            next_logits_with_padding = model(**model_kwargs).logits[:, -1, :]

            # They should result in very similar logits
            torch.testing.assert_close(next_logits_wo_padding, next_logits_with_padding, rtol=1e-5, atol=1e-5)

    def test_generate_continue_from_past_key_values(self):
        # Tests that we can continue generating from past key values, returned from a previous `generate` call
        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["imagegpt", "mllama"]):
                self.skipTest(reason="Won't fix: old model with unique inputs/caches/other")
            if any(model_name in model_class.__name__.lower() for model_name in ["umt5"]):
                self.skipTest(reason="TODO: needs modeling or test input preparation fixes for compatibility")

            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            # Let's make it always:
            # 1. use cache (for obvious reasons)
            # 2. generate to max length (which can be achieved by setting the eos token to an invalid value), which
            #    would make the test flaky (e.g. EOS is generated on iteration 1 on both generations, but the
            #    continuation would force it to generate beyond an EOS token)
            # 3. ignore `token_type_ids` for simplicity
            # 4. ignore `forced_eos_token_id`, which requires further manipulation of the continuation inputs and is
            #    active by default on some models
            # 5. ignore `encoder_no_repeat_ngram_size`, which is set by default in some encoder-decoder models. When
            #    we use their decoder as a stand-alone model, `encoder_no_repeat_ngram_size` actually prevents
            #    repetition exclusively from the prompt. This test relies on comparing one call vs 2 calls
            #    with cache, what is considered a prompt is different in the two cases.

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            model = model_class(config).to(torch_device)
            model.eval()

            # If "past_key_values" is not returned, skip the test (e.g. RWKV uses a different cache name and format)
            outputs = model(**inputs)
            if "past_key_values" not in outputs:
                self.skipTest(reason="This model doesn't return `past_key_values`")

            generate_kwargs = {
                "pad_token_id": -1,
                "eos_token_id": -1,
                "forced_eos_token_id": None,
                "encoder_no_repeat_ngram_size": 0,
                "use_cache": True,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            # Traditional way of generating text, with `return_dict_in_generate` to return the past key values
            _, inputs = self.prepare_config_and_inputs_for_generate()
            outputs = model.generate(**inputs, **generate_kwargs, max_new_tokens=3)

            # Let's generate again, but passing the past key values in between (2 + 1 = 3 tokens). Note that the
            # inputs may need to be tweaked across `generate` calls (like the attention mask).
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=2)

            # Continue from the tokens generated above, preparing the inputs accordingly
            inputs["past_key_values"] = outputs_cached.past_key_values
            new_attention_len = outputs_cached.sequences.shape[-1]
            if config.is_encoder_decoder:
                inputs["decoder_input_ids"] = outputs_cached.sequences
                if "decoder_attention_mask" in inputs:
                    inputs["decoder_attention_mask"] = torch.nn.functional.pad(
                        inputs["decoder_attention_mask"],
                        (0, new_attention_len - inputs["decoder_attention_mask"].shape[1]),
                        mode="constant",
                        value=1,
                    )
            else:
                inputs["input_ids"] = outputs_cached.sequences
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.nn.functional.pad(
                        inputs["attention_mask"],
                        (0, new_attention_len - inputs["attention_mask"].shape[1]),
                        mode="constant",
                        value=1,
                    )
            first_caches_scores = outputs_cached.scores
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=1)
            full_cached_scores = first_caches_scores + outputs_cached.scores
            outputs_cached.scores = full_cached_scores

            # The two sets of generated text and past kv should be equal to each other
            self.assertTrue(has_similar_generate_outputs(outputs, outputs_cached))
            for layer_idx in range(len(outputs_cached.past_key_values)):
                for kv_idx in range(len(outputs_cached.past_key_values[layer_idx])):
                    self.assertTrue(
                        torch.allclose(
                            outputs.past_key_values[layer_idx][kv_idx],
                            outputs_cached.past_key_values[layer_idx][kv_idx],
                        )
                    )

    # needs to be overridden to avoid to avoid casting of input_values to float16
    # indeed, the codec model is kept in fp32, so we need to avoid casting input_values to float16
    def _test_attention_implementation(self, attn_implementation):
        """
        Compares the output of generate with the eager attention implementation against other implementations.
        NOTE: despite the test logic being the same, different implementations actually need different decorators, hence
        this separate function.
        """
        max_new_tokens = 30
        support_flag = {
            "sdpa": "_supports_sdpa",
            "flash_attention_2": "_supports_flash_attn",
        }

        for model_class in self.all_generative_model_classes:
            if attn_implementation != "eager" and not getattr(model_class, support_flag[attn_implementation]):
                self.skipTest(f"{model_class.__name__} does not support `attn_implementation={attn_implementation}`")

            config, original_inputs_dict = self.prepare_config_and_inputs_for_generate()
            inputs_dict = {}
            for input_name, input_data in original_inputs_dict.items():
                if (
                    isinstance(input_data, torch.Tensor)
                    and input_data.dtype in [torch.float32, torch.bfloat16]
                    and input_name != "input_values"
                ):
                    inputs_dict[input_name] = input_data.to(torch.float16)
                else:
                    inputs_dict[input_name] = input_data
            main_input = inputs_dict[model_class.main_input_name]

            # FA2 doesn't accept masking in the middle of the sequence for now. We usually generate right-padded
            # attention masks at test time and, with generate, the mask will be appended with 1s on the right,
            # resulting in a mask with holes (not supported properly by FA2).
            if attn_implementation == "flash_attention_2":
                for input_name in ("attention_mask", "decoder_attention_mask", "encoder_attention_mask"):
                    if input_name in inputs_dict:
                        inputs_dict[input_name] = torch.ones_like(inputs_dict[input_name])

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + main_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                del model
                gc.collect()

                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "use_cache": True,
                }

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    dtype=torch.float16,
                    attn_implementation="eager",
                ).to(torch_device)
                res_eager = model_eager.generate(**inputs_dict, **generate_kwargs)
                del model_eager
                gc.collect()

                model_attn = model_class.from_pretrained(
                    tmpdirname,
                    dtype=torch.float16,
                    attn_implementation=attn_implementation,
                ).to(torch_device)
                res_attn = model_attn.generate(**inputs_dict, **generate_kwargs)
                del model_attn
                gc.collect()

                self.assertTrue(has_similar_generate_outputs(res_eager, res_attn, atol=1e-3, rtol=1e-3))


@require_torch
@require_accelerate
@slow
class KyutaiSpeechToTextBf16Test(unittest.TestCase):
    def test_bf16_fp32_conversion(self):
        r"""
        A test to check whether the argument `keep_in_fp32_modules` correctly does its job
        """
        model_checkpoint = "kyutai/stt-2.6b-en-trfs"
        orig_import = __import__
        accelerate_mock = unittest.mock.Mock()

        # mock import of accelerate
        def import_accelerate_mock(name, *args, **kwargs):
            if name == "accelerate":
                if accelerate_available:
                    return accelerate_mock
                else:
                    raise ImportError
            return orig_import(name, *args, **kwargs)

        # Load without using `accelerate`
        with unittest.mock.patch("builtins.__import__", side_effect=import_accelerate_mock):
            accelerate_available = False

            model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_checkpoint, dtype=torch.float16)
            self.assertTrue(model.codec_model.dtype == torch.float32)
            self.assertTrue(model.model.dtype == torch.float16)
            self.assertTrue(model.lm_head.weight.data.dtype == torch.float16)

            # Load without in bf16
            model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_checkpoint, dtype=torch.bfloat16)
            self.assertTrue(model.codec_model.dtype == torch.float32)
            self.assertTrue(model.model.dtype == torch.bfloat16)
            self.assertTrue(model.lm_head.weight.data.dtype == torch.bfloat16)

        # Load using `accelerate` in bf16
        model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            model_checkpoint, dtype=torch.bfloat16, device_map="auto"
        )
        self.assertTrue(model.codec_model.dtype == torch.float32)
        self.assertTrue(model.model.dtype == torch.bfloat16)
        self.assertTrue(model.lm_head.weight.data.dtype == torch.bfloat16)

        # Load using `accelerate` in bf16
        model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            model_checkpoint,
            dtype=torch.bfloat16,
        )
        self.assertTrue(model.codec_model.dtype == torch.float32)
        self.assertTrue(model.model.dtype == torch.bfloat16)
        self.assertTrue(model.lm_head.weight.data.dtype == torch.bfloat16)

        # Load without using `accelerate`
        model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            model_checkpoint,
            dtype=torch.float16,
        )
        self.assertTrue(model.codec_model.dtype == torch.float32)
        self.assertTrue(model.model.dtype == torch.float16)
        self.assertTrue(model.lm_head.weight.data.dtype == torch.float16)

        # Load using `accelerate`
        model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            model_checkpoint, dtype=torch.float16, device_map="auto"
        )
        self.assertTrue(model.codec_model.dtype == torch.float32)
        self.assertTrue(model.model.dtype == torch.float16)
        self.assertTrue(model.lm_head.weight.data.dtype == torch.float16)


class KyutaiSpeechToTextForConditionalGenerationIntegrationTests(unittest.TestCase):
    _dataset = None

    def setUp(self):
        self.model_checkpoint = "kyutai/stt-2.6b-en-trfs"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
        # Lazy loading of the dataset. Because it is a class method, it will only be loaded once per pytest process.
        if cls._dataset is None:
            cls._dataset = datasets.load_dataset(
                "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
            )
            # using 24000 here for simplicity, should rather be processor.feature_extractor.sampling_rate
            cls._dataset = cls._dataset.cast_column("audio", datasets.Audio(sampling_rate=24000))

    def _load_datasamples(self, num_samples):
        self._load_dataset()
        ds = self._dataset
        speech_samples = ds.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    @slow
    @require_torch_accelerator
    def test_generation(self):
        """
        reproduce test expected outputs using original codebase: https://gist.github.com/eustlb/7a9aa6139d11e0103c6b65bac103da52

        DISCLAIMER: we are testing for pretty short inputs. Indeed, reproducing correct expected outputs for longer is not possible
        as implementation choices (qkv matrix in one linear for original code vs three for hf) create growing divergence with context lenght,
        ultimately giving different outputs.
        """
        processor = KyutaiSpeechToTextProcessor.from_pretrained(self.model_checkpoint)
        model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            self.model_checkpoint, device_map=torch_device
        )

        samples = self._load_datasamples(1)
        inputs = processor(
            samples,
        ).to(torch_device)

        out = model.generate(**inputs)

        # fmt: off
        EXPECTED_TOKENS = torch.tensor([
            [48000, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 1519, 263, 3, 3, 0, 3635, 428, 641, 0, 277, 3, 0, 265, 0, 267, 1162, 261, 274, 410, 0,  272, 3, 0, 265, 0, 260, 1621, 0, 1174, 371, 262, 3, 3, 3, 0, 269, 0, 281, 0, 304, 0, 2433, 3, 0, 266, 3, 0, 281, 1661, 3, 0, 376, 3, 3, 0, 350, 261, 401, 516, 263, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]],
        )
        # fmt: on

        torch.testing.assert_close(out.cpu(), EXPECTED_TOKENS)

    @slow
    @require_torch_accelerator
    def test_generation_batched(self):
        """
        reproduce test expected outputs using original codebase: https://gist.github.com/eustlb/b58c217c75124d405ec1c13877c7ece8

        DISCLAIMER: we are testing for pretty short inputs. Indeed, reproducing correct expected outputs for longer is not possible
        as implementation choices (qkv matrix in one linear for original code vs three for hf) create growing divergence with context lenght,
        ultimately giving different outputs.
        """
        processor = KyutaiSpeechToTextProcessor.from_pretrained(self.model_checkpoint)
        model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            self.model_checkpoint, device_map=torch_device
        )

        samples = self._load_datasamples(4)
        inputs = processor(
            samples,
        ).to(torch_device)

        out = model.generate(**inputs)

        # fmt: off
        EXPECTED_TOKENS = torch.tensor([
            [48000, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 1519, 263, 3, 3, 0, 3635, 428, 641, 0, 277, 3, 0, 265, 0, 267, 1162, 261, 274, 410, 0, 272, 3, 0, 265, 0, 260, 1621, 0, 1174, 371, 262, 3, 3, 3, 0, 269, 0, 281, 0, 304, 0, 2433, 3, 0, 266, 3, 0, 281, 1661, 3, 0, 376, 3, 3, 0, 350, 261, 401, 516, 263, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [48000, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 500, 334, 0, 277, 3, 0, 1519, 263, 3, 3, 0, 3635, 428, 641, 264, 261, 0, 511, 1109, 3, 0, 1138, 3, 3, 3, 0, 508, 827, 3, 3, 3, 3, 0, 468, 3, 3, 0, 376, 3, 3, 3, 0, 260, 978, 263, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [48000, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 414, 0, 527, 261, 3, 0, 409, 3, 3, 3, 0, 271, 3, 0, 309, 3, 0, 285, 3, 0, 521, 371, 609, 3, 3, 0, 260, 959, 3, 3, 3, 0, 272, 3, 0, 265, 0, 546, 262, 3, 3, 3, 3, 3, 3, 0, 291, 3, 0, 975, 2203, 3, 3, 3, 3, 0, 269, 3, 0, 260, 489, 651, 274, 279, 1870, 3, 0, 1084, 873, 273, 3, 0, 260, 531, 3, 3, 0, 409, 262, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 1502, 1005, 836, 3, 3, 0, 1666, 306, 3, 0, 340, 3, 0, 260, 3232, 3, 0, 269, 3, 3, 0, 275, 261, 0, 260, 1379, 261, 0, 3324, 3, 3, 3, 3, 0, 549, 3, 3, 0, 693, 405, 323, 3, 0, 266, 3, 3, 0, 265, 0, 699, 263, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [48000, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 414, 0, 392, 3, 3, 0, 1269, 314, 0, 2607, 261, 3, 3, 3, 0, 1098, 295, 3, 3, 3, 0, 446, 625, 3, 0, 496, 280, 1205, 485, 1071, 1627, 449, 264, 261, 3, 0, 400, 0, 277, 3, 3, 3, 0, 260, 342, 3, 0, 618, 280, 1866, 3, 3, 0, 554, 3, 3, 3, 3, 0, 317, 262, 3, 3, 3, 3, 3, 3, 3, 3, 0, 269, 0, 303, 3, 0, 573, 2615, 3, 3, 0, 276, 3, 0, 275, 0, 305, 3, 0, 260, 415, 3, 3, 0, 272, 3, 3, 3, 3, 0, 1631, 327, 3, 3, 0, 333, 739, 841, 263, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        ])
        # fmt: on

        # See https://github.com/huggingface/transformers/pull/39416
        EXPECTED_TOKENS_2 = torch.clone(EXPECTED_TOKENS)
        EXPECTED_TOKENS_2[2, 159:162] = torch.tensor([3, 0, 269])

        try:
            torch.testing.assert_close(out.cpu(), EXPECTED_TOKENS)
        except AssertionError:
            torch.testing.assert_close(out.cpu(), EXPECTED_TOKENS_2)
