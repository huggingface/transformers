# coding=utf-8
# Copyright 2025 The TeleChat team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch TeleChat2 model."""

import gc
import unittest

import pytest
from packaging import version
from parameterized import parameterized

from transformers import AutoTokenizer, TeleChat2Config, is_torch_available, set_seed
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    backend_empty_cache,
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...models.llama.test_modeling_llama import LlamaModelTest, LlamaModelTester
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        TeleChat2ForCausalLM,
        TeleChat2ForQuestionAnswering,
        TeleChat2ForSequenceClassification,
        TeleChat2ForTokenClassification,
        TeleChat2Model,
    )


class TeleChat2ModelTester(LlamaModelTester):
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        max_window_layers=3,
        use_sliding_window=True,
        sliding_window=50,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        bos_token_id=1,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_window_layers = max_window_layers
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.scope = scope

    def get_config(self):
        return TeleChat2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            max_window_layers=self.max_window_layers,
            use_sliding_window=self.use_sliding_window,
            sliding_window=self.sliding_window,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TeleChat2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = TeleChat2Model(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = TeleChat2ForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))


@require_torch
class TeleChat2ModelTest(LlamaModelTest, unittest.TestCase):
    all_model_classes = (
        (
            TeleChat2Model,
            TeleChat2ForCausalLM,
            TeleChat2ForSequenceClassification,
            TeleChat2ForTokenClassification,
            TeleChat2ForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (TeleChat2ForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": TeleChat2Model,
            "text-classification": TeleChat2ForSequenceClassification,
            "token-classification": TeleChat2ForTokenClassification,
            "text-generation": TeleChat2ForCausalLM,
            "zero-shot": TeleChat2ForSequenceClassification,
            "question-answering": TeleChat2ForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = TeleChat2ForCausalLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = TeleChat2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TeleChat2Config, hidden_size=37)

    def test_llama_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = TeleChat2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_llama_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = TeleChat2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_llama_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = TeleChat2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_llama_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
        model = TeleChat2ForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
        )

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = TeleChat2Model(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = TeleChat2Model(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            torch.testing.assert_close(original_short_output, scaled_short_output, rtol=1e-5, atol=1e-5)
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    def test_model_loading_old_rope_configs(self):
        def _reinitialize_config(base_config, new_kwargs):
            # Reinitialize the config with the new kwargs, forcing the config to go through its __init__ validation
            # steps.
            base_config_dict = base_config.to_dict()
            new_config = TeleChat2Config.from_dict(config_dict={**base_config_dict, **new_kwargs})
            return new_config

        # from untouched config -> ✅
        base_config, model_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        original_model = TeleChat2ForCausalLM(base_config).to(torch_device)
        original_model(**model_inputs)

        # from a config with the expected rope configuration -> ✅
        config = _reinitialize_config(base_config, {"rope_scaling": {"rope_type": "linear", "factor": 10.0}})
        original_model = TeleChat2ForCausalLM(config).to(torch_device)
        original_model(**model_inputs)

        # from a config with the old rope configuration ('type' instead of 'rope_type')  -> ✅ we gracefully handle BC
        config = _reinitialize_config(base_config, {"rope_scaling": {"type": "linear", "factor": 10.0}})
        original_model = TeleChat2ForCausalLM(config).to(torch_device)
        original_model(**model_inputs)

        # from a config with both 'type' and 'rope_type'  -> ✅ they can coexist (and both are present in the config)
        config = _reinitialize_config(
            base_config, {"rope_scaling": {"type": "linear", "rope_type": "linear", "factor": 10.0}}
        )
        self.assertTrue(config.rope_scaling["type"] == "linear")
        self.assertTrue(config.rope_scaling["rope_type"] == "linear")
        original_model = TeleChat2ForCausalLM(config).to(torch_device)
        original_model(**model_inputs)

        # from a config with parameters in a bad range ('factor' should be >= 1.0) -> ⚠️ throws a warning
        with self.assertLogs("transformers.modeling_rope_utils", level="WARNING") as logs:
            config = _reinitialize_config(base_config, {"rope_scaling": {"rope_type": "linear", "factor": -999.0}})
            original_model = TeleChat2ForCausalLM(config).to(torch_device)
            original_model(**model_inputs)
            self.assertEqual(len(logs.output), 1)
            self.assertIn("factor field", logs.output[0])

        # from a config with unknown parameters ('foo' isn't a rope option) -> ⚠️ throws a warning
        with self.assertLogs("transformers.modeling_rope_utils", level="WARNING") as logs:
            config = _reinitialize_config(
                base_config, {"rope_scaling": {"rope_type": "linear", "factor": 10.0, "foo": "bar"}}
            )
            original_model = TeleChat2ForCausalLM(config).to(torch_device)
            original_model(**model_inputs)
            self.assertEqual(len(logs.output), 1)
            self.assertIn("Unrecognized keys", logs.output[0])

        # from a config with specific rope type but missing one of its mandatory parameters -> ❌ throws exception
        with self.assertRaises(KeyError):
            config = _reinitialize_config(base_config, {"rope_scaling": {"rope_type": "linear"}})  # missing "factor"


@require_torch
class TeleChat2IntegrationTest(unittest.TestCase):
    @slow
    def test_model_450m_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = TeleChat2ForCausalLM.from_pretrained("TeleAI/TeleChat2-3B", device_map="auto")
        input_ids = torch.tensor([input_ids]).to(model.model.word_embeddings.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-1.9537, -1.6193, -1.4123, -1.4673, -1.8511, -1.9309, -1.9826, -2.1776]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([3.2025, 7.1265, 4.6058, 3.6423, 1.6357, 3.9265, 5.1883, 5.8760, 2.7942, 4.4823, 3.2571, 2.1063, 3.4275, 4.2028, 1.9767, 5.2115, 6.6756, 6.3999, 6.0483, 5.7378, 5.6660, 5.2298, 5.4103, 5.1248, 5.4376, 2.4570, 2.6107, 5.4039, 2.8077, 4.7777])  # fmt: skip
        print(out[0, 0, :30])
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-4, rtol=1e-4)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_model_450m_generation(self):
        EXPECTED_TEXT_COMPLETION = (
            """My favourite condiment is 100% natural, organic and vegan. I love to use it in my cooking and I"""
        )
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("TeleAI/TeleChat2-3B", use_fast=False)
        model = TeleChat2ForCausalLM.from_pretrained("TeleAI/TeleChat2-3B", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.word_embeddings.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @require_bitsandbytes
    @slow
    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_model_450m_long_prompt(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = TeleChat2ForCausalLM.from_pretrained(
            "TeleAI/TeleChat2-3B",
            device_map="auto",
            load_in_4bit=True,
            attn_implementation="flash_attention_2",
        )
        input_ids = torch.tensor([input_ids]).to(model.model.word_embeddings.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        # Assisted generation
        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 2
        assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        del assistant_model
        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    @require_torch_sdpa
    def test_model_450m_long_prompt_sdpa(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = TeleChat2ForCausalLM.from_pretrained(
            "TeleAI/TeleChat2-3B", device_map="auto", attn_implementation="sdpa"
        )
        input_ids = torch.tensor([input_ids]).to(model.model.word_embeddings.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        # Assisted generation
        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 2
        assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
        generated_ids = assistant_model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        del assistant_model

        backend_empty_cache(torch_device)
        gc.collect()

        EXPECTED_TEXT_COMPLETION = (
            """My favourite condiment is 100% natural, organic and vegan. I love to use it in my cooking and I"""
        )
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("TeleAI/TeleChat2-3B", use_fast=False)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.word_embeddings.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    def test_speculative_generation(self):
        EXPECTED_TEXT_COMPLETION = (
            "My favourite condiment is 100% natural honey, and I always like to use it in my recipes. I love"
        )
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("TeleAIt/TeleChat2-7B", use_fast=False)
        model = TeleChat2ForCausalLM.from_pretrained(
            "TeleAI/TeleChat2-3B", device_map="auto", torch_dtype=torch.float16
        )
        assistant_model = TeleChat2ForCausalLM.from_pretrained(
            "TeleChat/TeleChat2-3B", device_map="auto", torch_dtype=torch.float16
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.word_embeddings.weight.device)

        # greedy generation outputs
        set_seed(0)
        generated_ids = model.generate(
            input_ids, max_new_tokens=20, do_sample=True, temperature=0.3, assistant_model=assistant_model
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )

        telechat_model = "TeleAI/TeleChat2-3B"

        tokenizer = AutoTokenizer.from_pretrained(telechat_model, pad_token="<_pad>", padding_side="right")
        EXPECTED_TEXT_COMPLETION = [
            "My favourite condiment is 100% natural, organic, gluten free, vegan, and free from preservatives. I"
        ]
        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = TeleChat2ForCausalLM.from_pretrained(
            telechat_model,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation=cache_implementation,
                max_length=max_generation_length,
                cache_config={
                    "batch_size": batch_size,
                    "max_cache_len": max_generation_length,
                },
            ),
        )

        prompt = ["My favourite condiment is "]
        prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + export
        exported_program = convert_and_export_with_cache(model)
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)
