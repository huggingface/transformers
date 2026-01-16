# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Zamba model."""

import tempfile
import unittest

import pytest
from parameterized import parameterized

from transformers import AutoTokenizer, BitsAndBytesConfig, Zamba2Config, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Zamba2ForCausalLM,
        Zamba2ForSequenceClassification,
        Zamba2Model,
    )
    from transformers.models.zamba2.modeling_zamba2 import (
        Zamba2HybridDynamicCache,
    )


class Zamba2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=16,
        mamba_d_state=2,
        chunk_size=8,
        mamba_dt_rank="auto",
        num_hidden_layers=2,
        num_attention_heads=2,
        n_mamba_heads=8,
        mamba_ngroups=8,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_mamba_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
        layers_block_type=["mamba", "hybrid"],
        num_mem_blocks=1,
        use_mem_rope=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mamba_dt_rank = mamba_dt_rank
        self.mamba_d_state = mamba_d_state
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_mamba_heads = n_mamba_heads
        self.mamba_ngroups = mamba_ngroups
        self.chunk_size = chunk_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_mamba_act = hidden_mamba_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.layers_block_type = layers_block_type
        self.num_mem_blocks = num_mem_blocks
        self.use_mem_rope = use_mem_rope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return Zamba2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            mamba_dt_rank=self.mamba_dt_rank,
            mamba_d_state=self.mamba_d_state,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            n_mamba_heads=self.n_mamba_heads,
            intermediate_size=self.intermediate_size,
            chunk_size=self.chunk_size,
            hidden_act=self.hidden_act,
            mamba_ngroups=self.mamba_ngroups,
            hidden_mamba_act=self.hidden_mamba_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=True,
            initializer_range=self.initializer_range,
            use_mamba_kernels=False,
            layers_block_type=self.layers_block_type,
            num_mem_blocks=self.num_mem_blocks,
            use_mem_rope=self.use_mem_rope,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = Zamba2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = Zamba2ForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids, labels=token_labels)
        result = model(input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.is_decoder = True
        config.add_cross_attention = False
        model = Zamba2ForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        # Attention: Zamba2 needs the cache to be initialized to return a cache!
        past_key_values = Zamba2HybridDynamicCache(config, input_ids.shape[0], model.dtype, device=model.device)
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 1), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            cache_position=torch.arange(
                input_ids.shape[1], input_ids.shape[1] + next_tokens.shape[1], device=model.device
            ),
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = Zamba2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class Zamba2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Zamba2Model,
            Zamba2ForCausalLM,
            Zamba2ForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "text-embedding": Zamba2Model,
            "text-classification": Zamba2ForSequenceClassification,
            "text-generation": Zamba2ForCausalLM,
            "zero-shot": Zamba2ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        self.assertIsInstance(past_key_values, Zamba2HybridDynamicCache)

        # (batch, kv heads, seq_length, head_dim)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        attention_shape = (batch_size, num_heads, seq_length, head_dim)

        intermediate_size = config.mamba_expand * config.hidden_size
        conv_shape = (
            batch_size,
            intermediate_size + 2 * config.mamba_ngroups * config.mamba_d_state,
            config.mamba_d_conv,
        )
        ssm_shape = (batch_size, config.n_mamba_heads, config.mamba_headdim, config.mamba_d_state)

        self.assertTrue(config.num_hidden_layers, len(past_key_values))

        for idx in range(len(past_key_values)):
            if config.layers_block_type[idx] == "mamba":
                self.assertEqual(past_key_values.conv_states[idx].shape, conv_shape)
                self.assertEqual(past_key_values.ssm_states[idx].shape, ssm_shape)
            else:
                self.assertEqual(past_key_values.key_cache[idx].shape, attention_shape)
                self.assertEqual(past_key_values.value_cache[idx].shape, attention_shape)

    def _check_caches_are_equal(self, cache1: Zamba2HybridDynamicCache, cache2: Zamba2HybridDynamicCache):
        if not isinstance(cache1, Zamba2HybridDynamicCache) or not isinstance(cache2, Zamba2HybridDynamicCache):
            raise ValueError("The wrong cache is being used!")

        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            torch.testing.assert_close(cache1.key_cache[idx], cache2.key_cache[idx])
            torch.testing.assert_close(cache1.value_cache[idx], cache2.value_cache[idx])
            torch.testing.assert_close(cache1.conv_states[idx], cache2.conv_states[idx])
            torch.testing.assert_close(cache1.ssm_states[idx], cache2.ssm_states[idx])

    def setUp(self):
        self.model_tester = Zamba2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Zamba2Config, hidden_size=37)

    @unittest.skip("position_ids cannot be used to pad due to Mamba2 layers")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(reason="Zamba2 has hybrid cache.")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip(reason="A large mamba2 would be necessary (and costly) for that")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_attention_outputs(self):
        r"""
        Overriding the test_attention_outputs test as the Zamba2 model outputs attention only for its attention layers
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    def _get_input_ids_and_config(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        return config, input_ids, input_mask

    @require_flash_attn
    @require_torch_accelerator
    @require_bitsandbytes
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_fp32_ln(self):
        r"""
        Overriding the test_flash_attn_2_fp32_ln test as the Zamba2 model, like Mixtral, doesn't support
        right padding + use cache with FA2
        """
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_input = inputs_dict[model.main_input_name]
                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
                # NOTE: Zamba2 does not support right padding + use_cache with FA2.
                dummy_attention_mask[:, -1] = 1

                model = model_class.from_pretrained(
                    tmpdirname,
                    dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                )

                for _, param in model.named_parameters():
                    # upcast only layer norms
                    if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                        param.data = param.data.to(torch.float32)

                _ = model(dummy_input)
                # with attention mask
                _ = model(dummy_input, attention_mask=dummy_attention_mask)

    @unittest.skip(reason="Zamba2 has its own special cache type")
    @parameterized.expand([(1, False), (1, True), (4, False)])
    def test_new_cache_format(self, num_beams, do_sample):
        pass

    @require_torch_accelerator
    def test_flex_attention_with_grads(self):
        """
        Overwriting as the base hidden size is big enough for compile.
        Manipulation of dims causes issues due to other constraints not being satisfied anymore.
        """
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config._attn_implementation = "flex_attention"

            model = model_class(config).to(device=torch_device)
            self.assertTrue(model.config._attn_implementation == "flex_attention")

            # Elaborate workaround for encoder-decoder models as some do not specify their main input
            dummy_inputs = {model.main_input_name: inputs_dict[model.main_input_name].to(torch_device)}
            if config.is_encoder_decoder:
                dummy_inputs["decoder_input_ids"] = inputs_dict["decoder_input_ids"].to(torch_device)
                dummy_inputs["decoder_attention_mask"] = inputs_dict["decoder_attention_mask"].to(torch_device)

            # If this does not raise an error, the test passes (see https://github.com/huggingface/transformers/pull/35605)
            _ = model(**dummy_inputs)


@require_torch
class Zamba2ModelIntegrationTest(unittest.TestCase):
    model = None
    tokenizer = None

    @classmethod
    @slow
    def setUpClass(cls):
        model_id = "Zyphra/Zamba2-1.2B"
        cls.model = Zamba2ForCausalLM.from_pretrained(model_id, dtype=torch.float32, revision="PR")
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id, revision="PR")

    @parameterized.expand([(torch_device,), ("cpu",)])
    @slow
    def test_simple_generate(self, torch_device):
        self.model.to(torch_device)

        input_ids = self.tokenizer("Hey how are you doing on this lovely evening?", return_tensors="pt")[
            "input_ids"
        ].to(torch_device)
        out = self.model.generate(input_ids, do_sample=False, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(out[0, :])
        self.assertEqual(
            output_sentence,
            "<s> Hey how are you doing on this lovely evening?\n\nI'm doing well, thanks for",
        )

        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits.to(dtype=torch.float32)

        EXPECTED_LOGITS_NO_GRAD = torch.tensor(
            [
               -5.9587, 10.5152,  7.0382, -2.8728, -4.8143, -4.8142, -4.8142, -4.8144,
               -4.8143, -4.8143, -4.8142, -4.8142,  6.0185, 18.0037, -4.8142, -4.8144,
               -4.8143, -4.8142, -4.8143, -4.8143, -4.8143, -4.8143, -4.8142, -4.8143,
               -4.8144, -4.8143, -4.8143, -4.8141, -4.8142, -4.8142, -4.8142, -4.8144,
               -4.8143, -4.8143, -4.8143, -4.8142, -4.8144, -4.8144, -4.8142, -4.8142
            ]
            , dtype=torch.float32)  # fmt: skip
        torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD, rtol=1e-3, atol=1e-3)

    @parameterized.expand([(torch_device,), ("cpu",)])
    @slow
    def test_simple_batched_generate_with_padding(self, torch_device):
        self.model.to(torch_device)

        inputs = self.tokenizer(
            ["Hey how are you doing on this lovely evening?", "When did the Roman empire "],
            padding=True,
            return_tensors="pt",
        ).to(torch_device)
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=10)
        output_sentences = self.tokenizer.batch_decode(out)
        self.assertEqual(
            output_sentences[0],
            "<s> Hey how are you doing on this lovely evening?\n\nI'm doing well, thanks for",
        )

        self.assertEqual(
            output_sentences[1],
            "[PAD][PAD][PAD][PAD]<s> When did the Roman empire 1st fall?\nThe Roman Empire fell in",
        )

        with torch.no_grad():
            logits = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits.to(
                dtype=torch.float32
            )

        EXPECTED_LOGITS_NO_GRAD_0 = torch.tensor(
            [
                -5.9611, 10.5208,  7.0411, -2.8743, -4.8167, -4.8167, -4.8167, -4.8168,
                -4.8167, -4.8167, -4.8167, -4.8166,  6.0218, 18.0062, -4.8167, -4.8168,
                -4.8167, -4.8167, -4.8167, -4.8168, -4.8168, -4.8168, -4.8167, -4.8167,
                -4.8168, -4.8167, -4.8167, -4.8165, -4.8167, -4.8167, -4.8167, -4.8169,
                -4.8168, -4.8168, -4.8168, -4.8166, -4.8169, -4.8168, -4.8167, -4.8167
            ]
            , dtype=torch.float32)  # fmt: skip

        EXPECTED_LOGITS_NO_GRAD_1S = Expectations(
            {
                ("xpu", 3): torch.tensor([0.2027,  6.3481,  3.8392, -5.7279, -6.5090, -6.5088, -6.5087, -6.5088,
                                          -6.5087, -6.5088, -6.5090, -6.5089,  7.8796, 13.5483, -6.5088, -6.5080,
                                          -6.5090, -6.5086, -6.5090, -6.5090, -6.5089, -6.5090, -6.5088, -6.5090,
                                          -6.5089, -6.5090, -6.5090, -6.5097, -6.5086, -6.5089, -6.5092, -6.5089,
                                          -6.5088, -6.5090, -6.5090, -6.5088, -6.5090, -6.5091, -6.5087, -6.5089],
                                         dtype=torch.float32),
                ("cuda", None): torch.tensor([0.1966,  6.3449,  3.8350, -5.7291, -6.5106, -6.5104, -6.5103, -6.5104,
                                              -6.5103, -6.5104, -6.5106, -6.5105,  7.8700, 13.5434, -6.5104, -6.5096,
                                              -6.5106, -6.5102, -6.5106, -6.5106, -6.5105, -6.5106, -6.5104, -6.5106,
                                              -6.5105, -6.5106, -6.5106, -6.5113, -6.5102, -6.5105, -6.5108, -6.5105,
                                              -6.5104, -6.5106, -6.5106, -6.5104, -6.5106, -6.5107, -6.5103, -6.5105],
                                             dtype=torch.float32),
            }
        )  # fmt: skip
        EXPECTED_LOGITS_NO_GRAD_1 = EXPECTED_LOGITS_NO_GRAD_1S.get_expectation()

        torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD_0, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(
            logits[1, -1, :40].cpu(),
            EXPECTED_LOGITS_NO_GRAD_1,
            rtol=1e-3,
            atol=6e-3 if torch_device == "cpu" else 1e-3,
        )
