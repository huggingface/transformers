# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GLM-4-MoE model."""

import unittest

from parameterized import parameterized

from transformers import AutoTokenizer, Glm4MoeConfig, is_torch_available, set_seed
from transformers.testing_utils import (
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_large_accelerator,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Glm4MoeForCausalLM,
        Glm4MoeForQuestionAnswering,
        Glm4MoeForSequenceClassification,
        Glm4MoeForTokenClassification,
        Glm4MoeModel,
    )
    from transformers.models.glm4_moe.modeling_glm4_moe import (
        Glm4MoeRotaryEmbedding,
    )


class Glm4MoeModelTester:
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
        hidden_size=64,
        intermediate_size=37,
        moe_intermediate_size=12,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        num_experts=8,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        qk_nope_head_dim=32,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=2,
        norm_topk_prob=True,
        aux_loss_alpha=0.001,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
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
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.num_experts = num_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return Glm4MoeConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            n_shared_experts=self.n_shared_experts,
            num_experts=self.num_experts,
            n_routed_experts=self.n_routed_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            kv_lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim,
            n_group=self.n_group,
            topk_group=self.topk_group,
            num_experts_per_tok=self.num_experts_per_tok,
            first_k_dense_replace=self.first_k_dense_replace,
            norm_topk_prob=self.norm_topk_prob,
            aux_loss_alpha=self.aux_loss_alpha,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            use_cache=True,
            pad_token_id=self.pad_token_id,
            attention_dropout=self.attention_probs_dropout_prob,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Glm4MoeModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class Glm4MoeModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Glm4MoeModel,
            Glm4MoeForSequenceClassification,
            Glm4MoeForTokenClassification,
            Glm4MoeForCausalLM,
            Glm4MoeForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (Glm4MoeForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Glm4MoeModel,
            "text-classification": Glm4MoeForSequenceClassification,
            "token-classification": Glm4MoeForTokenClassification,
            "text-generation": Glm4MoeForCausalLM,
            "question-answering": Glm4MoeForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Glm4MoeForCausalLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = Glm4MoeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Glm4MoeConfig, hidden_size=37)

    @unittest.skip(reason="GLM-4-MOE's output different if you pad left or right. This is expected")
    def test_left_padding_compatibility(self):
        pass

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    @parameterized.expand([("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = Glm4MoeModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = Glm4MoeModel(config)
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

    def test_model_rope_scaling(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(1, dtype=torch.float32, device=torch_device)  # used exlusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        original_rope = Glm4MoeRotaryEmbedding(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        linear_scaling_rope = Glm4MoeRotaryEmbedding(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short)
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
        ntk_scaling_rope = Glm4MoeRotaryEmbedding(config=config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short)
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        config.rope_scaling = {"type": "yarn", "factor": scaling_factor}
        yarn_scaling_rope = Glm4MoeRotaryEmbedding(config=config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short)
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long)
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

    @require_torch_large_accelerator
    @require_torch_sdpa
    @slow
    def test_eager_matches_sdpa_generate(self):
        """
        Overwriting the common test as the test is flaky on tiny models
        """
        max_new_tokens = 30
        tokenizer = AutoTokenizer.from_pretrained("THUDM/GLM-4-MoE")

        model_sdpa = Glm4MoeForCausalLM.from_pretrained(
            "THUDM/GLM-4-MoE",
            torch_dtype=torch.bfloat16,
        ).to(torch_device)

        self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

        model_eager = Glm4MoeForCausalLM.from_pretrained(
            "THUDM/GLM-4-MoE",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(torch_device)

        self.assertTrue(model_eager.config._attn_implementation == "eager")

        texts = [
            "hi here's a longer context, getting longer and",
            "Hello this is a very long sentence my friend, very long for real",
            "Today I am in Paris and",
        ]

        for padding_side in ["left", "right"]:
            tokenizer.padding_side = padding_side
            tokenizer.pad_token = tokenizer.eos_token

            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(torch_device)

            res_eager = model_eager.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            res_sdpa = model_sdpa.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            with self.subTest(f"{padding_side}"):
                torch.testing.assert_close(
                    res_eager,
                    res_sdpa,
                    msg=f"\n{tokenizer.batch_decode(res_eager)} \nvs\n{tokenizer.batch_decode(res_sdpa)}",
                )

    @require_torch_gpu
    def test_flex_attention_with_grads(self):
        """
        Overwriting as the namings/functionality on the attention part are different; for now it's more of a unique model.
        Original issue is also due to dimensionalities, here specifically due to dims not being a multiple of 2.
        """
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config._attn_implementation = "flex_attention"

            # Disable dropout
            config.attention_dropout = 0.0

            # Deepseek 3 specific - manipulate nope and adjust calculated total head dim
            config.qk_nope_head_dim = 16
            config.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

            model = model_class(config).to(device=torch_device)
            self.assertTrue(model.config._attn_implementation == "flex_attention")

            # Elaborate workaround for encoder-decoder models as some do not specify their main input
            dummy_inputs = {model.main_input_name: inputs_dict[model.main_input_name].to(torch_device)}
            if config.is_encoder_decoder:
                dummy_inputs["decoder_input_ids"] = inputs_dict["decoder_input_ids"].to(torch_device)
                dummy_inputs["decoder_attention_mask"] = inputs_dict["decoder_attention_mask"].to(torch_device)

            # If this does not raise an error, the test passes (see https://github.com/huggingface/transformers/pull/35605)
            _ = model(**dummy_inputs)


@require_torch_accelerator
class Glm4MoeIntegrationTest(unittest.TestCase):
    def tearDown(self):
        # See LlamaIntegrationTest.tearDown(). Can be removed once LlamaIntegrationTest.tearDown() is removed.
        cleanup(torch_device, gc_collect=False)

    @slow
    @require_torch_accelerator
    @require_read_token
    def test_compile_static_cache(self):
        NUM_TOKENS_TO_GENERATE = 40
        # https://github.com/huggingface/transformers/pull/38562#issuecomment-2939209171
        EXPECTED_TEXT_COMPLETION = [
            "[gMASK]<sop><|user|>\nhello, who are you<|assistant|>\n<think>\nThe user is asking me to introduce myself. I need to respond according to the guidelines provided, making it clear that I am GLM, a large language model trained by Zhipu",
            '[gMASK]<sop><|user|>\nwhat is the answer of 1 + 1?<|assistant|>\n<think>\nFirst, the user asked: "what is the answer of 1 + 1?" This seems like a very basic math question. I need to provide a straightforward answer.\n\nThe expression',
        ]

        prompts = [
            "hello, who are you",
            "what is the answer of 1 + 1?",
        ]
        tokenizer = AutoTokenizer.from_pretrained("THUDM/GLM-4-MoE")
        model = Glm4MoeForCausalLM.from_pretrained(
            "THUDM/GLM-4-MoE", device_map=torch_device, torch_dtype=torch.bfloat16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        # Static Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

        # Static Cache + compile
        model._cache = None  # clear cache object, initialized when we pass `cache_implementation="static"`
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_compiled_text)
