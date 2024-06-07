# coding=utf-8

"""Testing suite for the PyTorch Yuan2M32 model."""

import gc
import tempfile
import unittest

import pytest

from transformers import AutoTokenizer, Yuan2M32Config, is_torch_available, set_seed
from transformers.testing_utils import (
    backend_empty_cache,
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
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
        YuanForCausalLM,
        YuanForSequenceClassification,
        YuanModel,
    )


class Yuan2M32ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        intermediate_size=37,
        hidden_act="silu",
        dropout=0.1,
        max_position_embeddings=512,
        num_attention_heads=4,
        rms_norm_eps=1e-06,
        tie_word_embeddings=True,
        torch_dtype='bfloat16',
        use_cache=True,
        causal_mask=True,
        use_flash_attention=True,
        reset_attention_mask=True,
        reset_position_ids=True,
        use_loss_mask=False,
        use_moe=True,
        moe_config={
            "moe_num_experts":8,
            "moe_top_k":2,
            "ffn_hidden_size":64,
            "norm_topk_prob":True,
            "gated_linear_unit":True
        },
        output_router_logits=True,
        attention_projection_size=64,
        pad_token_id=0,
        bos_token_id=1,
    ):
        
        self.parent=parent
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.is_training=is_training
        self.use_input_mask=use_input_mask
        self.use_labels=use_labels
        self.vocab_size=vocab_size
        self.hidden_size=hidden_size
        self.num_hidden_layers=num_hidden_layers
        self.intermediate_size=intermediate_size
        self.hidden_act=hidden_act
        self.dropout=dropout
        self.max_position_embeddings=max_position_embeddings
        self.num_attention_heads=num_attention_heads
        self.rms_norm_eps=rms_norm_eps
        self.tie_word_embeddings=tie_word_embeddings
        self.torch_dtype=torch_dtype
        self.use_cache=use_cache
        self.causal_mask=causal_mask
        self.use_flash_attention=use_flash_attention
        self.reset_attention_mask=reset_attention_mask
        self.reset_position_ids=reset_position_ids
        self.use_loss_mask=use_loss_mask
        self.use_moe=use_moe
        self.moe_config=moe_config
        self.output_router_logits=output_router_logits
        self.attention_projection_size=attention_projection_size
        self.pad_token_id=pad_token_id
        self.bos_token_id=bos_token_id

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones(self.batch_size, self.seq_length)).to(torch_device)

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
        return Yuan2M32Config(
            vocab_size=self.vocab_size,
            parent=self.parent
            batch_size=self.batch_size
            seq_length=self.seq_length
            is_training=self.is_training
            use_input_mask=self.use_input_mask
            use_labels=self.use_labels
            vocab_size=self.vocab_size
            hidden_size=self.hidden_size
            num_hidden_layers=self.num_hidden_layers
            intermediate_size=self.intermediate_size
            hidden_act=self.hidden_act
            dropout=self.dropout
            max_position_embeddings=self.max_position_embeddings
            num_attention_heads=self.num_attention_heads
            rms_norm_eps=self.rms_norm_eps
            tie_word_embeddings=self.tie_word_embeddings
            torch_dtype=self.torch_dtype
            use_cache=self.use_cache
            causal_mask=self.causal_mask
            use_flash_attention=self.use_flash_attention
            reset_attention_mask=self.reset_attention_mask
            reset_position_ids=self.reset_position_ids
            use_loss_mask=self.use_loss_mask
            use_moe=self.use_moe
            moe_config=self.moe_config
            output_router_logits=self.output_router_logits
            attention_projection_size=self.attention_projection_size
            pad_token_id=self.pad_token_id
            bos_token_id=self.bos_token_id
        )

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model with Llama->Yuan2M32
    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = YuanModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model_as_decoder with Llama->Yuan2M32
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
        model = YuanModel(config)
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

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_for_causal_lm with Llama->Yuan2M32
    def create_and_check_for_causal_lm(
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
        model = YuanForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_decoder_model_past_large_inputs with Llama->Yuan2M32
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
        model = YuanForCausalLM(config=config)
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

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs_for_common
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
# Copied from tests.models.mistral.test_modeling_mistral.MistralModelTest with Mistral->Yuan2M32
class Yuan2M32ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (YuanModel, YuanForCausalLM, YuanForSequenceClassification)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (YuanForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": YuanModel,
            "text-classification": YuanForSequenceClassification,
            "text-generation": YuanForCausalLM,
            "zero-shot": YuanForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = True

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        return True

    # Ignore copy
    @require_torch_sdpa
    @slow
    def test_eager_matches_sdpa_generate(self):
        super().test_eager_matches_sdpa_generate()

    def setUp(self):
        self.model_tester = Yuan2M32ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Yuan2M32Config, hidden_size=37)

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

    def test_Yuan2M32_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        print(config)
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = YuanForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Yuan2M32_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = YuanForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Yuan2M32_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = YuanForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))


    @unittest.skip("Yuan2M32 buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip("Yuan2M32 uses GQA on all models so the KV cache is a non standard format")
    def test_past_key_values_format(self):
        pass

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_generate_padding_right(self):
        import torch

        for model_class in self.all_generative_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(
                    torch_device
                )

                dummy_input = torch.LongTensor([[0, 2, 3, 4], [0, 2, 3, 4]]).to(torch_device)
                dummy_attention_mask = torch.LongTensor([[1, 1, 1, 1], [1, 1, 1, 0]]).to(torch_device)

                model.generate(dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=1, do_sample=False)

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                with self.assertRaises(ValueError):
                    _ = model.generate(
                        dummy_input, attention_mask=dummy_attention_mask, max_new_tokens=1, do_sample=False
                    )

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_generate_use_cache(self):
        import torch

        max_new_tokens = 30

        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                dummy_input = dummy_input.to(torch.float16)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
                # NOTE: Yuan2M32 apparently does not support right padding + use_cache with FA2.
                dummy_attention_mask[:, -1] = 1

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                ).to(torch_device)

                # Just test that a large cache works as expected
                _ = model.generate(
                    dummy_input,
                    attention_mask=dummy_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest("Yuan2M32 flash attention does not support right padding")

    # Ignore copy
    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_experts = 8
        config.expert_interval = 2
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = YuanForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(result.router_logits[0].shape, (91, config.num_experts))
        torch.testing.assert_close(result.aux_loss.cpu(), torch.tensor(2, dtype=torch.float32), rtol=1e-2, atol=1e-2)

        # First, we make sure that adding padding tokens doesn't change the loss
        # loss(input_ids, attention_mask=None) == loss(input_ids + padding, attention_mask=attention_mask_with_padding)
        pad_length = 1000
        # Add padding tokens (assume that pad_token_id=1) to input_ids
        padding_block = torch.ones(input_ids.shape[0], pad_length, dtype=torch.int32).to(torch_device)
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)  # this is to simulate padding to the left
        padded_attention_mask = padded_input_ids.ne(1).to(torch_device)

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)

        # We make sure that the loss of includding padding tokens != the loss without padding tokens
        # if attention_mask=None --> we don't exclude padding tokens
        include_padding_result = model(padded_input_ids, attention_mask=None)

        # This is to mimic torch.testing.assert_not_close
        self.assertNotAlmostEqual(include_padding_result.aux_loss.item(), result.aux_loss.item())


@require_torch
class Yuan2M32IntegrationTest(unittest.TestCase):
    @slow
    def test_model_a2_7b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        # path?
        model = YuanForCausalLM.from_pretrained("Yuan/Yuan2-m32", device_map="auto")
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-4.2125, -3.6416, -4.9136, -4.3005, -4.9938, -3.4393, -3.5195, -4.1621]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([2.3013, -0.6595, -0.1389, -1.4095, -1.7381, -1.7609, -2.0449, -2.4289, -3.0271, -2.1351, -0.6568, -4.6012, -1.9102, -0.7475, -3.1377, 4.6904, 7.1936, 7.0991, 6.4414, 6.1720, 6.2617, 5.8751, 5.6997, 5.6011, 5.5828, -3.9505, -0.5384, -0.3392, 1.2445, 2.0714])  # fmt: skip
        print(out[0, 0, :30])
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-4, rtol=1e-4)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_model_a2_7b_generation(self):
        EXPECTED_TEXT_COMPLETION = """To be or not to be, that is the question. This is the question that has been asked by many people over the"""
        prompt = "To be or not to"
        tokenizer = AutoTokenizer.from_pretrained("Yuan/Yuan2-m32", use_fast=False)
        model = YuanForCausalLM.from_pretrained("Yuan/Yuan2-m32", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

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
    def test_model_a2_7b_long_prompt(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = YuanForCausalLM.from_pretrained(
            "Yuan/Yuan2-m32",
            device_map="auto"
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
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
    def test_model_a2_7b_long_prompt_sdpa(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        #sdpa?
        model = YuanForCausalLM.from_pretrained(
            "Yuan/Yuan2-m32",
            device_map="auto",
            attn_implementation="sdpa",
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
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

        EXPECTED_TEXT_COMPLETION = """To be or not to be, that is the question. This is the question that has been asked by many people over the"""
        prompt = "To be or not to"
        tokenizer = AutoTokenizer.from_pretrained("Yuan/Yuan2-m32", use_fast=False)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    def test_speculative_generation(self):
        EXPECTED_TEXT_COMPLETION = (
            "To be or not to be, that is the question.\nThe answer is to be, of course. But what does it"
        )
        prompt = "To be or not to"
        tokenizer = AutoTokenizer.from_pretrained("Yuan/Yuan2-m32", use_fast=False)
        model = YuanForCausalLM.from_pretrained(
            "Yuan/Yuan2-m32", device_map="auto", torch_dtype=torch.float16
        )
        assistant_model = YuanForCausalLM.from_pretrained(
            "Yuan/Yuan2-m32", device_map="auto", torch_dtype=torch.float16
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

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
