# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import datetime
import gc
import math
import unittest

from transformers import XGLMConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import XGLM_PRETRAINED_MODEL_ARCHIVE_LIST, XGLMForCausalLM, XGLMModel, XGLMTokenizer


class XGLMModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        d_model=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        ffn_dim=37,
        activation_function="gelu",
        activation_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = d_model
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_dim = ffn_dim
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = None
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.pad_token_id = 1

    def get_large_model_config(self):
        return XGLMConfig.from_pretrained("facebook/xglm-564M")

    def prepare_config_and_inputs(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(3)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config(gradient_checkpointing=gradient_checkpointing)

        head_mask = ids_tensor([self.num_hidden_layers, self.num_attention_heads], 2)

        return (
            config,
            input_ids,
            input_mask,
            head_mask,
        )

    def get_config(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        return XGLMConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            num_layers=self.num_hidden_layers,
            attention_heads=self.num_attention_heads,
            ffn_dim=self.ffn_dim,
            activation_function=self.activation_function,
            activation_dropout=self.activation_dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            head_mask,
        ) = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            input_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_xglm_model(self, config, input_ids, input_mask, head_mask, *args):
        model = XGLMModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, head_mask=head_mask)
        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(len(result.past_key_values), config.num_hidden_layers)

    def create_and_check_xglm_model_past(self, config, input_ids, input_mask, head_mask, *args):
        model = XGLMModel(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        output, past = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and token_type_ids
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_xglm_model_attention_mask_past(self, config, input_ids, input_mask, head_mask, *args):
        model = XGLMModel(config=config)
        model.to(torch_device)
        model.eval()

        # create attention mask
        attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        half_seq_length = self.seq_length // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        output, past = model(input_ids, attention_mask=attn_mask).to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and attn_mask
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attn_mask = torch.cat(
            [attn_mask, torch.zeros((attn_mask.shape[0], 1), dtype=torch.long, device=torch_device)],
            dim=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past, attention_mask=attn_mask)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_xglm_model_past_large_inputs(self, config, input_ids, input_mask, head_mask, *args):
        model = XGLMModel(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True)

        output, past = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=1)

        # append to next input_ids
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past)[
            "last_hidden_state"
        ]
        self.parent.assertTrue(output_from_past.shape[1] == next_tokens.shape[1])

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, head_mask, *args):
        model = XGLMForCausalLM(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_forward_and_backwards(
        self, config, input_ids, input_mask, head_mask, *args, gradient_checkpointing=False
    ):
        model = XGLMForCausalLM(config)
        model.to(torch_device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        result = model(input_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        result.loss.backward()

    def create_and_check_xglm_weight_initialization(self, config, *args):
        model = XGLMModel(config)
        model_std = model.config.initializer_range / math.sqrt(2 * model.config.num_hidden_layers)
        for key in model.state_dict().keys():
            if "c_proj" in key and "weight" in key:
                self.parent.assertLessEqual(abs(torch.std(model.state_dict()[key]) - model_std), 0.001)
                self.parent.assertLessEqual(abs(torch.mean(model.state_dict()[key]) - 0.0), 0.01)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (
            config,
            input_ids,
            input_mask,
            head_mask,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "head_mask": head_mask,
        }

        return config, inputs_dict


@require_torch
class XGLMModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (XGLMModel, XGLMForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (XGLMForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": XGLMModel, "text-generation": XGLMForCausalLM} if is_torch_available() else {}
    )
    fx_compatible = True
    test_missing_keys = False
    test_pruning = False

    def setUp(self):
        self.model_tester = XGLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XGLMConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_xglm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xglm_model(*config_and_inputs)

    def test_xglm_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xglm_model_past(*config_and_inputs)

    def test_xglm_model_att_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xglm_model_attention_mask_past(*config_and_inputs)

    def test_xglm_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xglm_model_past_large_inputs(*config_and_inputs)

    def test_xglm_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_xglm_gradient_checkpointing(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward_and_backwards(*config_and_inputs, gradient_checkpointing=True)

    def test_xglm_weight_initialization(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xglm_weight_initialization(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in XGLM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = XGLMModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @unittest.skip("Does not work on the tiny model as we keep hitting edge cases.")
    def test_model_parallelism(self):
        super().test_model_parallelism()


@require_torch
class XGLMModelLanguageGenerationTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        gc.collect()
        torch.cuda.empty_cache()

    def _test_lm_generate_xglm_helper(
        self,
        gradient_checkpointing=False,
        verify_outputs=True,
    ):
        model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        model.to(torch_device)
        input_ids = torch.tensor([[2, 268, 9865]], dtype=torch.long, device=torch_device)  # The dog
        # </s> The dog is a very friendly dog. He is very affectionate and loves to play with other
        expected_output_ids = [2, 268, 9865, 67, 11, 1988, 57252, 9865, 5, 984, 67, 1988, 213838, 1658, 53, 70446, 33, 6657, 278, 1581]  # fmt: skip
        output_ids = model.generate(input_ids, do_sample=False, num_beams=1)
        if verify_outputs:
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @slow
    def test_batch_generation(self):
        model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")
        model.to(torch_device)
        tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")

        tokenizer.padding_side = "left"

        # use different length sentences to test batching
        sentences = [
            "This is an extremelly long sentence that only exists to test the ability of the model to cope with "
            "left-padding, such as in batched generation. The output for the sequence below should be the same "
            "regardless of whether left padding is applied or not. When",
            "Hello, my dog is a little",
        ]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch_device)

        outputs = model.generate(
            input_ids=input_ids, attention_mask=inputs["attention_mask"].to(torch_device), max_new_tokens=12
        )

        inputs_non_padded = tokenizer(sentences[0], return_tensors="pt").input_ids.to(torch_device)
        output_non_padded = model.generate(input_ids=inputs_non_padded, max_new_tokens=12)

        inputs_padded = tokenizer(sentences[1], return_tensors="pt").input_ids.to(torch_device)
        output_padded = model.generate(input_ids=inputs_padded, max_new_tokens=12)

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            "This is an extremelly long sentence that only exists to test the ability of the model to cope with "
            "left-padding, such as in batched generation. The output for the sequence below should be the same "
            "regardless of whether left padding is applied or not. When left padding is applied, the sequence will be "
            "a single",
            "Hello, my dog is a little bit of a shy one, but he is very friendly",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(expected_output_sentence, [non_padded_sentence, padded_sentence])

    @slow
    def test_lm_generate_xglm(self):
        self._test_lm_generate_xglm_helper()

    @slow
    def test_lm_generate_xglm_with_gradient_checkpointing(self):
        self._test_lm_generate_xglm_helper(gradient_checkpointing=True)

    @slow
    def test_xglm_sample(self):
        tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
        model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")

        torch.manual_seed(0)
        tokenized = tokenizer("Today is a nice day and", return_tensors="pt")
        input_ids = tokenized.input_ids
        output_ids = model.generate(input_ids, do_sample=True, num_beams=1)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        EXPECTED_OUTPUT_STRS = [
            # TODO: remove this once we move to torch 2.0
            # torch 1.13.1 + cu116
            "Today is a nice day and the sun is shining. A nice day with warm rainy",
            # torch 2.0 + cu117
            "Today is a nice day and the water is still cold. We just stopped off for some fresh",
        ]
        self.assertIn(output_str, EXPECTED_OUTPUT_STRS)

    @slow
    def test_xglm_sample_max_time(self):
        tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
        model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")
        model.to(torch_device)

        torch.manual_seed(0)
        tokenized = tokenizer("Today is a nice day and", return_tensors="pt")
        input_ids = tokenized.input_ids.to(torch_device)

        MAX_TIME = 0.15

        start = datetime.datetime.now()
        model.generate(input_ids, do_sample=True, max_time=MAX_TIME, max_length=256)
        duration = datetime.datetime.now() - start
        self.assertGreater(duration, datetime.timedelta(seconds=MAX_TIME))
        self.assertLess(duration, datetime.timedelta(seconds=1.5 * MAX_TIME))

        start = datetime.datetime.now()
        model.generate(input_ids, do_sample=False, max_time=MAX_TIME, max_length=256)
        duration = datetime.datetime.now() - start
        self.assertGreater(duration, datetime.timedelta(seconds=MAX_TIME))
        self.assertLess(duration, datetime.timedelta(seconds=1.5 * MAX_TIME))

        start = datetime.datetime.now()
        model.generate(input_ids, do_sample=False, num_beams=2, max_time=MAX_TIME, max_length=256)
        duration = datetime.datetime.now() - start
        self.assertGreater(duration, datetime.timedelta(seconds=MAX_TIME))
        self.assertLess(duration, datetime.timedelta(seconds=1.5 * MAX_TIME))

        start = datetime.datetime.now()
        model.generate(input_ids, do_sample=True, num_beams=2, max_time=MAX_TIME, max_length=256)
        duration = datetime.datetime.now() - start
        self.assertGreater(duration, datetime.timedelta(seconds=MAX_TIME))
        self.assertLess(duration, datetime.timedelta(seconds=1.5 * MAX_TIME))

        start = datetime.datetime.now()
        model.generate(input_ids, do_sample=False, max_time=None, max_length=256)
        duration = datetime.datetime.now() - start
        self.assertGreater(duration, datetime.timedelta(seconds=1.25 * MAX_TIME))

    @require_torch_accelerator
    @require_torch_fp16
    def test_batched_nan_fp16(self):
        model_name = "facebook/xglm-564M"
        tokenizer = XGLMTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")

        model = XGLMForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).to(torch_device)
        model = model.eval()

        batch = tokenizer(["Who are you?", "Joe Biden is the president of"], padding=True, return_tensors="pt")

        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            self.assertFalse(
                torch.isnan(outputs.logits[0]).any().item()
            )  # the first logits could contain NaNs if it fails
