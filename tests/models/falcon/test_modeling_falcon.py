# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Falcon model. """


import unittest

from parameterized import parameterized

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    FalconConfig,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger, require_bitsandbytes, require_torch, slow, tooslow, torch_device
from transformers.utils import logging as transformers_logging

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        FalconForCausalLM,
        FalconForQuestionAnswering,
        FalconForSequenceClassification,
        FalconForTokenClassification,
        FalconModel,
    )


class FalconModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
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
        self.num_attention_heads = num_attention_heads
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
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None

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
        return FalconConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=1,
            new_decoder_architecture=True,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = FalconModel(config=config)
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
        model = FalconModel(config)
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
        model = FalconForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

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
        model = FalconForCausalLM(config=config)
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
class FalconModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            FalconModel,
            FalconForCausalLM,
            FalconForSequenceClassification,
            FalconForTokenClassification,
            FalconForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (FalconForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": FalconModel,
            "question-answering": FalconForQuestionAnswering,
            "text-classification": FalconForSequenceClassification,
            "text-generation": FalconForCausalLM,
            "token-classification": FalconForTokenClassification,
            "zero-shot": FalconForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = FalconModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FalconConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_position_embedding_types(self):
        config, *inputs = self.model_tester.prepare_config_and_inputs()
        for alibi in [True, False]:
            config.alibi = alibi
            self.model_tester.create_and_check_model(config, *inputs)

    def test_falcon_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = FalconForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_falcon_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = FalconForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_cache_conversions(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = input_dict["input_ids"]
        model = FalconForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, use_cache=True)
        batch_size = input_ids.shape[0]
        rw_cache = model._convert_to_rw_cache(result.past_key_values)
        standard_cache = model._convert_cache_to_standard_format(rw_cache, batch_size)
        for layer in range(len(rw_cache)):
            for tensor_idx in range(2):
                self.assertTrue(rw_cache[layer][tensor_idx].ndim == 3)
                self.assertTrue(result.past_key_values[layer][tensor_idx].ndim == 4)
                self.assertTrue(
                    torch.all(result.past_key_values[layer][tensor_idx] == standard_cache[layer][tensor_idx])
                )

    def test_falcon_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = FalconForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_past_key_values_format(self):
        # Falcon can have different numbers of KV-heads than the number of query heads, so we need
        # to override this test to use the right head counts.
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            # If it doesn't support cache, pass the test
            if not hasattr(config, "use_cache"):
                return

            model = model_class(config).to(torch_device)
            if "use_cache" not in inputs:
                inputs["use_cache"] = True
            outputs = model(**inputs)

            # If "past_key_values" is not returned, pass the test (e.g. RWKV uses a different cache name and format)
            if "past_key_values" not in outputs:
                return

            num_hidden_layers = (
                getattr(config, "decoder_layers", None)
                or getattr(config, "num_decoder_layers", None)
                or config.num_hidden_layers
            )
            num_attention_heads = getattr(config, "num_kv_heads", config.num_attention_heads)
            embed_dim = getattr(config, "d_model", config.hidden_size)
            per_head_embed_dim = embed_dim // num_attention_heads

            past_kv = outputs["past_key_values"]
            self.assertEqual(len(past_kv), num_hidden_layers)

            batch_size, seq_length = inputs["input_ids"].shape
            for i in range(num_hidden_layers):
                if config.new_decoder_architecture:
                    num_attention_heads = config.num_attention_heads
                elif config.multi_query:
                    num_attention_heads = 1
                self.assertEqual(len(past_kv[0]), 2)  # K V for the decoder = 2
                self.assertEqual(
                    past_kv[i][0].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                )
                self.assertEqual(
                    past_kv[i][1].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                )

    @parameterized.expand([("linear",), ("dynamic",)])
    def test_model_rope_scaling(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = FalconModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = FalconModel(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            self.assertTrue(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))


@require_torch
class FalconLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_falcon(self):
        tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
        model = FalconForCausalLM.from_pretrained("Rocketknight1/falcon-rw-1b")
        model.eval()
        model.to(torch_device)
        inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)

        EXPECTED_OUTPUT = (
            "My favorite food is pizza. I love it so much that I have a pizza party every year for my birthday."
        )

        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=19)
        output_str = tokenizer.batch_decode(output_ids)[0]

        self.assertEqual(output_str, EXPECTED_OUTPUT)

    @slow
    def test_lm_generation_big_models(self):
        # The big models are way too big for the CI, so we use tiny random models that resemble their
        # architectures but with much smaller and fewer layers
        for repo in ["Rocketknight1/tiny-random-falcon-7b", "Rocketknight1/tiny-random-falcon-40b"]:
            tokenizer = AutoTokenizer.from_pretrained(repo)
            model = FalconForCausalLM.from_pretrained(repo)
            model.eval()
            model.to(torch_device)
            inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)

            # We just test that these run without errors - the models are randomly initialized
            # and so the actual text outputs will be garbage
            model.generate(**inputs, do_sample=False, max_new_tokens=4)
            model.generate(**inputs, do_sample=True, max_new_tokens=4)
            model.generate(**inputs, num_beams=2, max_new_tokens=4)

    @slow
    def test_lm_generation_use_cache(self):
        # The big models are way too big for the CI, so we use tiny random models that resemble their
        # architectures but with much smaller and fewer layers
        with torch.no_grad():
            for repo in [
                "Rocketknight1/falcon-rw-1b",
                "Rocketknight1/tiny-random-falcon-7b",
                "Rocketknight1/tiny-random-falcon-40b",
            ]:
                tokenizer = AutoTokenizer.from_pretrained(repo)
                model = FalconForCausalLM.from_pretrained(repo)
                model.eval()
                model.to(device=torch_device)
                inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)

                # Test results are the same with and without cache
                outputs_no_cache = model.generate(**inputs, do_sample=False, max_new_tokens=20, use_cache=False)
                outputs_cache = model.generate(**inputs, do_sample=False, max_new_tokens=20, use_cache=True)
                self.assertTrue((outputs_cache - outputs_no_cache).sum().item() == 0)

    @require_bitsandbytes
    @slow
    def test_batched_generation(self):
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            "tiiuae/falcon-7b",
            device_map="auto",
            load_in_4bit=True,
        )

        test_text = "A sequence: 1, 2"  # should generate the rest of the sequence

        unpadded_inputs = tokenizer([test_text], return_tensors="pt").to("cuda:0")
        unpadded_inputs.pop("token_type_ids")
        unpadded_gen_out = model.generate(**unpadded_inputs, max_new_tokens=20)
        unpadded_gen_text = tokenizer.batch_decode(unpadded_gen_out, skip_special_tokens=True)

        dummy_text = "This is a longer text " * 2  # forces left-padding on `test_text`
        padded_inputs = tokenizer([test_text, dummy_text], return_tensors="pt", padding=True).to("cuda:0")
        padded_inputs.pop("token_type_ids")
        padded_gen_out = model.generate(**padded_inputs, max_new_tokens=20)
        padded_gen_text = tokenizer.batch_decode(padded_gen_out, skip_special_tokens=True)

        expected_output = "A sequence: 1, 2, 3, 4, 5, 6, 7, 8, "
        self.assertLess(unpadded_inputs.input_ids.shape[-1], padded_inputs.input_ids.shape[-1])  # left-padding exists
        self.assertEqual(unpadded_gen_text[0], expected_output)
        self.assertEqual(padded_gen_text[0], expected_output)


# TODO Lysandre: Remove this in version v4.34
class FalconOverrideTest(unittest.TestCase):
    supported_checkpoints = [
        "tiiuae/falcon-7b",
        "tiiuae/falcon-7b-instruct",
        "tiiuae/falcon-40b",
        "tiiuae/falcon-40b-instruct",
    ]

    latest_revisions = {
        "tiiuae/falcon-7b": "f7796529e36b2d49094450fb038cc7c4c86afa44",
        "tiiuae/falcon-7b-instruct": "eb410fb6ffa9028e97adb801f0d6ec46d02f8b07",
        "tiiuae/falcon-40b": "561820f7eef0cc56a31ea38af15ca1acb07fab5d",
        "tiiuae/falcon-40b-instruct": "ca78eac0ed45bf64445ff0687fabba1598daebf3",
    }

    def test_config_without_remote_code(self):
        logger_ = transformers_logging.get_logger("transformers.models.auto.configuration_auto")

        for supported_checkpoint in self.supported_checkpoints:
            with CaptureLogger(logger_) as cm:
                config1 = FalconConfig.from_pretrained(supported_checkpoint, trust_remote_code=False)
                config2 = FalconConfig.from_pretrained(supported_checkpoint)

            self.assertIn(
                "The Falcon model was initialized without `trust_remote_code=True`, and will therefore leverage the "
                "transformers library implementation.",
                cm.out,
            )

            self.assertEqual(config1.to_dict(), config2.to_dict())

    def test_auto_config_without_remote_code(self):
        logger_ = transformers_logging.get_logger("transformers.models.auto.configuration_auto")

        for supported_checkpoint in self.supported_checkpoints:
            with CaptureLogger(logger_) as cm:
                config1 = AutoConfig.from_pretrained(supported_checkpoint, trust_remote_code=False)
                config2 = AutoConfig.from_pretrained(supported_checkpoint)

            self.assertIn(
                "The Falcon model was initialized without `trust_remote_code=True`, and will therefore leverage the "
                "transformers library implementation.",
                cm.out,
            )

            self.assertEqual(config1.to_dict(), config2.to_dict())

    def test_config_with_remote_code(self):
        for supported_checkpoint in self.supported_checkpoints:
            config = FalconConfig.from_pretrained(supported_checkpoint, trust_remote_code=True)

            self.assertIn(config.model_type, ["RefinedWebModel", "RefinedWeb"])

    def test_auto_config_with_remote_code(self):
        for supported_checkpoint in self.supported_checkpoints:
            config = AutoConfig.from_pretrained(supported_checkpoint, trust_remote_code=True)

            self.assertIn(config.model_type, ["RefinedWebModel", "RefinedWeb"])

    def test_config_with_specific_revision(self):
        for supported_checkpoint in self.supported_checkpoints:
            config = FalconConfig.from_pretrained(
                supported_checkpoint, revision=self.latest_revisions[supported_checkpoint], trust_remote_code=True
            )

            self.assertIn(config.model_type, ["RefinedWebModel", "RefinedWeb"])

    def test_auto_config_with_specific_revision(self):
        for supported_checkpoint in self.supported_checkpoints:
            config = AutoConfig.from_pretrained(
                supported_checkpoint, revision=self.latest_revisions[supported_checkpoint], trust_remote_code=True
            )

            self.assertIn(config.model_type, ["RefinedWebModel", "RefinedWeb"])

    @tooslow
    def test_model_without_remote_code(self):
        logger_ = transformers_logging.get_logger("transformers.models.auto.configuration_auto")
        for supported_checkpoint in self.supported_checkpoints:
            with CaptureLogger(logger_) as cm:
                config1 = FalconModel.from_pretrained(supported_checkpoint, trust_remote_code=False).config
                config2 = FalconModel.from_pretrained(supported_checkpoint).config

                # trust_remote_code only works with Auto Classes !
                config3 = FalconModel.from_pretrained(supported_checkpoint, trust_remote_code=True).config

            self.assertIn(
                "The Falcon model was initialized without `trust_remote_code=True`, and will therefore leverage the "
                "transformers library implementation.",
                cm.out,
            )

            self.assertEqual(config1.to_dict(), config2.to_dict())
            self.assertEqual(config1.to_dict(), config3.to_dict())

    @tooslow
    def test_auto_model_without_remote_code(self):
        logger_ = transformers_logging.get_logger("transformers.models.auto.configuration_auto")
        for supported_checkpoint in self.supported_checkpoints:
            with CaptureLogger(logger_) as cm:
                config1 = AutoModel.from_pretrained(supported_checkpoint, trust_remote_code=False).config
                config2 = AutoModel.from_pretrained(supported_checkpoint).config

            self.assertIn(
                "The Falcon model was initialized without `trust_remote_code=True`, and will therefore leverage the "
                "transformers library implementation.",
                cm.out,
            )

            self.assertEqual(config1.to_dict(), config2.to_dict())

    @tooslow
    def test_auto_model_with_remote_code(self):
        for supported_checkpoint in self.supported_checkpoints:
            config = AutoModel.from_pretrained(supported_checkpoint, trust_remote_code=True).config

            self.assertIn(config.model_type, ["RefinedWebModel", "RefinedWeb"])

    @tooslow
    def test_auto_model_with_specific_revision(self):
        for supported_checkpoint in self.supported_checkpoints:
            config = AutoModel.from_pretrained(
                supported_checkpoint, revision=self.latest_revisions[supported_checkpoint], trust_remote_code=True
            ).config

            self.assertIn(config.model_type, ["RefinedWebModel", "RefinedWeb"])
