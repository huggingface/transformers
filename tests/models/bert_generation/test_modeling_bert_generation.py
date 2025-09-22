# Copyright 2020 The HuggingFace Team. All rights reserved.
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


import inspect
import tempfile
import unittest

from transformers import BertGenerationConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import BertGenerationDecoder, BertGenerationEncoder, DataCollatorWithFlattening


class BertGenerationEncoderTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=50,
        initializer_range=0.02,
        use_labels=True,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_labels = use_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return config, input_ids, input_mask, token_labels

    def get_config(self):
        return BertGenerationConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            token_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            input_mask,
            token_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
        token_labels,
        **kwargs,
    ):
        model = BertGenerationEncoder(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        input_mask,
        token_labels,
        encoder_hidden_states,
        encoder_attention_mask,
        **kwargs,
    ):
        config.add_cross_attention = True
        model = BertGenerationEncoder(config=config)
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
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
        token_labels,
        encoder_hidden_states,
        encoder_attention_mask,
        **kwargs,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = BertGenerationDecoder(config=config).to(torch_device).eval()

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

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        token_labels,
        *args,
    ):
        model = BertGenerationDecoder(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask, token_labels = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class BertGenerationEncoderTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (BertGenerationEncoder, BertGenerationDecoder) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": BertGenerationEncoder, "text-generation": BertGenerationDecoder}
        if is_torch_available()
        else {}
    )

    # Overwriting to add `is_decoder` flag
    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs = super().prepare_config_and_inputs_for_generate(batch_size)
        config.is_decoder = True
        return config, inputs

    def setUp(self):
        self.model_tester = BertGenerationEncoderTester(self)
        self.config_tester = ConfigTester(self, config_class=BertGenerationConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_as_bert(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs()
        config.model_type = "bert"
        self.model_tester.create_and_check_model(config, input_ids, input_mask, token_labels)

    def test_model_as_decoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_model_as_decoder_with_default_input_mask(self):
        (
            config,
            input_ids,
            input_mask,
            token_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = self.model_tester.prepare_config_and_inputs_for_decoder()

        input_mask = None

        self.model_tester.create_and_check_model_as_decoder(
            config,
            input_ids,
            input_mask,
            token_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model = BertGenerationEncoder.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
        self.assertIsNotNone(model)

    def attention_mask_padding_matches_padding_free_with_position_ids(
        self, attn_implementation: str, fa_kwargs: bool = False
    ):
        """
        Overwritten to account for the embeddings that rely on position ids.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        max_new_tokens = 30
        support_flag = {
            "sdpa": "_supports_sdpa",
            "flash_attention_2": "_supports_flash_attn",
            "flash_attention_3": "_supports_flash_attn",
        }

        for model_class in self.all_generative_model_classes:
            if attn_implementation != "eager" and not getattr(model_class, support_flag[attn_implementation]):
                self.skipTest(f"{model_class.__name__} does not support {attn_implementation}")

            # can't infer if new attn mask API is supported by assume that only model with attention backend support it
            if not model_class._supports_attention_backend:
                self.skipTest(f"{model_class.__name__} does not support new attention mask API")

            if model_class._is_stateful:  # non-transformer models most probably have no packing support
                self.skipTest(f"{model_class.__name__} doesn't support packing!")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            if config.is_encoder_decoder:
                self.skipTest("Model is an encoder-decoder")

            if 0 not in inputs_dict.get("attention_mask", []) or "attention_mask" not in inputs_dict:
                self.skipTest("Model dummy inputs should contain padding in their attention mask")

            if "input_ids" not in inputs_dict or inputs_dict["input_ids"].ndim != 2:
                self.skipTest("Model dummy inputs should contain text input ids")

            # make sure that all models have enough positions for generation
            dummy_input_ids = inputs_dict["input_ids"]
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input_ids.shape[1] + 1

            model = model_class(config)
            if "position_ids" not in inspect.signature(model.forward).parameters:
                self.skipTest("Model does not support position_ids")

            if (not fa_kwargs) and "position_ids" not in inspect.signature(model.forward).parameters:
                continue  # this model doesn't accept position ids as input

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Drop all keys except for the minimal set. Hard to manipulate with multimodals/head_mask/etc
                inputs_dict = {k: v for k, v in inputs_dict.items() if k in ["input_ids", "attention_mask"]}

                # Ensure left padding, to adapt for some models
                if 0 in inputs_dict["attention_mask"][:, -1]:
                    inputs_dict["attention_mask"] = inputs_dict["attention_mask"].flip(1)
                dummy_attention_mask = inputs_dict["attention_mask"]
                dummy_input_ids[~dummy_attention_mask.bool()] = config.get_text_config().pad_token_id

                # Main difference to other models, we need to prepare position ids according to the attention mask
                # as we use it to extract embeddings that rely on the correct position - naively increasing sequences do
                # not suffice anymore atp. The solution here calculates an increasing sequences for all 1s and puts 0s else.
                inputs_dict["position_ids"] = ((inputs_dict["attention_mask"] == 1).long().cumsum(dim=1) - 1) * (
                    inputs_dict["attention_mask"] == 1
                ).long()

                model = (
                    model_class.from_pretrained(
                        tmpdirname,
                        dtype=torch.bfloat16,
                        attn_implementation=attn_implementation,
                    )
                    .to(torch_device)
                    .eval()
                )

                if fa_kwargs:
                    # flatten
                    features = [
                        {"input_ids": i[a.bool()].tolist()} for i, a in zip(dummy_input_ids, dummy_attention_mask)
                    ]

                    # add position_ids + fa_kwargs
                    data_collator = DataCollatorWithFlattening(return_tensors="pt", return_flash_attn_kwargs=True)
                    batch = data_collator(features)
                    padfree_inputs_dict = {
                        k: t.to(torch_device) if torch.is_tensor(t) else t for k, t in batch.items()
                    }
                else:
                    # create packed position_ids
                    position_ids = (
                        torch.cat([torch.arange(length) for length in dummy_attention_mask.sum(1).tolist()])
                        .long()
                        .unsqueeze(0)
                        .to(torch_device)
                    )
                    padfree_inputs_dict = {
                        "input_ids": dummy_input_ids[dummy_attention_mask.bool()].unsqueeze(0),
                        "position_ids": position_ids,
                    }

                # We need to do simple forward without cache in order to trigger packed SDPA/flex/eager attention path
                res_padded = model(**inputs_dict, use_cache=False)
                res_padfree = model(**padfree_inputs_dict, use_cache=False)

                logits_padded = res_padded.logits[dummy_attention_mask.bool()]
                logits_padfree = res_padfree.logits[0]

                # acceptable numerical instability
                tol = torch.finfo(torch.bfloat16).eps
                torch.testing.assert_close(logits_padded, logits_padfree, rtol=tol, atol=tol)


@require_torch
class BertGenerationEncoderIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head_absolute_embedding(self):
        model = BertGenerationEncoder.from_pretrained(
            "google/bert_for_seq_generation_L-24_bbc_encoder", attn_implementation="eager"
        )
        input_ids = torch.tensor([[101, 7592, 1010, 2026, 3899, 2003, 10140, 102]])
        with torch.no_grad():
            output = model(input_ids)[0]
        expected_shape = torch.Size([1, 8, 1024])
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor(
            [[[0.1775, 0.0083, -0.0321], [1.6002, 0.1287, 0.3912], [2.1473, 0.5791, 0.6066]]]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)


@require_torch
class BertGenerationDecoderIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head_absolute_embedding(self):
        model = BertGenerationDecoder.from_pretrained(
            "google/bert_for_seq_generation_L-24_bbc_encoder", attn_implementation="eager"
        )
        input_ids = torch.tensor([[101, 7592, 1010, 2026, 3899, 2003, 10140, 102]])
        with torch.no_grad():
            output = model(input_ids)[0]
        expected_shape = torch.Size([1, 8, 50358])
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor(
            [[[-0.5788, -2.5994, -3.7054], [0.0438, 4.7997, 1.8795], [1.5862, 6.6409, 4.4638]]]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
