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

import copy
import unittest

from transformers import MT5Config, is_torch_available
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        MT5EncoderModel,
        MT5ForConditionalGeneration,
        MT5ForQuestionAnswering,
        MT5ForSequenceClassification,
        MT5ForTokenClassification,
        MT5Model,
    )


class MT5ModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        decoder_seq_length=7,
        # For common tests
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        dropout_rate=0.1,
        initializer_factor=0.002,
        eos_token_id=1,
        pad_token_id=0,
        decoder_start_token_id=0,
        scope=None,
        decoder_layers=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.scope = None
        self.decoder_layers = decoder_layers

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size).clamp(2)
        input_ids[:, -1] = self.eos_token_id  # Eos Token
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        config = self.get_config()

        return (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def get_pipeline_config(self):
        return MT5Config(
            vocab_size=166,  # t5 forces 100 extra tokens
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def get_config(self):
        return MT5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def check_prepare_lm_labels_via_shift_left(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5Model(config=config)
        model.to(torch_device)
        model.eval()

        # make sure that lm_labels are correctly padded from the right
        lm_labels.masked_fill_((lm_labels == self.decoder_start_token_id), self.eos_token_id)

        # add casaul pad token mask
        triangular_mask = torch.tril(lm_labels.new_ones(lm_labels.shape)).logical_not()
        lm_labels.masked_fill_(triangular_mask, self.pad_token_id)
        decoder_input_ids = model._shift_right(lm_labels)

        for i, (decoder_input_ids_slice, lm_labels_slice) in enumerate(zip(decoder_input_ids, lm_labels)):
            # first item
            self.parent.assertEqual(decoder_input_ids_slice[0].item(), self.decoder_start_token_id)
            if i < decoder_input_ids_slice.shape[-1]:
                if i < decoder_input_ids.shape[-1] - 1:
                    # items before diagonal
                    self.parent.assertListEqual(
                        decoder_input_ids_slice[1 : i + 1].tolist(), lm_labels_slice[:i].tolist()
                    )
                # pad items after diagonal
                if i < decoder_input_ids.shape[-1] - 2:
                    self.parent.assertListEqual(
                        decoder_input_ids_slice[i + 2 :].tolist(), lm_labels_slice[i + 1 : -1].tolist()
                    )
            else:
                # all items after square
                self.parent.assertListEqual(decoder_input_ids_slice[1:].tolist(), lm_labels_slice[:-1].tolist())

    def create_and_check_model(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        result = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        decoder_output = result.last_hidden_state
        decoder_past = result.past_key_values
        encoder_output = result.encoder_last_hidden_state

        self.parent.assertEqual(encoder_output.size(), (self.batch_size, self.encoder_seq_length, self.hidden_size))
        self.parent.assertEqual(decoder_output.size(), (self.batch_size, self.decoder_seq_length, self.hidden_size))
        # There should be `num_layers` key value embeddings stored in decoder_past
        self.parent.assertEqual(len(decoder_past), config.num_layers)

    def create_and_check_with_lm_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5ForConditionalGeneration(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
        self.parent.assertEqual(len(outputs), 4)
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.decoder_seq_length, self.vocab_size))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_with_sequence_classification_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        labels = torch.tensor([1] * self.batch_size, dtype=torch.long, device=torch_device)
        model = MT5ForSequenceClassification(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=input_ids,
            labels=labels,
        )
        # self.parent.assertEqual(len(outputs), 4)
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_decoder_model_past(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5Model(config=config).get_decoder().to(torch_device).eval()
        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_decoder_model_attention_mask_past(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5Model(config=config).get_decoder()
        model.to(torch_device)
        model.eval()

        # create attention mask
        attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        half_seq_length = input_ids.shape[-1] // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        output, past_key_values = model(input_ids, attention_mask=attn_mask, use_cache=True).to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size).squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones((attn_mask.shape[0], 1), dtype=torch.long, device=torch_device)],
            dim=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values, attention_mask=attn_mask)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5Model(config=config).get_decoder().to(torch_device).eval()
        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_generate_with_past_key_values(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5ForConditionalGeneration(config=config).to(torch_device).eval()
        torch.manual_seed(0)
        output_without_past_cache = model.generate(
            input_ids[:1], num_beams=2, max_length=5, do_sample=True, use_cache=False
        )
        torch.manual_seed(0)
        output_with_past_cache = model.generate(input_ids[:1], num_beams=2, max_length=5, do_sample=True)
        self.parent.assertTrue(torch.all(output_with_past_cache == output_without_past_cache))

    def create_and_check_model_fp16_forward(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = MT5Model(config=config).to(torch_device).half().eval()
        output = model(input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
        self.parent.assertFalse(torch.isnan(output).any().item())

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict


@require_torch
class MT5ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (MT5Model, MT5ForConditionalGeneration, MT5ForSequenceClassification, MT5ForQuestionAnswering)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": MT5Model,
            "question-answering": MT5ForQuestionAnswering,
            "summarization": MT5ForConditionalGeneration,
            "text-classification": MT5ForSequenceClassification,
            "text2text-generation": MT5ForConditionalGeneration,
            "translation": MT5ForConditionalGeneration,
            "zero-shot": MT5ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    test_resize_embeddings = True
    is_encoder_decoder = True
    # The small MT5 model needs higher percentages for CPU/MP tests
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = MT5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MT5Config, d_model=37)

    # `QAPipelineTests` is not working well with slow tokenizers (for some models) and we don't want to touch the file
    # `src/transformers/data/processors/squad.py` (where this test fails for this model)
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        if tokenizer_name is None:
            return True
        if pipeline_test_case_name == "QAPipelineTests" and not tokenizer_name.endswith("Fast"):
            return True

        return False

    # overwrite because MT5 doesn't accept position ids as input and expects `decoder_input_ids`
    def test_custom_4d_attention_mask(self):
        for model_class in self.all_generative_model_classes:
            config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config).to(device=torch_device, dtype=torch.float32)

            (
                input_ids,
                _,
                input_ids_shared_prefix,
                mask_shared_prefix,
                _,
            ) = self._get_custom_4d_mask_test_data()

            logits = model.forward(
                decoder_input_ids=input_ids,
                input_ids=input_dict["input_ids"][:3],
            ).logits
            # logits.shape == torch.Size([3, 4, ...])

            logits_shared_prefix = model(
                input_ids=input_dict["input_ids"][:1],
                decoder_input_ids=input_ids_shared_prefix,
                decoder_attention_mask=mask_shared_prefix,
            )[0]
            # logits_shared_prefix.shape == torch.Size([1, 6, ...])

            out_last_tokens = logits[:, -1, :]  # last tokens in each batch line
            out_shared_prefix_last_tokens = logits_shared_prefix[0, -3:, :]  # last three tokens

            # comparing softmax-normalized logits:
            normalized_0 = F.softmax(out_last_tokens)
            normalized_1 = F.softmax(out_shared_prefix_last_tokens)
            torch.testing.assert_close(normalized_0, normalized_1, rtol=1e-3, atol=1e-4)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_tie_word_embeddings(self):
        # Force it to True, see https://github.com/huggingface/transformers/pull/43880
        config = MT5Config(tie_word_embeddings=False)
        self.assertTrue(config.tie_word_embeddings)

    def test_shift_right(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_prepare_lm_labels_via_shift_left(*config_and_inputs)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        if config_and_inputs[0].__class__.__name__ == "T" + "5Config":
            self.assertTrue(config_and_inputs[0].scale_decoder_outputs)
        self.model_tester.create_and_check_model(*config_and_inputs)

    # MT5ForSequenceClassification does not support inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (MT5Model, MT5ForConditionalGeneration, MT5ForQuestionAnswering):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    def test_config_and_model_silu_gated(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        config.feed_forward_proj = "gated-silu"
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_with_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_lm_head(*config_and_inputs)

    def test_with_sequence_classification_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_sequence_classification_head(*config_and_inputs)

    def test_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past(*config_and_inputs)

    def test_decoder_model_past_with_attn_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_attention_mask_past(*config_and_inputs)

    def test_decoder_model_past_with_3d_attn_mask(self):
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = self.model_tester.prepare_config_and_inputs()

        attention_mask = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.encoder_seq_length, self.model_tester.encoder_seq_length],
            vocab_size=2,
        )
        decoder_attention_mask = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.decoder_seq_length, self.model_tester.decoder_seq_length],
            vocab_size=2,
        )

        self.model_tester.create_and_check_decoder_model_attention_mask_past(
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_generate_with_past_key_values(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_generate_with_past_key_values(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Can't do half precision")
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/mt5-small"
        model = MT5Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @unittest.skip(reason="MT5 has no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass


# Copied from tests.models.t5.test_modeling_t5.T5EncoderOnlyModelTester with T5->MT5
class MT5EncoderOnlyModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        # For common tests
        use_attention_mask=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        is_training=False,
        dropout_rate=0.1,
        initializer_factor=0.002,
        is_encoder_decoder=False,
        eos_token_id=1,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        # For common tests
        self.seq_length = self.encoder_seq_length
        self.use_attention_mask = use_attention_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.scope = None
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        config = MT5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )

        return (
            config,
            input_ids,
            attention_mask,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        model = MT5EncoderModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        result = model(input_ids=input_ids)
        encoder_output = result.last_hidden_state

        self.parent.assertEqual(encoder_output.size(), (self.batch_size, self.encoder_seq_length, self.hidden_size))

    def create_and_check_model_fp16_forward(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        model = MT5EncoderModel(config=config).to(torch_device).half().eval()
        output = model(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        self.parent.assertFalse(torch.isnan(output).any().item())

    def create_and_check_with_token_classification_head(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        labels = torch.tensor([1] * self.seq_length * self.batch_size, dtype=torch.long, device=torch_device)
        model = MT5ForTokenClassification(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.seq_length, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


# Copied from tests.models.t5.test_modeling_t5.T5EncoderOnlyModelTest with T5->MT5
class MT5EncoderOnlyModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MT5EncoderModel, MT5ForTokenClassification) if is_torch_available() else ()

    test_resize_embeddings = False
    pipeline_model_mapping = (
        {
            "token-classification": MT5ForTokenClassification,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = MT5EncoderOnlyModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MT5Config, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Can't do half precision")
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    def test_with_token_classification_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_token_classification_head(*config_and_inputs)

    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        if tokenizer_name is None:
            return True

        # `MT5EncoderOnlyModelTest` is not working well with slow tokenizers (for some models) and we don't want to touch the file
        # `src/transformers/data/processors/squad.py` (where this test fails for this model)
        if pipeline_test_case_name == "TokenClassificationPipelineTests" and not tokenizer_name.endswith("Fast"):
            return True

        return False


@require_torch
@require_sentencepiece
@require_tokenizers
class MT5IntegrationTest(unittest.TestCase):
    @slow
    def test_small_integration_test(self):
        """
        For comparison run:
        >>> import t5  # pip install t5==0.7.1
        >>> from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary

        >>> path_to_mtf_small_mt5_checkpoint = '<fill_in>'
        >>> path_to_mtf_small_mt5_spm_model_path = '<fill_in>'
        >>> t5_model = t5.models.MtfModel(model_dir=path_to_mtf_small_mt5_checkpoint, batch_size=1, tpu=None)
        >>> vocab = SentencePieceVocabulary(path_to_mtf_small_mt5_spm_model_path)
        >>> score = t5_model.score(inputs=["Hello there"], targets=["Hi I am"], vocabulary=vocab)
        """

        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small", return_dict=True).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

        input_ids = tokenizer("Hello there", return_tensors="pt").input_ids
        labels = tokenizer("Hi I am", return_tensors="pt").input_ids

        loss = model(input_ids.to(torch_device), labels=labels.to(torch_device)).loss
        mtf_score = -(labels.shape[-1] * loss.item())

        EXPECTED_SCORE = -84.9127
        self.assertLess(abs(mtf_score - EXPECTED_SCORE), 2e-4)
