# coding=utf-8
# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch MBART model. """
import copy
import inspect
import unittest

from transformers import EncodecConfig, MusicgenConfig, MusicgenDecoderConfig, PretrainedConfig, T5Config
from transformers.generation import (
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
)
from transformers.testing_utils import is_torch_available, require_torch, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        InfNanRemoveLogitsProcessor,
        LogitsProcessorList,
        MusicgenForCausalLM,
        MusicgenForConditionalGeneration,
        MusicgenModel,  # TODO(SG): swap for automodel
    )


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
        if isinstance(getattr(configs_no_init, key, None), PretrainedConfig):
            no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
            setattr(configs_no_init, key, no_init_subconfig)
    return configs_no_init


def prepare_musicgen_decoder_inputs_dict(
    config,
    input_ids,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.reshape(-1, config.num_codebooks, input_ids.shape[-1])[:, 0, :]
        attention_mask = attention_mask.ne(config.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads, device=torch_device)
    if encoder_attention_mask is None and encoder_hidden_states is not None:
        encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2], device=torch_device)
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads, device=torch_device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "head_mask": head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class MusicgenDecoderTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=False,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=99,
        bos_token_id=99,
        num_codebooks=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
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
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.num_codebooks = num_codebooks

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size * self.num_codebooks, self.seq_length], self.vocab_size)
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])

        config = self.get_config()
        inputs_dict = prepare_musicgen_decoder_inputs_dict(
            config, input_ids, encoder_hidden_states=encoder_hidden_states
        )
        return config, inputs_dict

    def get_config(self):
        config = MusicgenDecoderConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            d_ff=self.intermediate_size,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.bos_token_id,
            bos_token_id=self.bos_token_id,
            num_codebooks=self.num_codebooks,
            tie_word_embeddings=False,
        )
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
class MusicgenDecoderTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MusicgenModel, MusicgenForCausalLM) if is_torch_available() else ()
    generative_model_classes = (MusicgenForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "text-generation": MusicgenForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = MusicgenDecoderTester(self)
        self.config_tester = ConfigTester(self, config_class=MusicgenDecoderConfig, hidden_size=16)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]

            embed_tokens = model.get_input_embeddings()

            input_ids = input_ids.reshape(-1, config.num_codebooks, input_ids.shape[-1])

            inputs["inputs_embeds"] = sum(
                [embed_tokens[codebook](input_ids[:, codebook]) for codebook in range(config.num_codebooks)]
            )

            with torch.no_grad():
                model(**inputs)[0]

    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            first_embed = model.get_input_embeddings()[0]
            self.assertIsInstance(first_embed, torch.nn.Embedding)
            lm_heads = model.get_output_embeddings()
            self.assertTrue(lm_heads is None or isinstance(lm_heads[0], torch.nn.Linear))

    # skip as this model doesn't support all arguments tested
    def test_model_outputs_equivalence(self):
        pass

    # skip as this model has multiple inputs embeds and lm heads that should not be tied
    def test_tie_model_weights(self):
        pass

    # skip as this model has multiple inputs embeds and lm heads that should not be tied
    def test_tied_weights_keys(self):
        pass

    def _get_input_ids_and_config(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["input_ids"]

        # take max batch_size
        sequence_length = input_ids.shape[-1]
        input_ids = input_ids[: batch_size * config.num_codebooks, :]

        # generate max 3 tokens
        max_length = input_ids.shape[-1] + 3
        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
        return config, input_ids, attention_mask, max_length

    @staticmethod
    def _get_logits_processor_and_kwargs(
        input_length,
        eos_token_id,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        max_length=None,
        diversity_penalty=None,
    ):
        process_kwargs = {
            "min_length": input_length + 1 if max_length is None else max_length - 1,
        }
        logits_processor = LogitsProcessorList()
        return process_kwargs, logits_processor

    def test_greedy_generate_dict_outputs(self):
        for model_class in self.generative_model_classes:
            # disable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            config.use_cache = False
            model = model_class(config).to(torch_device).eval()
            output_greedy, output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertIsInstance(output_greedy, GreedySearchDecoderOnlyOutput)
            self.assertIsInstance(output_generate, GreedySearchDecoderOnlyOutput)

            self.assertNotIn(config.pad_token_id, output_generate)

    def test_greedy_generate_dict_outputs_use_cache(self):
        for model_class in self.generative_model_classes:
            # enable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            config.use_cache = True
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            output_greedy, output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertIsInstance(output_greedy, GreedySearchDecoderOnlyOutput)
            self.assertIsInstance(output_generate, GreedySearchDecoderOnlyOutput)

    def test_sample_generate(self):
        for model_class in self.generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            model = model_class(config).to(torch_device).eval()

            process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                model.config.eos_token_id,
                forced_bos_token_id=model.config.forced_bos_token_id,
                forced_eos_token_id=model.config.forced_eos_token_id,
                max_length=max_length,
            )
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=2)

            # check `generate()` and `sample()` are equal
            output_sample, output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
                process_kwargs=process_kwargs,
            )
            self.assertIsInstance(output_sample, torch.Tensor)
            self.assertIsInstance(output_generate, torch.Tensor)

    def test_sample_generate_dict_output(self):
        for model_class in self.generative_model_classes:
            # disable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            config.use_cache = False
            model = model_class(config).to(torch_device).eval()

            process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                model.config.eos_token_id,
                forced_bos_token_id=model.config.forced_bos_token_id,
                forced_eos_token_id=model.config.forced_eos_token_id,
                max_length=max_length,
            )
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=1)

            output_sample, output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
                process_kwargs=process_kwargs,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertIsInstance(output_sample, SampleDecoderOnlyOutput)
            self.assertIsInstance(output_generate, SampleDecoderOnlyOutput)


def prepare_musicgen_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if decoder_attention_mask is None:
        decoder_attention_mask = decoder_input_ids.reshape(
            -1, config.decoder.num_codebooks, decoder_input_ids.shape[-1]
        )[:, 0, :]
        decoder_attention_mask = decoder_attention_mask.ne(config.decoder.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(
            config.text_encoder.num_hidden_layers, config.text_encoder.num_attention_heads, device=torch_device
        )
    if decoder_head_mask is None:
        decoder_head_mask = torch.ones(
            config.decoder.num_hidden_layers, config.decoder.num_attention_heads, device=torch_device
        )
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(
            config.decoder.num_hidden_layers, config.decoder.num_attention_heads, device=torch_device
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class MusicgenTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=False,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=99,
        bos_token_id=99,
        num_codebooks=4,
        num_filters=4,
        codebook_size=128,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
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
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.num_codebooks = num_codebooks
        self.num_filters = num_filters
        self.codebook_size = codebook_size

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size * self.num_codebooks, self.seq_length], self.vocab_size)

        config = self.get_config()
        inputs_dict = prepare_musicgen_inputs_dict(config, input_ids, decoder_input_ids=decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        text_encoder_config = T5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.intermediate_size,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
        )
        audio_encoder_config = EncodecConfig(
            hidden_size=self.vocab_size,
            compress=1,
            num_filters=self.num_filters,
            codebook_size=self.codebook_size,
            codebook_dim=self.vocab_size,
        )
        decoder_config = MusicgenDecoderConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.bos_token_id,
            bos_token_id=self.bos_token_id,
            num_codebooks=self.num_codebooks,
            tie_word_embeddings=False,
        )
        config = MusicgenConfig.from_sub_model_configs(text_encoder_config, audio_encoder_config, decoder_config)
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
class MusicgenTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MusicgenForConditionalGeneration,) if is_torch_available() else ()
    generative_model_classes = (MusicgenForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {}
    test_pruning = False  # training is not supported yet for MusicGen
    test_headmasking = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = MusicgenTester(self)

    def _check_output_with_attentions(self, outputs, config, input_ids, decoder_input_ids):
        text_encoder_config = config.text_encoder
        decoder_config = config.decoder

        encoder_attentions = outputs["encoder_attentions"]
        self.assertEqual(len(encoder_attentions), text_encoder_config.num_hidden_layers)

        self.assertEqual(
            encoder_attentions[0].shape[-3:],
            (text_encoder_config.num_attention_heads, input_ids.shape[-1], input_ids.shape[-1]),
        )

        decoder_attentions = outputs["decoder_attentions"]
        num_decoder_layers = decoder_config.num_hidden_layers
        self.assertEqual(len(decoder_attentions), num_decoder_layers)

        self.assertEqual(
            decoder_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, decoder_input_ids.shape[-1], decoder_input_ids.shape[-1]),
        )

        cross_attentions = outputs["cross_attentions"]
        self.assertEqual(len(cross_attentions), num_decoder_layers)

        cross_attention_input_seq_len = decoder_input_ids.shape[-1]
        self.assertEqual(
            cross_attentions[0].shape[-3:],
            (decoder_config.num_attention_heads, cross_attention_input_seq_len, input_ids.shape[-1]),
        )

    def check_musicgen_model_output_attentions(
        self,
        model_class,
        config,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=True,
                **kwargs,
            )
        self._check_output_with_attentions(outputs, config, input_ids, decoder_input_ids)

    def check_musicgen_model_output_attentions_from_config(
        self,
        model_class,
        config,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        # Similar to `check_encoder_decoder_model_output_attentions`, but with `output_attentions` triggered from the
        # config file. Contrarily to most models, changing the model's config won't work -- the defaults are loaded
        # from the inner models' configurations.
        config.output_attentions = True  # model config -> won't work]

        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                **kwargs,
            )
        self.assertTrue(
            all(key not in outputs for key in ["encoder_attentions", "decoder_attentions", "cross_attentions"])
        )
        config.text_encoder.output_attentions = True  # inner model config -> will work
        config.audio_encoder.output_attentions = True
        config.decoder.output_attentions = True

        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                **kwargs,
            )
        self._check_output_with_attentions(outputs, config, input_ids, decoder_input_ids)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            self.check_musicgen_model_output_attentions(model_class, config, **inputs_dict)
            self.check_musicgen_model_output_attentions_from_config(model_class, config, **inputs_dict)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "input_ids",
                "attention_mask",
                "input_values",
                "padding_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
            ]
            expected_arg_names.extend(
                ["head_mask", "decoder_head_mask", "cross_attn_head_mask", "encoder_outputs"]
                if "head_mask" and "decoder_head_mask" and "cross_attn_head_mask" in arg_names
                else ["encoder_outputs"]
            )
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_gradient_checkpointing_backward_compatibility(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            config.text_encoder.gradient_checkpointing = True
            config.audio_encoder.gradient_checkpointing = True
            config.decoder.gradient_checkpointing = True
            model = model_class(config)
            self.assertTrue(model.is_gradient_checkpointing)

    # skip as this model has multiple inputs embeds and lm heads that should not be tied
    def test_tie_model_weights(self):
        pass

    # skip as this model has multiple inputs embeds and lm heads that should not be tied
    def test_tied_model_weights_key_ignore(self):
        pass

    # skip as this model has multiple inputs embeds and lm heads that should not be tied
    def test_tied_weights_keys(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_encoder.output_hidden_states = True
        config.audio_encoder.output_hidden_states = True
        config.decoder.output_hidden_states = True

        config.text_encoder.output_attentions = True
        config.decoder.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_hidden_states.retain_grad()

        decoder_hidden_states = outputs.decoder_hidden_states[0]
        decoder_hidden_states.retain_grad()

        if self.has_attentions:
            encoder_attentions = outputs.encoder_attentions[0]
            encoder_attentions.retain_grad()

            decoder_attentions = outputs.decoder_attentions[0]
            decoder_attentions.retain_grad()

            cross_attentions = outputs.cross_attentions[0]
            cross_attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(decoder_hidden_states.grad)

        if self.has_attentions:
            self.assertIsNotNone(encoder_attentions.grad)
            self.assertIsNotNone(decoder_attentions.grad)
            self.assertIsNotNone(cross_attentions.grad)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states

            expected_num_layers = self.model_tester.num_hidden_layers + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.seq_length
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            hidden_states = outputs.decoder_hidden_states
            self.assertIsInstance(hidden_states, (list, tuple))
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.text_encoder.output_hidden_states = True
            config.audio_encoder.output_hidden_states = True
            config.decoder.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = ["conv"]
                ignore_init = ["lstm"]
                if param.requires_grad:
                    if any([x in name for x in uniform_init_parms]):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    elif not any([x in name for x in ignore_init]):
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def _get_input_ids_and_config(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["input_ids"]

        # take max batch_size
        sequence_length = input_ids.shape[-1]
        input_ids = input_ids[:batch_size, :]
        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)

        # generate max 3 tokens
        decoder_input_ids = inputs_dict["decoder_input_ids"]
        max_length = decoder_input_ids.shape[-1] + 3
        decoder_input_ids = decoder_input_ids[: batch_size * config.decoder.num_codebooks, :]
        return config, input_ids, attention_mask, decoder_input_ids, max_length

    def _greedy_generate(
        self,
        model,
        input_ids,
        attention_mask,
        decoder_input_ids,
        max_length,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
            input_ids.shape[-1],
            eos_token_id=model.config.eos_token_id,
            forced_bos_token_id=model.config.forced_bos_token_id,
            forced_eos_token_id=model.config.forced_eos_token_id,
            max_length=max_length,
        )

        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        output_generate = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_length=max_length,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **logits_process_kwargs,
            **model_kwargs,
        )

        encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
            model,
            input_ids,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        with torch.no_grad():
            model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
            output_greedy = model.greedy_search(
                decoder_input_ids,
                max_length=max_length,
                logits_processor=logits_processor,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                encoder_outputs=encoder_outputs,
                **model_kwargs,
            )
        return output_greedy, output_generate

    def _sample_generate(
        self,
        model,
        input_ids,
        attention_mask,
        decoder_input_ids,
        max_length,
        num_return_sequences,
        logits_processor,
        logits_warper,
        logits_warper_kwargs,
        process_kwargs,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        torch.manual_seed(0)
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        output_generate = model.generate(
            input_ids,
            do_sample=True,
            num_beams=1,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **logits_warper_kwargs,
            **process_kwargs,
            **model_kwargs,
        )

        torch.manual_seed(0)
        encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
            model,
            input_ids,
            attention_mask,
            num_interleave=num_return_sequences,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # prevent flaky generation test failures
        logits_processor.append(InfNanRemoveLogitsProcessor())

        with torch.no_grad():
            model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
            output_sample = model.sample(
                decoder_input_ids.repeat_interleave(num_return_sequences, dim=0),
                max_length=max_length,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                output_scores=output_scores,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                encoder_outputs=encoder_outputs,
                **model_kwargs,
            )

        return output_sample, output_generate

    @staticmethod
    def _get_logits_processor_and_kwargs(
        input_length,
        eos_token_id,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        max_length=None,
        diversity_penalty=None,
    ):
        process_kwargs = {
            "min_length": input_length + 1 if max_length is None else max_length - 1,
        }
        logits_processor = LogitsProcessorList()
        return process_kwargs, logits_processor

    def test_greedy_generate_dict_outputs(self):
        for model_class in self.generative_model_classes:
            # disable cache
            config, input_ids, attention_mask, decoder_input_ids, max_length = self._get_input_ids_and_config()
            config.use_cache = False
            model = model_class(config).to(torch_device).eval()
            output_greedy, output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertIsInstance(output_greedy, GreedySearchEncoderDecoderOutput)
            self.assertIsInstance(output_generate, GreedySearchEncoderDecoderOutput)

            self.assertNotIn(config.pad_token_id, output_generate)

    def test_greedy_generate_dict_outputs_use_cache(self):
        for model_class in self.generative_model_classes:
            # enable cache
            config, input_ids, attention_mask, decoder_input_ids, max_length = self._get_input_ids_and_config()

            config.use_cache = True
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            output_greedy, output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertIsInstance(output_greedy, GreedySearchEncoderDecoderOutput)
            self.assertIsInstance(output_generate, GreedySearchEncoderDecoderOutput)

    def test_sample_generate(self):
        for model_class in self.generative_model_classes:
            config, input_ids, attention_mask, decoder_input_ids, max_length = self._get_input_ids_and_config()
            model = model_class(config).to(torch_device).eval()

            process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                model.config.eos_token_id,
                forced_bos_token_id=model.config.forced_bos_token_id,
                forced_eos_token_id=model.config.forced_eos_token_id,
                max_length=max_length,
            )
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=2)

            # check `generate()` and `sample()` are equal
            output_sample, output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                num_return_sequences=1,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
                process_kwargs=process_kwargs,
            )
            self.assertIsInstance(output_sample, torch.Tensor)
            self.assertIsInstance(output_generate, torch.Tensor)

    def test_sample_generate_dict_output(self):
        for model_class in self.generative_model_classes:
            # disable cache
            config, input_ids, attention_mask, decoder_input_ids, max_length = self._get_input_ids_and_config()
            config.use_cache = False
            model = model_class(config).to(torch_device).eval()

            process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                model.config.eos_token_id,
                forced_bos_token_id=model.config.forced_bos_token_id,
                forced_eos_token_id=model.config.forced_eos_token_id,
                max_length=max_length,
            )
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=1)

            output_sample, output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                num_return_sequences=1,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
                process_kwargs=process_kwargs,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertIsInstance(output_sample, SampleEncoderDecoderOutput)
            self.assertIsInstance(output_generate, SampleEncoderDecoderOutput)

    def test_generate_without_input_ids(self):
        config, _, _, _, max_length = self._get_input_ids_and_config()

        # if no bos token id => cannot generate from None
        if config.bos_token_id is None:
            return

        for model_class in self.all_generative_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()

            output_ids_generate = model.generate(do_sample=False, max_length=max_length, remove_invalid_values=True)
            self.assertIsNotNone(output_ids_generate)

    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), torch.nn.Embedding)
            lm_heads = model.get_output_embeddings()
            self.assertTrue(lm_heads is None or isinstance(lm_heads[0], torch.nn.Linear))
