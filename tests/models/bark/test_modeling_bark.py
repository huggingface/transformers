# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Bark model. """


import copy
import inspect
import tempfile
import unittest

from transformers import (
    BarkCoarseAcousticsConfig,
    BarkFineAcousticsConfig,
    BarkSemanticConfig,
    is_torch_available,
)
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import cached_property

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        BarkCoarseAcousticsModule,
        BarkFineAcousticsModule,
        BarkModel,
        BarkProcessor,
        BarkSemanticModule,
    )


class BarkModuleTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=4,
        is_training=False,  # for now training is not supported
        use_input_mask=True,
        use_labels=True,
        vocab_size=33,
        output_vocab_size=33,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=15,
        dropout=0.1,
        window_size=256,
        initializer_range=0.02,
        n_codes_total=8,  # for BarkFineAcousticsModel
        n_codes_given=1,  # for BarkFineAcousticsModel
        config_class=None,
        model_class=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.window_size = window_size
        self.initializer_range = initializer_range
        self.bos_token_id = output_vocab_size - 1
        self.eos_token_id = output_vocab_size - 1
        self.pad_token_id = output_vocab_size - 1

        self.n_codes_total = n_codes_total
        self.n_codes_given = n_codes_given

        self.is_encoder_decoder = False
        self.config_class = config_class
        self.model_class = model_class

    def get_large_model_config(self):
        return self.config_class.from_pretrained("ylacombe/bark-large")

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        head_mask = ids_tensor([self.num_hidden_layers, self.num_attention_heads], 2)

        inputs_dict = {
            "input_ids": input_ids,
            "head_mask": head_mask,
            "attention_mask": input_mask,
        }

        return (
            config,
            inputs_dict,
        )

    def get_config(self):
        return self.config_class(
            vocab_size=self.vocab_size,
            output_vocab_size=self.output_vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            window_size=self.window_size,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = self.model_class(config=config).to(torch_device).eval()

        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]
        head_mask = inputs_dict["head_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_attn_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["logits"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "logits"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

        # test no attention_mask works
        outputs = model(input_ids, use_cache=True)
        _, past_key_values = outputs.to_tuple()
        output_from_no_past = model(next_input_ids)["logits"]

        output_from_past = model(next_tokens, past_key_values=past_key_values)["logits"]

        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()
        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))


class BarkFineAcousticsModuleTester(BarkModuleTester):
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length, self.n_codes_total], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        head_mask = ids_tensor([self.num_hidden_layers, self.num_attention_heads], 2)

        # randint between self.n_codes_given - 1 and self.n_codes_total - 1
        codebook_idx = ids_tensor((1,), self.n_codes_total - self.n_codes_given).item() + self.n_codes_given

        inputs_dict = {
            "codebook_idx": codebook_idx,
            "input_ids": input_ids,
            "head_mask": head_mask,
            "attention_mask": input_mask,
        }

        return (
            config,
            inputs_dict,
        )


@require_torch
class BarkSemanticModuleTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (BarkSemanticModule,) if is_torch_available() else ()
    all_generative_model_classes = (BarkSemanticModule,) if is_torch_available() else ()

    is_encoder_decoder = False
    fx_compatible = False
    test_missing_keys = False
    test_pruning = False
    test_model_parallel = False
    # no model_parallel for now

    def setUp(self):
        self.model_tester = BarkModuleTester(self, config_class=BarkSemanticConfig, model_class=BarkSemanticModule)
        self.config_tester = ConfigTester(self, config_class=BarkSemanticConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
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
                inputs["input_embeds"] = wte(input_ids)
            else:
                inputs["input_embeds"] = wte(encoder_input_ids)
                inputs["decoder_input_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = self.all_model_classes[0](config).eval().to(torch_device)
        if torch_device == "cuda":
            model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)


@require_torch
class BarkCoarseAcousticsModuleTest(BarkSemanticModuleTest):
    # Same tester as BarkSemanticModuleTest, except for model_class and config_class
    all_model_classes = (BarkCoarseAcousticsModule,) if is_torch_available() else ()
    all_generative_model_classes = (BarkCoarseAcousticsModule,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = BarkModuleTester(
            self, config_class=BarkCoarseAcousticsConfig, model_class=BarkCoarseAcousticsModule
        )
        self.config_tester = ConfigTester(self, config_class=BarkCoarseAcousticsConfig, n_embd=37)


@require_torch
class BarkFineAcousticsModuleTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (BarkFineAcousticsModule,) if is_torch_available() else ()

    is_encoder_decoder = False
    fx_compatible = False
    test_missing_keys = False
    test_pruning = False
    # no model_parallel for now
    test_model_parallel = False
    
    # torchscript disabled for now because forward with an int
    test_torchscript = False

    def setUp(self):
        self.model_tester = BarkFineAcousticsModuleTester(
            self, config_class=BarkFineAcousticsConfig, model_class=BarkFineAcousticsModule
        )
        self.config_tester = ConfigTester(self, config_class=BarkFineAcousticsConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]

            wte = model.get_input_embeddings()[inputs_dict["codebook_idx"]]

            inputs["input_embeds"] = wte(input_ids[:, :, inputs_dict["codebook_idx"]])

            with torch.no_grad():
                model(**inputs)[0]

    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_ids = input_dict["input_ids"]
        # take first codebook channel
        attention_mask = input_ids[:, :, 0].ne(1).to(torch_device)
        model = self.all_model_classes[0](config).eval().to(torch_device)
        codebook_idx = 4
        if torch_device == "cuda":
            model.half()
        model(codebook_idx, input_ids, attention_mask=attention_mask)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["codebook_idx", "input_ids"]
            self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_model_common_attributes(self):
        # one embedding layer per codebook
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings()[0], (torch.nn.Embedding))
            model.set_input_embeddings(
                torch.nn.ModuleList([torch.nn.Embedding(10, 10) for _ in range(config.n_codes_total)])
            )
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x[0], torch.nn.Linear))

    def test_resize_tokens_embeddings(self):
        # resizing tokens_embeddings of a ModuleList
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed_list = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings_list = [model_embed.weight.clone() for model_embed in model_embed_list]

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed_list = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)

            # Check that it actually resizes the embeddings matrix for each codebook
            for model_embed, cloned_embeddings in zip(model_embed_list, cloned_embeddings_list):
                self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed_list = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            for model_embed, cloned_embeddings in zip(model_embed_list, cloned_embeddings_list):
                self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)

            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            # only check for the first embedding matrix
            models_equal = True
            for p1, p2 in zip(cloned_embeddings_list[0], model_embed_list[0].weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_resize_embeddings_untied(self):
        # resizing tokens_embeddings of a ModuleList
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        original_config.tie_word_embeddings = False

        # if model cannot untied embeddings -> leave test
        if original_config.tie_word_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)

            # if no output embeddings -> leave test
            if model.get_output_embeddings() is None:
                continue

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            output_embeds_list = model.get_output_embeddings()

            for output_embeds in output_embeds_list:
                self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)

                # Check bias if present
                if output_embeds.bias is not None:
                    self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds_list = model.get_output_embeddings()

            for output_embeds in output_embeds_list:
                self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
                # Check bias if present
                if output_embeds.bias is not None:
                    self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))


@require_torch
class BarkModelTest(unittest.TestCase):
    @cached_property
    def model(self):
        return BarkModel.from_pretrained("ylacombe/bark-large").to(torch_device)

    @cached_property
    def processor(self):
        return BarkProcessor(repo_id="ylacombe/bark-large", subfolder="speaker_embeddings")

    @cached_property
    def inputs(self):
        input_ids, history_prompt = self.processor(
            "In the light of the moon, a little egg lay on a leaf", voice_preset="en_speaker_6"
        )

        input_ids = {key: input_ids[key].to(torch_device) for key in input_ids}

        return input_ids, history_prompt

    @slow
    def test_generate_text_semantic(self):
        input_ids, history_prompt = self.inputs

        expected_output_ids = [
            7363,
            321,
            41,
            1461,
            6915,
            952,
            326,
            41,
            41,
            927,
            4335,
            2523,
            2523,
            1022,
            1723,
            163,
            17,
            298,
            8531,
            5819,
            521,
            245,
            6821,
            4228,
            41,
            41,
            41,
            41,
            373,
            711,
            4480,
            4480,
            4236,
            4236,
            4236,
            407,
            407,
            407,
            3734,
            627,
            171,
            171,
            171,
            138,
            10,
            138,
            266,
            276,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            3208,
            397,
            215,
            215,
            1293,
            4004,
            4004,
            206,
            206,
            206,
            206,
            6992,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
            206,
        ]

        # greedy decoding
        output_ids = self.model.generate_text_semantic(**input_ids, history_prompt=history_prompt, do_sample=False)

        # transformers' implementation of the Bark algorithm pads by default to the maximum length possible
        # since it will be padded anyways in generate_coarse.
        self.assertListEqual(output_ids[0, : len(expected_output_ids)].tolist(), expected_output_ids)

    @slow
    def test_generate_coarse(self):
        input_ids, history_prompt = self.inputs

        first_three_hundred_expected_output_ids = [
            11018,
            11391,
            10651,
            11418,
            10857,
            11620,
            10642,
            11366,
            10312,
            11528,
            10531,
            11516,
            10474,
            11051,
            10524,
            11051,
            10524,
            11846,
            10776,
            11846,
            10784,
            11444,
            10868,
            11444,
            10776,
            11846,
            10689,
            11846,
            10990,
            11846,
            10396,
            11846,
            10065,
            11051,
            10206,
            11163,
            10206,
            11288,
            10206,
            11025,
            10206,
            11025,
            10206,
            11484,
            10479,
            11551,
            10479,
            11251,
            10502,
            11484,
            10687,
            11484,
            10479,
            11151,
            10206,
            11235,
            10361,
            11371,
            10573,
            11391,
            10699,
            11675,
            10136,
            12010,
            10651,
            11855,
            10573,
            11884,
            10474,
            11858,
            10431,
            12038,
            10255,
            11717,
            10687,
            11441,
            10143,
            11130,
            10605,
            11130,
            10312,
            11130,
            10124,
            11130,
            10069,
            11130,
            10562,
            11444,
            10010,
            11130,
            10659,
            12038,
            10721,
            11130,
            10790,
            11130,
            10361,
            11130,
            10431,
            11130,
            10322,
            11130,
            10721,
            11130,
            10790,
            11130,
            10361,
            11130,
            10431,
            11130,
            10659,
            11130,
            10790,
            11130,
            10361,
            11130,
            10431,
            11130,
            10531,
            11878,
            10651,
            11696,
            10058,
            11553,
            10238,
            11553,
            10069,
            11553,
            10687,
            11238,
            10224,
            11588,
            10738,
            11387,
            10835,
            11301,
            10779,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
            10025,
            11301,
        ]

        output_ids = self.model.generate_text_semantic(**input_ids, history_prompt=history_prompt, do_sample=False)

        output_ids = self.model.generate_coarse(
            output_ids, history_prompt=history_prompt, max_coarse_history=630, sliding_window_len=60, do_sample=False
        )

        self.assertListEqual(output_ids[0, :300].tolist(), first_three_hundred_expected_output_ids)

    @slow
    def test_generate_fine(self):
        input_ids, history_prompt = self.inputs

        first_hundred_expected_output_ids = [
            [
                1018,
                651,
                857,
                642,
                312,
                531,
                474,
                524,
                524,
                776,
                784,
                868,
                776,
                689,
                990,
                396,
                65,
                206,
                206,
                206,
                206,
                206,
                479,
                479,
                502,
                687,
                479,
                206,
                361,
                573,
                699,
                136,
                651,
                573,
                474,
                431,
                255,
                687,
                143,
                605,
                312,
                124,
                69,
                562,
                10,
                659,
                721,
                790,
                361,
                431,
                322,
                721,
                790,
                361,
                431,
                659,
                790,
                361,
                431,
                531,
                651,
                58,
                238,
                69,
                687,
                224,
                738,
                835,
                779,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
            ],
            [
                367,
                394,
                596,
                342,
                504,
                492,
                27,
                27,
                822,
                822,
                420,
                420,
                822,
                822,
                822,
                822,
                27,
                139,
                264,
                1,
                1,
                460,
                527,
                227,
                460,
                460,
                127,
                211,
                347,
                367,
                651,
                986,
                831,
                860,
                834,
                1014,
                693,
                417,
                106,
                106,
                106,
                106,
                106,
                420,
                106,
                1014,
                106,
                106,
                106,
                106,
                106,
                106,
                106,
                106,
                106,
                106,
                106,
                106,
                106,
                854,
                672,
                529,
                529,
                529,
                214,
                564,
                363,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
                277,
            ],
            [
                961,
                955,
                221,
                955,
                955,
                686,
                939,
                939,
                479,
                176,
                464,
                644,
                105,
                43,
                43,
                883,
                31,
                687,
                838,
                955,
                532,
                654,
                333,
                883,
                830,
                581,
                631,
                201,
                862,
                4,
                626,
                126,
                915,
                33,
                891,
                657,
                565,
                308,
                935,
                862,
                164,
                933,
                945,
                358,
                977,
                42,
                696,
                707,
                361,
                52,
                883,
                769,
                812,
                418,
                456,
                769,
                361,
                361,
                935,
                769,
                221,
                232,
                560,
                918,
                925,
                915,
                989,
                752,
                752,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
                893,
            ],
            [
                638,
                365,
                218,
                944,
                853,
                363,
                639,
                22,
                884,
                456,
                990,
                436,
                452,
                791,
                662,
                3,
                880,
                292,
                537,
                970,
                12,
                68,
                223,
                350,
                328,
                840,
                282,
                506,
                424,
                688,
                674,
                766,
                739,
                881,
                139,
                12,
                335,
                519,
                497,
                459,
                77,
                766,
                1003,
                151,
                1004,
                67,
                499,
                456,
                63,
                315,
                140,
                536,
                456,
                791,
                139,
                579,
                644,
                990,
                261,
                162,
                803,
                811,
                974,
                674,
                770,
                651,
                74,
                854,
                916,
                762,
                1014,
                1014,
                1014,
                1014,
                1014,
                1014,
                1014,
                1014,
                741,
                741,
                741,
                741,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
                255,
            ],
            [
                302,
                912,
                524,
                38,
                174,
                209,
                879,
                23,
                910,
                227,
                58,
                250,
                910,
                152,
                583,
                62,
                307,
                890,
                547,
                418,
                215,
                816,
                620,
                285,
                171,
                475,
                869,
                134,
                615,
                812,
                714,
                524,
                990,
                435,
                136,
                302,
                43,
                307,
                15,
                663,
                91,
                673,
                930,
                317,
                584,
                913,
                384,
                987,
                988,
                942,
                594,
                910,
                52,
                913,
                392,
                624,
                700,
                988,
                292,
                574,
                325,
                968,
                1016,
                890,
                742,
                862,
                528,
                862,
                336,
                983,
                712,
                712,
                712,
                712,
                712,
                712,
                712,
                712,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
                622,
            ],
            [
                440,
                673,
                861,
                666,
                372,
                558,
                49,
                172,
                232,
                342,
                758,
                710,
                570,
                245,
                918,
                570,
                211,
                659,
                999,
                125,
                1016,
                734,
                896,
                570,
                897,
                897,
                801,
                694,
                852,
                673,
                282,
                1005,
                505,
                683,
                770,
                726,
                781,
                966,
                349,
                373,
                773,
                640,
                913,
                219,
                134,
                657,
                34,
                783,
                876,
                294,
                978,
                265,
                177,
                237,
                572,
                57,
                125,
                31,
                675,
                57,
                673,
                238,
                434,
                673,
                218,
                435,
                881,
                293,
                781,
                881,
                435,
                41,
                435,
                435,
                435,
                435,
                435,
                435,
                41,
                41,
                41,
                293,
                881,
                881,
                881,
                881,
                881,
                881,
                881,
                881,
                881,
                881,
                881,
                881,
                881,
                881,
                881,
                881,
                881,
                881,
            ],
            [
                244,
                358,
                123,
                356,
                586,
                520,
                499,
                877,
                542,
                637,
                533,
                229,
                873,
                533,
                921,
                745,
                79,
                798,
                511,
                978,
                599,
                407,
                791,
                669,
                499,
                360,
                664,
                637,
                894,
                332,
                714,
                78,
                772,
                422,
                427,
                223,
                67,
                585,
                938,
                817,
                684,
                749,
                874,
                1019,
                134,
                749,
                468,
                110,
                421,
                656,
                912,
                979,
                561,
                561,
                676,
                28,
                481,
                741,
                325,
                569,
                354,
                935,
                342,
                127,
                585,
                764,
                782,
                853,
                782,
                860,
                772,
                772,
                772,
                355,
                355,
                355,
                355,
                355,
                782,
                782,
                355,
                782,
                782,
                1002,
                1002,
                1002,
                1002,
                1002,
                1002,
                1002,
                1002,
                1002,
                1002,
                1002,
                1002,
                1002,
                1002,
                1002,
                1002,
                1002,
            ],
            [
                806,
                685,
                905,
                848,
                803,
                810,
                921,
                208,
                625,
                203,
                595,
                850,
                840,
                499,
                653,
                999,
                254,
                361,
                996,
                901,
                433,
                803,
                602,
                982,
                504,
                214,
                229,
                718,
                18,
                371,
                534,
                534,
                1012,
                534,
                480,
                526,
                416,
                458,
                624,
                937,
                1015,
                495,
                1008,
                838,
                542,
                506,
                181,
                110,
                208,
                821,
                437,
                926,
                939,
                443,
                848,
                780,
                746,
                810,
                625,
                966,
                658,
                677,
                919,
                468,
                832,
                534,
                534,
                701,
                835,
                534,
                948,
                948,
                948,
                948,
                989,
                989,
                989,
                989,
                975,
                518,
                475,
                975,
                948,
                475,
                475,
                475,
                475,
                475,
                475,
                475,
                475,
                475,
                475,
                475,
                475,
                475,
                475,
                475,
                475,
                475,
            ],
        ]

        output_ids = self.model.generate_text_semantic(**input_ids, history_prompt=history_prompt, do_sample=False)

        output_ids = self.model.generate_coarse(
            output_ids, history_prompt=history_prompt, max_coarse_history=630, sliding_window_len=60, do_sample=False
        )

        # greedy decoding
        output_ids = self.model.generate_fine(
            output_ids,
            history_prompt=history_prompt,
            temperature=None,
        )

        self.assertListEqual(output_ids[0, :, :100].tolist(), first_hundred_expected_output_ids)

    @slow
    def test_generate(self):
        input_ids, history_prompt = self.inputs

        self.model.generate_audio(**input_ids, history_prompt=None)
        self.model.generate_audio(**input_ids, history_prompt=history_prompt)
        self.model.generate_audio(**input_ids, history_prompt=None, do_sample=True, temperature=0.6)
        self.model.generate_audio(**input_ids, history_prompt=None, do_sample=True, temperature=0.6, num_beams=4)
        self.model.generate_audio(**input_ids, history_prompt=None, do_sample=True, temperature=0.6, penalty_alpha=0.6)
        self.model.generate_audio(**input_ids, history_prompt=None, do_sample=True, temperature=0.6, top_p=0.6)


#
#    @slow
#    def test_lm_generate_bark(self):
#        for checkpointing in [True, False]:
#            model = self.model
#            if checkpointing:
#                model.gradient_checkpointing_enable()
#            else:
#                model.gradient_checkpointing_disable()
#            input_ids = torch.tensor([[464, 3290]], dtype=torch.long, device=torch_device)  # The dog
#            # fmt: off
#            # The dog-eared copy of the book, which is a collection of essays by the late author,
#            expected_output_ids = [464, 3290, 12, 3380, 4866, 286, 262, 1492, 11, 543, 318, 257, 4947, 286, 27126, 416, 262, 2739, 1772, 11]
#            # fmt: on
#            output_ids = model.generate(input_ids, do_sample=False)
#            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
#
#    @slow
#    def test_bark_sample(self):
#        model = self.model
#        tokenizer = self.tokenizer
#
#        torch.manual_seed(0)
#        tokenized = tokenizer("Today is a nice day and", return_tensors="pt", return_token_type_ids=True)
#        input_ids = tokenized.input_ids.to(torch_device)
#        output_ids = model.generate(input_ids, do_sample=True)
#        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#
#        EXPECTED_OUTPUT_STR = "Today is a nice day and if you don’t get the memo here is what you can"
#        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)
#
#    @slow
#    def test_batch_generation(self):
#        model = self.model
#        tokenizer = self.tokenizer
#
#        tokenizer.padding_side = "left"
#
#        # Define PAD Token = EOS Token = 50256
#        tokenizer.pad_token = tokenizer.eos_token
#        model.config.pad_token_id = model.config.eos_token_id
#
#        # use different length sentences to test batching
#        sentences = [
#            "Hello, my dog is a little",
#            "Today, I am",
#        ]
#
#        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
#        input_ids = inputs["input_ids"].to(torch_device)
#
#        outputs = model.generate(
#            input_ids=input_ids,
#            attention_mask=inputs["attention_mask"].to(torch_device),
#        )
#
#        inputs_non_padded = tokenizer(sentences[0], return_tensors="pt").input_ids.to(torch_device)
#        output_non_padded = model.generate(input_ids=inputs_non_padded)
#
#        num_paddings = inputs_non_padded.shape[-1] - inputs["attention_mask"][-1].long().sum().cpu().item()
#        inputs_padded = tokenizer(sentences[1], return_tensors="pt").input_ids.to(torch_device)
#        output_padded = model.generate(input_ids=inputs_padded, max_length=model.config.max_length - num_paddings)
#
#        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
#        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)
#
#        expected_output_sentence = [
#            "Hello, my dog is a little bit of a kitty. She is a very sweet and loving",
#            "Today, I am going to talk about the best way to get a job in the",
#        ]
#        self.assertListEqual(expected_output_sentence, batch_out_sentence)
#        self.assertListEqual(expected_output_sentence, [non_padded_sentence, padded_sentence])
#
#    @slow
#    def test_model_from_pretrained(self):
#        for model_name in BARK_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
#            model = BarkModel.from_pretrained(model_name)
#            self.assertIsNotNone(model)
