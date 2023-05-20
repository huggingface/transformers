# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch MplugOwl model. """


import inspect
import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_torch_multi_gpu,
    slow,
    torch_device,
)
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import MplugOwlConfig, MplugOwlForConditionalGeneration, LlamaTokenizer


@require_torch
class MplugOwlModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        num_query_tokens=10,
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
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(
            3,
        )
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        config = MplugOwlConfig(
            vision_config={
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "projection_dim": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
            },
            visual_abstractor_config={
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "projection_dim": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
            },
            text_config={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "dropout": self.hidden_dropout_prob,
                "attention_dropout": self.attention_probs_dropout_prob,
                "max_position_embeddings": self.max_position_embeddings,
                "eos_token_id": self.eos_token_id,
                "bos_token_id": self.bos_token_id,
                "pad_token_id": self.pad_token_id,
            },
        )
        mask_shape = (input_ids.shape[0], input_ids.shape[1] - 1)
        inputs_dict = {
            "input_ids": input_ids.long(),
            "labels": input_ids.clone(),
            "pixel_values": None,
            "num_images": torch.zeros(self.batch_size, device=input_ids.device).long(),
            "non_padding_mask": torch.ones(*mask_shape, device=input_ids.device).long(),
            "non_media_mask": torch.ones(*mask_shape, device=input_ids.device).long(),
            "prompt_mask": torch.ones(*mask_shape, device=input_ids.device).long(),
            "attention_mask": torch.ones(*(mask_shape[0], mask_shape[1] + 1), device=input_ids.device).long(),
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = MplugOwlForConditionalGeneration(config=config).language_model.to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_attn_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask, output_hidden_states=True)[
            "hidden_states"
        ][-1]
        output_from_past = model(
            next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values, output_hidden_states=True
        )["hidden_states"][-1]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-2))


@require_torch
class MplugOwlModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (MplugOwlForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (MplugOwlForConditionalGeneration,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    has_attentions = False

    def setUp(self):
        self.model_tester = MplugOwlModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MplugOwlConfig)

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        common_properties = ["vision_config", "visual_abstractor_config", "text_config"]

        # Add common fields for text models
        if self.has_text_modality:
            common_properties.extend(["vocab_size"])

        # Test that config has the common properties as getters
        for prop in common_properties:
            self.parent.assertTrue(hasattr(config, prop), msg=f"`{prop}` does not exist")

        # Test that config has the common properties as setter
        for idx, name in enumerate(common_properties):
            try:
                setattr(config, name, idx)
                self.parent.assertEqual(
                    getattr(config, name), idx, msg=f"`{name} value {idx} expected, but was {getattr(config, name)}"
                )
            except NotImplementedError:
                # Some models might not be able to implement setters for common_properties
                # In that case, a NotImplementedError is raised
                pass

        # Test if config class can be called with Config(prop_name=..)
        for idx, name in enumerate(common_properties):
            try:
                config = self.config_class(**{name: idx})
                self.parent.assertEqual(
                    getattr(config, name), idx, msg=f"`{name} value {idx} expected, but was {getattr(config, name)}"
                )
            except NotImplementedError:
                # Some models might not be able to implement setters for common_properties
                # In that case, a NotImplementedError is raised
                pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            if model.config.is_encoder_decoder:
                expected_arg_names = [
                    "input_ids",
                ]
                # expected_arg_names.extend(
                #     ["head_mask", "decoder_head_mask", "cross_attn_head_mask", "encoder_outputs"]
                #     if "head_mask" and "decoder_head_mask" and "cross_attn_head_mask" in arg_names
                #     else ["encoder_outputs"]
                # )
                self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)
            else:
                expected_arg_names = ["input_ids", "pixel_values"]
                self.assertListEqual(arg_names[:2], expected_arg_names)

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_encoder_decoder_model_standalone(self):
        pass

    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = MplugOwlForConditionalGeneration(config).eval().to(torch_device)
        if torch_device == "cuda":
            model.half()
        model.generate(pixel_values=None, input_ids=input_ids, attention_mask=attention_mask)
        model.generate(
            pixel_values=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=2,
            do_sample=True,
            early_stopping=False,
            num_return_sequences=1,
        )

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="MplugOwlModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_determinism(self):
        pass

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_can_use_safetensors(
        self,
    ):
        pass

    @require_torch_multi_gpu
    def test_multi_gpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["head_mask", "decoder_head_mask", "cross_attn_head_mask"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to cuda:O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                inputs_dict[k] = v.to(0)

        for model_class in self.all_model_classes:
            model = model_class(config=config).half()
            model.to(0)
            model.eval()

            # Wrap model in nn.DataParallel
            from torch import nn

            model = nn.DataParallel(model)
            with torch.no_grad():
                _ = model(**self._prepare_for_class(inputs_dict, model_class))

    @unittest.skip(reason="Use llama")
    def test_resize_tokens_embeddings(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        pass

    @unittest.skip(reason="Use llama")
    def test_resize_embeddings_untied(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        pass

    @unittest.skip(reason="Do not support")
    def test_disk_offload(self):
        pass


def assert_tensors_close(a, b, atol=1e-12, prefix=""):
    """If tensors have different shapes, different values or a and b are not both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        pct_different = (torch.gt((a - b).abs(), atol)).float().mean().item()
        if a.numel() > 100:
            msg = f"tensor values are {pct_different:.1%} percent different."
        else:
            msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device)





@require_torch
@require_sentencepiece
@require_tokenizers
@slow
class MplugOwlModelIntegrationTests(unittest.TestCase):
    TOLERANCE = 1e-4
    
    @cached_property
    def default_tokenizer(self):
        return LlamaTokenizer.from_pretrained("MAGAer13/mplug-owl-llama-7b")

    def test_inference_head(self):
        model = MplugOwlForConditionalGeneration.from_pretrained("MAGAer13/mplug-owl-llama-7b").to(torch_device)
        # change to intended input
        input_ids = _long_tensor([[1, 2, 232, 328, 740, 1140, 3, 69, 4, 5, 2]])
        inputs_dict = {
            "input_ids": input_ids,
        }
        with torch.no_grad():
            output = model(**inputs_dict)[0]
        expected_shape = torch.Size((1, 11, model.config.vocab_size))
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = torch.tensor(
            [[-6.0774, -14.2518, 0.9577], [-5.3115, -12.1374, 0.3657], [-5.7249, -14.9420, 0.9261]],
            device=torch_device,
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=TOLERANCE))

    def test_seq_to_seq_generation(self):
        hf = MplugOwlForConditionalGeneration.from_pretrained("MAGAer13/mplug-owl-llama-7b").to(torch_device)
        tok = LlamaTokenizer.from_pretrained("MAGAer13/mplug-owl-llama-7b")

        batch_input = [
            "Note that the methods of a pool should only ever be used by the process which created it.",
            "The multiprocessing package mostly replicates the API of the threading module.",
            "Start the process’s activity.",
            "Return the process ID. Before the process is spawned, this will be None.",
        ]

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        dct = tok.batch_encode_plus(
            batch_input,
            max_length=20,
            padding="max_length",
            truncation_strategy="only_first",
            truncation=True,
            return_tensors="pt",
        )

        hypotheses_batch = hf.generate(
            input_ids=dct["input_ids"].to(torch_device),
            attention_mask=dct["attention_mask"].to(torch_device),
            num_beams=2,
        )

        EXPECTED = [
            "r\nPool.create = function(size) {\n  var pool = new Pool(",
            ".\n\nThe `multiprocessing` module provides the `Process` class, which",
            '() method.\n        /// </summary>\n        /// <param name="activity">The',
            'pa_pid\n        """\n        return None\n\n    def wait(self):\n',
        ]

        generated = tok.batch_decode(
            hypotheses_batch.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        assert generated == EXPECTED

