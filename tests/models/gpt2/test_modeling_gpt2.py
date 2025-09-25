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

import unittest

import pytest

from transformers import GPT2Config, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        GPT2DoubleHeadsModel,
        GPT2ForQuestionAnswering,
        GPT2ForSequenceClassification,
        GPT2ForTokenClassification,
        GPT2LMHeadModel,
        GPT2Model,
        GPT2Tokenizer,
    )


class GPT2ModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = GPT2Config
        base_model_class = GPT2Model
        causal_lm_class = GPT2LMHeadModel
        sequence_classification_class = GPT2ForSequenceClassification
        token_classification_class = GPT2ForTokenClassification
        question_answering_class = GPT2ForQuestionAnswering

    def __init__(
        self,
        parent,
        use_token_type_ids=True,
        num_choices=4,
        **kwargs,
    ):
        super().__init__(parent, use_token_type_ids=use_token_type_ids, **kwargs)
        self.num_choices = num_choices

    def prepare_config_and_inputs(
        self, extra_inputs=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        # Overwritten: `GPT2DoubleHeadsModel` uses extra inputs
        (config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels) = (
            super().prepare_config_and_inputs()
        )

        if extra_inputs:
            mc_token_ids = ids_tensor([self.batch_size, self.num_choices], self.seq_length)
            head_mask = ids_tensor([self.num_hidden_layers, self.num_attention_heads], 2)
            config_and_inputs = (
                config,
                input_ids,
                input_mask,
                head_mask,
                token_type_ids,
                mc_token_ids,
                sequence_labels,
                token_labels,
                choice_labels,
            )
        else:
            config_and_inputs = (
                config,
                input_ids,
                token_type_ids,
                input_mask,
                sequence_labels,
                token_labels,
                choice_labels,
            )

        config = self.get_config(
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
        )

        return config_and_inputs

    def get_config(self, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False):
        # Overwritten: `GPT2Config` has extra flags and we want to test them
        config = super().get_config()
        config.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        config.reorder_and_upcast_attn = reorder_and_upcast_attn
        return config

    def prepare_config_and_inputs_for_common(self):
        # Overwritten: we want `token_type_ids` as part of the common inputs
        config_and_inputs = self.prepare_config_and_inputs(extra_inputs=True)
        config, input_ids, _, head_mask, token_type_ids, _, _, _, _ = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "head_mask": head_mask}
        return config, inputs_dict

    def prepare_config_and_inputs_for_decoder(self):
        # Extra function: used in `encoder_decoder` tests
        (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            _,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs(extra_inputs=True)

        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )


@require_torch
class GPT2ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            GPT2Model,
            GPT2LMHeadModel,
            GPT2DoubleHeadsModel,
            GPT2ForQuestionAnswering,
            GPT2ForSequenceClassification,
            GPT2ForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GPT2Model,
            "question-answering": GPT2ForQuestionAnswering,
            "text-classification": GPT2ForSequenceClassification,
            "text-generation": GPT2LMHeadModel,
            "token-classification": GPT2ForTokenClassification,
            "zero-shot": GPT2ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    all_parallelizable_model_classes = (GPT2LMHeadModel, GPT2DoubleHeadsModel) if is_torch_available() else ()
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    test_missing_keys = False
    test_model_parallel = True
    model_tester_class = GPT2ModelTester

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # Overwritten: special case for DoubleHeads model
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "GPT2DoubleHeadsModel":
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.num_choices, self.model_tester.seq_length),
                    dtype=torch.long,
                    device=torch_device,
                )
                inputs_dict["input_ids"] = inputs_dict["labels"]
                inputs_dict["token_type_ids"] = inputs_dict["labels"]
                inputs_dict["mc_token_ids"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.num_choices),
                    dtype=torch.long,
                    device=torch_device,
                )
                inputs_dict["mc_labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
        return inputs_dict

    def test_gpt2_double_lm_head_model(self):
        # extra test: model-specific class
        config_and_inputs = self.model_tester.prepare_config_and_inputs(extra_inputs=True)
        config, input_ids, input_mask, _, token_type_ids, mc_token_ids, _, _, _ = config_and_inputs
        model = GPT2DoubleHeadsModel(config)
        model.to(torch_device)
        model.eval()

        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.model_tester.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.model_tester.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = (
            token_type_ids.unsqueeze(1).expand(-1, self.model_tester.num_choices, -1).contiguous()
        )

        inputs = {
            "input_ids": multiple_choice_inputs_ids,
            "mc_token_ids": mc_token_ids,
            "attention_mask": multiple_choice_input_mask,
            "token_type_ids": multiple_choice_token_type_ids,
            "labels": multiple_choice_inputs_ids,
        }

        result = model(**inputs)
        self.assertEqual(result.loss.shape, ())
        self.assertEqual(
            result.logits.shape,
            (
                self.model_tester.batch_size,
                self.model_tester.num_choices,
                self.model_tester.seq_length,
                self.model_tester.vocab_size,
            ),
        )
        self.assertEqual(result.mc_logits.shape, (self.model_tester.batch_size, self.model_tester.num_choices))

    def test_gpt2_scale_attn_by_inverse_layer_idx(self):
        # extra test: model-specific flag
        config_and_inputs = self.model_tester.prepare_config_and_inputs(scale_attn_by_inverse_layer_idx=True)
        config, input_ids, token_type_ids, _, _, _, _ = config_and_inputs

        model = GPT2LMHeadModel(config)
        model.to(torch_device)
        result = model(input_ids, token_type_ids=token_type_ids, labels=input_ids)
        self.assertEqual(result.loss.shape, ())
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size),
        )
        result.loss.backward()

    def test_gpt2_reorder_and_upcast_attn(self):
        # extra test: model-specific flag
        config_and_inputs = self.model_tester.prepare_config_and_inputs(reorder_and_upcast_attn=True)
        config, input_ids, token_type_ids, _, _, _, _ = config_and_inputs

        model = GPT2LMHeadModel(config)
        model.to(torch_device)
        result = model(input_ids, token_type_ids=token_type_ids, labels=input_ids)
        self.assertEqual(result.loss.shape, ())
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size),
        )
        result.loss.backward()

    def test_training_gradient_checkpointing(self):
        # overwritten: GPT2DoubleHeadsModel fails this test, non-standard class
        self.original_all_model_classes = self.all_model_classes
        self.all_model_classes = (cls for cls in self.all_model_classes if cls.__name__ != "GPT2DoubleHeadsModel")
        super().test_training_gradient_checkpointing()
        self.all_model_classes = self.original_all_model_classes

    def test_training_gradient_checkpointing_use_reentrant(self):
        # overwritten: GPT2DoubleHeadsModel fails this test, non-standard class
        self.original_all_model_classes = self.all_model_classes
        self.all_model_classes = (cls for cls in self.all_model_classes if cls.__name__ != "GPT2DoubleHeadsModel")
        super().test_training_gradient_checkpointing_use_reentrant()
        self.all_model_classes = self.original_all_model_classes

    def test_training_gradient_checkpointing_use_reentrant_false(self):
        # overwritten: GPT2DoubleHeadsModel fails this test, non-standard class
        self.original_all_model_classes = self.all_model_classes
        self.all_model_classes = (cls for cls in self.all_model_classes if cls.__name__ != "GPT2DoubleHeadsModel")
        super().test_training_gradient_checkpointing_use_reentrant_false()
        self.all_model_classes = self.original_all_model_classes


@require_torch
class GPT2ModelLanguageGenerationTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        cleanup(torch_device, gc_collect=True)

    def _test_lm_generate_gpt2_helper(
        self,
        gradient_checkpointing=False,
        reorder_and_upcast_attn=False,
        scale_attn_by_inverse_layer_idx=False,
        verify_outputs=True,
    ):
        model = GPT2LMHeadModel.from_pretrained(
            "openai-community/gpt2",
            reorder_and_upcast_attn=reorder_and_upcast_attn,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
        )
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        model.to(torch_device)

        # The dog
        input_ids = torch.tensor([[464, 3290]], dtype=torch.long, device=torch_device)

        # The dog was found in a field near the intersection of West and West Streets.\n\nThe dog
        expected_output_ids = [464, 3290, 373, 1043, 287, 257, 2214, 1474, 262, 16246, 286, 2688, 290, 2688, 27262, 13, 198, 198, 464, 3290,]  # fmt: skip
        output_ids = model.generate(input_ids, do_sample=False, max_length=20)
        if verify_outputs:
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @slow
    def test_lm_generate_gpt2(self):
        self._test_lm_generate_gpt2_helper()

    @slow
    def test_lm_generate_gpt2_with_gradient_checkpointing(self):
        self._test_lm_generate_gpt2_helper(gradient_checkpointing=True)

    @slow
    def test_lm_generate_gpt2_with_reorder_and_upcast_attn(self):
        self._test_lm_generate_gpt2_helper(reorder_and_upcast_attn=True)

    @slow
    def test_lm_generate_gpt2_with_scale_attn_by_inverse_layer_idx(self):
        self._test_lm_generate_gpt2_helper(scale_attn_by_inverse_layer_idx=True, verify_outputs=False)

    @slow
    def test_gpt2_sample(self):
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        model.to(torch_device)

        torch.manual_seed(0)
        tokenized = tokenizer("Today is a nice day and", return_tensors="pt", return_token_type_ids=True)
        input_ids = tokenized.input_ids.to(torch_device)
        output_ids = model.generate(input_ids, do_sample=True, max_length=20)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        token_type_ids = tokenized.token_type_ids.to(torch_device)
        output_seq = model.generate(input_ids=input_ids, do_sample=True, num_return_sequences=5, max_length=20)
        output_seq_tt = model.generate(
            input_ids=input_ids, token_type_ids=token_type_ids, do_sample=True, num_return_sequences=5, max_length=20
        )
        output_seq_strs = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
        output_seq_tt_strs = tokenizer.batch_decode(output_seq_tt, skip_special_tokens=True)

        expected_outputs = Expectations(
            {
                ("rocm", None): 'Today is a nice day and we can do this again."\n\nDana said that she will',
                ("rocm", (9, 5)): "Today is a nice day and if you don't know anything about the state of play during your holiday",
                ("cuda", None): "Today is a nice day and if you don't know anything about the state of play during your holiday",
                ("xpu", 3): "Today is a nice day and if you don't know anything about the state of play during your holiday",
            }
        )  # fmt: skip
        EXPECTED_OUTPUT = expected_outputs.get_expectation()
        self.assertEqual(output_str, EXPECTED_OUTPUT)
        self.assertTrue(
            all(output_seq_strs[idx] != output_seq_tt_strs[idx] for idx in range(len(output_seq_tt_strs)))
        )  # token_type_ids should change output

    # TODO joao, manuel: remove this in v4.62.0
    @slow
    def test_contrastive_search_gpt2(self):
        article = (
            "DeepMind Technologies is a British artificial intelligence subsidiary of Alphabet Inc. and research "
            "laboratory founded in 2010. DeepMind was acquired by Google in 2014. The company is based"
        )

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-large")
        gpt2_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large").to(torch_device)
        input_ids = gpt2_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        outputs = gpt2_model.generate(
            input_ids,
            penalty_alpha=0.6,
            top_k=4,
            max_length=256,
            trust_remote_code=True,
            custom_generate="transformers-community/contrastive-search",
        )

        generated_text = gpt2_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "DeepMind Technologies is a British artificial intelligence subsidiary of Alphabet Inc. and research "
                "laboratory founded in 2010. DeepMind was acquired by Google in 2014. The company is based in London, "
                "United Kingdom\n\nGoogle has a lot of data on its users and uses it to improve its products, such as "
                "Google Now, which helps users find the information they're looking for on the web. But the company "
                "is not the only one to collect data on its users. Facebook, for example, has its own facial "
                "recognition technology, as well as a database of millions of photos that it uses to personalize its "
                "News Feed.\n\nFacebook's use of data is a hot topic in the tech industry, with privacy advocates "
                "concerned about the company's ability to keep users' information private. In a blog post last "
                'year, Facebook CEO Mark Zuckerberg said his company would "do our best to be transparent about our '
                'data use and how we use it."\n\n"We have made it clear that we do not sell or share your data with '
                'third parties," Zuckerberg wrote. "If you have questions or concerns, please reach out to us at '
                'privacy@facebook.com."\n\nGoogle declined to comment on the privacy implications of its use of data, '
                "but said in a statement to The Associated Press that"
            ],
        )

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_generate_padding_left(self):
        """
        Overwriting the common test as the test is flaky on tiny models
        """
        model = GPT2LMHeadModel.from_pretrained("gpt2", dtype=torch.float16).to(0)

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        texts = ["hi", "Hello this is a very long sentence"]

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(0)

        output_native = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_native = tokenizer.batch_decode(output_native)

        model = GPT2LMHeadModel.from_pretrained(
            "gpt2", device_map={"": 0}, attn_implementation="flash_attention_2", dtype=torch.float16
        )

        output_fa_2 = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_fa_2 = tokenizer.batch_decode(output_fa_2)

        expected_output = [
            "<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>hi, who was born in the city of Kolkata, was a member of the Kolkata",
            "Hello this is a very long sentence. I'm sorry. I'm sorry. I'm sorry. I'm sorry. I'm sorry",
        ]

        self.assertListEqual(output_native, output_fa_2)
        self.assertListEqual(output_native, expected_output)

    @slow
    def test_batch_generation(self):
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        model.to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        tokenizer.padding_side = "left"

        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        # use different length sentences to test batching
        sentences = [
            "Hello, my dog is a little",
            "Today, I",
        ]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch_device)
        token_type_ids = torch.cat(
            [
                input_ids.new_full((input_ids.shape[0], input_ids.shape[1] - 1), 0),
                input_ids.new_full((input_ids.shape[0], 1), 500),
            ],
            dim=-1,
        )

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(torch_device),
            max_length=20,
        )

        outputs_tt = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(torch_device),
            token_type_ids=token_type_ids,
            max_length=20,
        )

        inputs_non_padded = tokenizer(sentences[0], return_tensors="pt").input_ids.to(torch_device)
        output_non_padded = model.generate(input_ids=inputs_non_padded, max_length=20)

        num_paddings = inputs_non_padded.shape[-1] - inputs["attention_mask"][-1].long().sum().item()
        inputs_padded = tokenizer(sentences[1], return_tensors="pt").input_ids.to(torch_device)
        output_padded = model.generate(input_ids=inputs_padded, max_length=model.config.max_length - num_paddings)

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch_out_sentence_tt = tokenizer.batch_decode(outputs_tt, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            "Hello, my dog is a little bit of a mess. I'm not sure if he's going",
            "Today, I'm going to be doing a lot of research on this. I",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertTrue(batch_out_sentence_tt != batch_out_sentence)  # token_type_ids should change output
        self.assertListEqual(expected_output_sentence, [non_padded_sentence, padded_sentence])

    @slow
    def test_batch_generation_2heads(self):
        model = GPT2DoubleHeadsModel.from_pretrained("openai-community/gpt2")
        model.to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        tokenizer.padding_side = "left"

        # This tokenizer has no pad token, so we have to set it in some way
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        # use different length sentences to test batching
        sentences = [
            "Hello, my dog is a little",
            "Today, I",
        ]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch_device)
        token_type_ids = torch.cat(
            [
                input_ids.new_full((input_ids.shape[0], input_ids.shape[1] - 1), 0),
                input_ids.new_full((input_ids.shape[0], 1), 500),
            ],
            dim=-1,
        )

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(torch_device),
            max_length=20,
        )

        outputs_tt = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(torch_device),
            token_type_ids=token_type_ids,
            max_length=20,
        )

        inputs_non_padded = tokenizer(sentences[0], return_tensors="pt").input_ids.to(torch_device)
        output_non_padded = model.generate(input_ids=inputs_non_padded, max_length=20)

        num_paddings = inputs_non_padded.shape[-1] - inputs["attention_mask"][-1].long().sum().item()
        inputs_padded = tokenizer(sentences[1], return_tensors="pt").input_ids.to(torch_device)
        output_padded = model.generate(input_ids=inputs_padded, max_length=model.config.max_length - num_paddings)

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch_out_sentence_tt = tokenizer.batch_decode(outputs_tt, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            "Hello, my dog is a little bit of a mess. I'm not sure if he's going",
            "Today, I'm going to be doing a lot of research on this. I",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertTrue(batch_out_sentence_tt != batch_out_sentence)  # token_type_ids should change output
        self.assertListEqual(expected_output_sentence, [non_padded_sentence, padded_sentence])
