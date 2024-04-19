# coding=utf-8
# Copyright 2020 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
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
import warnings

import numpy as np
from parameterized import parameterized

from transformers import is_torch_available, pipeline, set_seed
from transformers.testing_utils import (
    is_flaky,
    require_accelerate,
    require_torch,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)

from ..test_modeling_common import floats_tensor, ids_tensor
from .test_framework_agnostic import GenerationIntegrationTestsMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSpeechSeq2Seq,
        AutoModelForVision2Seq,
        AutoTokenizer,
        BartForCausalLM,
        BartForConditionalGeneration,
        BartTokenizer,
        GPT2LMHeadModel,
        GPT2Tokenizer,
        ImageGPTForCausalImageModeling,
        SpeechEncoderDecoderModel,
    )
    from transformers.cache_utils import DynamicCache
    from transformers.generation import (
        BeamSampleDecoderOnlyOutput,
        BeamSampleEncoderDecoderOutput,
        BeamSearchDecoderOnlyOutput,
        BeamSearchEncoderDecoderOutput,
        DisjunctiveConstraint,
        GenerateBeamDecoderOnlyOutput,
        GenerateBeamEncoderDecoderOutput,
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        GreedySearchDecoderOnlyOutput,
        GreedySearchEncoderDecoderOutput,
        LogitsProcessorList,
        MaxLengthCriteria,
        MinLengthLogitsProcessor,
        PhrasalConstraint,
        SampleDecoderOnlyOutput,
        SampleEncoderDecoderOutput,
        StoppingCriteria,
        StoppingCriteriaList,
    )
    from transformers.generation.utils import _speculative_sampling


class GenerationTesterMixin:
    model_tester = None
    all_generative_model_classes = ()
    input_name = "input_ids"
    max_new_tokens = 3

    def _get_input_ids_and_config(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict[self.input_name]

        input_ids = input_ids[:batch_size]

        if config.eos_token_id is not None and config.pad_token_id is None:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            if isinstance(config.eos_token_id, int):
                config.eos_token_id = [config.eos_token_id]
            config.pad_token_id = config.eos_token_id[0]
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # It is important set set the eos_token_id to None to ensure that no sequences
        # shorter than `max_length` can be generated
        config.eos_token_id = None
        config.forced_eos_token_id = None

        return config, input_ids, attention_mask

    @staticmethod
    def _get_logits_processor_and_warper_kwargs(
        input_length,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
    ):
        process_kwargs = {
            "bad_words_ids": [[1, 0]],
            "repetition_penalty": 1.2,
            "remove_invalid_values": True,
        }
        # NoRepeatNGramLogitsProcessor + forced tokens may result in no valid continuations
        if forced_bos_token_id is None and forced_eos_token_id is None:
            process_kwargs["no_repeat_ngram_size"] = 2

        warp_kwargs = {"top_k": 10, "top_p": 0.7, "temperature": 0.7}
        return process_kwargs, warp_kwargs

    @staticmethod
    def _get_beam_kwargs(num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
        }
        return beam_kwargs

    @staticmethod
    def _get_diverse_beam_kwargs(num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
            "num_beam_groups": 2,  # one beam per group
            "diversity_penalty": 2.0,
        }
        return beam_kwargs

    @staticmethod
    def _get_constrained_beam_kwargs(num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": num_return_sequences * 4,
            "num_return_sequences": num_return_sequences,
        }
        return beam_kwargs

    @staticmethod
    def _get_encoder_outputs(
        model, input_ids, attention_mask, output_attentions=None, output_hidden_states=None, num_interleave=1
    ):
        encoder = model.get_encoder()
        encoder_outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
            num_interleave, dim=0
        )
        input_ids = torch.zeros_like(input_ids[:, :1]) + model._get_decoder_start_token_id()
        attention_mask = None
        return encoder_outputs, input_ids, attention_mask

    def _greedy_generate(
        self,
        model,
        input_ids,
        attention_mask,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        logits_process_kwargs, _ = self._get_logits_processor_and_warper_kwargs(
            input_ids.shape[-1],
            forced_bos_token_id=model.config.forced_bos_token_id,
            forced_eos_token_id=model.config.forced_eos_token_id,
        )

        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        output_generate = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            **logits_process_kwargs,
            **model_kwargs,
        )

        return output_generate

    def _sample_generate(
        self,
        model,
        input_ids,
        attention_mask,
        num_return_sequences,
        logits_warper_kwargs,
        process_kwargs,
        output_scores=False,
        output_logits=False,
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
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=num_return_sequences,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            **logits_warper_kwargs,
            **process_kwargs,
            **model_kwargs,
        )

        return output_generate

    def _beam_search_generate(
        self,
        model,
        input_ids,
        attention_mask,
        beam_kwargs,
        logits_process_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        output_generate = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            **beam_kwargs,
            **logits_process_kwargs,
            **model_kwargs,
        )

        return output_generate

    def _beam_sample_generate(
        self,
        model,
        input_ids,
        attention_mask,
        beam_kwargs,
        logits_warper_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        torch.manual_seed(0)
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        output_generate = model.generate(
            input_ids,
            do_sample=True,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            **beam_kwargs,
            **logits_warper_kwargs,
            **model_kwargs,
        )

        return output_generate

    def _group_beam_search_generate(
        self,
        model,
        input_ids,
        attention_mask,
        beam_kwargs,
        logits_process_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        output_generate = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            **beam_kwargs,
            **logits_process_kwargs,
            **model_kwargs,
        )

        return output_generate

    def _constrained_beam_search_generate(
        self,
        model,
        input_ids,
        attention_mask,
        constraints,
        beam_kwargs,
        logits_process_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        output_generate = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            constraints=constraints,
            **beam_kwargs,
            **logits_process_kwargs,
            **model_kwargs,
        )

        return output_generate

    def _contrastive_generate(
        self,
        model,
        input_ids,
        attention_mask,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        contrastive_search_kwargs = {
            "penalty_alpha": 0.6,
            "top_k": 5,
        }

        logits_process_kwargs, _ = self._get_logits_processor_and_warper_kwargs(
            input_ids.shape[-1],
            forced_bos_token_id=model.config.forced_bos_token_id,
            forced_eos_token_id=model.config.forced_eos_token_id,
        )

        model_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
        output_generate = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            **logits_process_kwargs,
            **model_kwargs,
            **contrastive_search_kwargs,
        )

        return output_generate

    def test_greedy_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(model=model, input_ids=input_ids, attention_mask=attention_mask)

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + input_ids.shape[-1])

    def test_greedy_generate_dict_outputs(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            config.use_cache = False
            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, GreedySearchEncoderDecoderOutput)
            else:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + input_ids.shape[-1])
                self.assertIsInstance(output_generate, GenerateDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, GreedySearchDecoderOnlyOutput)

            self._check_outputs(output_generate, input_ids, model.config)

    def test_greedy_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            if not hasattr(config, "use_cache"):
                self.skipTest("This model doesn't support caching")

            config.use_cache = True
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + input_ids.shape[-1])
            self._check_outputs(output_generate, input_ids, model.config, use_cache=True)

    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            model = model_class(config).to(torch_device).eval()
            process_kwargs, logits_warper_kwargs = self._get_logits_processor_and_warper_kwargs(
                input_ids.shape[-1],
                forced_bos_token_id=model.config.forced_bos_token_id,
                forced_eos_token_id=model.config.forced_eos_token_id,
            )

            output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=1,
                logits_warper_kwargs=logits_warper_kwargs,
                process_kwargs=process_kwargs,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + input_ids.shape[-1])

    def test_sample_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            config.use_cache = False
            model = model_class(config).to(torch_device).eval()

            process_kwargs, logits_warper_kwargs = self._get_logits_processor_and_warper_kwargs(
                input_ids.shape[-1],
                forced_bos_token_id=model.config.forced_bos_token_id,
                forced_eos_token_id=model.config.forced_eos_token_id,
            )

            output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=2,
                logits_warper_kwargs=logits_warper_kwargs,
                process_kwargs=process_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, SampleEncoderDecoderOutput)
            else:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + input_ids.shape[-1])
                self.assertIsInstance(output_generate, GenerateDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, SampleDecoderOnlyOutput)

            self._check_outputs(output_generate, input_ids, model.config, num_return_sequences=2)

    def test_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            model = model_class(config).to(torch_device).eval()

            logits_process_kwargs, _ = self._get_logits_processor_and_warper_kwargs(
                input_ids.shape[-1],
                config.forced_bos_token_id,
                config.forced_eos_token_id,
            )
            beam_kwargs = self._get_beam_kwargs()

            output_generate = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + input_ids.shape[-1])

    def test_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            # disable cache
            config.use_cache = False

            model = model_class(config).to(torch_device).eval()
            logits_process_kwargs, _ = self._get_logits_processor_and_warper_kwargs(
                input_ids.shape[-1],
                config.forced_bos_token_id,
                config.forced_eos_token_id,
            )
            beam_kwargs = self._get_beam_kwargs()
            output_generate = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateBeamEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + input_ids.shape[-1])
                self.assertIsInstance(output_generate, GenerateBeamDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self._check_outputs(
                output_generate, input_ids, model.config, num_return_sequences=beam_kwargs["num_beams"]
            )

    def test_beam_search_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            # enable cache
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            if not hasattr(config, "use_cache"):
                self.skipTest("This model doesn't support caching")

            model = model_class(config).to(torch_device).eval()
            logits_process_kwargs, _ = self._get_logits_processor_and_warper_kwargs(
                input_ids.shape[-1],
                config.forced_bos_token_id,
                config.forced_eos_token_id,
            )

            beam_kwargs = self._get_beam_kwargs()

            config.use_cache = True
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            output_generate = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + input_ids.shape[-1])
            self._check_outputs(
                output_generate, input_ids, model.config, use_cache=True, num_return_sequences=beam_kwargs["num_beams"]
            )

    @require_accelerate
    @require_torch_multi_accelerator
    def test_model_parallel_beam_search(self):
        for model_class in self.all_generative_model_classes:
            if "xpu" in torch_device:
                return unittest.skip("device_map='auto' does not work with XPU devices")

            if model_class._no_split_modules is None:
                continue

            config, input_ids, attention_mask = self._get_input_ids_and_config()

            model = model_class(config).eval()
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)
                new_model = model_class.from_pretrained(tmp_dir, device_map="auto")

                new_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=2,
                )

    def test_beam_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            _, logits_warper_kwargs = self._get_logits_processor_and_warper_kwargs(input_ids.shape[-1])

            model = model_class(config).to(torch_device).eval()
            beam_kwargs = self._get_beam_kwargs()

            output_generate = self._beam_sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                beam_kwargs=beam_kwargs,
                logits_warper_kwargs=logits_warper_kwargs,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + input_ids.shape[-1])

            if "inputs_embeds" in set(inspect.signature(model.prepare_inputs_for_generation).parameters):
                input_embeds = model.get_input_embeddings()(input_ids)
                beam_kwargs.update({"inputs_embeds": input_embeds})
                output_generate2 = self._beam_sample_generate(
                    model=model,
                    input_ids=None,
                    attention_mask=attention_mask,
                    beam_kwargs=beam_kwargs,
                    logits_warper_kwargs=logits_warper_kwargs,
                )

                torch.testing.assert_close(output_generate[:, input_embeds.shape[1] :], output_generate2)

    def test_beam_sample_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            # disable cache
            config.use_cache = False

            model = model_class(config).to(torch_device).eval()
            _, logits_warper_kwargs = self._get_logits_processor_and_warper_kwargs(input_ids.shape[-1])
            beam_kwargs = self._get_beam_kwargs()

            output_generate = self._beam_sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                beam_kwargs=beam_kwargs,
                logits_warper_kwargs=logits_warper_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateBeamEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSampleEncoderDecoderOutput)
            else:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + input_ids.shape[-1])
                self.assertIsInstance(output_generate, GenerateBeamDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSampleDecoderOnlyOutput)

            self._check_outputs(
                output_generate, input_ids, model.config, num_return_sequences=beam_kwargs["num_beams"]
            )

    def test_generate_without_input_ids(self):
        config, _, _ = self._get_input_ids_and_config()

        # if no bos token id => cannot generate from None
        if config.bos_token_id is None:
            return

        # hack in case they are equal, otherwise the attn mask will be [0]
        if config.bos_token_id == config.pad_token_id:
            config.pad_token_id = None

        for model_class in self.all_generative_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()

            output_ids_generate = model.generate(
                do_sample=False, max_new_tokens=self.max_new_tokens, remove_invalid_values=True
            )
            self.assertIsNotNone(output_ids_generate)

    def test_group_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            model = model_class(config).to(torch_device).eval()
            logits_process_kwargs, _ = self._get_logits_processor_and_warper_kwargs(
                input_ids.shape[-1],
                config.forced_bos_token_id,
                config.forced_eos_token_id,
            )

            # check `generate()` and `group_beam_search()` are equal
            beam_kwargs = self._get_diverse_beam_kwargs()
            output_generate = self._group_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
            )
            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + input_ids.shape[-1])

            # check `group_beam_search` for higher than 1 `num_return_sequences`
            num_return_sequences = 2
            beam_kwargs = self._get_diverse_beam_kwargs(num_return_sequences=num_return_sequences)
            output_generate = self._group_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
            )
            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + input_ids.shape[-1])

    def test_group_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()
            config.use_cache = False

            model = model_class(config).to(torch_device).eval()
            logits_process_kwargs, _ = self._get_logits_processor_and_warper_kwargs(
                input_ids.shape[-1],
                config.forced_bos_token_id,
                config.forced_eos_token_id,
            )

            beam_kwargs = self._get_diverse_beam_kwargs()
            output_generate = self._group_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateBeamEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + input_ids.shape[-1])
                self.assertIsInstance(output_generate, GenerateBeamDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self._check_outputs(
                output_generate, input_ids, model.config, num_return_sequences=beam_kwargs["num_beams"]
            )

    # TODO: @gante
    @is_flaky()
    def test_constrained_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            model = model_class(config).to(torch_device).eval()

            logits_process_kwargs, _ = self._get_logits_processor_and_warper_kwargs(
                input_ids.shape[-1],
                config.forced_bos_token_id,
                config.forced_eos_token_id,
            )

            # Sample constraints
            min_id = 3
            max_id = config.vocab_size

            force_tokens = torch.randint(min_id, max_id, (1, 2)).tolist()[0]
            constraints = [
                PhrasalConstraint(force_tokens),
            ]

            beam_kwargs = self._get_constrained_beam_kwargs()
            output_generate = self._constrained_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                constraints=constraints,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + input_ids.shape[-1])

            for generation_output in output_generate:
                self._check_sequence_inside_sequence(force_tokens, generation_output)

            # check`constrained_beam_search` for higher than 1 `num_return_sequences`
            # Sample constraints
            force_tokens = torch.randint(min_id, max_id, (1, 2)).tolist()[0]
            constraints = [
                PhrasalConstraint(force_tokens),
            ]

            beam_kwargs = self._get_constrained_beam_kwargs(num_return_sequences=2)

            output_generate = self._constrained_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                constraints=constraints,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + input_ids.shape[-1])

            for generation_output in output_generate:
                self._check_sequence_inside_sequence(force_tokens, generation_output)

    def test_constrained_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()

            # disable cache
            config.use_cache = False

            model = model_class(config).to(torch_device).eval()
            logits_process_kwargs, _ = self._get_logits_processor_and_warper_kwargs(
                input_ids.shape[-1],
                config.forced_bos_token_id,
                config.forced_eos_token_id,
            )

            # Sample constraints
            min_id = 3
            max_id = model.config.vocab_size
            force_tokens = torch.randint(min_id, max_id, (1, 2)).tolist()[0]
            constraints = [
                PhrasalConstraint(force_tokens),
            ]

            beam_kwargs = self._get_constrained_beam_kwargs()
            output_generate = self._constrained_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                constraints=constraints,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateBeamEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + input_ids.shape[-1])
                self.assertIsInstance(output_generate, GenerateBeamDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self._check_outputs(
                output_generate, input_ids, model.config, num_return_sequences=beam_kwargs["num_beams"]
            )

    def test_contrastive_generate(self):
        for model_class in self.all_generative_model_classes:
            # won't fix: FSMT and Reformer have a different cache variable type (and format).
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest("Won't fix: old model with different cache format")

            config, input_ids, attention_mask = self._get_input_ids_and_config()

            # NOTE: contrastive search only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest("This model doesn't support caching")
            config.use_cache = True
            config.is_decoder = True

            # test old generation output for backwards compatibility
            model = model_class(config).to(torch_device).eval()
            output_generate = self._contrastive_generate(
                model=model, input_ids=input_ids, attention_mask=attention_mask
            )
            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + input_ids.shape[-1])

    def test_contrastive_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            # won't fix: FSMT and Reformer have a different cache variable type (and format).
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest("Won't fix: old model with different cache format")

            config, input_ids, attention_mask = self._get_input_ids_and_config()

            # NOTE: contrastive search only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest("This model doesn't support caching")
            config.use_cache = True
            config.is_decoder = True

            model = model_class(config).to(torch_device).eval()
            output_generate = self._contrastive_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + input_ids.shape[-1])
            self._check_outputs(output_generate, input_ids, model.config, use_cache=True)

    def test_contrastive_generate_low_memory(self):
        # Check that choosing 'low_memory' does not change the model output
        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer", "speech2text"]):
                self.skipTest("Won't fix: old model with different cache format")
            if any(model_name in model_class.__name__.lower() for model_name in ["gptbigcode", "jamba"]):
                self.skipTest("TODO: fix me")

            config, input_ids, attention_mask = self._get_input_ids_and_config(batch_size=1)

            # NOTE: contrastive search only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest("This model doesn't support caching")

            config.use_cache = True
            config.is_decoder = True

            # test output equality of low versus high memory
            model = model_class(config).to(torch_device).eval()

            low_output = model.generate(
                input_ids,
                top_k=4,
                penalty_alpha=0.6,
                low_memory=True,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
            )

            high_output = model.generate(
                input_ids,
                top_k=4,
                penalty_alpha=0.6,
                low_memory=False,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
            )
            self.assertListEqual(low_output.tolist(), high_output.tolist())

    def test_beam_search_low_memory(self):
        # Check that choosing 'low_memory' does not change the model output
        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest("Won't fix: old model with different cache format")
            if any(
                model_name in model_class.__name__.lower()
                for model_name in [
                    "bloom",
                    "ctrl",
                    "gptbigcode",
                    "transo_xl",
                    "xlnet",
                    "cpm",
                    "jamba",
                ]
            ):
                self.skipTest("May fix in the future: need model-specific fixes")
            config, input_ids, _ = self._get_input_ids_and_config(batch_size=2)
            # batch_size=1 is ok, but batch_size>1 will cause non-identical output

            config.use_cache = True
            config.is_decoder = True

            # test output equality of low versus high memory
            model = model_class(config).to(torch_device).eval()

            low_output = model.generate(input_ids, max_new_tokens=8, num_beams=5, early_stopping=True, low_memory=True)

            high_output = model.generate(
                input_ids, max_new_tokens=8, num_beams=5, early_stopping=True, low_memory=False
            )
            self.assertListEqual(low_output.tolist(), high_output.tolist())

    @is_flaky()  # Read NOTE (1) below. If there are API issues, all attempts will fail.
    def test_assisted_decoding_matches_greedy_search(self):
        # This test ensures that the assisted generation does not introduce output changes over greedy search.
        # NOTE (1): The sentence above is true most of the time, there is a tiny difference in the logits due to matmul
        # shape differences -- and it may result in a different output. The input shape difference happens in the
        # main model, that runs the forward pass with several candidates at once (as opposed to generating one token at
        # a time). See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535 for more info.
        # NOTE (2): It breaks the pattern in the tests above, for multiple reasons:
        # - assisted_decoding, contrarily to the other methods, can't be called on its own (e.g. needs to
        # prepare the assistant encoder outputs in the main generate body);
        # - assisted_decoding does not support `use_cache = False`
        # - assisted_decoding does not support `batch_size > 1`

        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest("Won't fix: old model with different cache format")
            if any(
                model_name in model_class.__name__.lower()
                for model_name in [
                    "bigbirdpegasus",
                    "led",
                    "mega",
                    "speech2text",
                    "git",
                    "prophetnet",
                    "seamlessm4t",
                    "clvp",
                ]
            ):
                self.skipTest("May fix in the future: need model-specific fixes")

            # enable cache
            config, input_ids, attention_mask = self._get_input_ids_and_config(batch_size=1)

            # NOTE: assisted generation only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest("This model doesn't support caching")

            config.use_cache = True
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            # Sets assisted generation arguments such that:
            # a) no EOS is generated, to ensure generation doesn't break early
            # b) the assistant model always generates two tokens when it is called, to ensure the input preparation of
            #    the assistant model is correct
            # c) there are at least two forward passes in the main model, to ensure the input preparation of
            #    the main model is correct
            generation_kwargs = {
                "eos_token_id": -1,  # see a)
                "max_new_tokens": 4,  # see c)
                "num_beams": 1,
                "do_sample": False,
                "output_scores": True,
                "output_logits": True,
                "output_hidden_states": True,
                "output_attentions": True,
                "return_dict_in_generate": True,
            }
            output_greedy = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)

            assistant_model = model
            assistant_model.generation_config.num_assistant_tokens = 2  # see b)
            assistant_model.generation_config.num_assistant_tokens_schedule = "constant"  # see b)
            generation_kwargs.update({"assistant_model": assistant_model})
            output_assisted = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)

            # The two outputs must match and their shape must be as expected
            self.assertListEqual(output_greedy.sequences.tolist(), output_assisted.sequences.tolist())
            for output in (output_greedy, output_assisted):
                self._check_outputs(output, input_ids, model.config, use_cache=True)

    @is_flaky()
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        # This test ensures that the prompt lookup generation does not introduce output changes over greedy search.
        # This test is mostly a copy of test_assisted_decoding_matches_greedy_search

        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest("Won't fix: old model with different cache format")
            if any(
                model_name in model_class.__name__.lower()
                for model_name in [
                    "bigbirdpegasus",
                    "led",
                    "mega",
                    "speech2text",
                    "git",
                    "prophetnet",
                    "seamlessm4t",
                    "clvp",
                ]
            ):
                self.skipTest("May fix in the future: need model-specific fixes")

            # enable cache
            config, input_ids, attention_mask = self._get_input_ids_and_config(batch_size=1)

            # NOTE: assisted generation only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest("This model doesn't support caching")

            config.use_cache = True
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            # Sets assisted generation arguments such that:
            # a) no EOS is generated, to ensure generation doesn't break early
            # b) the prompt lookup tries to give the model 2 tokens, to ensure the input preparation of
            #    prompt lookup is correct
            # c) there are at least two forward passes in the main model, to ensure the input preparation of
            #    the main model is correct
            generation_kwargs = {
                "eos_token_id": -1,  # see a)
                "max_new_tokens": 4,  # see c)
                "num_beams": 1,
                "do_sample": False,
                "output_scores": True,
                "output_logits": True,
                "output_hidden_states": True,
                "output_attentions": True,
                "return_dict_in_generate": True,
            }

            output_greedy = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)

            generation_kwargs.update({"prompt_lookup_num_tokens": 2})  # see b)
            output_prompt_lookup = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)

            # The two outputs must match and their shape must be as expected
            self.assertListEqual(output_greedy.sequences.tolist(), output_prompt_lookup.sequences.tolist())
            for output in (output_greedy, output_prompt_lookup):
                self._check_outputs(output, input_ids, model.config, use_cache=True)

    def test_assisted_decoding_sample(self):
        # In this test we don't check assisted vs non-assisted output -- seeded assisted decoding with sample will not
        # match sample for the same seed, as the forward pass does not return the exact same logits (due to matmul with
        # different shapes, see https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535).
        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest("Won't fix: old model with different cache format")
            if any(
                model_name in model_class.__name__.lower()
                for model_name in [
                    "bigbirdpegasus",
                    "led",
                    "mega",
                    "speech2text",
                    "git",
                    "prophetnet",
                    "seamlessm4t",
                    "clvp",
                ]
            ):
                self.skipTest("May fix in the future: need model-specific fixes")

            # enable cache
            config, input_ids, attention_mask = self._get_input_ids_and_config(batch_size=1)

            # NOTE: assisted generation only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest("This model doesn't support caching")

            config.use_cache = True
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            # Sets assisted generation arguments such that:
            # a) no EOS is generated, to ensure generation doesn't break early
            # b) the assistant model always generates two tokens when it is called, to ensure the input preparation of
            #    the assistant model is correct
            # c) there are at least two forward passes in the main model, to ensure the input preparation of
            #    the main model is correct
            assistant_model = model
            assistant_model.generation_config.num_assistant_tokens = 2  # see b)
            assistant_model.generation_config.num_assistant_tokens_schedule = "constant"  # see b)
            generation_kwargs = {
                "eos_token_id": -1,  # see a)
                "max_new_tokens": 4,  # see c)
                "num_beams": 1,
                "do_sample": True,
                "assistant_model": assistant_model,
                "output_scores": True,
                "output_logits": True,
                "output_hidden_states": True,
                "output_attentions": True,
                "return_dict_in_generate": True,
            }
            output_assisted = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)

            self._check_outputs(output_assisted, input_ids, model.config, use_cache=True)

    def test_generate_with_head_masking(self):
        """Test designed for encoder-decoder models to ensure the attention head masking is used."""
        attention_names = ["encoder_attentions", "decoder_attentions", "cross_attentions"]
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()
            # We want to test only encoder-decoder models
            if not config.is_encoder_decoder:
                continue
            model = model_class(config).to(torch_device)

            head_masking = {
                "head_mask": torch.zeros(config.encoder_layers, config.encoder_attention_heads, device=torch_device),
                "decoder_head_mask": torch.zeros(
                    config.decoder_layers, config.decoder_attention_heads, device=torch_device
                ),
                "cross_attn_head_mask": torch.zeros(
                    config.decoder_layers, config.decoder_attention_heads, device=torch_device
                ),
            }

            signature = inspect.signature(model.forward)
            # We want to test only models where encoder/decoder head masking is implemented
            if not set(head_masking.keys()) < {*signature.parameters.keys()}:
                continue

            for attn_name, (name, mask) in zip(attention_names, head_masking.items()):
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    num_beams=1,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    remove_invalid_values=True,
                    **{name: mask},
                )
                # We check the state of decoder_attentions and cross_attentions just from the last step
                attn_weights = out[attn_name] if attn_name == attention_names[0] else out[attn_name][-1]
                self.assertEqual(sum([w.sum().item() for w in attn_weights]), 0.0)

    def test_left_padding_compatibility(self):
        # NOTE: left-padding results in small numerical differences. This is expected.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

        # First, filter out models that don't support left padding
        # - The model must have generative capabilities
        if len(self.all_generative_model_classes) == 0:
            self.skipTest(reason="No generative architecture available for this model.")

        # - The model must be a decoder-only architecture (encoder-based architectures use right-padding)
        decoder_only_classes = []
        for model_class in self.all_generative_model_classes:
            config, _, _ = self._get_input_ids_and_config()
            if config.is_encoder_decoder:
                continue
            else:
                decoder_only_classes.append(model_class)
        if len(decoder_only_classes) == 0:
            self.skipTest(reason="No decoder-only architecture available for this model.")

        # - Decoder-only architectures derived from encoder-decoder models could support it in theory, but we haven't
        #   added support for it yet. We skip these models for now.
        has_encoder_attributes = any(
            attr_name
            for attr_name in config.to_dict().keys()
            if attr_name.startswith("encoder") and attr_name != "encoder_no_repeat_ngram_size"
        )
        if has_encoder_attributes:
            self.skipTest(
                reason="The decoder-only derived from encoder-decoder models are not expected to support left-padding."
            )

        # Then, test left-padding
        def _prepare_model_kwargs(input_ids, attention_mask, signature):
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "position_ids" in signature:
                position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                model_kwargs["position_ids"] = position_ids
            if "cache_position" in signature:
                cache_position = torch.arange(input_ids.shape[-1], device=torch_device)
                model_kwargs["cache_position"] = cache_position
            return model_kwargs

        for model_class in decoder_only_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()
            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, signature)
            next_logits_wo_padding = model(**model_kwargs).logits[:, -1, :]

            # With left-padding (length 32)
            pad_size = (input_ids.shape[0], 32)
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * config.pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat((torch.zeros_like(padding), attention_mask), dim=1)
            model_kwargs = _prepare_model_kwargs(padded_input_ids, padded_attention_mask, signature)
            next_logits_with_padding = model(**model_kwargs).logits[:, -1, :]

            # They should result in very similar logits
            self.assertTrue(torch.allclose(next_logits_wo_padding, next_logits_with_padding, atol=1e-5))

    def test_past_key_values_format(self):
        # Test that the KV cache is formatted correctly. Exceptions need to explicitly overwrite this test. Having a
        # standard KV cache format is important for a consistent API (and for advanced generation methods).
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            # If it doesn't support cache, pass the test
            if not hasattr(config, "use_cache"):
                self.skipTest("This model doesn't support caching")

            model = model_class(config).to(torch_device)
            if "use_cache" not in inputs:
                inputs["use_cache"] = True
            outputs = model(**inputs)

            # If "past_key_values" is not returned, pass the test (e.g. RWKV uses a different cache name and format)
            if "past_key_values" not in outputs:
                self.skipTest("This model doesn't return `past_key_values`")

            num_hidden_layers = (
                getattr(config, "decoder_layers", None)
                or getattr(config, "num_decoder_layers", None)
                or config.num_hidden_layers
            )
            num_attention_heads = getattr(config, "decoder_attention_heads", config.num_attention_heads)
            embed_dim = getattr(config, "d_model", config.hidden_size)
            per_head_embed_dim = embed_dim // num_attention_heads

            past_kv = outputs["past_key_values"]
            self.assertEqual(len(past_kv), num_hidden_layers)

            # Encoder-Decoder checks
            if config.is_encoder_decoder:
                encoder_num_attention_heads = config.encoder_attention_heads
                encoder_per_head_embed_dim = embed_dim // encoder_num_attention_heads
                batch_size, seq_length = inputs["decoder_input_ids"].shape
                for i in range(num_hidden_layers):
                    self.assertEqual(len(past_kv[i]), 4)  # K V for the decoder + K V for the encoder = 4
                    self.assertEqual(
                        past_kv[i][0].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    self.assertEqual(
                        past_kv[i][1].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    # The sequence length for the encoder K V depends on the model. Since it is not manipulated in
                    # autoregressive generation, I'm keeping the test general and not checking the 3rd dim
                    self.assertEqual(
                        (past_kv[i][2].shape[0], past_kv[i][2].shape[1], past_kv[i][2].shape[3]),
                        (batch_size, encoder_num_attention_heads, encoder_per_head_embed_dim),
                    )
                    self.assertEqual(
                        (past_kv[i][3].shape[0], past_kv[i][3].shape[1], past_kv[i][3].shape[3]),
                        (batch_size, encoder_num_attention_heads, encoder_per_head_embed_dim),
                    )

            # Decoder-only checks
            else:
                # TODO: this line is only needed because of imagegpt, where "pixel_values" = "input_ids". Fix the
                # tests in imagegpt such that `prepare_config_and_inputs_for_common` returns the later (and the other
                # tests use it)
                key = "input_ids" if "input_ids" in inputs else "pixel_values"
                batch_size, seq_length = inputs[key].shape
                for i in range(num_hidden_layers):
                    self.assertEqual(len(past_kv[0]), 2)  # K V for the decoder = 2
                    self.assertEqual(
                        past_kv[i][0].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    self.assertEqual(
                        past_kv[i][1].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )

    def test_generate_from_inputs_embeds_decoder_only(self):
        # When supported, tests that the decoder model can generate from `inputs_embeds` instead of `input_ids`
        # if fails, you should probably update the `prepare_inputs_for_generation` function
        for model_class in self.all_generative_model_classes:
            config, input_ids, _ = self._get_input_ids_and_config()

            # Ignore:
            # a) eos (to always output 20 tokens) and pad (so we don't try to infer the attn mask from the input_ids,
            #   which would cause a mismatch),
            config.pad_token_id = config.eos_token_id = -1
            # b) embedding scaling, the scaling factor applied after embeding from input_ids (requires knowledge of the
            #   variable that holds the scaling factor, which is model-dependent)
            if hasattr(config, "scale_embedding"):
                config.scale_embedding = False

            # This test is for decoder-only models (encoder-decoder models have native input embeddings support in the
            # decoder)
            if config.is_encoder_decoder:
                continue

            # Skip models without explicit support
            model = model_class(config).to(torch_device).eval()
            if "inputs_embeds" not in inspect.signature(model.prepare_inputs_for_generation).parameters.keys():
                continue

            # Traditional way of generating text
            outputs_from_ids = model.generate(input_ids)
            self.assertEqual(outputs_from_ids.shape, (2, 20))

            # Same thing, but from input embeddings (`input_ids` is passed so the prompt is present in the output)
            inputs_embeds = model.get_input_embeddings()(input_ids)
            outputs_from_embeds = model.generate(input_ids, inputs_embeds=inputs_embeds)
            self.assertListEqual(outputs_from_ids.tolist(), outputs_from_embeds.tolist())

            # But if we pass different inputs_embeds, we should get different outputs
            torch.manual_seed(0)
            random_embeds = torch.rand_like(inputs_embeds)
            outputs_from_rand_embeds = model.generate(input_ids, inputs_embeds=random_embeds)
            with self.assertRaises(AssertionError):
                self.assertListEqual(outputs_from_rand_embeds.tolist(), outputs_from_embeds.tolist())

            # input_ids is not a required input -- if we don't pass it, the newly generated tokens will be the same
            outputs_from_embeds_wo_ids = model.generate(
                inputs_embeds=inputs_embeds, max_new_tokens=20 - inputs_embeds.shape[1]
            )
            self.assertListEqual(
                outputs_from_embeds[:, inputs_embeds.shape[1] :].tolist(),
                outputs_from_embeds_wo_ids.tolist(),
            )

    def test_generate_continue_from_past_key_values(self):
        # Tests that we can continue generating from past key values, returned from a previous `generate` call
        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["imagegpt"]):
                self.skipTest("Won't fix: old model with unique inputs/caches/other")
            if any(model_name in model_class.__name__.lower() for model_name in ["umt5"]):
                self.skipTest("TODO: needs modeling or test input preparation fixes for compatibility")

            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            if not hasattr(config, "use_cache"):
                self.skipTest("This model doesn't support caching")

            # Let's make it always:
            # 1. use cache (for obvious reasons)
            # 2. generate to max length (which can be achieved by setting the eos token to an invalid value), which
            #    would make the test flaky (e.g. EOS is generated on iteration 1 on both generations, but the
            #    continuation would force it to generate beyond an EOS token)
            # 3. ignore `token_type_ids` for simplicity
            # 4. ignore `forced_eos_token_id`, which requires further manipulation of the continuation inputs and is
            #    active by default on some models
            config.use_cache = True
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            model = model_class(config).to(torch_device)
            model.eval()
            model.generation_config.pad_token_id = model.generation_config.eos_token_id = -1
            model.generation_config.forced_eos_token_id = None

            # If "past_key_values" is not returned, skip the test (e.g. RWKV uses a different cache name and format)
            outputs = model(**inputs)
            if "past_key_values" not in outputs:
                self.skipTest("This model doesn't return `past_key_values`")

            # Traditional way of generating text, with `return_dict_in_generate` to return the past key values
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=4, return_dict_in_generate=True)

            # Let's generate again, but passing the past key values in between (3 + 1 = 4 tokens). Note that the
            # inputs may need to be tweaked across `generate` calls (like the attention mask).
            outputs_cached = model.generate(**inputs, do_sample=False, max_new_tokens=3, return_dict_in_generate=True)

            # Continue from the tokens generated above, preparing the inputs accordingly
            inputs["past_key_values"] = outputs_cached.past_key_values
            new_attention_len = outputs_cached.sequences.shape[-1]
            if config.is_encoder_decoder:
                inputs["decoder_input_ids"] = outputs_cached.sequences
                if "decoder_attention_mask" in inputs:
                    inputs["decoder_attention_mask"] = torch.nn.functional.pad(
                        inputs["decoder_attention_mask"],
                        (0, new_attention_len - inputs["decoder_attention_mask"].shape[1]),
                        mode="constant",
                        value=1,
                    )
            else:
                inputs["input_ids"] = outputs_cached.sequences
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.nn.functional.pad(
                        inputs["attention_mask"],
                        (0, new_attention_len - inputs["attention_mask"].shape[1]),
                        mode="constant",
                        value=1,
                    )
            outputs_cached = model.generate(**inputs, do_sample=False, max_new_tokens=1, return_dict_in_generate=True)

            # The two sets of generated text and past kv should be equal to each other
            self.assertListEqual(outputs.sequences.tolist(), outputs_cached.sequences.tolist())
            for layer_idx in range(len(outputs_cached.past_key_values)):
                for kv_idx in range(len(outputs_cached.past_key_values[layer_idx])):
                    self.assertTrue(
                        torch.allclose(
                            outputs.past_key_values[layer_idx][kv_idx],
                            outputs_cached.past_key_values[layer_idx][kv_idx],
                        )
                    )

    @parameterized.expand([(1, False), (1, True), (4, False)])
    def test_new_cache_format(self, num_beams, do_sample):
        # Tests that generating with the new format is exactly the same as the legacy one (for models that support it).
        # 👉 tests with and without beam search so that we can test with and without cache reordering.
        # 👉 tests with and without sampling so we can cover the most common use cases.
        for model_class in self.all_generative_model_classes:
            if not model_class._supports_cache_class:
                self.skipTest("This model does not support the new cache format")

            config, input_ids, attention_mask = self._get_input_ids_and_config()
            config.use_cache = True
            config.is_decoder = True

            model = model_class(config).to(torch_device).eval()
            generation_kwargs = {
                "max_new_tokens": 5,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "num_return_sequences": num_beams,
                "return_dict_in_generate": True,  # Required to return `past_key_values`
            }

            # Sets seed before calling `generate` for the case with do_sample=True
            seed = torch.randint(0, 1000000, (1,)).item()
            set_seed(seed)
            legacy_results = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)
            set_seed(seed)
            new_results = model.generate(
                input_ids, attention_mask=attention_mask, past_key_values=DynamicCache(), **generation_kwargs
            )

            # The two sets of generated sequences must match, despite the cache format between forward passes being
            # different
            self.assertListEqual(legacy_results.sequences.tolist(), new_results.sequences.tolist())
            self.assertTrue(isinstance(legacy_results.past_key_values, tuple))
            self.assertTrue(isinstance(new_results.past_key_values, DynamicCache))

            # The contents of the two caches, when converted to the same format (in both directions!), must match
            legacy_cache = legacy_results.past_key_values
            new_cache_converted = new_results.past_key_values.to_legacy_cache()
            for layer_idx in range(len(legacy_cache)):
                for kv_idx in range(len(legacy_cache[layer_idx])):
                    self.assertTrue(
                        torch.allclose(
                            legacy_cache[layer_idx][kv_idx],
                            new_cache_converted[layer_idx][kv_idx],
                        )
                    )

            new_cache = new_results.past_key_values
            legacy_cache_converted = DynamicCache.from_legacy_cache(legacy_results.past_key_values)
            for layer_idx in range(len(new_cache)):
                for kv_idx in range(len(new_cache[layer_idx])):
                    self.assertTrue(
                        torch.allclose(
                            new_cache[layer_idx][kv_idx],
                            legacy_cache_converted[layer_idx][kv_idx],
                        )
                    )

    def _check_outputs(self, output, input_ids, config, use_cache=False, num_return_sequences=1):
        batch_size, seq_length = input_ids.shape
        num_sequences_in_output = batch_size * num_return_sequences

        gen_len = (
            output.sequences.shape[-1] - 1 if config.is_encoder_decoder else output.sequences.shape[-1] - seq_length
        )

        # scores
        self._check_scores(num_sequences_in_output, output.scores, length=gen_len, config=config)

        # unprocessed logits
        self._check_logits(num_sequences_in_output, output.logits, config=config)

        # Attentions
        if config.is_encoder_decoder:
            # encoder
            self._check_encoder_attention_for_generate(output.encoder_attentions, batch_size, config, seq_length)
            # decoder
            self._check_attentions_for_generate(
                num_sequences_in_output,
                output.decoder_attentions,
                min_length=1,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )
        else:
            # if use_cache first input is equal to no use_cache, so skip here
            attentions = output.attentions if not use_cache else output.attentions[1:]
            min_length = seq_length if not use_cache else seq_length + 1
            self._check_attentions_for_generate(
                num_sequences_in_output,
                attentions=attentions,
                min_length=min_length,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )

        # Hidden States
        if config.is_encoder_decoder:
            # encoder
            self._check_encoder_hidden_states_for_generate(
                output.encoder_hidden_states, batch_size, config, seq_length
            )

            # decoder
            self._check_hidden_states_for_generate(
                num_sequences_in_output,
                output.decoder_hidden_states,
                min_length=1,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )
        else:
            # if use_cache first input is equal to no use_cache, so skip here
            hidden_states = output.hidden_states if not use_cache else output.hidden_states[1:]
            min_length = seq_length if not use_cache else seq_length + 1
            self._check_hidden_states_for_generate(
                num_sequences_in_output,
                hidden_states,
                min_length=min_length,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )

        # Past Key Value States -- a few notes here:
        # 1. Its inner sequence length is with respect to the inputs of the latest forward pass, hence the "-1"
        # 2. Some old models still return `output.past_key_values` even without `use_cache=True`
        # 3. TODO (joao): A few models have different formats/types, skipping those until the cache refactor is
        # complete
        models_without_standard_cache = ("bloom", "ctrl", "fsmt", "gptbigcode", "mega", "reformer", "jamba")
        has_standard_cache = not any(
            model_name in config.__class__.__name__.lower() for model_name in models_without_standard_cache
        )
        if use_cache and has_standard_cache:
            past_key_values = output.past_key_values
            past_sequence_length = output.sequences.shape[-1] - 1
            self._check_past_key_values_for_generate(
                num_sequences_in_output,
                past_key_values,
                seq_length=past_sequence_length,
                config=config,
            )

    def _check_scores(self, batch_size, scores, length, config):
        expected_shape = (batch_size, config.vocab_size)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(len(scores), length)
        self.assertListEqual([iter_scores.shape for iter_scores in scores], [expected_shape] * len(scores))

    def _check_logits(self, batch_size, scores, config):
        self.assertIsInstance(scores, tuple)
        self.assertListEqual([iter_scores.shape[0] for iter_scores in scores], [batch_size] * len(scores))
        # vocabulary difference equal to one (imagegptmodel?) or zero (all other models)
        vocab_diff = config.vocab_size - scores[0].shape[-1]
        self.assertTrue(vocab_diff in [0, 1])
        self.assertListEqual([config.vocab_size - score.shape[-1] for score in scores], [vocab_diff] * len(scores))

    def _check_attentions_for_generate(
        self, batch_size, attentions, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (max_length - min_length) * num_beam_groups)

        for idx, iter_attentions in enumerate(attentions):
            tgt_len = min_length + idx if not use_cache else 1
            src_len = min_length + idx

            expected_shape = (
                batch_size * num_beam_groups,
                config.num_attention_heads,
                tgt_len,
                src_len,
            )
            # check attn size
            self.assertListEqual(
                [layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions)
            )

    def _check_encoder_attention_for_generate(self, attentions, batch_size, config, seq_length):
        encoder_expected_shape = (batch_size, config.num_attention_heads, seq_length, seq_length)
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [layer_attentions.shape for layer_attentions in attentions],
            [encoder_expected_shape] * len(attentions),
        )

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (max_length - min_length) * num_beam_groups)

        for idx, iter_hidden_states in enumerate(hidden_states):
            seq_len = min_length + idx if not use_cache else 1
            expected_shape = (batch_size * num_beam_groups, seq_len, config.hidden_size)
            # check hidden size
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [expected_shape] * len(iter_hidden_states),
            )

    def _check_encoder_hidden_states_for_generate(self, hidden_states, batch_size, config, seq_length):
        encoder_expected_shape = (batch_size, seq_length, config.hidden_size)
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [layer_hidden_states.shape for layer_hidden_states in hidden_states],
            [encoder_expected_shape] * len(hidden_states),
        )

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config, num_beam_groups=1):
        self.assertIsInstance(past_key_values, tuple)
        self.assertListEqual(
            [isinstance(iter_past_key_values, tuple) for iter_past_key_values in past_key_values],
            [True] * len(past_key_values),
        )

        # (batch, head, seq_length, head_features)
        expected_shape = (
            batch_size * num_beam_groups,
            config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads,
            seq_length,
            config.hidden_size // config.num_attention_heads,
        )
        # check shape key, value
        self.assertListEqual(
            [layer_past_key_values[0].shape for layer_past_key_values in past_key_values],
            [expected_shape] * len(past_key_values),
        )
        self.assertListEqual(
            [layer_past_key_values[1].shape for layer_past_key_values in past_key_values],
            [expected_shape] * len(past_key_values),
        )

    def _check_sequence_inside_sequence(self, tensor_1, tensor_2):
        # check if tensor_1 inside tensor_2 or tensor_2 inside tensor_1.
        # set to same device. we don't care what device.

        if not isinstance(tensor_1, list):
            tensor_1 = tensor_1.cpu().tolist()
        if not isinstance(tensor_2, list):
            tensor_2 = tensor_2.cpu().tolist()

        in_order = len(tensor_1) <= len(tensor_2)
        longer = tensor_2 if in_order else tensor_1
        shorter = tensor_1 if in_order else tensor_2

        flag = False
        chunk_size = len(shorter)
        for chunk_idx in range(len(longer) - chunk_size + 1):
            subseq = longer[chunk_idx : chunk_idx + chunk_size]
            if subseq == shorter:
                flag = True
                break

        self.assertTrue(flag)


@require_torch
class UtilsFunctionsTest(unittest.TestCase):
    def test_speculative_sampling(self):
        # assume vocab size 10, input length 5 + 3 generated candidates
        candidate_input_ids = torch.tensor([[8, 0, 3, 9, 8, 1, 4, 5]])  # input tokens
        candidate_logits = torch.tensor(
            [
                [
                    [-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # generated 1
                    [-10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # generated 4
                    [-10.0, -10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0],  # generated 5
                ]
            ]
        )
        candidate_length = 3
        inf = float("inf")
        new_logits = torch.tensor(
            [
                [
                    [-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # accepts 1
                    [-10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # accepts 4
                    [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, 10.0, -inf],  # rejects 5, accepts 8
                    [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # N/A
                ]
            ]
        )
        last_assistant_token_is_eos = False
        validated_tokens, n_matches = _speculative_sampling(
            candidate_input_ids,
            candidate_logits,
            candidate_length,
            new_logits,
            last_assistant_token_is_eos,
        )
        self.assertTrue(n_matches.item() == 2)
        self.assertTrue(validated_tokens.tolist()[0] == [1, 4, 8])


@require_torch
class GenerationIntegrationTests(unittest.TestCase, GenerationIntegrationTestsMixin):
    # setting framework_dependent_parameters needs to be gated, just like its contents' imports
    if is_torch_available():
        framework_dependent_parameters = {
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoModelForSpeechSeq2Seq": AutoModelForSpeechSeq2Seq,
            "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
            "AutoModelForVision2Seq": AutoModelForVision2Seq,
            "LogitsProcessorList": LogitsProcessorList,
            "MinLengthLogitsProcessor": MinLengthLogitsProcessor,
            "create_tensor_fn": torch.tensor,
            "floats_tensor": floats_tensor,
            "return_tensors": "pt",
        }

    @slow
    def test_diverse_beam_search(self):
        # PT-only test: TF doesn't have a diverse beam search implementation
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood.
        The celebrity couple announced the arrival of their son, Silas Randall Timberlake, in statements to People.
        "Silas was the middle name of Timberlake's maternal grandfather Bill Bomar, who died in 2012, while Randall is the musician's own middle name, as well as his father's first," People reports.
        The couple announced the pregnancy in January, with an Instagram post. It is the first baby for both."""

        bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        outputs = bart_model.generate(
            input_ids,
            num_beams=4,
            num_return_sequences=2,
            num_beam_groups=4,
            diversity_penalty=2.0,
            remove_invalid_values=True,
        )

        generated_text = bart_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The couple announced the birth of their son, Silas Randall Timberlake, in a statement. Silas was the"
                " middle name of Timberlake's maternal grandfather Bill Bomar. Randall is the musician's own middle"
                " name, as well as his father's first. It is the first baby for both of them.",
                "Justin Timberlake and Jessica Biel have a son. The baby is named Silas Randall Timberlake. It is the"
                " first child for both. The couple announced the pregnancy in January. The name Silas is the middle"
                " name of Timberlake's maternal grandfather. It's also his own middle name.",
            ],
        )

    def test_max_length_if_input_embeds(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = "Today a dragon flew over Paris."
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        input_ids = tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        max_length = 20
        input_len = input_ids.shape[-1]
        out_gen = model.generate(input_ids=input_ids, max_length=max_length)
        out_gen_embeds = model.generate(inputs_embeds=inputs_embeds, max_length=max_length)
        self.assertEqual(out_gen.shape[-1], input_len + out_gen_embeds.shape[-1])

    def test_min_length_if_input_embeds(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = "Today a dragon flew over Paris."
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        input_ids = tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        min_length = 10
        input_len = input_ids.shape[-1]
        out_gen = model.generate(input_ids=input_ids, min_length=min_length)
        out_gen_embeds = model.generate(inputs_embeds=inputs_embeds, min_length=min_length)
        self.assertEqual(out_gen.shape[-1], input_len + out_gen_embeds.shape[-1])

    def test_custom_stopping_criteria_overload_error(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)

        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=42))
        with self.assertRaises(ValueError):
            bart_model.generate(input_ids, stopping_criteria=stopping_criteria)
        with self.assertRaises(ValueError):
            bart_model.generate(input_ids, stopping_criteria=stopping_criteria, max_length=32)

    def test_custom_stopping_criteria(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        class DummyCriteria(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                return input_ids.shape[-1] >= 20

        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(DummyCriteria())

        self.assertEqual(
            list(bart_model.generate(input_ids, stopping_criteria=stopping_criteria, max_length=22).shape),
            [1, 20],
        )
        self.assertEqual(
            list(bart_model.generate(input_ids, stopping_criteria=stopping_criteria, max_length=18).shape),
            [1, 18],
        )

    def test_stop_sequence_stopping_criteria(self):
        # PT-only test: TF doesn't have StoppingCriteria
        prompt = """Hello I believe in"""
        generator = pipeline("text-generation", model="hf-internal-testing/tiny-random-bart")
        output = generator(prompt)
        self.assertEqual(
            output,
            [
                {
                    "generated_text": (
                        "Hello I believe in in in number number number number number number number number number"
                    )
                }
            ],
        )

        output = generator(prompt, stop_sequence=" number")
        self.assertEqual(output, [{"generated_text": "Hello I believe in in in number"}])

    def test_generate_non_nlp_input_ids_as_kwarg(self):
        # PT-only test: AFAIK there's no non-NLP model architecture in TF that supports `input_ids` as its only input
        model = ImageGPTForCausalImageModeling.from_pretrained(
            "hf-internal-testing/tiny-random-imagegpt", max_length=10
        ).to(torch_device)
        input_ids = ids_tensor((3, 5), vocab_size=10)

        output_sequences_kwargs = model.generate(input_ids=input_ids).cpu()
        output_sequences = model.generate(input_ids).cpu()

        self.assertListEqual(output_sequences.tolist(), output_sequences_kwargs.tolist())
        self.assertEqual(output_sequences.shape, (3, 10))

    def test_generate_input_values_as_encoder_kwarg(self):
        # PT-only test: AFAIK there's no generate-capable architecture in TF that supports `input_values` as its input
        input_values = floats_tensor((2, 250))
        model = SpeechEncoderDecoderModel.from_pretrained("hf-internal-testing/tiny-random-speech-encoder-decoder")
        model = model.to(torch_device)
        output_sequences_kwargs = model.generate(input_values=input_values, max_length=5).cpu()
        output_sequences = model.generate(input_values, max_length=5).cpu()

        self.assertListEqual(output_sequences.tolist(), output_sequences_kwargs.tolist())
        self.assertEqual(output_sequences.shape, (2, 5))

    def test_transition_scores_group_beam_search_encoder_decoder(self):
        # PT-only test: TF doesn't have group beam search
        articles = [
            "Justin Timberlake and Jessica Biel, welcome to parenthood.",
            "Michael Phelps is arguably the most decorated Olympian of all time.",
        ]
        tokenizer = BartTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        model = BartForConditionalGeneration.from_pretrained(
            "hf-internal-testing/tiny-random-bart",
            max_length=10,
            num_beams=2,
            num_beam_groups=2,
            num_return_sequences=2,
            diversity_penalty=1.0,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=0.0,
        )
        model = model.to(torch_device)

        input_ids = tokenizer(articles, return_tensors="pt", padding=True).input_ids.to(torch_device)
        outputs = model.generate(input_ids=input_ids)

        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices)
        transition_scores_sum = transition_scores.sum(-1)

        self.assertTrue(torch.allclose(transition_scores_sum, outputs.sequences_scores, atol=1e-3))

    def test_beam_search_low_memory(self):
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model_inputs = tokenizer("I", return_tensors="pt")["input_ids"]

        low_output = model.generate(model_inputs, max_new_tokens=40, num_beams=5, early_stopping=True, low_memory=True)

        high_output = model.generate(
            model_inputs, max_new_tokens=40, num_beams=5, early_stopping=True, low_memory=False
        )
        self.assertListEqual(low_output.tolist(), high_output.tolist())

    @slow
    def test_beam_search_example_integration(self):
        # PT-only test: TF doesn't have a BeamSearchScorer
        # exactly the example provided in the docstrings of beam search, which previously
        # failed after directly copying from it. Refer to PR #15555
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        encoder_input_str = "translate English to German: How old are you?"
        encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        # lets run beam search using 3 beams
        num_beams = 3
        # define decoder start token ids
        input_ids = torch.ones((1, 1), device=model.device, dtype=torch.long)
        input_ids = input_ids * model.config.decoder_start_token_id

        # add encoder_outputs to model keyword arguments
        model_kwargs = {"encoder_outputs": model.get_encoder()(encoder_input_ids, return_dict=True)}

        outputs = model.generate(
            input_ids, num_beams=num_beams, min_length=5, eos_token_id=model.config.eos_token_id, **model_kwargs
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(outputs, ["Wie alt bist du?"])

    @slow
    def test_constrained_beam_search(self):
        # PT-only test: TF doesn't have constrained beam search
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        force_tokens = tokenizer("scared", add_prefix_space=True, add_special_tokens=False).input_ids
        force_tokens_2 = tokenizer("big weapons", add_prefix_space=True, add_special_tokens=False).input_ids

        constraints = [
            PhrasalConstraint(force_tokens),
            PhrasalConstraint(force_tokens_2),
        ]

        starting_text = ["The soldiers were not prepared and"]

        input_ids = tokenizer(starting_text, return_tensors="pt").input_ids.to(torch_device)

        outputs = model.generate(
            input_ids,
            constraints=constraints,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            max_length=30,
            remove_invalid_values=True,
        )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The soldiers were not prepared and didn't know what to do. They had no idea how they would react if"
                " the enemy attacked them, big weapons scared"
            ],
        )

    @slow
    def test_constrained_beam_search_mixed(self):
        # PT-only test: TF doesn't have constrained beam search
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        force_phrase = tokenizer("scared", add_prefix_space=True, add_special_tokens=False).input_ids
        flexible_phrases = tokenizer(
            ["scream", "screams", "screaming", "screamed"], add_prefix_space=True, add_special_tokens=False
        ).input_ids

        constraints = [
            PhrasalConstraint(force_phrase),
            DisjunctiveConstraint(flexible_phrases),
        ]

        starting_text = ["The soldiers", "The child"]

        input_ids = tokenizer(starting_text, return_tensors="pt").input_ids.to(torch_device)

        outputs = model.generate(
            input_ids,
            constraints=constraints,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            # max_length=20,
            remove_invalid_values=True,
        )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The soldiers, who had been stationed at the base for more than a year before being evacuated"
                " screaming scared",
                "The child was taken to a local hospital where he died.\n 'I don't think screaming scared",
            ],
        )

    @slow
    def test_constrained_beam_search_mixed_mixin(self):
        # PT-only test: TF doesn't have constrained beam search
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        force_word = "scared"
        force_flexible = ["scream", "screams", "screaming", "screamed"]

        force_words_ids = [
            tokenizer([force_word], add_prefix_space=True, add_special_tokens=False).input_ids,
            tokenizer(force_flexible, add_prefix_space=True, add_special_tokens=False).input_ids,
        ]

        starting_text = ["The soldiers", "The child"]

        input_ids = tokenizer(starting_text, return_tensors="pt").input_ids.to(torch_device)

        outputs = model.generate(
            input_ids,
            force_words_ids=force_words_ids,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The soldiers, who had been stationed at the base for more than a year before being evacuated"
                " screaming scared",
                "The child was taken to a local hospital where he died.\n 'I don't think screaming scared",
            ],
        )

    @slow
    def test_cfg_mixin(self):
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        input = tokenizer(["The dragon flew over Paris,"], return_tensors="pt", return_attention_mask=True)
        input["input_ids"] = input["input_ids"].to(torch_device)
        input["attention_mask"] = input["attention_mask"].to(torch_device)

        outputs = model.generate(**input, max_new_tokens=32, guidance_scale=1.5)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The dragon flew over Paris, landing in the Rue de la Bastille. The crowd was so excited "
                'that they had to leave the city.\n\n"We\'re going to Paris!"\n'
            ],
        )

        neg = tokenizer(["France,"], return_tensors="pt", return_attention_mask=True)
        neg["input_ids"] = neg["input_ids"].to(torch_device)
        neg["attention_mask"] = neg["attention_mask"].to(torch_device)
        outputs = model.generate(
            **input,
            max_new_tokens=32,
            guidance_scale=1.5,
            negative_prompt_ids=neg["input_ids"],
            negative_prompt_attention_mask=neg["attention_mask"],
        )
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                'The dragon flew over Paris, landing on the pavement.\n\n"Paris!"\n\n"Paris!"\n\n"'
                'Paris!"\n\n"Paris!"\n\n"Paris!"\n\n'
            ],
        )

    @slow
    def test_constrained_beam_search_example_translation_mixin(self):
        # PT-only test: TF doesn't have constrained beam search
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        encoder_input_str = "translate English to German: How old are you?"
        force_words = ["sind"]

        input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
        force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids

        outputs = model.generate(
            input_ids,
            force_words_ids=force_words_ids,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )

        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(outputs, ["Wie alt sind Sie?"])

    @slow
    def test_constrained_beam_search_example_integration(self):
        # PT-only test: TF doesn't have constrained beam search
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        encoder_input_str = "translate English to German: How old are you?"
        encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        # lets run beam search using 5 beams
        num_beams = 5
        # define decoder start token ids
        input_ids = torch.ones((1, 1), device=model.device, dtype=torch.long)
        input_ids = input_ids * model.config.decoder_start_token_id

        # add encoder_outputs to model keyword arguments
        model_kwargs = {"encoder_outputs": model.get_encoder()(encoder_input_ids, return_dict=True)}

        constraint_str = "sind"
        constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  # remove eos token

        outputs = model.generate(
            input_ids,
            num_beams=num_beams,
            force_words_ids=[constraint_token_ids],
            min_length=5,
            eos_token_id=model.config.eos_token_id,
            **model_kwargs,
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(outputs, ["Wie alt sind Sie?"])

    def test_constrained_beam_search_mixin_type_checks(self):
        # PT-only test: TF doesn't have constrained beam search
        tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/t5-tiny-random")
        model = AutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/t5-tiny-random")

        encoder_input_str = "translate English to German: How old are you?"
        input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        with self.assertRaises(ValueError):
            force_words = ["sind"]
            force_words_ids = tokenizer(force_words, return_tensors="pt").input_ids
            model.generate(
                input_ids,
                force_words_ids=force_words_ids,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
            )

        with self.assertRaises(ValueError):
            force_words = ["sind"]
            force_words_ids = [tokenizer(force_words, return_tensors="pt").input_ids]
            model.generate(
                input_ids,
                force_words_ids=force_words_ids,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
            )

        with self.assertRaises(ValueError):
            model.generate(input_ids, force_words_ids=[])

        with self.assertRaises(ValueError):
            model.generate(input_ids, force_words_ids=[[-1]])

        with self.assertRaises(ValueError):
            model.generate(input_ids, force_words_ids=[[[-1]]])

    def test_batched_decoder_start_id(self):
        # PT-only test: TF doesn't support batched_decoder_start_id
        articles = [
            "Justin Timberlake and Jessica Biel, welcome to parenthood.",
            "Michael Phelps is arguably the most decorated Olympian of all time.",
        ]
        bart_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        bart_model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(
            torch_device
        )
        input_ids = bart_tokenizer(articles, return_tensors="pt", padding=True).input_ids.to(torch_device)
        decoder_start_token_id = bart_model.generation_config.decoder_start_token_id
        decoder_start_token_id_batch = [decoder_start_token_id] * input_ids.shape[0]

        outputs = bart_model.generate(input_ids, decoder_start_token_id=decoder_start_token_id)

        outputs_batched_ids = bart_model.generate(input_ids, decoder_start_token_id=decoder_start_token_id_batch)

        self.assertListEqual(outputs.tolist(), outputs_batched_ids.tolist())

    def test_contrastive_search_batched(self):
        # PT-only test: TF doesn't have constrained beam search
        # Tests that contrastive search works with batched inputs (i.e. has the same output as for non-batched inputs)
        articles = ["Foo", "Bar Baz"]
        tokenizer = BartTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(torch_device)

        model.config.eos_token_id = None
        input_ids_batched = tokenizer(articles, padding=True, return_tensors="pt").input_ids.to(torch_device)
        input_ids = tokenizer(articles[1], return_tensors="pt").input_ids.to(torch_device)

        output_sequences_batched = model.generate(
            input_ids=input_ids_batched, penalty_alpha=0.6, top_k=4, return_dict_in_generate=True, output_scores=True
        )
        output_sequences = model.generate(
            input_ids=input_ids, penalty_alpha=0.6, top_k=4, return_dict_in_generate=True, output_scores=True
        )

        batched_out = tokenizer.decode(output_sequences_batched.sequences[1], skip_special_tokens=True)
        out = tokenizer.decode(output_sequences.sequences[0], skip_special_tokens=True)
        self.assertEqual(batched_out, out)

        # output_sequences_batched.scores[0][1] -> 1st set of logits, 2nd sequence
        max_score_diff = (output_sequences_batched.scores[0][1] - output_sequences.scores[0][0]).abs().max()
        self.assertTrue(max_score_diff < 1e-5)

    def test_logits_processor_not_inplace(self):
        # PT-only test: TF fixes were not made
        article = "Today a dragon flew over Paris."
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        input_ids = tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        out = model.generate(input_ids, output_logits=True, output_scores=True, return_dict_in_generate=True)
        out_with_temp = model.generate(
            input_ids,
            temperature=0.5,
            do_sample=True,
            output_logits=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # if no logits processor is used, scores == logits. Otherwise, the processor has to modify the scores
        self.assertListEqual(out.logits[-1].tolist(), out.scores[-1].tolist())
        self.assertNotEqual(out_with_temp.logits[-1].tolist(), out_with_temp.scores[-1].tolist())

    def test_eos_token_id_int_and_list_top_k_top_sampling(self):
        # Has TF equivalent: this test relies on random sampling
        generation_kwargs = {
            "do_sample": True,
            "num_beams": 1,
            "top_p": 0.7,
            "top_k": 10,
            "temperature": 0.7,
        }
        expectation = 20

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        text = """Hello, my dog is cute and"""
        tokens = tokenizer(text, return_tensors="pt").to(torch_device)
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)

        # Only some seeds will work both on CPU/GPU for a fixed `expectation` value.
        # The selected seed is not guaranteed to work on all torch versions.
        torch.manual_seed(1)
        eos_token_id = 846
        generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

        torch.manual_seed(1)
        eos_token_id = [846, 198]
        generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

    def test_model_kwarg_encoder_signature_filtering(self):
        # Has TF equivalent: ample use of framework-specific code
        bart_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        article = """Hugging Face is a technology company based in New York and Paris."""
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        bart_model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(
            torch_device
        )
        output = bart_model.generate(input_ids).cpu().numpy()

        # Let's create a fake model that has a different signature. In particular, this fake model accepts "foo" as an
        # argument. Because "foo" is not in the encoder signature and doesn't start with "decoder_", it will be part of
        # the encoder kwargs prior to signature filtering, which would lead to an exception. But filtering kicks in and
        # saves the day.
        class FakeBart(BartForConditionalGeneration):
            def forward(self, input_ids, foo=None, **kwargs):
                return super().forward(input_ids, **kwargs)

        bart_model = FakeBart.from_pretrained("hf-internal-testing/tiny-random-bart").to(torch_device)
        fake_output = bart_model.generate(input_ids, foo="bar").cpu().numpy()
        self.assertTrue(np.array_equal(output, fake_output))

        # Encoder signature filtering only kicks in if it doesn't accept wildcard kwargs. The following test will fail
        # because it doesn't do signature filtering.
        class FakeEncoder(bart_model.model.encoder.__class__):
            def forward(self, input_ids, **kwargs):
                return super().forward(input_ids, **kwargs)

        fake_encoder = FakeEncoder(bart_model.config, bart_model.model.shared).to(torch_device)
        bart_model.model.encoder = fake_encoder

        # Normal generation still works (the output will be different because the encoder weights are different)
        fake_output = bart_model.generate(input_ids).cpu().numpy()
        with self.assertRaises(TypeError):
            # FakeEncoder.forward() accepts **kwargs -> no filtering -> type error due to unexpected input "foo"
            bart_model.generate(input_ids, foo="bar")

    def test_default_max_length_warning(self):
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model.config.pad_token_id = tokenizer.eos_token_id

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        # Default generation config value of 20 -> emits warning
        with self.assertWarns(UserWarning):
            model.generate(input_ids)

        # Explicitly setting max_length to 20 -> no warning
        with warnings.catch_warnings(record=True) as warning_list:
            model.generate(input_ids, max_length=20)
            self.assertEqual(len(warning_list), 0)

        # Generation config max_length != 20 -> no warning
        with warnings.catch_warnings(record=True) as warning_list:
            # generation_config is modified -> legacy mode is disabled = generation_config takes precedence
            model.generation_config.max_length = 10
            model.generate(input_ids)
            self.assertEqual(len(warning_list), 0)

    def test_length_warning_assisted_generation(self):
        # PT-only test: TF doesn't support assisted decoding yet.
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        assistant = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model.config.pad_token_id = tokenizer.eos_token_id
        assistant.config.pad_token_id = tokenizer.eos_token_id

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        # This should not raise any warning that min length is not feasible in candidate generation
        with warnings.catch_warnings(record=True) as warning_list:
            model.generate(
                input_ids,
                assistant_model=assistant,
                min_new_tokens=10,
                max_length=20,
            )
            self.assertEqual(len(warning_list), 0)

    def test_generated_length_assisted_generation(self):
        # PT-only test: TF doesn't support assisted decoding yet.
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        assistant = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model.config.pad_token_id = tokenizer.eos_token_id
        assistant.config.pad_token_id = tokenizer.eos_token_id

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)
        input_length = input_ids.shape[-1]

        out = model.generate(
            input_ids,
            assistant_model=assistant,
            min_new_tokens=10,
            max_new_tokens=20,
        )
        self.assertTrue((10 + input_length) <= out.shape[-1] <= (20 + input_length))

        out = model.generate(
            input_ids,
            assistant_model=assistant,
            min_new_tokens=10,
        )
        self.assertTrue((input_length + 10) <= out.shape[-1] <= 20)

    def test_model_kwarg_assisted_decoding_decoder_only(self):
        # PT-only test: TF doesn't support assisted decoding yet.
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model.config.pad_token_id = tokenizer.eos_token_id

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        # Traditional way of generating text
        outputs_normal = model.generate(input_ids)
        self.assertEqual(outputs_normal.shape, (1, 20))

        # Should be different with token_type_ids
        outputs_tti = model.generate(
            input_ids,
            token_type_ids=torch.zeros(input_ids.shape, dtype=torch.long).to(torch_device),
        )
        with self.assertRaises(AssertionError):
            self.assertListEqual(outputs_tti.tolist(), outputs_normal.tolist())

        # Assistant model
        assistant = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        assistant.config.pad_token_id = tokenizer.eos_token_id

        # If assisted generation passes model_kwargs correctly, should be same as previous
        outputs_assisted = model.generate(
            input_ids,
            token_type_ids=torch.zeros(input_ids.shape, dtype=torch.long).to(torch_device),
            assistant_model=assistant,
        )
        self.assertListEqual(outputs_assisted.tolist(), outputs_tti.tolist())

    def test_model_kwarg_assisted_decoding_encoder_decoder(self):
        """
        Tests that the following scenario is compatible with assisted generation:
        1. encoder-decoder main model
        2. encoder-decoder assistant model
        3. both have a custom input
        (e.g. Whisper)
        """

        # PT-only test: TF doesn't support assisted decoding yet.
        # Bart subclass with a kwarg that distorts the output
        class FakeBart(BartForConditionalGeneration):
            def forward(self, input_ids, past_key_values, foo=False, **kwargs):
                outs = super().forward(input_ids, past_key_values=past_key_values, **kwargs)
                if foo:
                    outs["logits"][:, :, :] = 0.0
                return outs

            def prepare_inputs_for_generation(self, *args, foo=False, encoder_outputs=None, **kwargs):
                kwargs["encoder_outputs"] = encoder_outputs
                inputs = super().prepare_inputs_for_generation(*args, **kwargs)
                inputs["foo"] = foo
                return inputs

        model = FakeBart.from_pretrained("hf-internal-testing/tiny-random-BartForConditionalGeneration").to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BartForConditionalGeneration")

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        # Traditional way of generating text
        outputs_normal = model.generate(input_ids)
        self.assertEqual(outputs_normal.shape, (1, 20))

        # Should be different with foo
        outputs_foo = model.generate(input_ids, foo=True)
        with self.assertRaises(AssertionError):
            self.assertListEqual(outputs_foo.tolist(), outputs_normal.tolist())

        # Assistant model
        assistant = FakeBart.from_pretrained("hf-internal-testing/tiny-random-BartForConditionalGeneration").to(
            torch_device
        )

        # If assisted generation passes model_kwargs correctly, should be same as previous
        outputs_assisted = model.generate(
            input_ids,
            foo=True,
            assistant_model=assistant,
        )
        self.assertListEqual(outputs_assisted.tolist(), outputs_foo.tolist())

        # Check that passing encoder_outputs directly also works as expected
        encoder_outputs = assistant.get_encoder()(input_ids)

        outputs_assisted = model.generate(
            foo=True,
            assistant_model=assistant,
            encoder_outputs=encoder_outputs,
            assistant_encoder_outputs=encoder_outputs,
        )
        self.assertListEqual(outputs_assisted.tolist(), outputs_foo.tolist())

    def test_assisted_decoding_encoder_decoder_shared_encoder(self):
        """
        Tests that the following scenario is compatible with assisted generation:
        1. encoder-decoder main model
        2. decoder-only assistant model
        3. both have a custom input
        (e.g. DistilWhisper)
        """

        # PT-only test: TF doesn't support assisted decoding yet.
        # Bart subclass with a kwarg called foo that distorts the output
        class FakeBartSeq2Seq(BartForConditionalGeneration):
            def forward(self, input_ids, foo=False, **kwargs):
                outs = super().forward(input_ids, **kwargs)
                if foo:
                    outs["logits"][:, :, :] = 0.0
                return outs

            def prepare_inputs_for_generation(self, *args, foo=False, encoder_outputs=None, **kwargs):
                kwargs["encoder_outputs"] = encoder_outputs
                inputs = super().prepare_inputs_for_generation(*args, **kwargs)
                inputs["foo"] = foo
                return inputs

        class FakeBartCausalLM(BartForCausalLM):
            def forward(self, input_ids, attention_mask, past_key_values, foo=False, **kwargs):
                outs = super().forward(input_ids, attention_mask, past_key_values=past_key_values, **kwargs)
                if foo:
                    outs["logits"][:, :, :] = 0.0
                return outs

            def prepare_inputs_for_generation(self, *args, foo=False, encoder_outputs=None, **kwargs):
                kwargs["encoder_outputs"] = encoder_outputs
                inputs = super().prepare_inputs_for_generation(*args, **kwargs)
                inputs["foo"] = foo
                return inputs

        model = FakeBartSeq2Seq.from_pretrained("hf-internal-testing/tiny-random-BartForConditionalGeneration").to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BartForConditionalGeneration")

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        # Traditional way of generating text
        outputs_normal = model.generate(input_ids)
        self.assertEqual(outputs_normal.shape, (1, 20))

        # Should be different with foo
        outputs_foo = model.generate(input_ids, foo=True)
        with self.assertRaises(AssertionError):
            self.assertListEqual(outputs_foo.tolist(), outputs_normal.tolist())

        # Assistant model
        assistant = FakeBartCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-BartForConditionalGeneration"
        ).to(torch_device)

        # If assisted generation passes model_kwargs correctly, should be same as previous
        outputs_assisted = model.generate(
            input_ids,
            foo=True,
            assistant_model=assistant,
        )
        self.assertListEqual(outputs_assisted.tolist(), outputs_foo.tolist())

        # Check that passing encoder_outputs directly also works as expected
        encoder_outputs = model.get_encoder()(input_ids)

        outputs_assisted = model.generate(
            foo=True,
            assistant_model=assistant,
            encoder_outputs=encoder_outputs,
        )
        self.assertListEqual(outputs_assisted.tolist(), outputs_foo.tolist())

    def test_assisted_decoding_num_assistant_tokens_heuristic_schedule(self):
        # This test ensures that the assisted generation num_assistant_tokens 'heuristic' schedule works properly.

        prompt = "Alice and Bob"
        checkpoint = "EleutherAI/pythia-160m-deduped"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        inputs = tokenizer(prompt, return_tensors="pt")

        model = AutoModelForCausalLM.from_pretrained(checkpoint)

        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 5
        assistant_model.generation_config.num_assistant_tokens_schedule = "heuristic"
        generation_kwargs = {
            "eos_token_id": -1,
            "max_new_tokens": 5,
            "do_sample": False,
            "assistant_model": assistant_model,
        }
        model.generate(**inputs, **generation_kwargs)
        # update_candidate_strategy is called only once and therefore, assistant_model.generation_config.num_assistant_tokens should be either 4 or 7
        self.assertTrue(assistant_model.generation_config.num_assistant_tokens in (4, 7))

    def test_assisted_decoding_num_assistant_tokens_heuristic_transient_schedule(self):
        # This test ensures that the assisted generation num_assistant_tokens 'heuristic' schedule works properly.

        prompt = "Alice and Bob"
        checkpoint = "EleutherAI/pythia-160m-deduped"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        inputs = tokenizer(prompt, return_tensors="pt")

        model = AutoModelForCausalLM.from_pretrained(checkpoint)

        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 5
        assistant_model.generation_config.num_assistant_tokens_schedule = "heuristic_transient"
        generation_kwargs = {
            "eos_token_id": -1,
            "max_new_tokens": 5,
            "do_sample": False,
            "assistant_model": assistant_model,
        }
        model.generate(**inputs, **generation_kwargs)
        # update_candidate_strategy is called once but assistant_model.generation_config.num_assistant_tokens should stay 5
        self.assertEqual(assistant_model.generation_config.num_assistant_tokens, 5)

    def test_compare_unprocessed_logit_scores(self):
        # Get unprocessed logit scores back from model generate function.
        # Assert that unprocessed logits from generate() are same as those from modal eval()

        # tell model to generate text and return unprocessed/unwarped logit scores
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        text = "generate yes or no: "
        input_ids = tokenizer([text], return_tensors="pt").input_ids.to(torch_device)

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)

        with torch.no_grad():
            # Get logits for the next token from fwd pass
            logits_fwd = model(input_ids).logits[:, -1, :][0]

        # Get logits for the next token from generate function
        outputs = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_logits=True,
            max_new_tokens=1,
            do_sample=True,
        )
        logits_gen = outputs.logits[0][0]

        # assert that unprocessed logits from generate() are same as those from modal eval()
        self.assertListEqual(logits_fwd.tolist(), logits_gen.tolist())

    def test_return_unprocessed_logit_scores(self):
        # tell model to generate text and return unprocessed/unwarped logit scores
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        text = "generate yes or no: "
        input_ids = tokenizer([text], return_tensors="pt").input_ids.to(torch_device)
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)

        outputs = model.generate(
            input_ids=input_ids, return_dict_in_generate=True, output_logits=True, max_new_tokens=3
        )

        # perform dummy check if unpreprocessed logits make sense.
        # do preselection on high probabilities; find scores of y and n tokens
        probs_all = torch.nn.functional.softmax(outputs.logits[2][0], dim=-1)
        indices = torch.argwhere(probs_all > 0.001)
        indices = indices[:, -1]
        tokens_max = tokenizer.batch_decode(indices, skip_special_tokens=True)
        probs_max = probs_all[probs_all > 0.001]

        self.assertTrue(len(indices) >= 2)
        next_token_dict = {str(t): p for t, p in zip(tokens_max, probs_max)}
        self.assertTrue("n" in next_token_dict)
        self.assertTrue("y" in next_token_dict)
        y_prob = next_token_dict["y"]
        n_prob = next_token_dict["n"]

        self.assertTrue(y_prob > 0.001 and n_prob > 0.001)
        self.assertTrue(y_prob <= 1.0 and n_prob <= 1.0)

    def test_generate_from_inputs_embeds_with_bos_token_id_is_none(self):
        article = "Today a dragon flew over Paris."
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        input_ids = tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        model.generate(inputs_embeds=inputs_embeds, max_length=20, bos_token_id=None)

        # bos_token_id is required when no input ids nor inputs_embeds is passed
        with self.assertRaises(ValueError):
            model.generate(max_length=20, bos_token_id=None)
