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
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import BartForConditionalGeneration, BartTokenizer, top_k_top_p_filtering
    from transformers.generation_beam_search import BeamSearchScorer
    from transformers.generation_logits_process import (
        ForcedBOSTokenLogitsProcessor,
        ForcedEOSTokenLogitsProcessor,
        HammingDiversityLogitsProcessor,
        InfNanRemoveLogitsProcessor,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        NoBadWordsLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
    from transformers.generation_stopping_criteria import MaxLengthCriteria, StoppingCriteriaList
    from transformers.generation_utils import (
        BeamSampleDecoderOnlyOutput,
        BeamSampleEncoderDecoderOutput,
        BeamSearchDecoderOnlyOutput,
        BeamSearchEncoderDecoderOutput,
        GreedySearchDecoderOnlyOutput,
        GreedySearchEncoderDecoderOutput,
        SampleDecoderOnlyOutput,
        SampleEncoderDecoderOutput,
    )


class GenerationTesterMixin:
    model_tester = None
    all_generative_model_classes = ()
    input_name = "input_ids"

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        input_ids = inputs_dict[self.input_name]
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # cut to half length & take max batch_size 3
        max_batch_size = 2
        sequence_length = input_ids.shape[-1] // 2
        input_ids = input_ids[:max_batch_size, :sequence_length]
        attention_mask = attention_mask[:max_batch_size, :sequence_length]

        # generate max 3 tokens
        max_length = input_ids.shape[-1] + 3
        if config.eos_token_id is not None and config.pad_token_id is None:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            config.pad_token_id = config.eos_token_id
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
            "min_length": input_length + 1,
            "bad_words_ids": [[1, 0]],
            "no_repeat_ngram_size": 2,
            "repetition_penalty": 1.2,
        }
        logits_processor = LogitsProcessorList(
            (
                [
                    HammingDiversityLogitsProcessor(diversity_penalty, num_beams=2, num_beam_groups=2),
                ]
                if diversity_penalty is not None
                else []
            )
            + (
                [
                    MinLengthLogitsProcessor(process_kwargs["min_length"], eos_token_id),
                ]
                if eos_token_id is not None
                else []
            )
            + (
                [
                    ForcedBOSTokenLogitsProcessor(forced_bos_token_id),
                ]
                if forced_bos_token_id is not None
                else []
            )
            + (
                [ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id)]
                if forced_eos_token_id is not None
                else []
            )
            + [
                NoBadWordsLogitsProcessor(process_kwargs["bad_words_ids"], eos_token_id),
                NoRepeatNGramLogitsProcessor(process_kwargs["no_repeat_ngram_size"]),
                RepetitionPenaltyLogitsProcessor(process_kwargs["repetition_penalty"]),
            ]
        )
        return process_kwargs, logits_processor

    @staticmethod
    def _get_warper_and_kwargs(num_beams):
        warp_kwargs = {"top_k": 10, "top_p": 0.7, "temperature": 0.7}
        logits_warper = LogitsProcessorList(
            [
                TemperatureLogitsWarper(warp_kwargs["temperature"]),
                TopKLogitsWarper(top_k=warp_kwargs["top_k"], min_tokens_to_keep=(2 if num_beams > 1 else 1)),
                TopPLogitsWarper(top_p=warp_kwargs["top_p"], min_tokens_to_keep=(2 if num_beams > 1 else 1)),
            ]
        )
        return warp_kwargs, logits_warper

    @staticmethod
    def _get_beam_scorer_and_kwargs(batch_size, max_length, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
        }
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=beam_kwargs["num_beams"],
            device=torch_device,
            length_penalty=beam_kwargs["length_penalty"],
            do_early_stopping=beam_kwargs["early_stopping"],
            num_beam_hyps_to_keep=num_return_sequences,
        )
        return beam_kwargs, beam_scorer

    @staticmethod
    def _get_diverse_beam_scorer_and_kwargs(batch_size, max_length, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
            "num_beam_groups": 2,  # one beam per group
            "diversity_penalty": 2.0,
        }
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=beam_kwargs["num_beams"],
            device=torch_device,
            length_penalty=beam_kwargs["length_penalty"],
            do_early_stopping=beam_kwargs["early_stopping"],
            num_beam_hyps_to_keep=num_return_sequences,
            num_beam_groups=beam_kwargs["num_beam_groups"],
        )
        return beam_kwargs, beam_scorer

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
        max_length,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        if model.config.is_encoder_decoder:
            max_length = 4
        logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
            input_ids.shape[-1],
            eos_token_id=model.config.eos_token_id,
            forced_bos_token_id=model.config.forced_bos_token_id,
            forced_eos_token_id=model.config.forced_eos_token_id,
            max_length=max_length,
        )

        kwargs = {}

        output_generate = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_length=max_length,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **logits_process_kwargs,
        )

        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs

        with torch.no_grad():
            output_greedy = model.greedy_search(
                input_ids,
                max_length=max_length,
                attention_mask=attention_mask,
                logits_processor=logits_processor,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )
        return output_greedy, output_generate

    def _sample_generate(
        self,
        model,
        input_ids,
        attention_mask,
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
        output_generate = model.generate(
            input_ids,
            do_sample=True,
            num_beams=1,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            attention_mask=attention_mask,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **logits_warper_kwargs,
            **process_kwargs,
        )

        torch.manual_seed(0)
        kwargs = {}
        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                num_interleave=num_return_sequences,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs
            input_ids_clone = input_ids_clone.repeat_interleave(num_return_sequences, dim=0)
        else:
            attention_mask_clone = attention_mask.repeat_interleave(num_return_sequences, dim=0)
            input_ids_clone = input_ids.repeat_interleave(num_return_sequences, dim=0)

        # prevent flaky generation test failures
        logits_processor.append(InfNanRemoveLogitsProcessor())

        with torch.no_grad():
            output_sample = model.sample(
                input_ids_clone,
                attention_mask=attention_mask_clone,
                max_length=max_length,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                output_scores=output_scores,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )
        return output_sample, output_generate

    def _beam_search_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        beam_scorer,
        beam_kwargs,
        logits_processor,
        logits_process_kwargs,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        output_generate = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_length=max_length,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **beam_kwargs,
            **logits_process_kwargs,
        )

        # beam_search does not automatically interleave `batch_size` dim for `num_beams`
        kwargs = {}
        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                num_interleave=beam_scorer.num_beams,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs
            input_ids_clone = input_ids_clone.repeat_interleave(beam_scorer.num_beams, dim=0)
        else:
            attention_mask_clone = attention_mask.repeat_interleave(beam_scorer.num_beams, dim=0)
            input_ids_clone = input_ids.repeat_interleave(beam_scorer.num_beams, dim=0)

        with torch.no_grad():
            output_beam_search = model.beam_search(
                input_ids_clone,
                beam_scorer,
                max_length=max_length,
                attention_mask=attention_mask_clone,
                logits_processor=logits_processor,
                output_scores=output_scores,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )
        return output_generate, output_beam_search

    def _beam_sample_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        num_return_sequences,
        beam_scorer,
        beam_kwargs,
        logits_warper,
        logits_warper_kwargs,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        torch.manual_seed(0)
        output_generate = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_length=max_length,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **beam_kwargs,
            **logits_warper_kwargs,
        )
        # beam_search does not automatically interleave `batch_size` dim for `num_beams * num_return_sequences`
        kwargs = {}
        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                num_interleave=beam_scorer.num_beams * num_return_sequences,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs
        else:
            attention_mask = attention_mask.repeat_interleave(beam_scorer.num_beams * num_return_sequences, dim=0)

        # prevent flaky generation test failures
        logits_processor = LogitsProcessorList()
        logits_processor.append(InfNanRemoveLogitsProcessor())

        torch.manual_seed(0)
        with torch.no_grad():
            output_beam_sample = model.beam_sample(
                input_ids.repeat_interleave(beam_scorer.num_beams * num_return_sequences, dim=0),
                beam_scorer,
                max_length=max_length,
                attention_mask=attention_mask,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                output_scores=output_scores,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )

        return output_generate, output_beam_sample

    def _group_beam_search_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        beam_scorer,
        beam_kwargs,
        logits_processor,
        logits_process_kwargs,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):
        output_generate = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_length=max_length,
            output_scores=output_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            remove_invalid_values=True,
            **beam_kwargs,
            **logits_process_kwargs,
        )

        # group_beam_search does not automatically interleave `batch_size` dim for `num_beams`
        kwargs = {}
        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                num_interleave=beam_scorer.num_beams,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs
            input_ids_clone = input_ids_clone.repeat_interleave(beam_scorer.num_beams, dim=0)
        else:
            attention_mask_clone = attention_mask.repeat_interleave(beam_scorer.num_beams, dim=0)
            input_ids_clone = input_ids.repeat_interleave(beam_scorer.num_beams, dim=0)

        with torch.no_grad():
            output_group_beam_search = model.group_beam_search(
                input_ids_clone,
                beam_scorer,
                max_length=max_length,
                attention_mask=attention_mask_clone,
                logits_processor=logits_processor,
                output_scores=output_scores,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )
        return output_generate, output_group_beam_search

    def test_greedy_generate(self):
        # check `generate()` and `greedy_search()` are equal
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            # test old generation output for backwards compatibility
            model = model_class(config).to(torch_device).eval()
            output_greedy, output_generate = self._greedy_generate(
                model=model, input_ids=input_ids, attention_mask=attention_mask, max_length=max_length
            )
            self.assertListEqual(output_greedy.tolist(), output_generate.tolist())

    def test_greedy_generate_dict_outputs(self):
        for model_class in self.all_generative_model_classes:
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

            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_greedy, GreedySearchEncoderDecoderOutput)
                self.assertIsInstance(output_generate, GreedySearchEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_greedy, GreedySearchDecoderOnlyOutput)
                self.assertIsInstance(output_generate, GreedySearchDecoderOnlyOutput)

            self.assertListEqual(output_generate.sequences.tolist(), output_greedy.sequences.tolist())

            for output in (output_greedy, output_generate):
                self._check_outputs(output, input_ids, model.config)

    def test_greedy_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            # enable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            if not hasattr(config, "use_cache"):
                # only relevant if model has "use_cache"
                return

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

            self.assertListEqual(output_generate.sequences.tolist(), output_greedy.sequences.tolist())

            for output in (output_greedy, output_generate):
                self._check_outputs(output, input_ids, model.config, use_cache=True)

    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            model = model_class(config).to(torch_device).eval()

            if model.config.is_encoder_decoder:
                max_length = 4

            process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                model.config.eos_token_id,
                forced_bos_token_id=model.config.forced_bos_token_id,
                forced_eos_token_id=model.config.forced_eos_token_id,
                max_length=max_length,
            )
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=1)

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
            self.assertListEqual(output_sample.tolist(), output_generate.tolist())

            # check `generate()` and `sample()` yield equal results for `num_return_sequences`
            output_sample, output_generate = self._sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=3,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
                process_kwargs=process_kwargs,
            )
            self.assertListEqual(output_sample.tolist(), output_generate.tolist())

    def test_sample_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            # disable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            config.use_cache = False
            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 4

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
                num_return_sequences=2,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
                process_kwargs=process_kwargs,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_sample, SampleEncoderDecoderOutput)
                self.assertIsInstance(output_generate, SampleEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_sample, SampleDecoderOnlyOutput)
                self.assertIsInstance(output_generate, SampleDecoderOnlyOutput)

            self.assertListEqual(output_generate.sequences.tolist(), output_sample.sequences.tolist())

            for output in (output_sample, output_generate):
                self._check_outputs(output, input_ids, model.config, num_return_sequences=2)

    def test_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 4

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                config.eos_token_id,
                config.forced_bos_token_id,
                config.forced_eos_token_id,
                max_length,
            )
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(input_ids.shape[0], max_length)

            # check `generate()` and `beam_search()` are equal
            output_generate, output_beam_search = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                logits_processor=logits_processor,
            )
            self.assertListEqual(output_generate.tolist(), output_beam_search.tolist())

            # check `generate()` and `beam_search()` are equal for `num_return_sequences`
            num_return_sequences = 2
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(
                input_ids.shape[0], max_length, num_return_sequences=num_return_sequences
            )

            output_generate, output_beam_search = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                logits_processor=logits_processor,
            )
            self.assertListEqual(output_generate.tolist(), output_beam_search.tolist())

    def test_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # disable cache
            config.use_cache = False

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 4

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                config.eos_token_id,
                config.forced_bos_token_id,
                config.forced_eos_token_id,
                max_length,
            )
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(input_ids.shape[0], max_length)
            output_generate, output_beam_search = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                logits_processor=logits_processor,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_beam_search, BeamSearchEncoderDecoderOutput)
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_beam_search, BeamSearchDecoderOnlyOutput)
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self.assertListEqual(output_generate.sequences.tolist(), output_beam_search.sequences.tolist())
            self.assertTrue(
                torch.allclose(output_generate["sequences_scores"], output_beam_search["sequences_scores"], atol=1e-3)
            )
            self.assertTrue(output_generate["sequences_scores"].shape == (output_generate["sequences"].shape[0],))
            self.assertTrue((output_generate["sequences_scores"] < 0).all().item())

            for output in (output_beam_search, output_generate):
                self._check_outputs(output, input_ids, model.config, num_return_sequences=beam_scorer.num_beams)

    def test_beam_search_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            # enable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            if not hasattr(config, "use_cache"):
                # only relevant if model has "use_cache"
                return

            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 4

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                config.eos_token_id,
                config.forced_bos_token_id,
                config.forced_eos_token_id,
                max_length,
            )

            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(input_ids.shape[0], max_length)

            config.use_cache = True
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            output_beam, output_generate = self._beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_process_kwargs=logits_process_kwargs,
                logits_processor=logits_processor,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertListEqual(output_generate.sequences.tolist(), output_beam.sequences.tolist())

            for output in (output_beam, output_generate):
                self._check_outputs(
                    output, input_ids, model.config, use_cache=True, num_return_sequences=beam_scorer.num_beams
                )

    def test_beam_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=1)

            model = model_class(config).to(torch_device).eval()

            # check `generate()` and `beam_search()` are equal
            # change `num_return_sequences = 2` but not for `beam_scorer`
            num_return_sequences = 2
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(
                input_ids.shape[0] * num_return_sequences, max_length
            )
            beam_kwargs["num_return_sequences"] = num_return_sequences

            output_generate, output_beam_sample = self._beam_sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
            )
            self.assertListEqual(output_generate.tolist(), output_beam_sample.tolist())

    def test_beam_sample_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # disable cache
            config.use_cache = False

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            model = model_class(config).to(torch_device).eval()
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=1)

            num_return_sequences = 2
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(
                input_ids.shape[0] * num_return_sequences, max_length
            )
            beam_kwargs["num_return_sequences"] = num_return_sequences

            output_beam_sample, output_generate = self._beam_sample_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_warper=logits_warper,
                logits_warper_kwargs=logits_warper_kwargs,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_beam_sample, BeamSampleEncoderDecoderOutput)
                self.assertIsInstance(output_generate, BeamSampleEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_beam_sample, BeamSampleDecoderOnlyOutput)
                self.assertIsInstance(output_generate, BeamSampleDecoderOnlyOutput)

            self.assertListEqual(output_generate.sequences.tolist(), output_beam_sample.sequences.tolist())
            self.assertTrue(
                torch.allclose(output_generate["sequences_scores"], output_beam_sample["sequences_scores"], atol=1e-3)
            )
            self.assertTrue(output_generate["sequences_scores"].shape == (output_generate["sequences"].shape[0],))
            self.assertTrue((output_generate["sequences_scores"] < 0).all().item())

            for output in (output_beam_sample, output_generate):
                self._check_outputs(
                    output, input_ids, model.config, num_return_sequences=num_return_sequences * beam_scorer.num_beams
                )

    def test_generate_without_input_ids(self):
        config, _, _, max_length = self._get_input_ids_and_config()

        # if no bos token id => cannot generate from None
        if config.bos_token_id is None:
            return

        for model_class in self.all_generative_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()

            output_ids_generate = model.generate(
                do_sample=False,
                max_length=max_length,
                remove_invalid_values=True,
            )

            self.assertIsNotNone(output_ids_generate)

    def test_group_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 4

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                config.eos_token_id,
                config.forced_bos_token_id,
                config.forced_eos_token_id,
                max_length,
                diversity_penalty=2.0,
            )

            # check `generate()` and `group_beam_search()` are equal
            beam_kwargs, beam_scorer = self._get_diverse_beam_scorer_and_kwargs(input_ids.shape[0], max_length)
            output_generate, output_group_beam_search = self._group_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_processor=logits_processor,
                logits_process_kwargs=logits_process_kwargs,
            )
            self.assertListEqual(output_generate.tolist(), output_group_beam_search.tolist())

            # check `generate()` and `group_beam_search()` are equal for `num_return_sequences`
            num_return_sequences = 2
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_diverse_beam_scorer_and_kwargs(
                input_ids.shape[0], max_length, num_return_sequences=num_return_sequences
            )
            output_generate, output_group_beam_search = self._group_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_processor=logits_processor,
                logits_process_kwargs=logits_process_kwargs,
            )
            self.assertListEqual(output_generate.tolist(), output_group_beam_search.tolist())

    def test_group_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            config.use_cache = False

            # It is important set set the eos_token_id to None to ensure that no sequences
            # shorter than `max_length` can be generated which could lead to flaky circle ci
            # failures if the top `num_return_sequences` beams are all shorter than the longest beam
            config.eos_token_id = None
            config.forced_eos_token_id = None

            model = model_class(config).to(torch_device).eval()
            if model.config.is_encoder_decoder:
                max_length = 4

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1],
                config.eos_token_id,
                config.forced_bos_token_id,
                config.forced_eos_token_id,
                max_length,
                diversity_penalty=2.0,
            )

            num_return_sequences = 1
            beam_kwargs, beam_scorer = self._get_diverse_beam_scorer_and_kwargs(
                input_ids.shape[0], max_length, num_return_sequences=num_return_sequences
            )
            output_generate, output_group_beam_search = self._group_beam_search_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                beam_scorer=beam_scorer,
                beam_kwargs=beam_kwargs,
                logits_processor=logits_processor,
                logits_process_kwargs=logits_process_kwargs,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_group_beam_search, BeamSearchEncoderDecoderOutput)
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_group_beam_search, BeamSearchDecoderOnlyOutput)
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self.assertListEqual(output_generate.sequences.tolist(), output_group_beam_search.sequences.tolist())
            self.assertTrue(
                torch.allclose(
                    output_generate["sequences_scores"], output_group_beam_search["sequences_scores"], atol=1e-3
                )
            )
            self.assertTrue(output_generate["sequences_scores"].shape == (output_generate["sequences"].shape[0],))
            self.assertTrue((output_generate["sequences_scores"] < 0).all().item())

            for output in (output_group_beam_search, output_generate):
                self._check_outputs(
                    output, input_ids, model.config, num_return_sequences=num_return_sequences * beam_scorer.num_beams
                )

    def test_generate_with_head_masking(self):
        """Test designed for encoder-decoder models to ensure the attention head masking is used."""
        attention_names = ["encoder_attentions", "decoder_attentions", "cross_attentions"]
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            model = model_class(config)
            # We want to test only encoder-decoder models
            if not config.is_encoder_decoder:
                continue

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
            if set(head_masking.keys()) < set([*signature.parameters.keys()]):
                continue

            for attn_name, (name, mask) in zip(attention_names, head_masking.items()):
                out = model.generate(
                    input_ids,
                    num_beams=1,
                    max_length=max_length,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    **{name: mask},
                )
                # We check the state of decoder_attentions and cross_attentions just from the last step
                attn_weights = out[attn_name] if attn_name == attention_names[0] else out[attn_name][-1]
                self.assertEqual(sum([w.sum().item() for w in attn_weights]), 0.0)

    def _check_outputs(self, output, input_ids, config, use_cache=False, num_return_sequences=1):
        batch_size, seq_length = input_ids.shape
        num_sequences_in_output = batch_size * num_return_sequences
        gen_len = (
            output.sequences.shape[-1] - 1 if config.is_encoder_decoder else output.sequences.shape[-1] - seq_length
        )

        # scores
        self._check_scores(num_sequences_in_output, output.scores, length=gen_len, config=config)

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

    def _check_scores(self, batch_size, scores, length, config):
        expected_shape = (batch_size, config.vocab_size)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(len(scores), length)
        self.assertListEqual([iter_scores.shape for iter_scores in scores], [expected_shape] * len(scores))

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


@require_torch
class UtilsFunctionsTest(unittest.TestCase):

    # tests whether the top_k_top_p function behaves as expected
    def test_top_k_top_p_filtering(self):
        logits = torch.tensor(
            [
                [
                    8.2220991,  # 3rd highest value; idx. 0
                    -0.5620044,
                    5.23229752,
                    4.0386393,
                    -6.8798378,
                    -0.54785802,
                    -3.2012153,
                    2.92777176,
                    1.88171953,
                    7.35341276,
                    8.43207833,  # 2nd highest value; idx. 10
                    -9.85711836,
                    -5.96209236,
                    -1.13039161,
                    -7.1115294,
                    -0.8369633,
                    -5.3186408,
                    7.06427407,
                    0.81369344,
                    -0.82023817,
                    -5.9179796,
                    0.58813443,
                    -6.99778438,
                    4.71551189,
                    -0.18771637,
                    7.44020759,  # 4th highest value; idx. 25
                    9.38450987,  # 1st highest value; idx. 26
                    2.12662941,
                    -9.32562038,
                    2.35652522,
                ],  # cummulative prob of 4 highest values <= 0.6
                [
                    0.58425518,
                    4.53139238,
                    -5.57510464,
                    -6.28030699,
                    -7.19529503,
                    -4.02122551,
                    1.39337037,
                    -6.06707057,
                    1.59480517,
                    -9.643119,
                    0.03907799,
                    0.67231762,
                    -8.88206726,
                    6.27115922,  # 4th highest value; idx. 13
                    2.28520723,
                    4.82767506,
                    4.30421368,
                    8.8275313,  # 2nd highest value; idx. 17
                    5.44029958,
                    -4.4735794,
                    7.38579536,  # 3rd highest value; idx. 20
                    -2.91051663,
                    2.61946077,
                    -2.5674762,
                    -9.48959302,
                    -4.02922645,
                    -1.35416918,
                    9.67702323,  # 1st highest value; idx. 27
                    -5.89478553,
                    1.85370467,
                ],  # cummulative prob of 4 highest values <= 0.6
            ],
            dtype=torch.float,
            device=torch_device,
        )

        non_inf_expected_idx = torch.tensor(
            [[0, 0], [0, 10], [0, 25], [0, 26], [1, 13], [1, 17], [1, 20], [1, 27]],
            dtype=torch.long,
            device=torch_device,
        )  # expected non filtered idx as noted above

        non_inf_expected_output = torch.tensor(
            [
                8.2221,
                8.4321,
                7.4402,
                9.3845,
                6.2712,
                8.8275,
                7.3858,
                9.6770,
            ],  # expected non filtered values as noted above
            dtype=torch.float,
            device=torch_device,
        )

        output = top_k_top_p_filtering(logits, top_k=10, top_p=0.6, min_tokens_to_keep=4)
        non_inf_output = output[output != -float("inf")].to(device=torch_device)
        non_inf_idx = (output != -float("inf")).nonzero().to(device=torch_device)

        self.assertTrue(torch.allclose(non_inf_expected_output, non_inf_output, atol=1e-12))
        self.assertTrue(torch.all(torch.eq(non_inf_expected_idx, non_inf_idx)))


@require_torch
class GenerationIntegrationTests(unittest.TestCase):
    @slow
    def test_diverse_beam_search(self):
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
                "The couple announced the birth of their son, Silas Randall Timberlake, in a statement. Silas was the middle name of Timberlake's maternal grandfather Bill Bomar. Randall is the musician's own middle name, as well as his father's first. It is the first baby for both of them.",
                "Justin Timberlake and Jessica Biel have a son. The baby is named Silas Randall Timberlake. It is the first child for both. The couple announced the pregnancy in January. The name Silas is the middle name of Timberlake's maternal grandfather. It's also his own middle name.",
            ],
        )

    def test_max_length_backward_compat_greedy(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        max_length = 20
        input_ids = input_ids.expand(2, -1)
        model_kwargs = bart_model._prepare_encoder_decoder_kwargs_for_generation(input_ids, {})
        input_ids = bart_model._prepare_decoder_input_ids_for_generation(
            input_ids,
            decoder_start_token_id=bart_model.config.decoder_start_token_id,
            bos_token_id=bart_model.config.bos_token_id,
        )

        with self.assertWarns(UserWarning):
            bart_model.greedy_search(
                input_ids,
                max_length=max_length,
                pad_token_id=bart_model.config.pad_token_id,
                eos_token_id=bart_model.config.eos_token_id,
                **model_kwargs,
            )

    def test_max_length_backward_compat_sample(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        max_length = 20
        input_ids = input_ids.expand(2, -1)
        model_kwargs = bart_model._prepare_encoder_decoder_kwargs_for_generation(input_ids, {})
        input_ids = bart_model._prepare_decoder_input_ids_for_generation(
            input_ids,
            decoder_start_token_id=bart_model.config.decoder_start_token_id,
            bos_token_id=bart_model.config.bos_token_id,
        )
        with torch.no_grad():
            with self.assertWarns(UserWarning):
                bart_model.sample(
                    input_ids,
                    max_length=max_length,
                    pad_token_id=bart_model.config.pad_token_id,
                    eos_token_id=bart_model.config.eos_token_id,
                    **model_kwargs,
                )

    def test_max_length_backward_compat_beam_search(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        batch_size = 1
        max_length = 20
        num_beams = 2

        input_ids = input_ids.expand(2, -1)
        model_kwargs = bart_model._prepare_encoder_decoder_kwargs_for_generation(input_ids, {})
        input_ids = bart_model._prepare_decoder_input_ids_for_generation(
            input_ids,
            decoder_start_token_id=bart_model.config.decoder_start_token_id,
            bos_token_id=bart_model.config.bos_token_id,
        )

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=torch_device,
        )
        with self.assertWarns(UserWarning):
            _ = bart_model.beam_search(
                input_ids, num_beams=num_beams, max_length=max_length, beam_scorer=beam_scorer, **model_kwargs
            )

    def test_max_length_backward_compat_group_beam_search(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        batch_size = 1
        max_length = 20
        num_beams = 6
        num_beam_groups = 3
        num_return_sequences = num_beams * batch_size

        input_ids = input_ids.expand(6, -1)
        model_kwargs = bart_model._prepare_encoder_decoder_kwargs_for_generation(input_ids, {})
        input_ids = bart_model._prepare_decoder_input_ids_for_generation(
            input_ids,
            decoder_start_token_id=bart_model.config.decoder_start_token_id,
            bos_token_id=bart_model.config.bos_token_id,
        )

        diverse_beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=torch_device,
            num_beam_hyps_to_keep=num_return_sequences,
            num_beam_groups=num_beam_groups,
        )
        with self.assertWarns(UserWarning):
            bart_model.group_beam_search(
                input_ids, diverse_beam_scorer, num_beams=num_beams, max_length=max_length, **model_kwargs
            )

    def test_max_length_warning_if_different(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        batch_size = 1

        max_length = 20
        num_beams = 6
        num_beam_groups = 3
        num_return_sequences = num_beams * batch_size
        stopping_criteria_max_length = 18
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=stopping_criteria_max_length)])

        # Greedy
        input_ids = input_ids.expand(6, -1)
        model_kwargs = bart_model._prepare_encoder_decoder_kwargs_for_generation(input_ids, {})
        input_ids = bart_model._prepare_decoder_input_ids_for_generation(
            input_ids,
            decoder_start_token_id=bart_model.config.decoder_start_token_id,
            bos_token_id=bart_model.config.bos_token_id,
        )

        with self.assertWarns(UserWarning):
            bart_model.greedy_search(
                input_ids,
                max_length=max_length,
                pad_token_id=bart_model.config.pad_token_id,
                stopping_criteria=stopping_criteria,
                eos_token_id=bart_model.config.eos_token_id,
                **model_kwargs,
            )

        # Sample
        with self.assertWarns(UserWarning):
            with torch.no_grad():
                bart_model.sample(
                    input_ids,
                    max_length=max_length,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=bart_model.config.pad_token_id,
                    eos_token_id=bart_model.config.eos_token_id,
                    **model_kwargs,
                )

        # Beam
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=torch_device,
        )
        with self.assertWarns(UserWarning):
            with torch.no_grad():
                bart_model.beam_search(
                    input_ids,
                    num_beams=num_beams,
                    stopping_criteria=stopping_criteria,
                    max_length=max_length,
                    beam_scorer=beam_scorer,
                    **model_kwargs,
                )

        # Grouped beam search
        diverse_beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=torch_device,
            num_beam_hyps_to_keep=num_return_sequences,
            num_beam_groups=num_beam_groups,
        )
        with self.assertWarns(UserWarning):
            bart_model.group_beam_search(
                input_ids,
                diverse_beam_scorer,
                stopping_criteria=stopping_criteria,
                num_beams=num_beams,
                max_length=max_length,
                **model_kwargs,
            )

    def test_beam_search_warning_if_max_length_is_passed(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)

        batch_size = 1
        num_beams = 3

        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        input_ids = input_ids.expand(num_beams, -1)
        model_kwargs = bart_model._prepare_encoder_decoder_kwargs_for_generation(input_ids, {})

        stopping_criteria_max_length = 18
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=stopping_criteria_max_length)])

        with self.assertWarns(UserWarning):
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=torch_device,
                max_length=10,
            )

        generated_ids = bart_model.beam_search(
            input_ids,
            num_beams=num_beams,
            stopping_criteria=stopping_criteria,
            beam_scorer=beam_scorer,
            **model_kwargs,
        )

        beam_scorer_no_max_len = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=torch_device,
        )

        generated_ids_no_max_len = bart_model.beam_search(
            input_ids,
            num_beams=num_beams,
            stopping_criteria=stopping_criteria,
            beam_scorer=beam_scorer_no_max_len,
            **model_kwargs,
        )

        # BeamSearchScorer max_length should not influence "real" max_length
        self.assertEqual(generated_ids.tolist(), generated_ids_no_max_len.tolist())

    def test_max_new_tokens(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        self.assertEqual(list(input_ids.shape), [1, 15])

        # Encoder decoder call
        max_new_tokens = 3
        outputs = bart_model.generate(input_ids, max_new_tokens=max_new_tokens)
        # 1 BOS + 3 new tokens
        self.assertEqual(list(outputs.shape), [1, 4])

        # Decoder only call
        outputs = bart_model.generate(decoder_input_ids=input_ids, max_new_tokens=max_new_tokens)
        # 15 + 3 new tokens
        self.assertEqual(list(outputs.shape), [1, 18])

        # max_new_tokens and max_length serve the same purpose and should not be used together.
        with self.assertWarns(UserWarning):
            outputs = bart_model.generate(decoder_input_ids=input_ids, max_new_tokens=10, max_length=20)
