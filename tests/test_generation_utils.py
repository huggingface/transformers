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


import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import top_k_top_p_filtering
    from transformers.generation_beam_search import BeamSearchScorer
    from transformers.generation_logits_process import (
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        NoBadWordsLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )


class GenerationTesterMixin:
    model_tester = None
    all_generative_model_classes = ()

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        input_ids = inputs_dict["input_ids"]
        attention_mask = torch.ones_like(input_ids)

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
    def _get_logits_processor_and_kwargs(input_length, eos_token_id):
        process_kwargs = {
            "min_length": input_length + 1,
            "bad_words_ids": [[1, 0]],
            "no_repeat_ngram_size": 2,
            "repetition_penalty": 1.2,
        }
        logits_processor = LogitsProcessorList(
            (
                [
                    MinLengthLogitsProcessor(process_kwargs["min_length"], eos_token_id),
                ]
                if eos_token_id is not None
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
            max_length=max_length,
            num_beams=beam_kwargs["num_beams"],
            device=torch_device,
            length_penalty=beam_kwargs["length_penalty"],
            do_early_stopping=beam_kwargs["early_stopping"],
            num_beam_hyps_to_keep=num_return_sequences,
        )
        return beam_kwargs, beam_scorer

    @staticmethod
    def _get_encoder_outputs(model, input_ids, attention_mask, num_interleave=1):
        encoder = model.get_encoder()
        encoder_outputs = encoder(input_ids, attention_mask=attention_mask)
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
            num_interleave, dim=0
        )
        input_ids = torch.zeros_like(input_ids[:, :1]) + model._get_decoder_start_token_id()
        attention_mask = None
        return encoder_outputs, input_ids, attention_mask

    def test_greedy_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1], config.eos_token_id
            )

            model = model_class(config).to(torch_device)
            model.eval()

            # check `generate()` and `greedy_search()` are equal
            kwargs = {}
            if model.config.is_encoder_decoder:
                max_length = 4

            output_ids_generate = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_length=max_length,
                **logits_process_kwargs,
            )

            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                    model, input_ids, attention_mask
                )
                kwargs["encoder_outputs"] = encoder_outputs

            with torch.no_grad():
                output_ids_greedy = model.greedy_search(
                    input_ids,
                    max_length=max_length,
                    attention_mask=attention_mask,
                    logits_processor=logits_processor,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_greedy.tolist())

    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1], config.eos_token_id
            )
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=1)

            model = model_class(config).to(torch_device)
            model.eval()

            # check `generate()` and `sample()` are equal
            if model.config.is_encoder_decoder:
                max_length = 4

            torch.manual_seed(0)
            output_ids_generate = model.generate(
                input_ids,
                do_sample=True,
                num_beams=1,
                max_length=max_length,
                attention_mask=attention_mask,
                **logits_warper_kwargs,
                **process_kwargs,
            )

            torch.manual_seed(0)
            kwargs = {}
            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                    model, input_ids, attention_mask
                )
                kwargs["encoder_outputs"] = encoder_outputs
            else:
                attention_mask_clone = attention_mask
                input_ids_clone = input_ids

            with torch.no_grad():
                output_ids_sample = model.sample(
                    input_ids_clone,
                    attention_mask=attention_mask_clone,
                    max_length=max_length,
                    logits_processor=logits_processor,
                    logits_warper=logits_warper,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_sample.tolist())

            # check `generate()` and `sample()` yield equal results for `num_return_sequences`
            num_return_sequences = 3
            if model.config.is_encoder_decoder:
                max_length = 4

            torch.manual_seed(0)
            output_ids_generate = model.generate(
                input_ids,
                do_sample=True,
                num_beams=1,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                attention_mask=attention_mask,
                **logits_warper_kwargs,
                **process_kwargs,
            )

            torch.manual_seed(0)
            kwargs = {}
            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                    model, input_ids, attention_mask, num_interleave=num_return_sequences
                )
                kwargs["encoder_outputs"] = encoder_outputs
                input_ids_clone = input_ids_clone.repeat_interleave(num_return_sequences, dim=0)
            else:
                attention_mask_clone = attention_mask.repeat_interleave(num_return_sequences, dim=0)
                input_ids_clone = input_ids.repeat_interleave(num_return_sequences, dim=0)

            with torch.no_grad():
                output_ids_sample = model.sample(
                    input_ids_clone,
                    attention_mask=attention_mask_clone,
                    max_length=max_length,
                    logits_processor=logits_processor,
                    logits_warper=logits_warper,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_sample.tolist())

    def test_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1], config.eos_token_id
            )

            model = model_class(config).to(torch_device)
            model.eval()

            # check `generate()` and `beam_search()` are equal
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(input_ids.shape[0], max_length)
            output_ids_generate = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_length=max_length,
                **beam_kwargs,
                **logits_process_kwargs,
            )

            # beam_search does not automatically interleave `batch_size` dim for `num_beams`
            kwargs = {}
            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                    model, input_ids, attention_mask, num_interleave=beam_scorer.num_beams
                )
                kwargs["encoder_outputs"] = encoder_outputs
                input_ids_clone = input_ids_clone.repeat_interleave(beam_scorer.num_beams, dim=0)
            else:
                attention_mask_clone = attention_mask.repeat_interleave(beam_scorer.num_beams, dim=0)
                input_ids_clone = input_ids.repeat_interleave(beam_scorer.num_beams, dim=0)

            with torch.no_grad():
                output_ids_beam_search = model.beam_search(
                    input_ids_clone,
                    beam_scorer,
                    max_length=max_length,
                    attention_mask=attention_mask_clone,
                    logits_processor=logits_processor,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_beam_search.tolist())

            # check `generate()` and `beam_search()` are equal for `num_return_sequences`
            num_return_sequences = 2
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(
                input_ids.shape[0], max_length, num_return_sequences=num_return_sequences
            )

            output_ids_generate = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_length=max_length,
                **beam_kwargs,
                **logits_process_kwargs,
            )
            # beam_search does not automatically interleave `batch_size` dim for `num_beams`
            kwargs = {}
            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                    model, input_ids, attention_mask, num_interleave=beam_scorer.num_beams
                )
                kwargs["encoder_outputs"] = encoder_outputs
                input_ids_clone = input_ids_clone.repeat_interleave(beam_scorer.num_beams, dim=0)
            else:
                attention_mask_clone = attention_mask.repeat_interleave(beam_scorer.num_beams, dim=0)
                input_ids_clone = input_ids.repeat_interleave(beam_scorer.num_beams, dim=0)

            with torch.no_grad():
                output_ids_beam_search = model.beam_search(
                    input_ids_clone,
                    beam_scorer,
                    max_length=max_length,
                    attention_mask=attention_mask_clone,
                    logits_processor=logits_processor,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_beam_search.tolist())

    def test_beam_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            print("Return dict", config.return_dict)
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=1)

            model = model_class(config).to(torch_device)
            model.eval()

            # check `generate()` and `beam_search()` are equal
            # change `num_return_sequences = 2` but not for `beam_scorer`
            num_return_sequences = 2
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(
                input_ids.shape[0] * num_return_sequences, max_length
            )
            beam_kwargs["num_return_sequences"] = num_return_sequences
            torch.manual_seed(0)
            output_ids_generate = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_length=max_length,
                **beam_kwargs,
                **logits_warper_kwargs,
            )
            # beam_search does not automatically interleave `batch_size` dim for `num_beams * num_return_sequences`
            kwargs = {}
            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                    model, input_ids, attention_mask, num_interleave=beam_scorer.num_beams * num_return_sequences
                )
                kwargs["encoder_outputs"] = encoder_outputs
            else:
                attention_mask = attention_mask.repeat_interleave(beam_scorer.num_beams * num_return_sequences, dim=0)

            torch.manual_seed(0)
            with torch.no_grad():
                output_ids_beam_sample = model.beam_sample(
                    input_ids.repeat_interleave(beam_scorer.num_beams * num_return_sequences, dim=0),
                    beam_scorer,
                    max_length=max_length,
                    attention_mask=attention_mask,
                    logits_warper=logits_warper,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_beam_sample.tolist())

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
                )

                self.assertIsNotNone(output_ids_generate)


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
