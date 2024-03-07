# coding=utf-8
# Copyright 2023 The HuggingFace Team Inc.
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

from collections import Counter
import copy
from queue import Empty
import random
from threading import Thread
import unittest
import pytest

from transformers import AutoTokenizer, TextIteratorStreamer, TextStreamer, is_torch_available #, OutputIteratorStreamer
from transformers.generation.streamers import OutputIteratorStreamer # TODO: fix import
from transformers.testing_utils import CaptureStdout, require_torch, torch_device

from ..test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import AutoModelForCausalLM

# for debugging only
import lovely_tensors as lt
lt.monkey_patch()

@require_torch
class StreamerTester(unittest.TestCase):
    def test_text_streamer_matches_non_streaming(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        greedy_text = tokenizer.decode(greedy_ids[0])

        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer)
            model.generate(input_ids, max_new_tokens=10, do_sample=False, streamer=streamer)
        # The greedy text should be printed to stdout, except for the final "\n" in the streamer
        streamer_text = cs.out[:-1]

        self.assertEqual(streamer_text, greedy_text)

    def test_iterator_streamer_matches_non_streaming(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        greedy_text = tokenizer.decode(greedy_ids[0])

        streamer = TextIteratorStreamer(tokenizer)
        generation_kwargs = {"input_ids": input_ids, "max_new_tokens": 10, "do_sample": False, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        streamer_text = ""
        for new_text in streamer:
            streamer_text += new_text

        self.assertEqual(streamer_text, greedy_text)

    def test_text_streamer_skip_prompt(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        greedy_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        new_greedy_ids = greedy_ids[:, input_ids.shape[1] :]
        new_greedy_text = tokenizer.decode(new_greedy_ids[0])

        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer, skip_prompt=True)
            model.generate(input_ids, max_new_tokens=10, do_sample=False, streamer=streamer)
        # The greedy text should be printed to stdout, except for the final "\n" in the streamer
        streamer_text = cs.out[:-1]

        self.assertEqual(streamer_text, new_greedy_text)

    def test_text_streamer_decode_kwargs(self):
        # Tests that we can pass `decode_kwargs` to the streamer to control how the tokens are decoded. Must be tested
        # with actual models -- the dummy models' tokenizers are not aligned with their models, and
        # `skip_special_tokens=True` has no effect on them
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = torch.ones((1, 5), device=torch_device).long() * model.config.bos_token_id
        with CaptureStdout() as cs:
            streamer = TextStreamer(tokenizer, skip_special_tokens=True)
            model.generate(input_ids, max_new_tokens=1, do_sample=False, streamer=streamer)

        # The prompt contains a special token, so the streamer should not print it. As such, the output text, when
        # re-tokenized, must only contain one token
        streamer_text = cs.out[:-1]  # Remove the final "\n"
        streamer_text_tokenized = tokenizer(streamer_text, return_tensors="pt")
        self.assertEqual(streamer_text_tokenized.input_ids.shape, (1, 1))

    def test_iterator_streamer_timeout(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)
        streamer = TextIteratorStreamer(tokenizer, timeout=0.001)
        generation_kwargs = {"input_ids": input_ids, "max_new_tokens": 10, "do_sample": False, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # The streamer will timeout after 0.001 seconds, so an exception will be raised
        with self.assertRaises(Empty):
            streamer_text = ""
            for new_text in streamer:
                streamer_text += new_text


@require_torch
#class OutputIteratorStreamerTester(unittest.TestCase): # incompatible with pytest.mark.parameterize
class TestOutputIteratorStreamer:
    @pytest.mark.parametrize("do_sample,top_k", [(False,None), (True,4)])
    @pytest.mark.parametrize("penalty_alpha", [None, 0.6])
    @pytest.mark.parametrize("output_scores", [False, True])
    @pytest.mark.parametrize("output_logits", [False, True])
    def test_outputs_match(self,
                           *,
                           do_sample,top_k,penalty_alpha,output_scores,output_logits,
                           max_new_tokens=10,
                           return_dict_in_generate=True,
                           output_attentions=False,
                           output_hidden_states=False
                           ):
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)

        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=return_dict_in_generate,
            do_sample=do_sample,
            top_k=top_k,
            penalty_alpha=penalty_alpha,
            output_scores=output_scores,
            output_logits=output_logits,
        )

        baseline_kwargs = copy.deepcopy(generation_kwargs)
        test_kwargs = copy.deepcopy(generation_kwargs)

        seed = random.randint(0, int(1e9))
        torch.manual_seed(seed)
        baseline_outputs = model.generate(**baseline_kwargs)

        streamer = OutputIteratorStreamer()
        test_kwargs['streamer'] = streamer
        torch.manual_seed(seed)
        thread = Thread(target=model.generate, kwargs=test_kwargs)
        thread.start()

        outputs = {'sequences':torch.Tensor()}
        if output_scores:
            outputs['scores'] = torch.Tensor()
        if output_logits:
            outputs['logits'] = torch.Tensor()
        if output_attentions:
            outputs['attentions'] = torch.Tensor()
        if output_hidden_states:
            outputs['hidden_states'] = torch.Tensor()

        n_times_field_extended = Counter()
        for answer in streamer:
            #if isinstance(answer, list):
            assert isinstance(answer, list)
            for output_object in answer:
                for output_name in outputs.keys():
                    new_values = getattr(output_object, output_name)
                    if (new_values is not None) and (len(new_values) > 0):
                        if output_name != 'sequences':
                            # unpack tuple
                            new_values = new_values[0]
                        new_values = new_values.cpu()
                        if new_values.ndim == 1:
                            new_values = new_values.unsqueeze(0)
                        outputs[output_name] = torch.cat([outputs[output_name], new_values], axis=-1)
                        n_times_field_extended[output_name] +=1

        for output_name in outputs.keys():
            print(output_name)
            baseline_values = getattr(baseline_outputs, output_name)
            assert baseline_values is not None
            assert type(baseline_values) == type(getattr(output_object, output_name))
            assert n_times_field_extended[output_name] > 1 # make sure we're not just comparing to the final output tensor

            if not isinstance(baseline_values, torch.Tensor):
                baseline_values = torch.cat(baseline_values, axis=-1)
            target_values = outputs[output_name]
            assert baseline_values.shape == target_values.shape
            assert baseline_values.tolist() == target_values.tolist()


    @pytest.mark.parametrize("do_sample,top_k", [(False,None), (True,4)])
    @pytest.mark.parametrize("penalty_alpha", [None, 0.6])
    def test_ids_only_match(self,
                            do_sample, top_k, penalty_alpha,
                            max_new_tokens=10,
                            return_dict_in_generate=False,
                       ):
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        model.config.eos_token_id = -1

        input_ids = ids_tensor((1, 5), vocab_size=model.config.vocab_size).to(torch_device)

        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=return_dict_in_generate,
            do_sample=do_sample,
            top_k=top_k,
            penalty_alpha=penalty_alpha,
            # output_scores=output_scores,
            # output_logits=output_logits,
        )

        baseline_kwargs = copy.deepcopy(generation_kwargs)

        seed = random.randint(0, int(1e9))
        torch.manual_seed(seed)
        baseline_values = model.generate(**baseline_kwargs)

        streamer = OutputIteratorStreamer()
        test_kwargs = copy.deepcopy(generation_kwargs)
        test_kwargs['streamer'] = streamer

        torch.manual_seed(seed)
        thread = Thread(target=model.generate, kwargs=test_kwargs)
        thread.start()

        target_values = torch.Tensor()
        for answer in streamer:
            assert isinstance(answer, list)
            for output_object in answer:
                new_ids = output_object.cpu()
                if new_ids.ndim == 1:
                    new_ids = new_ids.unsqueeze(0)
                target_values = torch.cat([target_values, new_ids], axis=-1)

        assert baseline_values.shape == target_values.shape
        assert baseline_values.tolist() == target_values.tolist()