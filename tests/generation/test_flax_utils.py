# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import random
import unittest

import numpy as np

import transformers
from transformers import is_flax_available, is_torch_available
from transformers.testing_utils import is_pt_flax_cross_test, require_flax


if is_flax_available():
    import os

    import jax.numpy as jnp
    from jax import jit

    from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
    from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.12"  # assumed parallelism: 8


if is_torch_available():
    import torch


def ids_tensor(shape, vocab_size, rng=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    output = np.array(values, dtype=jnp.int32).reshape(shape)

    return output


def random_attention_mask(shape, rng=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=rng)
    # make sure that at least one token is attended to for each batch
    attn_mask[:, -1] = 1
    return attn_mask


@require_flax
class FlaxGenerationTesterMixin:
    model_tester = None
    all_generative_model_classes = ()

    def _get_input_ids_and_config(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        # cut to half length & take max batch_size 3
        max_batch_size = 2
        sequence_length = inputs["input_ids"].shape[-1] // 2
        input_ids = inputs["input_ids"][:max_batch_size, :sequence_length]

        attention_mask = jnp.ones_like(input_ids)
        attention_mask = attention_mask[:max_batch_size, :sequence_length]

        # generate max 5 tokens
        max_length = input_ids.shape[-1] + 5
        if config.eos_token_id is not None and config.pad_token_id is None:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            config.pad_token_id = config.eos_token_id
        return config, input_ids, attention_mask, max_length

    @is_pt_flax_cross_test
    def test_greedy_generate_pt_fx(self):
        config, input_ids, _, max_length = self._get_input_ids_and_config()
        config.do_sample = False
        config.max_length = max_length
        config.decoder_start_token_id = 0

        for model_class in self.all_generative_model_classes:
            flax_model = model_class(config)

            pt_model_class_name = model_class.__name__[4:]  # Skip the "Flax" at the beginning
            pt_model_class = getattr(transformers, pt_model_class_name)
            pt_model = pt_model_class(config).eval()
            pt_model = load_flax_weights_in_pytorch_model(pt_model, flax_model.params)

            flax_generation_outputs = flax_model.generate(input_ids).sequences
            pt_generation_outputs = pt_model.generate(torch.tensor(input_ids, dtype=torch.long))

            if flax_generation_outputs.shape[-1] > pt_generation_outputs.shape[-1]:
                flax_generation_outputs = flax_generation_outputs[:, : pt_generation_outputs.shape[-1]]

            self.assertListEqual(pt_generation_outputs.numpy().tolist(), flax_generation_outputs.tolist())

    def test_greedy_generate(self):
        config, input_ids, _, max_length = self._get_input_ids_and_config()
        config.do_sample = False
        config.max_length = max_length

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            generation_outputs = model.generate(input_ids).sequences
            self.assertEqual(generation_outputs.shape[-1], max_length)

            jit_generate = jit(model.generate)
            jit_generation_outputs = jit_generate(input_ids).sequences

            self.assertListEqual(generation_outputs.tolist(), jit_generation_outputs.tolist())

    def test_sample_generate(self):
        config, input_ids, _, max_length = self._get_input_ids_and_config()
        config.do_sample = True
        config.max_length = max_length

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            generation_outputs = model.generate(input_ids).sequences
            self.assertEqual(generation_outputs.shape[-1], max_length)

            jit_generate = jit(model.generate)
            jit_generation_outputs = jit_generate(input_ids).sequences

            self.assertListEqual(generation_outputs.tolist(), jit_generation_outputs.tolist())

    def test_beam_search_generate(self):
        config, input_ids, _, max_length = self._get_input_ids_and_config()
        config.do_sample = False
        config.max_length = max_length
        config.num_beams = 2

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            generation_outputs = model.generate(input_ids).sequences
            self.assertEqual(generation_outputs.shape[-1], max_length)

            jit_generate = jit(model.generate)
            jit_generation_outputs = jit_generate(input_ids).sequences

            self.assertListEqual(generation_outputs.tolist(), jit_generation_outputs.tolist())

    def test_sample_generate_logits_warper(self):
        config, input_ids, _, max_length = self._get_input_ids_and_config()
        config.do_sample = True
        config.max_length = max_length
        config.temperature = 0.8
        config.top_k = 10
        config.top_p = 0.3
        config.min_length = 1
        config.forced_bos_token_id = 8
        config.forced_eos_token_id = 9

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            generation_outputs = model.generate(input_ids).sequences
            self.assertEqual(generation_outputs.shape[-1], max_length)

            jit_generate = jit(model.generate)
            jit_generation_outputs = jit_generate(input_ids).sequences

            self.assertListEqual(generation_outputs.tolist(), jit_generation_outputs.tolist())

    def test_greedy_generate_logits_warper(self):
        config, input_ids, _, max_length = self._get_input_ids_and_config()
        config.max_length = max_length
        config.min_length = 1
        config.forced_bos_token_id = 8
        config.forced_eos_token_id = 9

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            generation_outputs = model.generate(input_ids).sequences
            self.assertEqual(generation_outputs.shape[-1], max_length)

            jit_generate = jit(model.generate)
            jit_generation_outputs = jit_generate(input_ids).sequences

            self.assertListEqual(generation_outputs.tolist(), jit_generation_outputs.tolist())

    def test_beam_search_generate_logits_warper(self):
        config, input_ids, _, max_length = self._get_input_ids_and_config()
        config.max_length = max_length
        config.num_beams = 2
        config.min_length = 1
        config.forced_bos_token_id = 8
        config.forced_eos_token_id = 9

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            generation_outputs = model.generate(input_ids).sequences
            self.assertEqual(generation_outputs.shape[-1], max_length)

            jit_generate = jit(model.generate)
            jit_generation_outputs = jit_generate(input_ids).sequences

            self.assertListEqual(generation_outputs.tolist(), jit_generation_outputs.tolist())

    def test_greedy_generate_attn_mask(self):
        config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

        # pad attention mask on the left
        attention_mask = attention_mask.at[(0, 0)].set(0)

        config.do_sample = False
        config.max_length = max_length

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            generation_outputs = model.generate(input_ids, attention_mask=attention_mask).sequences
            self.assertEqual(generation_outputs.shape[-1], max_length)

            jit_generate = jit(model.generate)
            jit_generation_outputs = jit_generate(input_ids, attention_mask=attention_mask).sequences

            self.assertListEqual(generation_outputs.tolist(), jit_generation_outputs.tolist())

    def test_sample_generate_attn_mask(self):
        config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

        # pad attention mask on the left
        attention_mask = attention_mask.at[(0, 0)].set(0)

        config.do_sample = True
        config.max_length = max_length

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            generation_outputs = model.generate(input_ids, attention_mask=attention_mask).sequences
            self.assertEqual(generation_outputs.shape[-1], max_length)

            jit_generate = jit(model.generate)
            jit_generation_outputs = jit_generate(input_ids, attention_mask=attention_mask).sequences

            self.assertListEqual(generation_outputs.tolist(), jit_generation_outputs.tolist())

    def test_beam_search_generate_attn_mask(self):
        config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

        # pad attention mask on the left
        attention_mask = attention_mask.at[(0, 0)].set(0)

        config.num_beams = 2
        config.max_length = max_length

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            generation_outputs = model.generate(input_ids, attention_mask=attention_mask).sequences
            self.assertEqual(generation_outputs.shape[-1], max_length)

            jit_generate = jit(model.generate)
            jit_generation_outputs = jit_generate(input_ids, attention_mask=attention_mask).sequences

            self.assertListEqual(generation_outputs.tolist(), jit_generation_outputs.tolist())


@require_flax
class FlaxGenerationIntegrationTests(unittest.TestCase):
    def test_validate_generation_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-bert")
        model = FlaxAutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-bert-flax-only")

        encoder_input_str = "Hello world"
        input_ids = tokenizer(encoder_input_str, return_tensors="np").input_ids

        # typos are quickly detected (the correct argument is `do_sample`)
        with self.assertRaisesRegex(ValueError, "do_samples"):
            model.generate(input_ids, do_samples=True)

        # arbitrary arguments that will not be used anywhere are also not accepted
        with self.assertRaisesRegex(ValueError, "foo"):
            fake_model_kwargs = {"foo": "bar"}
            model.generate(input_ids, **fake_model_kwargs)
