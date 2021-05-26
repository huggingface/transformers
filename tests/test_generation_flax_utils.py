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

import numpy as np

from transformers import is_flax_available
from transformers.testing_utils import require_flax


if is_flax_available():
    import os

    import jax
    import jax.numpy as jnp
    from jax import jit

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.12"  # assumed parallelism: 8


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

    def test_sample_generate_logits_warper(self):
        config, input_ids, _, max_length = self._get_input_ids_and_config()
        config.do_sample = True
        config.max_length = max_length
        config.temperature = 0.8
        config.top_k = 10
        config.top_p = 0.3

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
        attention_mask = jax.ops.index_update(attention_mask, (0, 0), 0)

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
        attention_mask = jax.ops.index_update(attention_mask, (0, 0), 0)

        config.do_sample = True
        config.max_length = max_length

        for model_class in self.all_generative_model_classes:
            model = model_class(config)

            generation_outputs = model.generate(input_ids, attention_mask=attention_mask).sequences
            self.assertEqual(generation_outputs.shape[-1], max_length)

            jit_generate = jit(model.generate)
            jit_generation_outputs = jit_generate(input_ids, attention_mask=attention_mask).sequences

            self.assertListEqual(generation_outputs.tolist(), jit_generation_outputs.tolist())
