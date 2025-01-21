# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import numpy as np
import timeout_decorator  # noqa

from transformers import OPTConfig, is_flax_available
from transformers.testing_utils import require_flax, require_sentencepiece, slow

from ...generation.test_flax_utils import FlaxGenerationTesterMixin
from ...test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor


if is_flax_available():
    import os

    # The slow tests are often failing with OOM error on GPU
    # This makes JAX allocate exactly what is needed on demand, and deallocate memory that is no longer needed
    # but will be slower as stated here https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import jax
    import jax.numpy as jnp

    from transformers import FlaxOPTForCausalLM, FlaxOPTModel, GPT2Tokenizer


def prepare_opt_inputs_dict(config, input_ids, attention_mask=None, head_mask=None):
    if attention_mask is None:
        attention_mask = np.where(input_ids != config.pad_token_id, 1, 0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


@require_flax
class FlaxOPTModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
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
        embed_dim=16,
        word_embed_proj_dim=16,
        initializer_range=0.02,
        attn_implemetation="eager",
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
        self.embed_dim = embed_dim
        self.word_embed_proj_dim = word_embed_proj_dim
        self.initializer_range = initializer_range
        self.is_encoder_decoder = False
        self.attn_implementation = attn_implemetation

    def prepare_config_and_inputs(self):
        input_ids = np.clip(ids_tensor([self.batch_size, self.seq_length - 1], self.vocab_size), 3, self.vocab_size)
        input_ids = np.concatenate((input_ids, 2 * np.ones((self.batch_size, 1), dtype=np.int64)), -1)

        config = OPTConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            embed_dim=self.embed_dim,
            is_encoder_decoder=False,
            word_embed_proj_dim=self.word_embed_proj_dim,
            initializer_range=self.initializer_range,
            use_cache=False,
            attn_implementation=self.attn_implementation,
        )
        inputs_dict = prepare_opt_inputs_dict(config, input_ids)
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def check_use_cache_forward(self, model_class_name, config, inputs_dict):
        max_length = 20
        model = model_class_name(config)

        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        past_key_values = model.init_cache(input_ids.shape[0], max_length)
        attention_mask = jnp.ones((input_ids.shape[0], max_length), dtype="i4")

        position_ids = jnp.broadcast_to(
            jnp.arange(input_ids.shape[-1] - 1)[None, :],
            (input_ids.shape[0], input_ids.shape[-1] - 1),
        )
        outputs_cache = model(
            input_ids[:, :-1],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        position_ids = jnp.array(input_ids.shape[0] * [[input_ids.shape[-1] - 1]], dtype="i4")
        outputs_cache_next = model(
            input_ids[:, -1:],
            attention_mask=attention_mask,
            past_key_values=outputs_cache.past_key_values,
            position_ids=position_ids,
        )

        outputs = model(input_ids)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")

    def check_use_cache_forward_with_attn_mask(self, model_class_name, config, inputs_dict):
        max_length = 20
        model = model_class_name(config)

        input_ids, attention_mask = (
            inputs_dict["input_ids"],
            inputs_dict["attention_mask"],
        )

        attention_mask_cache = jnp.concatenate(
            [
                attention_mask,
                jnp.zeros((attention_mask.shape[0], max_length - attention_mask.shape[1])),
            ],
            axis=-1,
        )

        past_key_values = model.init_cache(input_ids.shape[0], max_length)
        position_ids = jnp.broadcast_to(
            jnp.arange(input_ids.shape[-1] - 1)[None, :],
            (input_ids.shape[0], input_ids.shape[-1] - 1),
        )

        outputs_cache = model(
            input_ids[:, :-1],
            attention_mask=attention_mask_cache,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_ids = jnp.array(input_ids.shape[0] * [[input_ids.shape[-1] - 1]], dtype="i4")
        outputs_cache_next = model(
            input_ids[:, -1:],
            past_key_values=outputs_cache.past_key_values,
            attention_mask=attention_mask_cache,
            position_ids=position_ids,
        )

        outputs = model(input_ids, attention_mask=attention_mask)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")


@require_flax
class FlaxOPTModelTest(FlaxModelTesterMixin, unittest.TestCase, FlaxGenerationTesterMixin):
    all_model_classes = (FlaxOPTModel, FlaxOPTForCausalLM) if is_flax_available() else ()
    all_generative_model_classes = () if is_flax_available() else ()

    def setUp(self):
        self.model_tester = FlaxOPTModelTester(self)

    def test_use_cache_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            self.model_tester.check_use_cache_forward(model_class, config, inputs_dict)

    def test_use_cache_forward_with_attn_mask(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            self.model_tester.check_use_cache_forward_with_attn_mask(model_class, config, inputs_dict)

    @slow
    def test_model_from_pretrained(self):
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained("facebook/opt-125m")
            input_ids = np.ones((1, 1)) * model.config.eos_token_id
            outputs = model(input_ids)
            self.assertIsNotNone(outputs)


@require_sentencepiece
@require_flax
class FlaxOPTModelIntegrationTests(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = FlaxOPTModel.from_pretrained("facebook/opt-350m")
        input_ids = jnp.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids=input_ids).last_hidden_state
        expected_shape = (1, 11, 512)
        self.assertEqual(output.shape, expected_shape)
        expected_slice = jnp.array(
            [[-0.2867, -1.9256, -0.3062], [-1.2711, -0.1337, -0.1897], [0.4109, 0.1187, -1.3142]]
        )
        self.assertTrue(jnp.allclose(output[:, :3, :3], expected_slice, atol=4e-2))


@require_flax
@slow
class FlaxOPTEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.path_model = "facebook/opt-350m"

    def test_logits(self):
        model = FlaxOPTForCausalLM.from_pretrained(self.path_model)
        tokenizer = GPT2Tokenizer.from_pretrained(self.path_model)

        prompts = [
            "Today is a beautiful day and I want to",
            "In the city of",
            "Paris is the capital of France and",
            "Computers and mobile phones have taken",
        ]
        # verify that prompt without BOS token is identical to Metaseq -> add_special_tokens=False
        inputs = tokenizer(prompts, return_tensors="jax", padding=True, add_special_tokens=False)
        logits = model(inputs.input_ids, attention_mask=inputs.attention_mask)[0].mean(axis=-1)
        logits_meta = jnp.array(
            [
                [1.3851, -13.8923, -10.5229, -10.7533, -0.2309, -10.2384, -0.5365, -9.0947, -5.1670],
                [-4.7073, -10.6276, -3.9415, -21.5242, -0.2822, -0.2822, -0.2822, -0.2822, -0.2822],
                [0.6247, -3.4229, -8.9179, -1.4297, -14.1650, 1.4146, -9.0218, -0.2703, -0.2703],
                [6.4783, -1.9913, -10.7926, -2.3336, 1.5092, -0.9974, -6.8213, 1.3477, 1.3477],
            ]
        )
        self.assertTrue(jnp.allclose(logits, logits_meta, atol=4e-2))

        model = jax.jit(model)
        logits = model(inputs.input_ids, attention_mask=inputs.attention_mask)[0].mean(axis=-1)
        self.assertTrue(jnp.allclose(logits, logits_meta, atol=4e-2))


@require_flax
@slow
class FlaxOPTGenerationTest(unittest.TestCase):
    @property
    def prompts(self):
        return [
            "Today is a beautiful day and I want",
            "In the city of",
            "Paris is the capital of France and",
            "Computers and mobile phones have taken",
        ]

    def test_generation_pre_attn_layer_norm(self):
        model_id = "facebook/opt-125m"

        EXPECTED_OUTPUTS = [
            "Today is a beautiful day and I want to",
            "In the city of New York, the city",
            "Paris is the capital of France and the capital",
            "Computers and mobile phones have taken over the",
        ]

        predicted_outputs = []

        model = FlaxOPTForCausalLM.from_pretrained(model_id)
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)

        for prompt in self.prompts:
            input_ids = tokenizer(prompt, return_tensors="jax").input_ids

            generated_ids = model.generate(input_ids, max_length=10)
            generated_ids = generated_ids[0]

            generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predicted_outputs += generated_string

        self.assertListEqual(predicted_outputs, EXPECTED_OUTPUTS)

    def test_generation_post_attn_layer_norm(self):
        model_id = "facebook/opt-350m"

        EXPECTED_OUTPUTS = [
            "Today is a beautiful day and I want to",
            "In the city of San Francisco, the city",
            "Paris is the capital of France and the capital",
            "Computers and mobile phones have taken over the",
        ]

        predicted_outputs = []
        model = FlaxOPTForCausalLM.from_pretrained(model_id)
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)

        for prompt in self.prompts:
            input_ids = tokenizer(prompt, return_tensors="jax").input_ids

            generated_ids = model.generate(input_ids, max_length=10)
            generated_ids = generated_ids[0]

            generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predicted_outputs += generated_string

        self.assertListEqual(predicted_outputs, EXPECTED_OUTPUTS)

    def test_jitted_batch_generation(self):
        model_id = "facebook/opt-125m"
        EXPECTED_OUTPUTS = [
            "Today is a beautiful day and I want to thank",
            "In the city of Rome Canaver Canaver Canaver Canaver",
        ]
        model = FlaxOPTForCausalLM.from_pretrained(model_id)
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        inputs = tokenizer(
            [
                "Today is a beautiful day and I want to",
                "In the city of",
            ],
            return_tensors="jax",
            padding=True,
        )

        jit_generate = jax.jit(model.generate)

        output_sequences = jit_generate(inputs["input_ids"], attention_mask=inputs["attention_mask"]).sequences

        output_string = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        self.assertIsNotNone(output_string, EXPECTED_OUTPUTS)

    def test_batch_generation(self):
        model_id = "facebook/opt-350m"

        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        model = FlaxOPTForCausalLM.from_pretrained(model_id)

        tokenizer.padding_side = "left"

        # use different length sentences to test batching
        sentences = [
            "Hello, my dog is a little",
            "Today, I",
        ]

        inputs = tokenizer(sentences, return_tensors="jax", padding=True)
        input_ids = inputs["input_ids"]

        outputs = model.generate(input_ids=input_ids, attention_mask=inputs["attention_mask"], trace=False)

        inputs_non_padded = tokenizer(sentences[0], return_tensors="jax").input_ids
        output_non_padded = model.generate(input_ids=inputs_non_padded)

        num_paddings = inputs_non_padded.shape[-1] - inputs["attention_mask"][-1].sum()
        inputs_padded = tokenizer(sentences[1], return_tensors="jax").input_ids
        output_padded = model.generate(input_ids=inputs_padded, max_length=model.config.max_length - num_paddings)

        batch_out_sentence = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0][0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0][0], skip_special_tokens=True)

        expected_output_sentence = [
            "Hello, my dog is a little bit of a dork.\nI'm a little bit",
            "Today, I was in the middle of a conversation with a friend about the",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(batch_out_sentence, [non_padded_sentence, padded_sentence])
