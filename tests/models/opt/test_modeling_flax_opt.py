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
from transformers.models.opt.modeling_flax_opt import FlaxOPTForCausalLM
from transformers.testing_utils import require_flax, slow

from ...generation.test_generation_flax_utils import FlaxGenerationTesterMixin
from ...test_modeling_flax_common import FlaxModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_flax_available():
    import os

    # The slow tests are often failing with OOM error on GPU
    # This makes JAX allocate exactly what is needed on demand, and deallocate memory that is no longer needed
    # but will be slower as stated here https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import jax
    import jax.numpy as jnp
    from transformers.models.opt.modeling_flax_opt import FlaxOPTModel, shift_tokens_right


def prepare_opt_inputs_dict(
    config,
    input_ids,
    attention_mask=None,
    head_mask=None,
):
    if attention_mask is None:
        attention_mask = np.where(input_ids != config.pad_token_id, 1, 0)
    if head_mask is None:
        head_mask = np.ones((config.num_hidden_layers, config.num_attention_heads))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "head_mask": head_mask,
    }


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
        )
        inputs_dict = prepare_opt_inputs_dict(config, input_ids)
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def check_use_cache_forward(self, model_class_name, config, inputs_dict):
        max_length = 20
        model = model_class_name(config)

        input_ids, attention_mask = (
            inputs_dict["input_ids"],
            inputs_dict["attention_mask"],
        )
        
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
    all_model_classes = (FlaxOPTModel,FlaxOPTForCausalLM) if is_flax_available() else ()
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

    #@slow
    def test_model_from_pretrained(self):
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained("facebook/opt-125m", from_pt=True)
            input_ids = np.ones((1, 1)) * model.config.eos_token_id
            outputs = model(input_ids)
            self.assertIsNotNone(outputs)


### Could either compare form the HF version or raw logits.
# TODO Add model integration tests

# TODO add embeddings tests

# TODO add OPTGenerationTest