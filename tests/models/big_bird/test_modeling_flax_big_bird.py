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

import numpy as np

from transformers import BigBirdConfig, is_flax_available
from transformers.testing_utils import require_flax, slow

from ...test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor, random_attention_mask


if is_flax_available():
    import jax
    from transformers.models.big_bird.modeling_flax_big_bird import (
        FlaxBigBirdForCausalLM,
        FlaxBigBirdForMaskedLM,
        FlaxBigBirdForMultipleChoice,
        FlaxBigBirdForPreTraining,
        FlaxBigBirdForQuestionAnswering,
        FlaxBigBirdForSequenceClassification,
        FlaxBigBirdForTokenClassification,
        FlaxBigBirdModel,
    )


class FlaxBigBirdModelTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=56,
        is_training=True,
        use_attention_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=7,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_choices=4,
        attention_type="block_sparse",
        use_bias=True,
        rescale_embeddings=False,
        block_size=2,
        num_random_blocks=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_token_type_ids = use_token_type_ids
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
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_choices = num_choices

        self.rescale_embeddings = rescale_embeddings
        self.attention_type = attention_type
        self.use_bias = use_bias
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        config = BigBirdConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            attention_type=self.attention_type,
            block_size=self.block_size,
            num_random_blocks=self.num_random_blocks,
            use_bias=self.use_bias,
            rescale_embeddings=self.rescale_embeddings,
        )

        return config, input_ids, token_type_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, token_type_ids, attention_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}
        return config, inputs_dict


@require_flax
class FlaxBigBirdModelTest(FlaxModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            FlaxBigBirdForCausalLM,
            FlaxBigBirdModel,
            FlaxBigBirdForPreTraining,
            FlaxBigBirdForMaskedLM,
            FlaxBigBirdForMultipleChoice,
            FlaxBigBirdForQuestionAnswering,
            FlaxBigBirdForSequenceClassification,
            FlaxBigBirdForTokenClassification,
        )
        if is_flax_available()
        else ()
    )

    test_attn_probs = False
    test_mismatched_shapes = False

    def setUp(self):
        self.model_tester = FlaxBigBirdModelTester(self)

    @slow
    # copied from `test_modeling_flax_common` because it takes much longer than other models
    def test_from_pretrained_save_pretrained(self):
        super().test_from_pretrained_save_pretrained()

    @slow
    # copied from `test_modeling_flax_common` because it takes much longer than other models
    def test_from_pretrained_with_no_automatic_init(self):
        super().test_from_pretrained_with_no_automatic_init()

    @slow
    # copied from `test_modeling_flax_common` because it takes much longer than other models
    def test_no_automatic_init(self):
        super().test_no_automatic_init()

    @slow
    # copied from `test_modeling_flax_common` because it takes much longer than other models
    def test_hidden_states_output(self):
        super().test_hidden_states_output()

    @slow
    def test_model_from_pretrained(self):
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained("google/bigbird-roberta-base")
            outputs = model(np.ones((1, 1)))
            self.assertIsNotNone(outputs)

    def test_attention_outputs(self):
        if self.test_attn_probs:
            super().test_attention_outputs()

    @slow
    # copied from `test_modeling_flax_common` because it takes much longer than other models
    def test_jit_compilation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                model = model_class(config)

                @jax.jit
                def model_jitted(input_ids, attention_mask=None, **kwargs):
                    return model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

                with self.subTest("JIT Enabled"):
                    jitted_outputs = model_jitted(**prepared_inputs_dict).to_tuple()

                with self.subTest("JIT Disabled"):
                    with jax.disable_jit():
                        outputs = model_jitted(**prepared_inputs_dict).to_tuple()

                self.assertEqual(len(outputs), len(jitted_outputs))
                for jitted_output, output in zip(jitted_outputs, outputs):

                    self.assertEqual(jitted_output.shape, output.shape)

    # overwrite from common in order to skip the check on `attentions`
    def check_pt_flax_outputs(self, fx_outputs, pt_outputs, model_class, tol=1e-5, name="outputs", attributes=None):
        # `bigbird_block_sparse_attention` in `FlaxBigBird` returns `attention_probs = None`, while in PyTorch version,
        # an effort was done to return `attention_probs` (yet to be verified).
        if name.startswith("outputs.attentions"):
            return
        else:
            super().check_pt_flax_outputs(fx_outputs, pt_outputs, model_class, tol, name, attributes)
