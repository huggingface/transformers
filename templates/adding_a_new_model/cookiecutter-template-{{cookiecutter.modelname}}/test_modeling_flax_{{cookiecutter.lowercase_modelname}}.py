# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

{% if cookiecutter.is_encoder_decoder_model == "False" %}

import unittest

from transformers import is_flax_available, {{cookiecutter.camelcase_modelname}}Config
from transformers.testing_utils import require_flax, slow

from .test_configuration_common import ConfigTester
from .test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor

if is_flax_available():
    import numpy as np
    from transformers import (
        Flax{{cookiecutter.camelcase_modelname}}ForCausalLM,
        Flax{{cookiecutter.camelcase_modelname}}ForMaskedLM,
        Flax{{cookiecutter.camelcase_modelname}}ForMultipleChoice,
        Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
        Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
        Flax{{cookiecutter.camelcase_modelname}}ForTokenClassification,
        Flax{{cookiecutter.camelcase_modelname}}Model,
    )


class Flax{{cookiecutter.camelcase_modelname}}ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_mask = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 16
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.num_labels = 3
        self.num_choices = 4
        self.scope = None

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = {{cookiecutter.camelcase_modelname}}Config(
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
            initializer_range=self.initializer_range,
            return_dict=True,
        )

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Flax{{cookiecutter.camelcase_modelname}}Model(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}

        inputs = [input_ids, input_mask]

        result = model(*inputs)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_lm_head(
            self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.is_decoder = True
        model = Flax{{cookiecutter.camelcase_modelname}}ForCausalLM(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        prediction_scores = model(**inputs)["logits"]
        self.parent.assertListEqual(
            list(prediction_scores.shape), [self.batch_size, self.seq_length, self.vocab_size]
        )

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Flax{{cookiecutter.camelcase_modelname}}ForMaskedLM(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        result = model(**inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }

        result = model(**inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = Flax{{cookiecutter.camelcase_modelname}}ForMultipleChoice(config=config)
        multiple_choice_inputs_ids = np.tile(np.expand_dims(input_ids, 1), (1, self.num_choices, 1))
        multiple_choice_input_mask = np.tile(np.expand_dims(input_mask, 1), (1, self.num_choices, 1))
        multiple_choice_token_type_ids = np.tile(np.expand_dims(token_type_ids, 1), (1, self.num_choices, 1))
        inputs = {
            "input_ids": multiple_choice_inputs_ids,
            "attention_mask": multiple_choice_input_mask,
            "token_type_ids": multiple_choice_token_type_ids,
        }
        result = model(**inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = Flax{{cookiecutter.camelcase_modelname}}ForTokenClassification(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        result = model(**inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }

        result = model(**inputs)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_flax
class Flax{{cookiecutter.camelcase_modelname}}ModelTest(FlaxModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            Flax{{cookiecutter.camelcase_modelname}}Model,
            Flax{{cookiecutter.camelcase_modelname}}ForCausalLM,
            Flax{{cookiecutter.camelcase_modelname}}ForMaskedLM,
            Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
            Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
            Flax{{cookiecutter.camelcase_modelname}}ForTokenClassification,
            Flax{{cookiecutter.camelcase_modelname}}ForMultipleChoice,
        )
        if is_flax_available()
        else ()
    )

    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = Flax{{cookiecutter.camelcase_modelname}}ModelTester(self)
        self.config_tester = ConfigTester(self, config_class={{cookiecutter.camelcase_modelname}}Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model = Flax{{cookiecutter.camelcase_modelname}}Model.from_pretrained("{{cookiecutter.checkpoint_identifier}}")
        self.assertIsNotNone(model)


def _assert_tensors_equal(a, b, atol=1e-12, prefix=""):
    """If tensors not close, or a and b arent both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if _assert_tensors_equal(a, b, atol=atol):
            return True
        raise
    except Exception:
        if len(prefix) > 0:
            prefix = f"{prefix}: "
        raise AssertionError(f"{prefix}{a} != {b}")


@require_flax
class Flax{{cookiecutter.camelcase_modelname}}ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = Flax{{cookiecutter.camelcase_modelname}}ForMaskedLM.from_pretrained("{{cookiecutter.checkpoint_identifier}}")
        input_ids = np.array([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        # TODO Replace vocab size
        vocab_size = 32000

        expected_shape = [1, 6, vocab_size]
        self.assertEqual(output.shape, expected_shape)

        print(output[:, :3, :3])

        # TODO Replace values below with what was printed above.
        expected_slice = np.array(
            [
                [
                    [-0.05243197, -0.04498899, 0.05512108],
                    [-0.07444685, -0.01064632, 0.04352357],
                    [-0.05020351, 0.05530146, 0.00700043],
                ]
            ]
        )
        _assert_tensors_equal(output[:, :3, :3], expected_slice, atol=1e-4)

{% else %}
import unittest

from transformers import (
    is_flax_available,
    {{cookiecutter.camelcase_modelname}}Config,
    {{cookiecutter.camelcase_modelname}}Tokenizer,
)
from transformers.testing_utils import require_sentencepiece, require_flax, require_tokenizers, slow

from .test_configuration_common import ConfigTester
from .test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor


if is_flax_available():
    import numpy as np
    import jax.numpy as jnp
    from transformers import (
        Flax{{cookiecutter.camelcase_modelname}}ForConditionalGeneration,
        Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
        Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
        Flax{{cookiecutter.camelcase_modelname}}Model,
    )


@require_flax
class Flax{{cookiecutter.camelcase_modelname}}ModelTester:
    config_cls = {{cookiecutter.camelcase_modelname}}Config
    config_updates = {}
    hidden_act = "gelu"

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
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

        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length - 1], self.vocab_size).clip(3, self.vocab_size)
        eos_tensor = np.expand_dims(np.array([self.eos_token_id] * self.batch_size), 1)
        input_ids = np.concatenate([input_ids, eos_tensor], axis=1)

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.config_cls(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_ids=[2],
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.pad_token_id,
            **self.config_updates,
        )
        inputs_dict = prepare_{{cookiecutter.lowercase_modelname}}_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def check_use_cache_forward(self, model_class_name, config, inputs_dict):
        max_decoder_length = 20
        model = model_class_name(config)

        encoder_outputs = model.encode(inputs_dict["input_ids"])

        decoder_input_ids, decoder_attention_mask = (
            inputs_dict["decoder_input_ids"],
            inputs_dict["decoder_attention_mask"],
        )

        past_key_values = model.init_cache(decoder_input_ids.shape[0], max_decoder_length, encoder_outputs)
        decoder_attention_mask = jnp.ones((decoder_input_ids.shape[0], max_decoder_length), dtype="i4")

        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(decoder_input_ids.shape[-1] - 1)[None, :],
            (decoder_input_ids.shape[0], decoder_input_ids.shape[-1] - 1),
        )
        outputs_cache = model.decode(
            decoder_input_ids[:, :-1],
            encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            decoder_position_ids=decoder_position_ids,
        )

        decoder_position_ids = jnp.array(decoder_input_ids.shape[0] * [[decoder_input_ids.shape[-1] - 1]], dtype="i4")
        outputs_cache_next = model.decode(
            decoder_input_ids[:, -1:],
            encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=outputs_cache.past_key_values,
            decoder_position_ids=decoder_position_ids,
        )

        outputs = model.decode(decoder_input_ids, encoder_outputs)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")

    def check_use_cache_forward_with_attn_mask(self, model_class_name, config, inputs_dict):
        max_decoder_length = 20
        model = model_class_name(config)

        encoder_outputs = model.encode(inputs_dict["input_ids"])

        decoder_input_ids, decoder_attention_mask = (
            inputs_dict["decoder_input_ids"],
            inputs_dict["decoder_attention_mask"],
        )

        decoder_attention_mask_cache = jnp.concatenate(
            [
                decoder_attention_mask,
                jnp.zeros((decoder_attention_mask.shape[0], max_decoder_length - decoder_attention_mask.shape[1])),
            ],
            axis=-1,
        )

        past_key_values = model.init_cache(decoder_input_ids.shape[0], max_decoder_length, encoder_outputs)
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(decoder_input_ids.shape[-1] - 1)[None, :],
            (decoder_input_ids.shape[0], decoder_input_ids.shape[-1] - 1),
        )

        outputs_cache = model.decode(
            decoder_input_ids[:, :-1],
            encoder_outputs,
            decoder_attention_mask=decoder_attention_mask_cache,
            past_key_values=past_key_values,
            decoder_position_ids=decoder_position_ids,
        )
        decoder_position_ids = jnp.array(decoder_input_ids.shape[0] * [[decoder_input_ids.shape[-1] - 1]], dtype="i4")
        outputs_cache_next = model.decode(
            decoder_input_ids[:, -1:],
            encoder_outputs,
            past_key_values=outputs_cache.past_key_values,
            decoder_attention_mask=decoder_attention_mask_cache,
            decoder_position_ids=decoder_position_ids,
        )

        outputs = model.decode(decoder_input_ids, encoder_outputs, decoder_attention_mask=decoder_attention_mask)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")


def prepare_{{cookiecutter.lowercase_modelname}}_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
):
    if attention_mask is None:
        attention_mask = np.not_equal(input_ids, config.pad_token_id).astype(np.int8)
    if decoder_attention_mask is None:
        decoder_attention_mask = np.concatenate([np.ones(decoder_input_ids[:, :1].shape, dtype=np.int8), np.not_equal(decoder_input_ids[:, 1:], config.pad_token_id).astype(np.int8)], axis=-1)
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
    }


@require_flax
class Flax{{cookiecutter.camelcase_modelname}}ModelTest(FlaxModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Flax{{cookiecutter.camelcase_modelname}}ForConditionalGeneration, 
            Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
            Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
            Flax{{cookiecutter.camelcase_modelname}}Model,
        ) if is_flax_available()
        else ()
    )
    all_generative_model_classes = (Flax{{cookiecutter.camelcase_modelname}}ForConditionalGeneration,) if is_flax_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = Flax{{cookiecutter.camelcase_modelname}}ModelTester(self)
        self.config_tester = ConfigTester(self, config_class={{cookiecutter.camelcase_modelname}}Config)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_use_cache_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            self.model_tester.check_use_cache_forward(model_class, config, inputs_dict)

    def test_use_cache_forward_with_attn_mask(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            self.model_tester.check_use_cache_forward_with_attn_mask(model_class, config, inputs_dict)


def _assert_tensors_equal(a, b, atol=1e-12, prefix=""):
    """If tensors not close, or a and b arent both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if _assert_tensors_equal(a, b, atol=atol):
            return True
        raise
    except Exception:
        if len(prefix) > 0:
            prefix = f"{prefix}: "
        raise AssertionError(f"{prefix}{a} != {b}")


def _long_tensor(tok_lst):
    return np.array(tok_lst, dtype=np.int32)


TOLERANCE = 1e-4


@slow
@require_sentencepiece
@require_tokenizers
@require_flax
class Flax{{cookiecutter.camelcase_modelname}}ModelIntegrationTest(unittest.TestCase):
    def test_inference_no_head(self):
        model = Flax{{cookiecutter.camelcase_modelname}}Model.from_pretrained('{{cookiecutter.checkpoint_identifier}}')
        # change to intended input here
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        decoder_input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        inputs_dict = prepare_{{cookiecutter.lowercase_modelname}}_inputs_dict(model.config, input_ids, decoder_input_ids)
        output = model(**inputs_dict)[0]
        expected_shape = (1, 11, 1024)
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = np.array(
            [[0.7144, 0.8143, -1.2813], [0.7144, 0.8143, -1.2813], [-0.0467, 2.5911, -2.1845]],
        )
        _assert_tensors_equal(output[:, :3, :3], expected_slice, atol=TOLERANCE)

    def test_inference_with_head(self):
        model = Flax{{cookiecutter.camelcase_modelname}}ForConditionalGeneration.from_pretrained('{{cookiecutter.checkpoint_identifier}}')
        # change to intended input here
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        decoder_input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        inputs_dict = prepare_{{cookiecutter.lowercase_modelname}}_inputs_dict(model.config, input_ids, decoder_input_ids)
        output = model(**inputs_dict)[0]
        expected_shape = (1, 11, 1024)
        self.assertEqual(output.shape, expected_shape)
        # change to expected output here
        expected_slice = np.array(
            [[0.7144, 0.8143, -1.2813], [0.7144, 0.8143, -1.2813], [-0.0467, 2.5911, -2.1845]],
        )
        _assert_tensors_equal(output[:, :3, :3], expected_slice, atol=TOLERANCE)

    def test_seq_to_seq_generation(self):
        hf = Flax{{cookiecutter.camelcase_modelname}}ForConditionalGeneration.from_pretrained('{{cookiecutter.checkpoint_identifier}}')
        tok = {{cookiecutter.camelcase_modelname}}Tokenizer.from_pretrained('{{cookiecutter.checkpoint_identifier}}')

        batch_input = [
            # string 1,
            # string 2,
            # string 3,
            # string 4,
        ]

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        dct = tok.batch_encode_plus(
            batch_input,
            max_length=512,
            padding="max_length",
            truncation_strategy="only_first",
            truncation=True,
            return_tensors="np",
        )

        hypotheses_batch = hf.generate(
            input_ids=dct["input_ids"],
            attention_mask=dct["attention_mask"],
            num_beams=2,
        )

        EXPECTED = [
            # here expected 1,
            # here expected 2,
            # here expected 3,
            # here expected 4,
        ]

        generated = tok.batch_decode(
            hypotheses_batch.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        assert generated == EXPECTED
{%- endif %}
