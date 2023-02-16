# coding=utf-8
# Copyright 2022 The HuggingFace Team Inc.
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

import tempfile
import unittest

import numpy as np

from transformers import is_tf_available
from transformers.testing_utils import require_tf, slow

from ..test_modeling_tf_common import floats_tensor
from .test_framework_agnostic import GenerationIntegrationTestsMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        AutoTokenizer,
        TFAutoModelForCausalLM,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSpeechSeq2Seq,
        TFAutoModelForVision2Seq,
        TFBartForConditionalGeneration,
        TFLogitsProcessorList,
        TFMinLengthLogitsProcessor,
        tf_top_k_top_p_filtering,
    )


@require_tf
class UtilsFunctionsTest(unittest.TestCase):
    # tests whether the top_k_top_p_filtering function behaves as expected
    def test_top_k_top_p_filtering(self):
        logits = tf.convert_to_tensor(
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
                    7.35341276,  # 5th highest value; idx. 9
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
                ],  # cummulative prob of 5 highest values <= 0.6
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
                    5.44029958,  # 5th highest value; idx. 18
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
                ],  # cummulative prob of 5 highest values <= 0.6
            ],
            dtype=tf.float32,
        )

        non_inf_expected_idx = tf.convert_to_tensor(
            [[0, 0], [0, 9], [0, 10], [0, 25], [0, 26], [1, 13], [1, 17], [1, 18], [1, 20], [1, 27]],
            dtype=tf.int32,
        )  # expected non filtered idx as noted above

        non_inf_expected_output = tf.convert_to_tensor(
            [8.222099, 7.3534126, 8.432078, 7.4402075, 9.38451, 6.271159, 8.827531, 5.4402995, 7.3857956, 9.677023],
            dtype=tf.float32,
        )  # expected non filtered values as noted above

        output = tf_top_k_top_p_filtering(logits, top_k=10, top_p=0.6, min_tokens_to_keep=4)

        non_inf_output = output[output != -float("inf")]
        non_inf_idx = tf.cast(
            tf.where(tf.not_equal(output, tf.constant(-float("inf"), dtype=tf.float32))),
            dtype=tf.int32,
        )

        tf.debugging.assert_near(non_inf_output, non_inf_expected_output, rtol=1e-12)
        tf.debugging.assert_equal(non_inf_idx, non_inf_expected_idx)


@require_tf
class TFGenerationIntegrationTests(unittest.TestCase, GenerationIntegrationTestsMixin):
    # setting framework_dependent_parameters needs to be gated, just like its contents' imports
    if is_tf_available():
        framework_dependent_parameters = {
            "AutoModelForCausalLM": TFAutoModelForCausalLM,
            "AutoModelForSpeechSeq2Seq": TFAutoModelForSpeechSeq2Seq,
            "AutoModelForSeq2SeqLM": TFAutoModelForSeq2SeqLM,
            "AutoModelForVision2Seq": TFAutoModelForVision2Seq,
            "LogitsProcessorList": TFLogitsProcessorList,
            "MinLengthLogitsProcessor": TFMinLengthLogitsProcessor,
            "create_tensor_fn": tf.convert_to_tensor,
            "floats_tensor": floats_tensor,
            "return_tensors": "tf",
        }

    @slow
    def test_generate_tf_function_export_fixed_input_length(self):
        # TF-only test: tf.saved_model export
        test_model = TFAutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        input_length = 2
        max_new_tokens = 2

        class DummyModel(tf.Module):
            def __init__(self, model):
                super(DummyModel, self).__init__()
                self.model = model

            @tf.function(
                input_signature=(
                    tf.TensorSpec((None, input_length), tf.int32, name="input_ids"),
                    tf.TensorSpec((None, input_length), tf.int32, name="attention_mask"),
                ),
                jit_compile=True,
            )
            def serving(self, input_ids, attention_mask):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                )
                return {"sequences": outputs["sequences"]}

        dummy_input_ids = [[2, 0], [102, 103]]
        dummy_attention_masks = [[1, 0], [1, 1]]
        dummy_model = DummyModel(model=test_model)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tf.saved_model.save(dummy_model, tmp_dir, signatures={"serving_default": dummy_model.serving})
            serving_func = tf.saved_model.load(tmp_dir).signatures["serving_default"]
            for batch_size in range(1, len(dummy_input_ids) + 1):
                inputs = {
                    "input_ids": tf.constant(dummy_input_ids[:batch_size]),
                    "attention_mask": tf.constant(dummy_attention_masks[:batch_size]),
                }
                tf_func_outputs = serving_func(**inputs)["sequences"]
                tf_model_outputs = test_model.generate(**inputs, max_new_tokens=max_new_tokens)
                tf.debugging.assert_equal(tf_func_outputs, tf_model_outputs)

    @slow
    def test_generate_tf_function_export_fixed_batch_size(self):
        # TF-only test: tf.saved_model export
        test_model = TFAutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        batch_size = 1
        max_new_tokens = 2

        class DummyModel(tf.Module):
            def __init__(self, model):
                super(DummyModel, self).__init__()
                self.model = model

            @tf.function(
                input_signature=(
                    tf.TensorSpec((batch_size, None), tf.int32, name="input_ids"),
                    tf.TensorSpec((batch_size, None), tf.int32, name="attention_mask"),
                ),
                jit_compile=True,
            )
            def serving(self, input_ids, attention_mask):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                )
                return {"sequences": outputs["sequences"]}

        dummy_input_ids = [[2], [102, 103]]
        dummy_attention_masks = [[1], [1, 1]]
        dummy_model = DummyModel(model=test_model)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tf.saved_model.save(dummy_model, tmp_dir, signatures={"serving_default": dummy_model.serving})
            serving_func = tf.saved_model.load(tmp_dir).signatures["serving_default"]
            for input_row in range(len(dummy_input_ids)):
                inputs = {
                    "input_ids": tf.constant([dummy_input_ids[input_row]]),
                    "attention_mask": tf.constant([dummy_attention_masks[input_row]]),
                }
                tf_func_outputs = serving_func(**inputs)["sequences"]
                tf_model_outputs = test_model.generate(**inputs, max_new_tokens=max_new_tokens)
                tf.debugging.assert_equal(tf_func_outputs, tf_model_outputs)

    def test_eos_token_id_int_and_list_top_k_top_sampling(self):
        # Has PT equivalent: this test relies on random sampling
        generation_kwargs = {
            "do_sample": True,
            "num_beams": 1,
            "top_p": 0.7,
            "top_k": 10,
            "temperature": 0.7,
        }
        expectation = 14

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        text = """Hello, my dog is cute and"""
        tokens = tokenizer(text, return_tensors="tf")
        model = TFAutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        eos_token_id = 638
        # forces the generation to happen on CPU, to avoid GPU-related quirks
        with tf.device(":/CPU:0"):
            tf.random.set_seed(0)
            generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

        eos_token_id = [638, 198]
        with tf.device(":/CPU:0"):
            tf.random.set_seed(0)
            generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

    def test_model_kwarg_encoder_signature_filtering(self):
        # Has PT equivalent: ample use of framework-specific code
        bart_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        article = """Hugging Face is a technology company based in New York and Paris."""
        input_ids = bart_tokenizer(article, return_tensors="tf").input_ids
        bart_model = TFBartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart")
        output = bart_model.generate(input_ids).numpy()

        # Let's create a fake model that has a different signature. In particular, this fake model accepts "foo" as an
        # argument. Because "foo" is not in the encoder signature and doesn't start with "decoder_", it will be part of
        # the encoder kwargs prior to signature filtering, which would lead to an exception. But filtering kicks in and
        # saves the day.
        class FakeBart(TFBartForConditionalGeneration):
            def call(self, input_ids, foo=None, **kwargs):
                return super().call(input_ids, **kwargs)

        bart_model = FakeBart.from_pretrained("hf-internal-testing/tiny-random-bart")
        fake_output = bart_model.generate(input_ids, foo="bar").numpy()
        self.assertTrue(np.array_equal(output, fake_output))

        # Encoder signature filtering only kicks in if it doesn't accept wildcard kwargs. The following test will fail
        # because it doesn't do signature filtering.
        class FakeEncoder(bart_model.model.encoder.__class__):
            def call(self, input_ids, **kwargs):
                return super().call(input_ids, **kwargs)

        fake_encoder = FakeEncoder(bart_model.config, bart_model.model.shared)
        bart_model.model.encoder = fake_encoder

        # Normal generation still works (the output will be different because the encoder weights are different)
        fake_output = bart_model.generate(input_ids).numpy()
        with self.assertRaises(ValueError):
            # FakeEncoder.call() accepts **kwargs -> no filtering -> value error due to unexpected input "foo"
            bart_model.generate(input_ids, foo="bar")
