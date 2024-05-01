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

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from huggingface_hub import hf_hub_download

from transformers import is_tensorflow_text_available, is_tf_available
from transformers.testing_utils import require_tensorflow_text, require_tf, slow

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
    )
    from transformers.modeling_tf_utils import keras

if is_tensorflow_text_available():
    import tensorflow_text as text


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

    @slow
    @require_tensorflow_text
    def test_generate_tf_function_export_with_tf_tokenizer(self):
        # TF-only test: tf.saved_model export
        with tempfile.TemporaryDirectory() as tmp_dir:
            # file needed to load the TF tokenizer
            hf_hub_download(repo_id="google/flan-t5-small", filename="spiece.model", local_dir=tmp_dir)

            class CompleteSentenceTransformer(keras.layers.Layer):
                def __init__(self):
                    super().__init__()
                    self.tokenizer = text.SentencepieceTokenizer(
                        model=tf.io.gfile.GFile(os.path.join(tmp_dir, "spiece.model"), "rb").read()
                    )
                    self.model = TFAutoModelForSeq2SeqLM.from_pretrained("hf-internal-testing/tiny-random-t5")

                def call(self, inputs, *args, **kwargs):
                    tokens = self.tokenizer.tokenize(inputs)
                    input_ids, attention_mask = text.pad_model_inputs(
                        tokens, max_seq_length=64, pad_value=self.model.config.pad_token_id
                    )
                    outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
                    return self.tokenizer.detokenize(outputs)

            complete_model = CompleteSentenceTransformer()
            inputs = keras.layers.Input(shape=(1,), dtype=tf.string, name="inputs")
            outputs = complete_model(inputs)
            keras_model = keras.Model(inputs, outputs)
            keras_model.save(tmp_dir)

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
