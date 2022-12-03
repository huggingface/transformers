# coding=utf-8
# Copyright 2020, The ATLAS Authors and The HuggingFace Inc. team.
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


import gc
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from transformers import BartTokenizer, T5Tokenizer
from transformers.testing_utils import (
    get_tests_dir,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_torch_non_multi_gpu,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_datasets_available, is_faiss_available, is_torch_available

TOLERANCE = 1e-3

T5_SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")
if is_torch_available() and is_datasets_available() and is_faiss_available():
    import torch
    from datasets import Dataset

    import faiss
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSeq2SeqLM,
        DPRContextEncoder,
        AtlasConfig,
        AtlasModel,
        AtlasRetriever,
        AtlasSequenceForGeneration,
        AtlasTokenForGeneration,
        AtlasTokenizer,
    )
    from transformers.modeling_outputs import BaseModelOutput


def _assert_tensors_equal(a, b, atol=1e-12, prefix=""):
    """If tensors not close, or a and b arent both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


def require_retrieval(test_case):
    """
    Decorator marking a test that requires a set of dependencies necessary for pefrorm retrieval with
    [`AtlasRetriever`].

    These tests are skipped when respective libraries are not installed.

    """
    if not (is_torch_available() and is_datasets_available() and is_faiss_available()):
        test_case = unittest.skip("test requires PyTorch, datasets and faiss")(test_case)
    return test_case


@require_torch
@require_retrieval
@require_sentencepiece
@require_tokenizers
@require_torch_non_multi_gpu
class AtlasModelIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        gc.collect()
        torch.cuda.empty_cache()

    @cached_property
    def model(self):
        return (
            AtlasModel.from_pretrained_question_encoder_generator(
                "facebook/dpr-question_encoder-single-nq-base", "facebook/bart-large-cnn"
            )
            .to(torch_device)
            .eval()
        )

    def get_atlas_config(self):
        question_encoder_config = AutoConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        generator_config = AutoConfig.from_pretrained("facebook/bart-large-cnn")
        return AtlasConfig.from_question_encoder_generator_configs(
            question_encoder_config,
            generator_config,
            bos_token_id=0,
            decoder_start_token_id=2,
            eos_token_id=2,
            is_encoder_decoder=True,
            pad_token_id=1,
            vocab_size=50264,
            title_sep=" / ",
            doc_sep=" // ",
            n_docs=5,
            max_combined_length=300,
            dataset="wiki_dpr",
            dataset_split="train",
            index_name="exact",
            index_path=None,
            use_dummy_dataset=True,
            retrieval_vector_size=768,
            retrieval_batch_size=8,
        )

    @slow
    def test_atlas_sequence_inference(self):
        atlas_config = self.get_atlas_config()
        atlas_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        atlas_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        atlas_retriever = AtlasRetriever(
            atlas_config,
            question_encoder_tokenizer=atlas_question_encoder_tokenizer,
            generator_tokenizer=atlas_decoder_tokenizer,
        )

        atlas_sequence = self.sequence_model
        atlas_sequence.set_retriever(atlas_retriever)

        input_ids = atlas_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="pt"
        ).input_ids
        decoder_input_ids = atlas_decoder_tokenizer("Linda Davis", return_tensors="pt").input_ids

        input_ids = input_ids.to(torch_device)
        decoder_input_ids = decoder_input_ids.to(torch_device)

        with torch.no_grad():
            output = atlas_sequence(
                input_ids,
                labels=decoder_input_ids,
            )

        expected_shape = torch.Size([5, 5, 50264])
        self.assertEqual(output.logits.shape, expected_shape)

        expected_doc_scores = torch.tensor([[75.0286, 74.4998, 74.0804, 74.0306, 73.9504]]).to(torch_device)
        _assert_tensors_equal(expected_doc_scores, output.doc_scores, atol=TOLERANCE)

        expected_loss = torch.tensor([36.7368]).to(torch_device)
        _assert_tensors_equal(expected_loss, output.loss, atol=TOLERANCE)


        atlas_config = self.get_atlas_config()
        atlas_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        atlas_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        atlas_retriever = AtlasRetriever(
            atlas_config,
            question_encoder_tokenizer=atlas_question_encoder_tokenizer,
            generator_tokenizer=atlas_decoder_tokenizer,
        )

        atlas_token = self.token_model
        atlas_token.set_retriever(atlas_retriever)

        input_ids = atlas_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="pt"
        ).input_ids
        decoder_input_ids = atlas_decoder_tokenizer("Linda Davis", return_tensors="pt").input_ids

        input_ids = input_ids.to(torch_device)
        decoder_input_ids = decoder_input_ids.to(torch_device)

        with torch.no_grad():
            output = atlas_token(
                input_ids,
                labels=decoder_input_ids,
            )

        expected_shape = torch.Size([5, 5, 50264])
        self.assertEqual(output.logits.shape, expected_shape)

        expected_doc_scores = torch.tensor([[75.0286, 74.4998, 74.0804, 74.0306, 73.9504]]).to(torch_device)
        _assert_tensors_equal(expected_doc_scores, output.doc_scores, atol=TOLERANCE)

        expected_loss = torch.tensor([36.3557]).to(torch_device)
        _assert_tensors_equal(expected_loss, output.loss, atol=TOLERANCE)


    @slow
    def test_atlas_sequence_generate_beam(self):
        atlas_config = self.get_atlas_config()
        atlas_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        atlas_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        atlas_retriever = AtlasRetriever(
            atlas_config,
            question_encoder_tokenizer=atlas_question_encoder_tokenizer,
            generator_tokenizer=atlas_decoder_tokenizer,
        )

        atlas_sequence = self.sequence_model
        atlas_sequence.set_retriever(atlas_retriever)

        input_ids = atlas_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="pt"
        ).input_ids

        input_ids = input_ids.to(torch_device)

        output_ids = atlas_sequence.generate(
            input_ids,
            decoder_start_token_id=atlas_sequence.generator.config.decoder_start_token_id,
            num_beams=2,
            num_return_sequences=2,
        )
        # sequence generate test
        output_text_1 = atlas_decoder_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_text_2 = atlas_decoder_tokenizer.decode(output_ids[1], skip_special_tokens=True)

        # Expected outputs as given by model at integration time.
        EXPECTED_OUTPUT_TEXT_1 = """\"She's My Kind of Girl\" was released through Epic Records in Japan in March 1972, giving the duo a Top 10 hit. Two more singles were released in Japan, \"En Carousel\" and \"Love Has Its Ways\" Ulvaeus and Andersson persevered with their songwriting and experimented with new sounds and vocal arrangements."""
        EXPECTED_OUTPUT_TEXT_2 = """In September 2018, Björn Ulvaeus revealed that the two new songs, \"I Still Have Faith In You\" and \"Don't Shut Me Down\", would be released no earlier than March 2019. The two new tracks will feature in a TV special set to air later in the year."""

        self.assertEqual(output_text_1, EXPECTED_OUTPUT_TEXT_1)
        self.assertEqual(output_text_2, EXPECTED_OUTPUT_TEXT_2)

    @slow
    def test_atlas_sequence_generate_batch(self):
        tokenizer = AtlasTokenizer.from_pretrained("facebook/atlas-sequence-nq")
        retriever = AtlasRetriever.from_pretrained(
            "facebook/atlas-sequence-nq", index_name="exact", use_dummy_dataset=True
        )
        atlas_sequence = AtlasSequenceForGeneration.from_pretrained("facebook/atlas-sequence-nq", retriever=retriever).to(
            torch_device
        )

        input_dict = tokenizer(
            self.test_data_questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = input_dict.input_ids.to(torch_device)
        attention_mask = input_dict.attention_mask.to(torch_device)

        output_ids = atlas_sequence.generate(
            input_ids,
            attention_mask=attention_mask,
        )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        EXPECTED_OUTPUTS = [
            " albert einstein",
            " june 22, 2018",
            " amplitude modulation",
            " tim besley ( chairman )",
            " june 20, 2018",
            " 1980",
            " 7.0",
            " 8",
        ]
        self.assertListEqual(outputs, EXPECTED_OUTPUTS)
