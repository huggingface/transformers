import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from transformers import BartTokenizer
from transformers.file_utils import cached_property, is_datasets_available, is_faiss_available, is_tf_available
from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES as DPR_VOCAB_FILES_NAMES
from transformers.models.dpr.tokenization_dpr import DPRQuestionEncoderTokenizer
from transformers.models.roberta.tokenization_roberta import VOCAB_FILES_NAMES as BART_VOCAB_FILES_NAMES
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow


if is_tf_available() and is_datasets_available() and is_faiss_available():
    import tensorflow as tf
    from datasets import Dataset
    import faiss

    from transformers import (
        AutoConfig,
        RagConfig,
        RagRetriever,
        RagTokenizer,
        TFAutoModel,
        TFAutoModelForSeq2SeqLM,
        TFRagModel,
        TFRagSequenceForGeneration,
        TFRagTokenForGeneration,
    )

    from transformers.modeling_tf_outputs import TFBaseModelOutput

from .test_modeling_tf_bart import TFBartModelTester
from .test_modeling_tf_dpr import TFDPRModelTester


TOLERANCE = 1e-3


def require_retrieval(test_case):
    """
    Decorator marking a test that requires a set of dependencies necessary for pefrorm retrieval with
    [`RagRetriever`].

    These tests are skipped when respective libraries are not installed.

    """
    if not (is_tf_available() and is_datasets_available() and is_faiss_available()):
        test_case = unittest.skip("test requires tensorflow, datasets and faiss")(test_case)
    return test_case


@require_tf
@require_retrieval
@require_sentencepiece
class TFRagTestMixin:

    all_model_classes = (
        (TFRagModel, TFRagTokenForGeneration, TFRagSequenceForGeneration)
        if is_tf_available() and is_datasets_available() and is_faiss_available()
        else ()
    )
    all_generative_model_classes = (
        (TFRagTokenForGeneration, TFRagSequenceForGeneration)
        if is_tf_available() and is_datasets_available() and is_faiss_available()
        else ()
    )

    retrieval_vector_size = 32
    n_docs = 3
    max_combined_length = 16

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        # DPR tok
        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]
        dpr_tokenizer_path = os.path.join(self.tmpdirname, "dpr_tokenizer")
        os.makedirs(dpr_tokenizer_path, exist_ok=True)
        self.vocab_file = os.path.join(dpr_tokenizer_path, DPR_VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        # BART tok
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        bart_tokenizer_path = os.path.join(self.tmpdirname, "bart_tokenizer")
        os.makedirs(bart_tokenizer_path, exist_ok=True)
        self.vocab_file = os.path.join(bart_tokenizer_path, BART_VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(bart_tokenizer_path, BART_VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    @cached_property
    def dpr_tokenizer(self) -> DPRQuestionEncoderTokenizer:
        return DPRQuestionEncoderTokenizer.from_pretrained(os.path.join(self.tmpdirname, "dpr_tokenizer"))

    @cached_property
    def bart_tokenizer(self) -> BartTokenizer:
        return BartTokenizer.from_pretrained(os.path.join(self.tmpdirname, "bart_tokenizer"))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_retriever(self, config):
        dataset = Dataset.from_dict(
            {
                "id": ["0", "1", "3"],
                "text": ["foo", "bar", "qux"],
                "title": ["Foo", "Bar", "Qux"],
                "embeddings": [
                    np.ones(self.retrieval_vector_size),
                    2 * np.ones(self.retrieval_vector_size),
                    3 * np.ones(self.retrieval_vector_size),
                ],
            }
        )
        dataset.add_faiss_index("embeddings", string_factory="Flat", metric_type=faiss.METRIC_INNER_PRODUCT)
        tokenizer = self.bart_tokenizer
        with patch("transformers.models.rag.retrieval_rag.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = dataset
            retriever = RagRetriever(
                config,
                question_encoder_tokenizer=self.dpr_tokenizer,
                generator_tokenizer=tokenizer,
            )
        return retriever

    def check_model_with_retriever(
        self, config, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs
    ):
        self.assertIsNotNone(config.question_encoder)
        self.assertIsNotNone(config.generator)

        for model_class in self.all_model_classes:
            model = model_class(config, retriever=self.get_retriever(config))

            self.assertTrue(model.config.is_encoder_decoder)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

            # logits
            self.assertEqual(
                outputs.logits.shape,
                (self.n_docs * decoder_input_ids.shape[0], decoder_input_ids.shape[1], config.generator.vocab_size),
            )
            # generator encoder last hidden states
            self.assertEqual(
                outputs.generator_enc_last_hidden_state.shape,
                (self.n_docs * decoder_input_ids.shape[0], self.max_combined_length, config.generator.hidden_size),
            )
            # doc scores
            self.assertEqual(outputs.doc_scores.shape, (input_ids.shape[0], self.n_docs))

    def check_model_generate_from_context_input_ids(
        self, config, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs
    ):
        self.assertIsNotNone(config.question_encoder)
        self.assertIsNotNone(config.generator)

        retriever = self.get_retriever(config)

        for i, model_class in enumerate(self.all_generative_model_classes):
            model = model_class(config)
            self.assertTrue(model.config.is_encoder_decoder)

            question_hidden_states = model.question_encoder(input_ids, attention_mask=attention_mask)[0]

            out = retriever(
                input_ids,
                question_hidden_states.numpy(),
                prefix=config.generator.prefix,
                return_tensors="tf",
            )

            context_input_ids, context_attention_mask, retrieved_doc_embeds = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
            )
            retrieved_doc_embeds = tf.cast(retrieved_doc_embeds, tf.float32)

            # compute doc_scores
            doc_scores = tf.squeeze(
                tf.matmul(tf.expand_dims(question_hidden_states, axis=[1]), retrieved_doc_embeds, transpose_b=True),
                axis=[1],
            )

            outputs = model.generate(
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                doc_scores=doc_scores,
            )

            self.assertIsNotNone(outputs)

    def check_model_generate(
        self, config, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs
    ):
        self.assertIsNotNone(config.question_encoder)
        self.assertIsNotNone(config.generator)

        for model_class in self.all_generative_model_classes:
            model = model_class(config, retriever=self.get_retriever(config))

            self.assertTrue(model.config.is_encoder_decoder)

            input_ids = tf.cast(input_ids, tf.int32)
            outputs = model.generate(
                input_ids=input_ids,
                num_beams=2,
                num_return_sequences=2,
                decoder_start_token_id=config.generator.eos_token_id,
            )

            self.assertIsNotNone(outputs)

    def check_model_without_retriever(
        self, config, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs
    ):
        self.assertIsNotNone(config.question_encoder)
        self.assertIsNotNone(config.generator)

        retriever = self.get_retriever(config)

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertTrue(model.config.is_encoder_decoder)

            question_hidden_states = model.question_encoder(input_ids, attention_mask=attention_mask)[0]

            out = retriever(
                input_ids,
                question_hidden_states.numpy(),
                prefix=config.generator.prefix,
                return_tensors="tf",
            )

            context_input_ids, context_attention_mask, retrieved_doc_embeds = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
            )

            retrieved_doc_embeds = tf.cast(retrieved_doc_embeds, tf.float32)

            # compute doc_scores
            doc_scores = tf.squeeze(
                tf.matmul(tf.expand_dims(question_hidden_states, axis=[1]), retrieved_doc_embeds, transpose_b=True),
                axis=[1],
            )

            outputs = model(
                input_ids=None,
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                doc_scores=doc_scores,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

            # logits
            self.assertEqual(
                outputs.logits.shape,
                (self.n_docs * decoder_input_ids.shape[0], decoder_input_ids.shape[1], config.generator.vocab_size),
            )

            # generator encoder last hidden states
            self.assertEqual(
                outputs.generator_enc_last_hidden_state.shape,
                (self.n_docs * decoder_input_ids.shape[0], self.max_combined_length, config.generator.hidden_size),
            )
            # doc scores
            self.assertEqual(outputs.doc_scores.shape, (input_ids.shape[0], self.n_docs))

    def check_model_custom_n_docs(
        self, config, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, n_docs, **kwargs
    ):
        self.assertIsNotNone(config.question_encoder)
        self.assertIsNotNone(config.generator)

        retriever = self.get_retriever(config)

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertTrue(model.config.is_encoder_decoder)

            question_hidden_states = model.question_encoder(input_ids, attention_mask=attention_mask)[0]

            out = retriever(
                input_ids,
                question_hidden_states.numpy(),
                prefix=config.generator.prefix,
                return_tensors="tf",
                n_docs=n_docs,
            )

            context_input_ids, context_attention_mask, retrieved_doc_embeds = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
            )

            retrieved_doc_embeds = tf.cast(retrieved_doc_embeds, tf.float32)

            # compute doc_scores
            doc_scores = tf.squeeze(
                tf.matmul(tf.expand_dims(question_hidden_states, axis=[1]), retrieved_doc_embeds, transpose_b=True),
                axis=[1],
            )

            outputs = model(
                input_ids=None,
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                doc_scores=doc_scores,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                n_docs=n_docs,
            )

            # logits
            self.assertEqual(
                outputs.logits.shape,
                (n_docs * decoder_input_ids.shape[0], decoder_input_ids.shape[1], config.generator.vocab_size),
            )
            # generator encoder last hidden states
            self.assertEqual(
                outputs.generator_enc_last_hidden_state.shape,
                (n_docs * decoder_input_ids.shape[0], self.max_combined_length, config.generator.hidden_size),
            )
            # doc scores
            self.assertEqual(outputs.doc_scores.shape, (input_ids.shape[0], n_docs))

    def check_model_with_mismatch_n_docs_value(
        self,
        config,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        retriever_n_docs,
        generator_n_docs,
        **kwargs
    ):
        self.assertIsNotNone(config.question_encoder)
        self.assertIsNotNone(config.generator)

        retriever = self.get_retriever(config)

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertTrue(model.config.is_encoder_decoder)

            question_hidden_states = model.question_encoder(input_ids, attention_mask=attention_mask)[0]

            out = retriever(
                input_ids,
                question_hidden_states.numpy(),
                prefix=config.generator.prefix,
                return_tensors="tf",
                n_docs=retriever_n_docs,
            )

            context_input_ids, context_attention_mask, retrieved_doc_embeds = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
            )

            retrieved_doc_embeds = tf.cast(retrieved_doc_embeds, tf.float32)

            # compute doc_scores
            doc_scores = tf.squeeze(
                tf.matmul(tf.expand_dims(question_hidden_states, axis=[1]), retrieved_doc_embeds, transpose_b=True),
                axis=[1],
            )

            self.assertRaises(
                AssertionError,
                model.__call__,
                input_ids=None,
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                doc_scores=doc_scores,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                n_docs=generator_n_docs,
            )

    def check_model_with_encoder_outputs(
        self, config, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs
    ):
        self.assertIsNotNone(config.question_encoder)
        self.assertIsNotNone(config.generator)

        for model_class in self.all_model_classes:
            model = model_class(config, retriever=self.get_retriever(config))

            self.assertTrue(model.config.is_encoder_decoder)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

            encoder_outputs = TFBaseModelOutput(outputs.generator_enc_last_hidden_state)

            # run only generator
            outputs = model(
                input_ids=None,
                encoder_outputs=encoder_outputs,
                doc_scores=outputs.doc_scores,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

            # logits
            self.assertEqual(
                outputs.logits.shape,
                (self.n_docs * decoder_input_ids.shape[0], decoder_input_ids.shape[1], config.generator.vocab_size),
            )
            # generator encoder last hidden states
            self.assertEqual(
                outputs.generator_enc_last_hidden_state.shape,
                (self.n_docs * decoder_input_ids.shape[0], self.max_combined_length, config.generator.hidden_size),
            )
            # doc scores
            self.assertEqual(outputs.doc_scores.shape, (input_ids.shape[0], self.n_docs))

    def test_model_with_retriever(self):
        inputs_dict = self.config_and_inputs
        self.check_model_with_retriever(**inputs_dict)

    def test_model_without_retriever(self):
        inputs_dict = self.config_and_inputs
        self.check_model_without_retriever(**inputs_dict)

    def test_model_generate_from_context_input_ids(self):
        inputs_dict = self.config_and_inputs
        self.check_model_generate_from_context_input_ids(**inputs_dict)

    def test_model_with_encoder_outputs(self):
        inputs_dict = self.config_and_inputs
        self.check_model_with_encoder_outputs(**inputs_dict)

    def test_model_generate(self):
        inputs_dict = self.config_and_inputs
        self.check_model_generate(**inputs_dict)

    def test_model_with_custom_n_docs(self):
        inputs_dict = self.config_and_inputs
        inputs_dict["n_docs"] = 1
        self.check_model_custom_n_docs(**inputs_dict)

    def test_model_with_mismatch_n_docs_value(self):
        inputs_dict = self.config_and_inputs
        inputs_dict["retriever_n_docs"] = 3
        inputs_dict["generator_n_docs"] = 2
        self.check_model_with_mismatch_n_docs_value(**inputs_dict)


@require_tf
@require_retrieval
class TFRagDPRBartTest(TFRagTestMixin, unittest.TestCase):
    @cached_property
    def config_and_inputs(self):
        question_encoder_tester = TFDPRModelTester(self)
        dpr_config_and_inputs = question_encoder_tester.prepare_config_and_inputs()
        generator_tester = TFBartModelTester(self)
        bart_config_and_inputs = generator_tester.prepare_config_and_inputs_for_common()

        (question_encoder_config, input_ids, _, input_mask, _, _, _) = dpr_config_and_inputs
        (generator_config, bart_inputs_dict) = bart_config_and_inputs
        decoder_input_ids, decoder_attention_mask = bart_inputs_dict["input_ids"], bart_inputs_dict["attention_mask"]

        config = RagConfig.from_question_encoder_generator_configs(
            question_encoder_config,
            generator_config,
            n_docs=self.n_docs,
            retrieval_vector_size=self.retrieval_vector_size,
            max_combined_length=self.max_combined_length,
        )

        return {
            "config": config,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }


@require_tf
@require_retrieval
@require_sentencepiece
@require_tokenizers
class TFRagModelIntegrationTests(unittest.TestCase):
    @cached_property
    def token_model(self):
        return TFRagTokenForGeneration.from_pretrained_question_encoder_generator(
            "facebook/dpr-question_encoder-single-nq-base", "facebook/bart-large-cnn"
        )

    @cached_property
    def sequence_model(self):
        return TFRagSequenceForGeneration.from_pretrained_question_encoder_generator(
            "facebook/dpr-question_encoder-single-nq-base", "facebook/bart-large-cnn"
        )

    def token_model_nq_checkpoint(self, retriever):
        return TFRagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

    def get_rag_config(self):
        question_encoder_config = AutoConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        generator_config = AutoConfig.from_pretrained("facebook/bart-large-cnn")
        return RagConfig.from_question_encoder_generator_configs(
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
    def test_rag_sequence_inference(self):
        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        rag_sequence = self.sequence_model
        rag_sequence.set_retriever(rag_retriever)

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="tf"
        ).input_ids
        decoder_input_ids = rag_decoder_tokenizer("Linda Davis", return_tensors="tf").input_ids

        output = rag_sequence(
            input_ids,
            labels=decoder_input_ids,
        )

        expected_shape = tf.TensorShape([5, 5, 50264])
        self.assertEqual(output.logits.shape, expected_shape)

        expected_doc_scores = tf.convert_to_tensor([[75.0286, 74.4998, 74.0804, 74.0306, 73.9504]])
        expected_loss = tf.convert_to_tensor([36.7368])

        tf.debugging.assert_near(output.loss, expected_loss, atol=1e-3)
        tf.debugging.assert_near(output.doc_scores, expected_doc_scores, atol=1e-3)

    @slow
    def test_rag_token_inference(self):
        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        rag_token = self.token_model
        rag_token.set_retriever(rag_retriever)

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="tf"
        ).input_ids
        decoder_input_ids = rag_decoder_tokenizer("Linda Davis", return_tensors="tf").input_ids

        output = rag_token(
            input_ids,
            labels=decoder_input_ids,
        )

        expected_shape = tf.TensorShape([5, 5, 50264])
        self.assertEqual(output.logits.shape, expected_shape)

        expected_doc_scores = tf.convert_to_tensor([[75.0286, 74.4998, 74.0804, 74.0306, 73.9504]])
        expected_loss = tf.convert_to_tensor([36.3557])

        tf.debugging.assert_near(output.loss, expected_loss, atol=1e-3)
        tf.debugging.assert_near(output.doc_scores, expected_doc_scores, atol=1e-3)

    @slow
    def test_rag_token_inference_nq_checkpoint(self):
        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        rag_token = self.token_model_nq_checkpoint(retriever=rag_retriever)

        # check that outputs after saving and loading are equal
        with tempfile.TemporaryDirectory() as tmpdirname:
            rag_token.save_pretrained(tmpdirname)
            rag_token = TFRagTokenForGeneration.from_pretrained(tmpdirname, retriever=rag_retriever)

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="tf"
        ).input_ids
        decoder_input_ids = rag_decoder_tokenizer("Linda Davis", return_tensors="tf").input_ids

        output = rag_token(
            input_ids,
            labels=decoder_input_ids,
        )

        expected_shape = tf.TensorShape([5, 5, 50265])
        self.assertEqual(output.logits.shape, expected_shape)

        expected_doc_scores = tf.convert_to_tensor([[62.9402, 62.7107, 62.2382, 62.1194, 61.8578]])
        expected_loss = tf.convert_to_tensor([32.521812])

        tf.debugging.assert_near(output.loss, expected_loss, atol=1e-3)
        tf.debugging.assert_near(output.doc_scores, expected_doc_scores, atol=1e-3)

    @slow
    def test_rag_token_inference_save_pretrained(self):
        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        rag_token = self.token_model
        rag_token.set_retriever(rag_retriever)

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="tf"
        ).input_ids
        decoder_input_ids = rag_decoder_tokenizer("Linda Davis", return_tensors="tf").input_ids

        # model must run once to be functional before loading/saving works
        rag_token(
            input_ids,
            labels=decoder_input_ids,
        )

        # check that outputs after saving and loading are equal
        with tempfile.TemporaryDirectory() as tmpdirname:
            rag_token.save_pretrained(tmpdirname)
            rag_token = TFRagTokenForGeneration.from_pretrained(tmpdirname, retriever=rag_retriever)

        output = rag_token(
            input_ids,
            labels=decoder_input_ids,
        )

        expected_shape = tf.TensorShape([5, 5, 50264])
        self.assertEqual(output.logits.shape, expected_shape)

        expected_doc_scores = tf.convert_to_tensor([[75.0286, 74.4998, 74.0804, 74.0306, 73.9504]])
        expected_loss = tf.convert_to_tensor([36.3557])

        tf.debugging.assert_near(output.loss, expected_loss, atol=1e-3)
        tf.debugging.assert_near(output.doc_scores, expected_doc_scores, atol=1e-3)

    @slow
    def test_init_and_from_pretrained(self):
        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        rag_config = RagConfig.from_pretrained("facebook/rag-sequence-base")
        rag = TFRagTokenForGeneration(rag_config, retriever=rag_retriever)

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="tf"
        ).input_ids
        decoder_input_ids = rag_decoder_tokenizer("Linda Davis", return_tensors="tf").input_ids

        rag(
            input_ids,
            decoder_input_ids=decoder_input_ids,
        )

        # this should not give any warnings
        with tempfile.TemporaryDirectory() as tmpdirname:
            rag.save_pretrained(tmpdirname)
            rag = TFRagTokenForGeneration.from_pretrained(tmpdirname, retriever=rag_retriever)

    @property
    def test_data_questions(self):
        return [
            "who got the first nobel prize in physics",
            "when is the next deadpool movie being released",
            "which mode is used for short wave broadcast service",
            "who is the owner of reading football club",
            "when is the next scandal episode coming out",
            "when is the last time the philadelphia won the superbowl",
            "what is the most current adobe flash player version",
            "how many episodes are there in dragon ball z",
        ]

    @slow
    def test_rag_token_greedy_search(self):
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
        rag_token = TFRagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

        # check first two questions
        input_dict = tokenizer(
            self.test_data_questions[:2],
            return_tensors="tf",
            padding=True,
            truncation=True,
        )

        input_ids = input_dict.input_ids
        attention_mask = input_dict.attention_mask

        # make sure only 1 beam is used
        rag_token.config.num_beams = 1

        output_ids = rag_token.generate(
            input_ids,
            attention_mask=attention_mask,
        )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        EXPECTED_OUTPUTS = [
            " albert einstein",
            " september 22, 2017",
        ]
        self.assertListEqual(outputs, EXPECTED_OUTPUTS)

    @slow
    def test_rag_token_generate_batch(self):
        # NOTE: gold labels comes from num_beam=4, so this is effectively beam-search test
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
        rag_token = TFRagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

        input_dict = tokenizer(
            self.test_data_questions,
            return_tensors="tf",
            padding=True,
            truncation=True,
        )

        input_ids = input_dict.input_ids
        attention_mask = input_dict.attention_mask

        output_ids = rag_token.generate(
            input_ids,
            attention_mask=attention_mask,
        )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        EXPECTED_OUTPUTS = [
            " albert einstein",
            " september 22, 2017",
            " amplitude modulation",
            " stefan persson",
            " april 20, 2018",
            " the 1970s",
            " 7.1. 2",
            " 13",
        ]
        self.assertListEqual(outputs, EXPECTED_OUTPUTS)

    @slow
    def test_rag_sequence_generate_batch(self):
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True
        )
        rag_sequence = TFRagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

        input_dict = tokenizer(
            self.test_data_questions,
            return_tensors="tf",
            padding=True,
            truncation=True,
        )

        input_ids = input_dict.input_ids
        attention_mask = input_dict.attention_mask

        output_ids = rag_sequence.generate(
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

    @slow
    def test_rag_sequence_generate_batch_from_context_input_ids(self):
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True
        )
        rag_sequence = TFRagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
        input_dict = tokenizer(
            self.test_data_questions,
            return_tensors="tf",
            padding=True,
            truncation=True,
        )

        input_ids = input_dict.input_ids

        question_hidden_states = rag_sequence.question_encoder(input_ids)[0]
        docs_dict = retriever(input_ids.numpy(), question_hidden_states.numpy(), return_tensors="tf")
        doc_scores = tf.squeeze(
            tf.matmul(
                tf.expand_dims(question_hidden_states, axis=[1]), docs_dict["retrieved_doc_embeds"], transpose_b=True
            ),
            axis=[1],
        )
        output_ids = rag_sequence.generate(
            context_input_ids=docs_dict["context_input_ids"],
            context_attention_mask=docs_dict["context_attention_mask"],
            doc_scores=doc_scores,
            do_deduplication=True,
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


@require_tf
@require_retrieval
class TFRagModelSaveLoadTests(unittest.TestCase):
    def get_rag_config(self):
        question_encoder_config = AutoConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        generator_config = AutoConfig.from_pretrained("facebook/bart-large-cnn")
        return RagConfig.from_question_encoder_generator_configs(
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
    def test_rag_sequence_from_pretrained(self):
        load_weight_prefix = "tf_rag_model_1"

        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="tf"
        ).input_ids
        decoder_input_ids = rag_decoder_tokenizer("Linda Davis", return_tensors="tf").input_ids

        with tempfile.TemporaryDirectory() as tmp_dirname:
            rag_sequence = TFRagSequenceForGeneration.from_pretrained_question_encoder_generator(
                "facebook/dpr-question_encoder-single-nq-base",
                "facebook/bart-large-cnn",
                retriever=rag_retriever,
                config=rag_config,
            )
            # check that the from pretrained methods work
            rag_sequence.save_pretrained(tmp_dirname)
            rag_sequence.from_pretrained(tmp_dirname, retriever=rag_retriever)

            output = rag_sequence(input_ids, labels=decoder_input_ids)

            loss_pretrained = output.loss
            del rag_sequence

        question_encoder = TFAutoModel.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        generator = TFAutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn", load_weight_prefix=load_weight_prefix, name="generator"
        )

        rag_sequence = TFRagSequenceForGeneration(
            config=rag_config, question_encoder=question_encoder, generator=generator, retriever=rag_retriever
        )

        output = rag_sequence(input_ids, labels=decoder_input_ids)

        loss_init = output.loss

        self.assertAlmostEqual(loss_pretrained, loss_init, places=4)

    @slow
    def test_rag_token_from_pretrained(self):
        load_weight_prefix = "tf_rag_model_1"

        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="tf"
        ).input_ids
        decoder_input_ids = rag_decoder_tokenizer("Linda Davis", return_tensors="tf").input_ids

        with tempfile.TemporaryDirectory() as tmp_dirname:
            rag_token = TFRagTokenForGeneration.from_pretrained_question_encoder_generator(
                "facebook/dpr-question_encoder-single-nq-base",
                "facebook/bart-large-cnn",
                retriever=rag_retriever,
                config=rag_config,
            )
            # check that the from pretrained methods work
            rag_token.save_pretrained(tmp_dirname)
            rag_token.from_pretrained(tmp_dirname, retriever=rag_retriever)

            output = rag_token(input_ids, labels=decoder_input_ids)

            loss_pretrained = output.loss
            del rag_token

        question_encoder = TFAutoModel.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        generator = TFAutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn", load_weight_prefix=load_weight_prefix, name="generator"
        )
        rag_token = TFRagTokenForGeneration(
            config=rag_config, question_encoder=question_encoder, generator=generator, retriever=rag_retriever
        )

        output = rag_token(input_ids, labels=decoder_input_ids)

        loss_init = output.loss

        self.assertAlmostEqual(loss_pretrained, loss_init, places=4)
