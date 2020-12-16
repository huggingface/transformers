import unittest
import tempfile

from transformers.file_utils import cached_property, is_datasets_available, is_faiss_available, is_tf_available
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow


if is_tf_available() and is_datasets_available() and is_faiss_available():
    import tensorflow as tf

    from transformers import (
        AutoConfig,
        BartTokenizer,
        DPRQuestionEncoderTokenizer,
        RagConfig,
        RagRetriever,
        RagTokenizer,
        TFRagTokenForGeneration,
    )


def require_retrieval(test_case):
    """
    Decorator marking a test that requires a set of dependencies necessary for pefrorm retrieval with
    :class:`~transformers.RagRetriever`.

    These tests are skipped when respective libraries are not installed.

    """
    if not (is_tf_available() and is_datasets_available() and is_faiss_available()):
        test_case = unittest.skip("test requires PyTorch, datasets and faiss")(test_case)
    return test_case


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
            "what is the first step in the evolution of the eye",
            "where is gall bladder situated in human body",
            "what is the main mineral in lithium batteries",
            "who is the president of usa right now?",
            "where do the greasers live in the outsiders",
            "panda is a national animal of which country",
#             "what is the name of manchester united stadium",
        ]

    @slow
    def test_rag_token_generate_batch(self):
        # NOTE: temporarily change to greedy-search gold labels instead of num_beam=4 gold labels
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
        rag_token = TFRagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever, from_pt=True)

        input_dict = tokenizer(
            self.test_data_questions,
            return_tensors="tf",
            padding=True,
            truncation=True,
        )

        input_ids = input_dict.input_ids
        attention_mask = input_dict.attention_mask

        rag_token.config.num_beams = 1
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
            " step by step",
            " stomach",
            " spodumene",
            " obama",
            " northern new jersey",
            " india",
#             " united stadium",
        ]
        self.assertListEqual(outputs, EXPECTED_OUTPUTS)
