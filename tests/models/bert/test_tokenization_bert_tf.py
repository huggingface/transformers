import unittest

from transformers import is_tensorflow_text_available, is_tf_available
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.testing_utils import require_tensorflow_text


if is_tensorflow_text_available():
    from transformers.models.bert import TFBertTokenizer

if is_tf_available():
    import tensorflow as tf


TEST_CHECKPOINT = "bert-base-cased"


@require_tensorflow_text
class BertTokenizationTest(unittest.TestCase):
    # The TF tokenizers are usually going to be used as pretrained tokenizers from existing model checkpoints,
    # so that's what we focus on here.

    def setUp(self):
        super().setUp()

        self.tokenizer = BertTokenizer.from_pretrained(TEST_CHECKPOINT)
        self.tf_tokenizer = TFBertTokenizer.from_pretrained(TEST_CHECKPOINT)

    def test_output_equivalence(self):
        test_sentences = [
            "This is a straightforward English test sentence.",
            "This one has some weird characters\rto\nsee\r\nif  those\u00E9break things.",
            "Now we're going to add some Chinese: 一 二 三",
            "And some much more rare Chinese: 齉 堃",
            "Je vais aussi écrire en français pour tester les accents",
            "Classical Irish also has some unusual characters, so in they go: Gaelaċ, ꝼ",
        ]

        python_outputs = self.tokenizer(test_sentences, return_tensors="tf", padding="longest")
        tf_outputs = self.tf_tokenizer(test_sentences)

        for key in python_outputs.keys():
            self.assertTrue(tf.reduce_all(tf.cast(python_outputs[key], tf.int64) == tf_outputs[key]))

        paired_sentences = list(zip(test_sentences, test_sentences[::-1]))

        python_outputs = self.tokenizer(paired_sentences, return_tensors="tf", padding="longest")
        tf_outputs = self.tf_tokenizer(paired_sentences)

        for key in python_outputs.keys():
            self.assertTrue(tf.reduce_all(tf.cast(python_outputs[key], tf.int64) == tf_outputs[key]))
