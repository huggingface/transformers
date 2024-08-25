import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoConfig, TFAutoModel, is_tensorflow_text_available, is_tf_available
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.testing_utils import require_tensorflow_text, require_tf, slow


if is_tf_available():
    import tensorflow as tf

if is_tensorflow_text_available():
    from transformers.models.bert import TFBertTokenizer


TOKENIZER_CHECKPOINTS = ["bert-base-uncased", "bert-base-cased"]
TINY_MODEL_CHECKPOINT = "hf-internal-testing/tiny-bert-tf-only"

if is_tf_available():

    class ModelToSave(tf.keras.Model):
        def __init__(self, tokenizer):
            super().__init__()
            self.tokenizer = tokenizer
            config = AutoConfig.from_pretrained(TINY_MODEL_CHECKPOINT)
            self.bert = TFAutoModel.from_config(config)

        def call(self, inputs):
            tokenized = self.tokenizer(inputs)
            out = self.bert(**tokenized)
            return out["pooler_output"]


@require_tf
@require_tensorflow_text
class BertTokenizationTest(unittest.TestCase):
    # The TF tokenizers are usually going to be used as pretrained tokenizers from existing model checkpoints,
    # so that's what we focus on here.

    def setUp(self):
        super().setUp()

        self.tokenizers = [
            BertTokenizer.from_pretrained(checkpoint) for checkpoint in (TOKENIZER_CHECKPOINTS * 2)
        ]  # repeat for when fast_bert_tokenizer=false
        self.tf_tokenizers = [TFBertTokenizer.from_pretrained(checkpoint) for checkpoint in TOKENIZER_CHECKPOINTS] + [
            TFBertTokenizer.from_pretrained(checkpoint, use_fast_bert_tokenizer=False)
            for checkpoint in TOKENIZER_CHECKPOINTS
        ]
        assert len(self.tokenizers) == len(self.tf_tokenizers)

        self.test_sentences = [
            "This is a straightforward English test sentence.",
            "This one has some weird characters\rto\nsee\r\nif  those\u00E9break things.",
            "Now we're going to add some Chinese: 一 二 三 一二三",
            "And some much more rare Chinese: 齉 堃 齉堃",
            "Je vais aussi écrire en français pour tester les accents",
            "Classical Irish also has some unusual characters, so in they go: Gaelaċ, ꝼ",
        ]
        self.paired_sentences = list(zip(self.test_sentences, self.test_sentences[::-1]))

    def test_output_equivalence(self):
        for tokenizer, tf_tokenizer in zip(self.tokenizers, self.tf_tokenizers):
            for test_inputs in (self.test_sentences, self.paired_sentences):
                python_outputs = tokenizer(test_inputs, return_tensors="tf", padding="longest")
                tf_outputs = tf_tokenizer(test_inputs)

                for key in python_outputs.keys():
                    self.assertTrue(tf.reduce_all(python_outputs[key].shape == tf_outputs[key].shape))
                    self.assertTrue(tf.reduce_all(tf.cast(python_outputs[key], tf.int64) == tf_outputs[key]))

    @slow
    def test_different_pairing_styles(self):
        for tf_tokenizer in self.tf_tokenizers:
            merged_outputs = tf_tokenizer(self.paired_sentences)
            separated_outputs = tf_tokenizer(
                text=[sentence[0] for sentence in self.paired_sentences],
                text_pair=[sentence[1] for sentence in self.paired_sentences],
            )
            for key in merged_outputs.keys():
                self.assertTrue(tf.reduce_all(tf.cast(merged_outputs[key], tf.int64) == separated_outputs[key]))

    @slow
    def test_graph_mode(self):
        for tf_tokenizer in self.tf_tokenizers:
            compiled_tokenizer = tf.function(tf_tokenizer)
            for test_inputs in (self.test_sentences, self.paired_sentences):
                test_inputs = tf.constant(test_inputs)
                compiled_outputs = compiled_tokenizer(test_inputs)
                eager_outputs = tf_tokenizer(test_inputs)

                for key in eager_outputs.keys():
                    self.assertTrue(tf.reduce_all(eager_outputs[key] == compiled_outputs[key]))

    @slow
    def test_saved_model(self):
        for tf_tokenizer in self.tf_tokenizers:
            model = ModelToSave(tokenizer=tf_tokenizer)
            test_inputs = tf.convert_to_tensor(self.test_sentences)
            out = model(test_inputs)  # Build model with some sample inputs
            with TemporaryDirectory() as tempdir:
                save_path = Path(tempdir) / "saved.model"
                model.save(save_path)
                loaded_model = tf.keras.models.load_model(save_path)
            loaded_output = loaded_model(test_inputs)
            # We may see small differences because the loaded model is compiled, so we need an epsilon for the test
            self.assertLessEqual(tf.reduce_max(tf.abs(out - loaded_output)), 1e-5)
