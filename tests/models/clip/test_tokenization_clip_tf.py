import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import is_keras_nlp_available, is_tf_available
from transformers.models.clip.tokenization_clip import CLIPTokenizer
from transformers.testing_utils import require_keras_nlp, slow


if is_keras_nlp_available():
    from transformers.models.clip import TFCLIPTokenizer

if is_tf_available():
    import tensorflow as tf


TOKENIZER_CHECKPOINTS = ["openai/clip-vit-large-patch14"]
TINY_MODEL_CHECKPOINT = "openai/clip-vit-large-patch14"

if is_tf_available():

    class ModelToSave(tf.Module):
        def __init__(self, tokenizer):
            super().__init__()
            self.tokenizer = tokenizer
            # self.model = TFCLIPTextModel.from_pretrained(TINY_MODEL_CHECKPOINT)

        @tf.function(input_signature=(tf.TensorSpec((None,), tf.string, name="text"),))
        def serving(self, text):
            tokenized = self.tokenizer(text)
            return tokenized
            # input_ids_dense = tokenized["input_ids"]

            # input_mask = tf.cast(input_ids_dense > 0, tf.int32)
            # outputs = self.model(input_ids=input_ids_dense, attention_mask=input_mask)["pooler_output"]

            # return outputs


@require_keras_nlp
class GPTTokenizationTest(unittest.TestCase):
    # The TF tokenizers are usually going to be used as pretrained tokenizers from existing model checkpoints,
    # so that's what we focus on here.

    def setUp(self):
        super().setUp()

        self.tokenizers = [CLIPTokenizer.from_pretrained(checkpoint) for checkpoint in (TOKENIZER_CHECKPOINTS)]
        self.tf_tokenizers = [TFCLIPTokenizer.from_pretrained(checkpoint) for checkpoint in TOKENIZER_CHECKPOINTS]
        assert len(self.tokenizers) == len(self.tf_tokenizers)

        self.test_sentences = [
            # "Hi",
            "This is a straightforward English test sentence.",
            # "This one has some weird characters\rto\nsee\r\nif  those\u00E9break things.",
            "Now we're going to add some Chinese: 一 二 三 一二三",
            "And some much more rare Chinese: 齉 堃 齉堃",
            "Je vais aussi écrire en français pour tester les accents",
            "Classical Irish also has some unusual characters, so in they go: Gaelaċ, ꝼ",
        ]
        self.paired_sentences = list(zip(self.test_sentences, self.test_sentences[::-1]))

    def test_output_equivalence(self):
        for tokenizer, tf_tokenizer in zip(self.tokenizers, self.tf_tokenizers):
            for test_inputs in self.test_sentences:
                python_outputs = tokenizer([test_inputs], return_tensors="tf")
                tf_outputs = tf_tokenizer(tf.convert_to_tensor([test_inputs]))

                for key in python_outputs.keys():
                    # convert them to numpy to avoid messing with ragged tensors
                    python_outputs_values = python_outputs[key].numpy()
                    tf_outputs_values = tf_outputs[key].numpy()

                    # print(test_inputs, python_outputs_values, tf_outputs_values)
                    # print(tokenizer.tokenize(test_inputs))
                    self.assertTrue(tf.reduce_all(python_outputs_values.shape == tf_outputs_values.shape))
                    self.assertTrue(tf.reduce_all(tf.cast(python_outputs_values, tf.int64) == tf_outputs_values))

    @slow
    def test_graph_mode(self):
        for tf_tokenizer in self.tf_tokenizers:
            compiled_tokenizer = tf.function(tf_tokenizer)
            for test_inputs in self.test_sentences:
                test_inputs = tf.constant([test_inputs])
                compiled_outputs = compiled_tokenizer(test_inputs)
                eager_outputs = tf_tokenizer(test_inputs)

                for key in eager_outputs.keys():
                    self.assertTrue(tf.reduce_all(eager_outputs[key] == compiled_outputs[key]))

    @slow
    def test_saved_model(self):
        for tf_tokenizer in self.tf_tokenizers:
            model = ModelToSave(tokenizer=tf_tokenizer)
            test_inputs = tf.convert_to_tensor([self.test_sentences[0]])
            out = model.serving(test_inputs)  # Build model with some sample inputs
            with TemporaryDirectory() as tempdir:
                save_path = Path(tempdir) / "saved.model"
                tf.saved_model.save(model, save_path, signatures={"serving_default": model.serving})
                loaded_model = tf.saved_model.load(save_path)
            loaded_output = loaded_model.signatures["serving_default"](test_inputs)  # ["output_0"]
            # We may see small differences because the loaded model is compiled, so we need an epsilon for the test
            self.assertTrue(tf.reduce_all(out["input_ids"].to_tensor() == loaded_output["input_ids"].to_tensor()))

    @slow
    def test_from_config(self):
        for tf_tokenizer in self.tf_tokenizers:
            test_inputs = tf.convert_to_tensor([self.test_sentences[0]])
            out = tf_tokenizer(test_inputs)  # Build model with some sample inputs

            config = tf_tokenizer.get_config()
            model_from_config = TFCLIPTokenizer.from_config(config)
            from_config_output = model_from_config(test_inputs)

            for key in from_config_output.keys():
                self.assertTrue(tf.reduce_all(from_config_output[key] == out[key]))

    @slow
    def test_padding(self):
        for tf_tokenizer in self.tf_tokenizers:
            # for the test to run
            tf_tokenizer.pad_token_id = 123123

            for max_length in [3, 5, 1024]:
                test_inputs = tf.convert_to_tensor([self.test_sentences[0]])
                out = tf_tokenizer(test_inputs, max_length=max_length)

                out_length = out["input_ids"].numpy().shape[1]

                assert out_length == max_length
