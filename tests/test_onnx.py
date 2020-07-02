import unittest
from os.path import dirname, exists
from shutil import rmtree
from tempfile import NamedTemporaryFile, TemporaryDirectory

from transformers import BertConfig, BertTokenizerFast, FeatureExtractionPipeline
from transformers.convert_graph_to_onnx import convert, ensure_valid_input, infer_shapes
from transformers.testing_utils import require_tf, require_torch, slow


class FuncContiguousArgs:
    def forward(self, input_ids, token_type_ids, attention_mask):
        return None


class FuncNonContiguousArgs:
    def forward(self, input_ids, some_other_args, token_type_ids, attention_mask):
        return None


class OnnxExportTestCase(unittest.TestCase):
    MODEL_TO_TEST = ["bert-base-cased", "gpt2", "roberta-base"]

    @require_tf
    @slow
    def test_export_tensorflow(self):
        for model in OnnxExportTestCase.MODEL_TO_TEST:
            self._test_export(model, "tf", 11)

    @require_torch
    @slow
    def test_export_pytorch(self):
        for model in OnnxExportTestCase.MODEL_TO_TEST:
            self._test_export(model, "pt", 11)

    @require_torch
    @slow
    def test_export_custom_bert_model(self):
        from transformers import BertModel

        vocab = ["[UNK]", "[SEP]", "[CLS]", "[PAD]", "[MASK]", "some", "other", "words"]
        with NamedTemporaryFile(mode="w+t") as vocab_file:
            vocab_file.write("\n".join(vocab))
            vocab_file.flush()
            tokenizer = BertTokenizerFast(vocab_file.name)

        with TemporaryDirectory() as bert_save_dir:
            model = BertModel(BertConfig(vocab_size=len(vocab)))
            model.save_pretrained(bert_save_dir)
            self._test_export(bert_save_dir, "pt", 11, tokenizer)

    def _test_export(self, model, framework, opset, tokenizer=None):
        try:
            # Compute path
            with TemporaryDirectory() as tempdir:
                path = tempdir + "/model.onnx"

            # Remove folder if exists
            if exists(dirname(path)):
                rmtree(dirname(path))

                # Export
                convert(framework, model, path, opset, tokenizer)
        except Exception as e:
            self.fail(e)

    @require_torch
    def test_infer_dynamic_axis_pytorch(self):
        """
        Validate the dynamic axis generated for each parameters are correct
        """
        from transformers import BertModel

        model = BertModel(BertConfig.from_pretrained("bert-base-cased"))
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        self._test_infer_dynamic_axis(model, tokenizer, "pt")

    @require_tf
    def test_infer_dynamic_axis_tf(self):
        """
        Validate the dynamic axis generated for each parameters are correct
        """
        from transformers import TFBertModel

        model = TFBertModel(BertConfig.from_pretrained("bert-base-cased"))
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        self._test_infer_dynamic_axis(model, tokenizer, "tf")

    def _test_infer_dynamic_axis(self, model, tokenizer, framework):
        nlp = FeatureExtractionPipeline(model, tokenizer)

        variable_names = ["input_ids", "token_type_ids", "attention_mask", "output_0", "output_1"]
        input_vars, output_vars, shapes, tokens = infer_shapes(nlp, framework)

        # Assert all variables are present
        self.assertEqual(len(shapes), len(variable_names))
        self.assertTrue(all([var_name in shapes for var_name in variable_names]))
        self.assertSequenceEqual(variable_names[:3], input_vars)
        self.assertSequenceEqual(variable_names[3:], output_vars)

        # Assert inputs are {0: batch, 1: sequence}
        for var_name in ["input_ids", "token_type_ids", "attention_mask"]:
            self.assertDictEqual(shapes[var_name], {0: "batch", 1: "sequence"})

        # Assert outputs are {0: batch, 1: sequence} and {0: batch}
        self.assertDictEqual(shapes["output_0"], {0: "batch", 1: "sequence"})
        self.assertDictEqual(shapes["output_1"], {0: "batch"})

    def test_ensure_valid_input(self):
        """
        Validate parameters are correctly exported
        GPT2 has "past" parameter in the middle of input_ids, token_type_ids and attention_mask.
        ONNX doesn't support export with a dictionary, only a tuple. Thus we need to ensure we remove
        token_type_ids and attention_mask for now to not having a None tensor in the middle
        """
        # All generated args are valid
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        tokens = {"input_ids": [1, 2, 3, 4], "attention_mask": [0, 0, 0, 0], "token_type_ids": [1, 1, 1, 1]}
        ordered_input_names, inputs_args = ensure_valid_input(FuncContiguousArgs(), tokens, input_names)

        # Should have exactly the same number of args (all are valid)
        self.assertEqual(len(inputs_args), 3)

        # Should have exactly the same input names
        self.assertEqual(set(ordered_input_names), set(input_names))

        # Parameter should be reordered according to their respective place in the function:
        # (input_ids, token_type_ids, attention_mask)
        self.assertEqual(inputs_args, (tokens["input_ids"], tokens["token_type_ids"], tokens["attention_mask"]))

        # Generated args are interleaved with another args (for instance parameter "past" in GPT2)
        ordered_input_names, inputs_args = ensure_valid_input(FuncNonContiguousArgs(), tokens, input_names)

        # Should have exactly the one arg (all before the one not provided "some_other_args")
        self.assertEqual(len(inputs_args), 1)
        self.assertEqual(len(ordered_input_names), 1)

        # Should have only "input_ids"
        self.assertEqual(inputs_args[0], tokens["input_ids"])
        self.assertEqual(ordered_input_names[0], "input_ids")
