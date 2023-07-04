# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from transformers import BertConfig, BertTokenizerFast, FeatureExtractionPipeline
from transformers.convert_graph_to_onnx import (
    convert,
    ensure_valid_input,
    generate_identified_filename,
    infer_shapes,
    quantize,
)
from transformers.testing_utils import require_tf, require_tokenizers, require_torch, slow


class FuncContiguousArgs:
    def forward(self, input_ids, token_type_ids, attention_mask):
        return None


class FuncNonContiguousArgs:
    def forward(self, input_ids, some_other_args, token_type_ids, attention_mask):
        return None


class OnnxExportTestCase(unittest.TestCase):
    MODEL_TO_TEST = [
        # (model_name, model_kwargs)
        ("bert-base-cased", {}),
        ("gpt2", {"use_cache": False}),  # We don't support exporting GPT2 past keys anymore
    ]

    @require_tf
    @slow
    def test_export_tensorflow(self):
        for model, model_kwargs in OnnxExportTestCase.MODEL_TO_TEST:
            self._test_export(model, "tf", 12, **model_kwargs)

    @require_torch
    @slow
    def test_export_pytorch(self):
        for model, model_kwargs in OnnxExportTestCase.MODEL_TO_TEST:
            self._test_export(model, "pt", 12, **model_kwargs)

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
            self._test_export(bert_save_dir, "pt", 12, tokenizer)

    @require_tf
    @slow
    def test_quantize_tf(self):
        for model, model_kwargs in OnnxExportTestCase.MODEL_TO_TEST:
            path = self._test_export(model, "tf", 12, **model_kwargs)
            quantized_path = quantize(Path(path))

            # Ensure the actual quantized model is not bigger than the original one
            if quantized_path.stat().st_size >= Path(path).stat().st_size:
                self.fail("Quantized model is bigger than initial ONNX model")

    @require_torch
    @slow
    def test_quantize_pytorch(self):
        for model, model_kwargs in OnnxExportTestCase.MODEL_TO_TEST:
            path = self._test_export(model, "pt", 12, **model_kwargs)
            quantized_path = quantize(path)

            # Ensure the actual quantized model is not bigger than the original one
            if quantized_path.stat().st_size >= Path(path).stat().st_size:
                self.fail("Quantized model is bigger than initial ONNX model")

    def _test_export(self, model, framework, opset, tokenizer=None, **model_kwargs):
        try:
            # Compute path
            with TemporaryDirectory() as tempdir:
                path = Path(tempdir).joinpath("model.onnx")

            # Remove folder if exists
            if path.parent.exists():
                path.parent.rmdir()

            # Export
            convert(framework, model, path, opset, tokenizer, **model_kwargs)

            return path
        except Exception as e:
            self.fail(e)

    @require_torch
    @require_tokenizers
    @slow
    def test_infer_dynamic_axis_pytorch(self):
        """
        Validate the dynamic axis generated for each parameters are correct
        """
        from transformers import BertModel

        model = BertModel(BertConfig.from_pretrained("lysandre/tiny-bert-random"))
        tokenizer = BertTokenizerFast.from_pretrained("lysandre/tiny-bert-random")
        self._test_infer_dynamic_axis(model, tokenizer, "pt")

    @require_tf
    @require_tokenizers
    @slow
    def test_infer_dynamic_axis_tf(self):
        """
        Validate the dynamic axis generated for each parameters are correct
        """
        from transformers import TFBertModel

        model = TFBertModel(BertConfig.from_pretrained("lysandre/tiny-bert-random"))
        tokenizer = BertTokenizerFast.from_pretrained("lysandre/tiny-bert-random")
        self._test_infer_dynamic_axis(model, tokenizer, "tf")

    def _test_infer_dynamic_axis(self, model, tokenizer, framework):
        feature_extractor = FeatureExtractionPipeline(model, tokenizer)

        variable_names = ["input_ids", "token_type_ids", "attention_mask", "output_0", "output_1"]
        input_vars, output_vars, shapes, tokens = infer_shapes(feature_extractor, framework)

        # Assert all variables are present
        self.assertEqual(len(shapes), len(variable_names))
        self.assertTrue(all(var_name in shapes for var_name in variable_names))
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

    def test_generate_identified_name(self):
        generated = generate_identified_filename(Path("/home/something/my_fake_model.onnx"), "-test")
        self.assertEqual("/home/something/my_fake_model-test.onnx", generated.as_posix())
