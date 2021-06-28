from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from transformers import BertConfig, PreTrainedModel
from transformers.models.bert.configuration_bert import BertOnnxConfig
from transformers.onnx import EXTERNAL_DATA_FORMAT_SIZE_LIMIT, OnnxConfig, ParameterFormat

# from transformers.onnx.convert import convert_pytorch
from transformers.onnx.utils import (
    compute_effective_axis_dimension,
    compute_serialized_parameters_size,
    flatten_output_collection_property,
    generate_identified_filename,
)
from transformers.testing_utils import require_onnx


@require_onnx
class OnnxUtilsTestCaseV2(TestCase):
    def test_compute_effective_axis_dimension(self):
        # Dynamic axis (batch, no token added by the tokenizer)
        self.assertEqual(compute_effective_axis_dimension(-1, fixed_dimension=2, num_token_to_add=0), 2)

        # Static axis (batch, no token added by the tokenizer)
        self.assertEqual(compute_effective_axis_dimension(0, fixed_dimension=2, num_token_to_add=0), 2)

        # Dynamic axis (sequence, token added by the tokenizer 2 (no pair))
        self.assertEqual(compute_effective_axis_dimension(0, fixed_dimension=8, num_token_to_add=2), 6)
        self.assertEqual(compute_effective_axis_dimension(0, fixed_dimension=8, num_token_to_add=2), 6)

        # Dynamic axis (sequence, token added by the tokenizer 3 (pair))
        self.assertEqual(compute_effective_axis_dimension(0, fixed_dimension=8, num_token_to_add=3), 5)
        self.assertEqual(compute_effective_axis_dimension(0, fixed_dimension=8, num_token_to_add=3), 5)

    def test_compute_parameters_serialized_size(self):
        self.assertEqual(compute_serialized_parameters_size(2, ParameterFormat.Float), 2 * ParameterFormat.Float.size)

    def test_generate_identified_filename(self):
        self.assertEqual(generate_identified_filename(Path("model.onnx"), "_suffix"), Path("model_suffix.onnx"))
        self.assertEqual(generate_identified_filename(Path("./model.onnx"), "_suffix"), Path("./model_suffix.onnx"))
        self.assertEqual(generate_identified_filename(Path("../model.onnx"), "_suffix"), Path("../model_suffix.onnx"))
        self.assertEqual(
            generate_identified_filename(Path("a/b/model.onnx"), "_suffix"), Path("a/b/model_suffix.onnx")
        )

    def test_flatten_output_collection_property(self):
        self.assertEqual(
            flatten_output_collection_property("past_key", [[0], [1], [2]]),
            {
                "past_key.0": 0,
                "past_key.1": 1,
                "past_key.2": 2,
            },
        )


class OnnxConfigTestCaseV2(TestCase):
    @patch.multiple(OnnxConfig, __abstractmethods__=set())
    def test_use_external_data_format(self):
        """
        External data format is required only if the serialized size of the parameters if bigger than 2Gb
        """
        LIMIT = EXTERNAL_DATA_FORMAT_SIZE_LIMIT

        # No parameters
        with patch.object(PreTrainedModel, "num_parameters", return_value=0):
            model = PreTrainedModel(BertConfig())
            onnx = OnnxConfig(model)
            self.assertFalse(onnx.use_external_data_format(model.num_parameters()))

        # Some parameters
        with patch.object(PreTrainedModel, "num_parameters", return_value=2):
            model = PreTrainedModel(BertConfig())
            onnx = OnnxConfig(model)
            self.assertFalse(onnx.use_external_data_format(model.num_parameters()))

        # Almost 2Gb parameters
        with patch.object(PreTrainedModel, "num_parameters", return_value=(LIMIT - 1) // ParameterFormat.Float.size):
            model = PreTrainedModel(BertConfig())
            onnx = OnnxConfig(model)
            self.assertFalse(onnx.use_external_data_format(model.num_parameters()))

        # Exactly 2Gb parameters
        with patch.object(PreTrainedModel, "num_parameters", return_value=LIMIT):
            model = PreTrainedModel(BertConfig())
            onnx = OnnxConfig(model)
            self.assertTrue(onnx.use_external_data_format(model.num_parameters()))

        # More than 2Gb parameters
        with patch.object(PreTrainedModel, "num_parameters", return_value=(LIMIT + 1) // ParameterFormat.Float.size):
            model = PreTrainedModel(BertConfig())
            onnx = OnnxConfig(model)
            self.assertTrue(onnx.use_external_data_format(model.num_parameters()))


class OnnxExportTestCaseV2(TestCase):
    EXPORT_DEFAULT_MODELS = {
        ("BERT", "bert-base-cased", BertOnnxConfig),
    }

    def export_default(self):
        pass

    def export_with_past(self):
        pass
