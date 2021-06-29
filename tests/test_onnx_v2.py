from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase
from unittest.mock import patch

from transformers import (
    AlbertConfig,
    AutoTokenizer,
    DistilBertConfig,
    # LongformerConfig,
    RobertaConfig,
    XLMRobertaConfig,
    is_torch_available, BartConfig, GPT2Config, T5Config,
)
from transformers.models.albert import AlbertOnnxConfig
from transformers.models.bart import BartOnnxConfig
from transformers.models.bert.configuration_bert import BertConfig, BertOnnxConfig
from transformers.models.distilbert import DistilBertOnnxConfig
# from transformers.models.longformer import LongformerOnnxConfig
from transformers.models.gpt2 import GPT2OnnxConfig
from transformers.models.roberta import RobertaOnnxConfig
from transformers.models.t5 import T5OnnxConfig
from transformers.models.xlm_roberta import XLMRobertaOnnxConfig
from transformers.onnx import EXTERNAL_DATA_FORMAT_SIZE_LIMIT, OnnxConfig, ParameterFormat

from transformers.onnx.config import DEFAULT_ONNX_OPSET
from transformers.onnx.utils import (
    compute_effective_axis_dimension,
    compute_serialized_parameters_size,
    flatten_output_collection_property,
)
from transformers.testing_utils import require_onnx, require_torch, slow


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
        TWO_GB_LIMIT = EXTERNAL_DATA_FORMAT_SIZE_LIMIT

        # No parameters
        self.assertFalse(OnnxConfig.use_external_data_format(0))

        # Some parameters
        self.assertFalse(OnnxConfig.use_external_data_format(1))

        # Almost 2Gb parameters
        self.assertFalse(OnnxConfig.use_external_data_format((TWO_GB_LIMIT - 1) // ParameterFormat.Float.size))

        # Exactly 2Gb parameters
        self.assertTrue(OnnxConfig.use_external_data_format(TWO_GB_LIMIT))

        # More than 2Gb parameters
        self.assertTrue(OnnxConfig.use_external_data_format((TWO_GB_LIMIT + 1) // ParameterFormat.Float.size))


if is_torch_available():
    from transformers import AlbertModel, BartModel, BertModel, DistilBertModel, GPT2Model, RobertaModel, T5Model, XLMRobertaModel

    PYTORCH_EXPORT_DEFAULT_MODELS = {
        ("ALBERT", "albert-base-v2", AlbertModel, AlbertConfig, AlbertOnnxConfig),
        ("BART", "facebook/bart-base", BartModel, BartConfig, BartOnnxConfig),
        ("BERT", "bert-base-cased", BertModel, BertConfig, BertOnnxConfig),
        ("DistilBERT", "distilbert-base-cased", DistilBertModel, DistilBertConfig, DistilBertOnnxConfig),
        ("GPT2", "gpt2", GPT2Model, GPT2Config, GPT2OnnxConfig),
        # ("LongFormer", "longformer-base-4096", LongformerModel, LongformerConfig, LongformerOnnxConfig),
        ("Roberta", "roberta-base", RobertaModel, RobertaConfig, RobertaOnnxConfig),
        ("XLM-Roberta", "roberta-base", XLMRobertaModel, XLMRobertaConfig, XLMRobertaOnnxConfig),
        ("T5", "t5-small", T5Model, T5Config, T5OnnxConfig)
    }

    PYTORCH_EXPORT_WITH_PAST_MODELS = {
        ("BART", ),
        ("GPT2", ),
        ("T5", )
    }


class OnnxExportTestCaseV2(TestCase):
    @slow
    @require_torch
    def test_pytorch_export_default(self):
        from transformers.onnx.convert import convert_pytorch

        for name, model, model_class, config_class, onnx_config_class in PYTORCH_EXPORT_DEFAULT_MODELS:

            with self.subTest(name):
                tokenizer = AutoTokenizer.from_pretrained(model)
                model = model_class(config_class())
                onnx_config = onnx_config_class.default(model.config)

                with NamedTemporaryFile("w") as output:
                    convert_pytorch(tokenizer, model, onnx_config, DEFAULT_ONNX_OPSET, Path(output.name))

    @slow
    @require_torch
    def test_pytorch_export_with_past(self):
        pass
