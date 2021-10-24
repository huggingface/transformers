from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase
from unittest.mock import patch

from transformers import (  # LongformerConfig,; T5Config,
    AlbertConfig,
    AutoTokenizer,
    BartConfig,
    DistilBertConfig,
    GPT2Config,
    GPTNeoConfig,
    LayoutLMConfig,
    MBartConfig,
    RobertaConfig,
    XLMRobertaConfig,
    is_torch_available,
)
from transformers.models.albert import AlbertOnnxConfig
from transformers.models.bart import BartOnnxConfig
from transformers.models.bert.configuration_bert import BertConfig, BertOnnxConfig
from transformers.models.distilbert import DistilBertOnnxConfig

# from transformers.models.longformer import LongformerOnnxConfig
from transformers.models.gpt2 import GPT2OnnxConfig
from transformers.models.gpt_neo import GPTNeoOnnxConfig
from transformers.models.layoutlm import LayoutLMOnnxConfig
from transformers.models.mbart import MBartOnnxConfig
from transformers.models.roberta import RobertaOnnxConfig

# from transformers.models.t5 import T5OnnxConfig
from transformers.models.xlm_roberta import XLMRobertaOnnxConfig
from transformers.onnx import (
    EXTERNAL_DATA_FORMAT_SIZE_LIMIT,
    OnnxConfig,
    ParameterFormat,
    export,
    validate_model_outputs,
)
from transformers.onnx.config import DEFAULT_ONNX_OPSET, OnnxConfigWithPast
from transformers.onnx.utils import compute_effective_axis_dimension, compute_serialized_parameters_size
from transformers.testing_utils import require_onnx, require_torch, slow


@require_onnx
class OnnxUtilsTestCaseV2(TestCase):
    """
    Cover all the utilities involved to export ONNX models
    """

    @require_torch
    @patch("transformers.onnx.convert.is_torch_onnx_dict_inputs_support_available", return_value=False)
    def test_ensure_pytorch_version_ge_1_8_0(self, mock_is_torch_onnx_dict_inputs_support_available):
        """
        Ensure we raise an Exception if the pytorch version is unsupported (< 1.8.0)
        """
        self.assertRaises(AssertionError, export, None, None, None, None, None)
        mock_is_torch_onnx_dict_inputs_support_available.assert_called()

    def test_compute_effective_axis_dimension(self):
        """
        When exporting ONNX model with dynamic axis (batch or sequence) we set batch_size and/or sequence_length = -1.
        We cannot generate an effective tensor with axis dim == -1, so we trick by using some "fixed" values
        (> 1 to avoid ONNX squeezing the axis).

        This test ensure we are correctly replacing generated batch / sequence tensor with axis > 1
        """

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
        """
        This test ensures we compute a "correct" approximation of the underlying storage requirement (size) for all the
        parameters for the specified parameter's dtype.
        """
        self.assertEqual(compute_serialized_parameters_size(2, ParameterFormat.Float), 2 * ParameterFormat.Float.size)

    def test_flatten_output_collection_property(self):
        """
        This test ensures we correctly flatten nested collection such as the one we use when returning past_keys.
        past_keys = Tuple[Tuple]

        ONNX exporter will export nested collections as ${collection_name}.${level_idx_0}.${level_idx_1}...${idx_n}
        """
        self.assertEqual(
            OnnxConfig.flatten_output_collection_property("past_key", [[0], [1], [2]]),
            {
                "past_key.0": 0,
                "past_key.1": 1,
                "past_key.2": 2,
            },
        )


class OnnxConfigTestCaseV2(TestCase):
    """
    Cover the test for models default.

    Default means no specific features is being enabled on the model.
    """

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


class OnnxConfigWithPastTestCaseV2(TestCase):
    """
    Cover the tests for model which have use_cache feature (i.e. "with_past" for ONNX)
    """

    SUPPORTED_WITH_PAST_CONFIGS = {
        ("BART", BartConfig),
        ("GPT2", GPT2Config),
        # ("T5", T5Config)
    }

    @patch.multiple(OnnxConfigWithPast, __abstractmethods__=set())
    def test_use_past(self):
        """
        Ensure the use_past variable is correctly being set
        """
        for name, config in OnnxConfigWithPastTestCaseV2.SUPPORTED_WITH_PAST_CONFIGS:
            with self.subTest(name):
                self.assertFalse(
                    OnnxConfigWithPast.from_model_config(config()).use_past,
                    "OnnxConfigWithPast.from_model_config() should not use_past",
                )

                self.assertTrue(
                    OnnxConfigWithPast.with_past(config()).use_past,
                    "OnnxConfigWithPast.from_model_config() should use_past",
                )

    @patch.multiple(OnnxConfigWithPast, __abstractmethods__=set())
    def test_values_override(self):
        """
        Ensure the use_past variable correctly set the `use_cache` value in model's configuration
        """
        for name, config in OnnxConfigWithPastTestCaseV2.SUPPORTED_WITH_PAST_CONFIGS:
            with self.subTest(name):

                # without past
                onnx_config_default = OnnxConfigWithPast.from_model_config(config())
                self.assertIsNotNone(onnx_config_default.values_override, "values_override should not be None")
                self.assertIn("use_cache", onnx_config_default.values_override, "use_cache should be present")
                self.assertFalse(
                    onnx_config_default.values_override["use_cache"], "use_cache should be False if not using past"
                )

                # with past
                onnx_config_default = OnnxConfigWithPast.with_past(config())
                self.assertIsNotNone(onnx_config_default.values_override, "values_override should not be None")
                self.assertIn("use_cache", onnx_config_default.values_override, "use_cache should be present")
                self.assertTrue(
                    onnx_config_default.values_override["use_cache"], "use_cache should be False if not using past"
                )


if is_torch_available():
    from transformers import (  # T5Model,
        AlbertModel,
        BartModel,
        BertModel,
        DistilBertModel,
        GPT2Model,
        GPTNeoModel,
        LayoutLMModel,
        MBartModel,
        RobertaModel,
        XLMRobertaModel,
    )

    PYTORCH_EXPORT_DEFAULT_MODELS = {
        ("ALBERT", "hf-internal-testing/tiny-albert", AlbertModel, AlbertConfig, AlbertOnnxConfig),
        ("BART", "facebook/bart-base", BartModel, BartConfig, BartOnnxConfig),
        ("BERT", "bert-base-cased", BertModel, BertConfig, BertOnnxConfig),
        ("DistilBERT", "distilbert-base-cased", DistilBertModel, DistilBertConfig, DistilBertOnnxConfig),
        ("GPT2", "gpt2", GPT2Model, GPT2Config, GPT2OnnxConfig),
        ("GPT-Neo", "EleutherAI/gpt-neo-125M", GPTNeoModel, GPTNeoConfig, GPTNeoOnnxConfig),
        # ("LongFormer", "longformer-base-4096", LongformerModel, LongformerConfig, LongformerOnnxConfig),
        ("Roberta", "roberta-base", RobertaModel, RobertaConfig, RobertaOnnxConfig),
        ("XLM-Roberta", "roberta-base", XLMRobertaModel, XLMRobertaConfig, XLMRobertaOnnxConfig),
        ("LayoutLM", "microsoft/layoutlm-base-uncased", LayoutLMModel, LayoutLMConfig, LayoutLMOnnxConfig),
        ("MBart", "sshleifer/tiny-mbart", MBartModel, MBartConfig, MBartOnnxConfig),
        # ("T5", "t5-small", T5Model, T5Config, T5OnnxConfig),
    }

    PYTORCH_EXPORT_WITH_PAST_MODELS = {
        # ("BART", "facebook/bart-base", BartModel, BartConfig, BartOnnxConfig),
        # ("GPT2", "gpt2", GPT2Model, GPT2Config, GPT2OnnxConfig),
        # ("T5", "t5-small", T5Model, T5Config, T5OnnxConfig)
    }


class OnnxExportTestCaseV2(TestCase):
    """
    Integration tests ensuring supported models are correctly exported
    """

    @slow
    @require_torch
    def test_pytorch_export_default(self):
        from transformers.onnx import export

        for name, model, model_class, config_class, onnx_config_class in PYTORCH_EXPORT_DEFAULT_MODELS:
            with self.subTest(name):
                self.assertTrue(hasattr(onnx_config_class, "from_model_config"))

                tokenizer = AutoTokenizer.from_pretrained(model)
                model = model_class(config_class.from_pretrained(model))
                onnx_config = onnx_config_class.from_model_config(model.config)

                with NamedTemporaryFile("w") as output:
                    onnx_inputs, onnx_outputs = export(
                        tokenizer, model, onnx_config, DEFAULT_ONNX_OPSET, Path(output.name)
                    )

                    try:
                        validate_model_outputs(onnx_config, tokenizer, model, Path(output.name), onnx_outputs, 1e-5)
                    except ValueError as ve:
                        self.fail(f"{name} -> {ve}")

    @slow
    @require_torch
    def test_pytorch_export_with_past(self):
        from transformers.onnx import export

        for name, model, model_class, config_class, onnx_config_class in PYTORCH_EXPORT_WITH_PAST_MODELS:
            with self.subTest(name):
                self.assertTrue(hasattr(onnx_config_class, "with_past"), "OnnxConfigWithPast should have with_past()")

                tokenizer = AutoTokenizer.from_pretrained(model)
                model = model_class(config_class())
                onnx_config = onnx_config_class.with_past(model.config)

                self.assertTrue(hasattr(onnx_config, "use_past"), "OnnxConfigWithPast should have use_past attribute.")
                self.assertTrue(
                    onnx_config.use_past, "OnnxConfigWithPast.use_past should be if called with with_past()"
                )

                with NamedTemporaryFile("w") as output:
                    output = Path(output.name)
                    onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, DEFAULT_ONNX_OPSET, output)

                    try:
                        validate_model_outputs(onnx_config, tokenizer, model, output, onnx_outputs, 1e-5)
                    except ValueError as ve:
                        self.fail(f"{name} -> {ve}")
