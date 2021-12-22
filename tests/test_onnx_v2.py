from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase
from unittest.mock import patch

from parameterized import parameterized
from transformers import AutoConfig, AutoTokenizer, is_torch_available
from transformers.onnx import (
    EXTERNAL_DATA_FORMAT_SIZE_LIMIT,
    OnnxConfig,
    ParameterFormat,
    export,
    validate_model_outputs,
)
from transformers.onnx.config import OnnxConfigWithPast


if is_torch_available():
    from transformers.onnx.features import FeaturesManager

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

    SUPPORTED_WITH_PAST_CONFIGS = {}
    # SUPPORTED_WITH_PAST_CONFIGS = {
    #     ("BART", BartConfig),
    #     ("GPT2", GPT2Config),
    #     # ("T5", T5Config)
    # }

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


PYTORCH_EXPORT_MODELS = {
    ("albert", "hf-internal-testing/tiny-albert"),
    ("bert", "bert-base-cased"),
    ("camembert", "camembert-base"),
    ("distilbert", "distilbert-base-cased"),
    # ("longFormer", "longformer-base-4096"),
    ("roberta", "roberta-base"),
    ("xlm-roberta", "xlm-roberta-base"),
    ("layoutlm", "microsoft/layoutlm-base-uncased"),
}

PYTORCH_EXPORT_WITH_PAST_MODELS = {
    ("gpt2", "gpt2"),
    ("gpt-neo", "EleutherAI/gpt-neo-125M"),
}

PYTORCH_EXPORT_SEQ2SEQ_WITH_PAST_MODELS = {
    ("bart", "facebook/bart-base"),
    ("mbart", "sshleifer/tiny-mbart"),
    ("t5", "t5-small"),
}


def _get_models_to_test(export_models_list):
    models_to_test = []
    if not is_torch_available():
        # Returning some dummy test that should not be ever called because of the @require_torch decorator.
        # The reason for not returning an empty list is because parameterized.expand complains when it's empty.
        return [("dummy", "dummy", "dummy", "dummy", OnnxConfig.from_model_config)]
    for (name, model) in export_models_list:
        for feature, onnx_config_class_constructor in FeaturesManager.get_supported_features_for_model_type(
            name
        ).items():
            models_to_test.append((f"{name}_{feature}", name, model, feature, onnx_config_class_constructor))
    return sorted(models_to_test)


class OnnxExportTestCaseV2(TestCase):
    """
    Integration tests ensuring supported models are correctly exported
    """

    def _pytorch_export(self, test_name, name, model_name, feature, onnx_config_class_constructor):
        from transformers.onnx import export

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        # Useful for causal lm models that do not use pad tokens.
        if not getattr(config, "pad_token_id", None):
            config.pad_token_id = tokenizer.eos_token_id

        model_class = FeaturesManager.get_model_class_for_feature(feature)
        model = model_class.from_config(config)
        onnx_config = onnx_config_class_constructor(model.config)

        with NamedTemporaryFile("w") as output:
            try:
                onnx_inputs, onnx_outputs = export(
                    tokenizer, model, onnx_config, onnx_config.default_onnx_opset, Path(output.name)
                )
                validate_model_outputs(
                    onnx_config,
                    tokenizer,
                    model,
                    Path(output.name),
                    onnx_outputs,
                    onnx_config.atol_for_validation,
                )
            except (RuntimeError, ValueError) as e:
                self.fail(f"{name}, {feature} -> {e}")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS))
    @slow
    @require_torch
    def test_pytorch_export(self, test_name, name, model_name, feature, onnx_config_class_constructor):
        self._pytorch_export(test_name, name, model_name, feature, onnx_config_class_constructor)

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_WITH_PAST_MODELS))
    @slow
    @require_torch
    def test_pytorch_export_with_past(self, test_name, name, model_name, feature, onnx_config_class_constructor):
        self._pytorch_export(test_name, name, model_name, feature, onnx_config_class_constructor)

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_SEQ2SEQ_WITH_PAST_MODELS))
    @slow
    @require_torch
    def test_pytorch_export_seq2seq_with_past(
        self, test_name, name, model_name, feature, onnx_config_class_constructor
    ):
        self._pytorch_export(test_name, name, model_name, feature, onnx_config_class_constructor)
