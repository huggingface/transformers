import os
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase
from unittest.mock import patch

import pytest
from parameterized import parameterized

from transformers import AutoConfig, PreTrainedTokenizerBase, is_tf_available, is_torch_available
from transformers.onnx import (
    EXTERNAL_DATA_FORMAT_SIZE_LIMIT,
    OnnxConfig,
    OnnxConfigWithPast,
    ParameterFormat,
    validate_model_outputs,
)
from transformers.onnx.utils import (
    compute_effective_axis_dimension,
    compute_serialized_parameters_size,
    get_preprocessor,
)
from transformers.testing_utils import require_onnx, require_rjieba, require_tf, require_torch, require_vision, slow


if is_torch_available() or is_tf_available():
    from transformers.onnx.features import FeaturesManager

if is_torch_available():
    import torch

    from transformers.models.deberta import modeling_deberta


@require_onnx
class OnnxUtilsTestCaseV2(TestCase):
    """
    Cover all the utilities involved to export ONNX models
    """

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
    ("albert", "hf-internal-testing/tiny-random-AlbertModel"),
    ("bert", "hf-internal-testing/tiny-random-BertModel"),
    ("beit", "microsoft/beit-base-patch16-224"),
    ("big-bird", "hf-internal-testing/tiny-random-BigBirdModel"),
    ("camembert", "camembert-base"),
    ("clip", "hf-internal-testing/tiny-random-CLIPModel"),
    ("convbert", "hf-internal-testing/tiny-random-ConvBertModel"),
    ("codegen", "hf-internal-testing/tiny-random-CodeGenModel"),
    ("data2vec-text", "hf-internal-testing/tiny-random-Data2VecTextModel"),
    ("data2vec-vision", "facebook/data2vec-vision-base"),
    ("deberta", "hf-internal-testing/tiny-random-DebertaModel"),
    ("deberta-v2", "hf-internal-testing/tiny-random-DebertaV2Model"),
    ("deit", "facebook/deit-small-patch16-224"),
    ("convnext", "facebook/convnext-tiny-224"),
    ("detr", "facebook/detr-resnet-50"),
    ("distilbert", "hf-internal-testing/tiny-random-DistilBertModel"),
    ("electra", "hf-internal-testing/tiny-random-ElectraModel"),
    ("groupvit", "nvidia/groupvit-gcc-yfcc"),
    ("ibert", "kssteven/ibert-roberta-base"),
    ("imagegpt", "openai/imagegpt-small"),
    ("levit", "facebook/levit-128S"),
    ("layoutlm", "hf-internal-testing/tiny-random-LayoutLMModel"),
    ("layoutlmv3", "microsoft/layoutlmv3-base"),
    ("longformer", "allenai/longformer-base-4096"),
    ("mobilebert", "hf-internal-testing/tiny-random-MobileBertModel"),
    ("mobilenet_v1", "google/mobilenet_v1_0.75_192"),
    ("mobilenet_v2", "google/mobilenet_v2_0.35_96"),
    ("mobilevit", "apple/mobilevit-small"),
    ("owlvit", "google/owlvit-base-patch32"),
    ("perceiver", "hf-internal-testing/tiny-random-PerceiverModel", ("masked-lm", "sequence-classification")),
    ("perceiver", "hf-internal-testing/tiny-random-PerceiverModel", ("image-classification",)),
    ("poolformer", "sail/poolformer_s12"),
    ("rembert", "google/rembert"),
    ("resnet", "microsoft/resnet-50"),
    ("roberta", "hf-internal-testing/tiny-random-RobertaModel"),
    ("roformer", "hf-internal-testing/tiny-random-RoFormerModel"),
    ("segformer", "nvidia/segformer-b0-finetuned-ade-512-512"),
    ("squeezebert", "hf-internal-testing/tiny-random-SqueezeBertModel"),
    ("swin", "microsoft/swin-tiny-patch4-window7-224"),
    ("vit", "google/vit-base-patch16-224"),
    ("yolos", "hustvl/yolos-tiny"),
    ("whisper", "openai/whisper-tiny.en"),
    ("xlm", "hf-internal-testing/tiny-random-XLMModel"),
    ("xlm-roberta", "hf-internal-testing/tiny-random-XLMRobertaXLModel"),
}

PYTORCH_EXPORT_ENCODER_DECODER_MODELS = {
    ("vision-encoder-decoder", "nlpconnect/vit-gpt2-image-captioning"),
}

PYTORCH_EXPORT_WITH_PAST_MODELS = {
    ("bloom", "hf-internal-testing/tiny-random-BloomModel"),
    ("gpt2", "hf-internal-testing/tiny-random-GPT2Model"),
    ("gpt-neo", "hf-internal-testing/tiny-random-GPTNeoModel"),
}

PYTORCH_EXPORT_SEQ2SEQ_WITH_PAST_MODELS = {
    ("bart", "hf-internal-testing/tiny-random-BartModel"),
    ("bigbird-pegasus", "hf-internal-testing/tiny-random-BigBirdPegasusModel"),
    ("blenderbot-small", "facebook/blenderbot_small-90M"),
    ("blenderbot", "hf-internal-testing/tiny-random-BlenderbotModel"),
    ("longt5", "hf-internal-testing/tiny-random-LongT5Model"),
    ("marian", "Helsinki-NLP/opus-mt-en-de"),
    ("mbart", "sshleifer/tiny-mbart"),
    ("mt5", "google/mt5-base"),
    ("m2m-100", "hf-internal-testing/tiny-random-M2M100Model"),
    ("t5", "hf-internal-testing/tiny-random-T5Model"),
}

# TODO(lewtun): Include the same model types in `PYTORCH_EXPORT_MODELS` once TensorFlow has parity with the PyTorch model implementations.
TENSORFLOW_EXPORT_DEFAULT_MODELS = {
    ("albert", "hf-internal-testing/tiny-albert"),
    ("bert", "hf-internal-testing/tiny-random-BertModel"),
    ("camembert", "camembert-base"),
    ("distilbert", "hf-internal-testing/tiny-random-DistilBertModel"),
    ("roberta", "hf-internal-testing/tiny-random-RobertaModel"),
}

# TODO(lewtun): Include the same model types in `PYTORCH_EXPORT_WITH_PAST_MODELS` once TensorFlow has parity with the PyTorch model implementations.
TENSORFLOW_EXPORT_WITH_PAST_MODELS = {}

# TODO(lewtun): Include the same model types in `PYTORCH_EXPORT_SEQ2SEQ_WITH_PAST_MODELS` once TensorFlow has parity with the PyTorch model implementations.
TENSORFLOW_EXPORT_SEQ2SEQ_WITH_PAST_MODELS = {}


def _get_models_to_test(export_models_list):
    models_to_test = []
    if is_torch_available() or is_tf_available():
        for name, model, *features in export_models_list:
            if features:
                feature_config_mapping = {
                    feature: FeaturesManager.get_config(name, feature) for _ in features for feature in _
                }
            else:
                # pre-process the model names
                model_type = name.replace("_", "-")
                model_name = getattr(model, "name", "")
                feature_config_mapping = FeaturesManager.get_supported_features_for_model_type(
                    model_type, model_name=model_name
                )

            for feature, onnx_config_class_constructor in feature_config_mapping.items():
                models_to_test.append((f"{name}_{feature}", name, model, feature, onnx_config_class_constructor))
        return sorted(models_to_test)
    else:
        # Returning some dummy test that should not be ever called because of the @require_torch / @require_tf
        # decorators.
        # The reason for not returning an empty list is because parameterized.expand complains when it's empty.
        return [("dummy", "dummy", "dummy", "dummy", OnnxConfig.from_model_config)]


class OnnxExportTestCaseV2(TestCase):
    """
    Integration tests ensuring supported models are correctly exported
    """

    def _onnx_export(
        self, test_name, name, model_name, feature, onnx_config_class_constructor, device="cpu", framework="pt"
    ):
        from transformers.onnx import export

        model_class = FeaturesManager.get_model_class_for_feature(feature, framework=framework)
        config = AutoConfig.from_pretrained(model_name)
        model = model_class.from_config(config)

        # Dynamic axes aren't supported for YOLO-like models. This means they cannot be exported to ONNX on CUDA devices.
        # See: https://github.com/ultralytics/yolov5/pull/8378
        if model.__class__.__name__.startswith("Yolos") and device != "cpu":
            return

        # ONNX inference fails with the following name, feature, framework parameterizations
        # See: https://github.com/huggingface/transformers/issues/19357
        if (name, feature, framework) in {
            ("deberta-v2", "question-answering", "pt"),
            ("deberta-v2", "multiple-choice", "pt"),
            ("roformer", "multiple-choice", "pt"),
            ("groupvit", "default", "pt"),
            ("perceiver", "masked-lm", "pt"),
            ("perceiver", "sequence-classification", "pt"),
            ("perceiver", "image-classification", "pt"),
            ("bert", "multiple-choice", "tf"),
            ("camembert", "multiple-choice", "tf"),
            ("roberta", "multiple-choice", "tf"),
        }:
            return

        onnx_config = onnx_config_class_constructor(model.config)

        if is_torch_available():
            from transformers.utils import torch_version

            if torch_version < onnx_config.torch_onnx_minimum_version:
                pytest.skip(
                    "Skipping due to incompatible PyTorch version. Minimum required is"
                    f" {onnx_config.torch_onnx_minimum_version}, got: {torch_version}"
                )

        preprocessor = get_preprocessor(model_name)

        # Useful for causal lm models that do not use pad tokens.
        if isinstance(preprocessor, PreTrainedTokenizerBase) and not getattr(config, "pad_token_id", None):
            config.pad_token_id = preprocessor.eos_token_id

        with NamedTemporaryFile("w") as output:
            try:
                onnx_inputs, onnx_outputs = export(
                    preprocessor, model, onnx_config, onnx_config.default_onnx_opset, Path(output.name), device=device
                )
                validate_model_outputs(
                    onnx_config,
                    preprocessor,
                    model,
                    Path(output.name),
                    onnx_outputs,
                    onnx_config.atol_for_validation,
                )
            except (RuntimeError, ValueError) as e:
                self.fail(f"{name}, {feature} -> {e}")

    def _onnx_export_encoder_decoder_models(
        self, test_name, name, model_name, feature, onnx_config_class_constructor, device="cpu"
    ):
        from transformers import AutoFeatureExtractor, AutoTokenizer
        from transformers.onnx import export

        model_class = FeaturesManager.get_model_class_for_feature(feature)
        config = AutoConfig.from_pretrained(model_name)
        model = model_class.from_config(config)

        onnx_config = onnx_config_class_constructor(model.config)

        if is_torch_available():
            from transformers.utils import torch_version

            if torch_version < onnx_config.torch_onnx_minimum_version:
                pytest.skip(
                    "Skipping due to incompatible PyTorch version. Minimum required is"
                    f" {onnx_config.torch_onnx_minimum_version}, got: {torch_version}"
                )

        encoder_model = model.get_encoder()
        decoder_model = model.get_decoder()

        encoder_onnx_config = onnx_config.get_encoder_config(encoder_model.config)
        decoder_onnx_config = onnx_config.get_decoder_config(encoder_model.config, decoder_model.config, feature)

        preprocessor = AutoFeatureExtractor.from_pretrained(model_name)

        onnx_opset = max(encoder_onnx_config.default_onnx_opset, decoder_onnx_config.default_onnx_opset)

        with NamedTemporaryFile("w") as encoder_output:
            onnx_inputs, onnx_outputs = export(
                preprocessor, encoder_model, encoder_onnx_config, onnx_opset, Path(encoder_output.name), device=device
            )
            validate_model_outputs(
                encoder_onnx_config,
                preprocessor,
                encoder_model,
                Path(encoder_output.name),
                onnx_outputs,
                encoder_onnx_config.atol_for_validation,
            )

        preprocessor = AutoTokenizer.from_pretrained(model_name)

        with NamedTemporaryFile("w") as decoder_output:
            _, onnx_outputs = export(
                preprocessor,
                decoder_model,
                decoder_onnx_config,
                onnx_config.default_onnx_opset,
                Path(decoder_output.name),
                device=device,
            )
            validate_model_outputs(
                decoder_onnx_config,
                preprocessor,
                decoder_model,
                Path(decoder_output.name),
                onnx_outputs,
                decoder_onnx_config.atol_for_validation,
            )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS))
    @slow
    @require_torch
    @require_vision
    @require_rjieba
    def test_pytorch_export(self, test_name, name, model_name, feature, onnx_config_class_constructor):
        self._onnx_export(test_name, name, model_name, feature, onnx_config_class_constructor)

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS))
    @slow
    @require_torch
    @require_vision
    @require_rjieba
    def test_pytorch_export_on_cuda(self, test_name, name, model_name, feature, onnx_config_class_constructor):
        self._onnx_export(test_name, name, model_name, feature, onnx_config_class_constructor, device="cuda")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_ENCODER_DECODER_MODELS))
    @slow
    @require_torch
    @require_vision
    @require_rjieba
    def test_pytorch_export_encoder_decoder_models(
        self, test_name, name, model_name, feature, onnx_config_class_constructor
    ):
        self._onnx_export_encoder_decoder_models(test_name, name, model_name, feature, onnx_config_class_constructor)

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_ENCODER_DECODER_MODELS))
    @slow
    @require_torch
    @require_vision
    @require_rjieba
    def test_pytorch_export_encoder_decoder_models_on_cuda(
        self, test_name, name, model_name, feature, onnx_config_class_constructor
    ):
        self._onnx_export_encoder_decoder_models(
            test_name, name, model_name, feature, onnx_config_class_constructor, device="cuda"
        )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_WITH_PAST_MODELS))
    @slow
    @require_torch
    def test_pytorch_export_with_past(self, test_name, name, model_name, feature, onnx_config_class_constructor):
        self._onnx_export(test_name, name, model_name, feature, onnx_config_class_constructor)

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_SEQ2SEQ_WITH_PAST_MODELS))
    @slow
    @require_torch
    def test_pytorch_export_seq2seq_with_past(
        self, test_name, name, model_name, feature, onnx_config_class_constructor
    ):
        self._onnx_export(test_name, name, model_name, feature, onnx_config_class_constructor)

    @parameterized.expand(_get_models_to_test(TENSORFLOW_EXPORT_DEFAULT_MODELS))
    @slow
    @require_tf
    @require_vision
    def test_tensorflow_export(self, test_name, name, model_name, feature, onnx_config_class_constructor):
        self._onnx_export(test_name, name, model_name, feature, onnx_config_class_constructor, framework="tf")

    @parameterized.expand(_get_models_to_test(TENSORFLOW_EXPORT_WITH_PAST_MODELS), skip_on_empty=True)
    @slow
    @require_tf
    def test_tensorflow_export_with_past(self, test_name, name, model_name, feature, onnx_config_class_constructor):
        self._onnx_export(test_name, name, model_name, feature, onnx_config_class_constructor, framework="tf")

    @parameterized.expand(_get_models_to_test(TENSORFLOW_EXPORT_SEQ2SEQ_WITH_PAST_MODELS), skip_on_empty=True)
    @slow
    @require_tf
    def test_tensorflow_export_seq2seq_with_past(
        self, test_name, name, model_name, feature, onnx_config_class_constructor
    ):
        self._onnx_export(test_name, name, model_name, feature, onnx_config_class_constructor, framework="tf")


class StableDropoutTestCase(TestCase):
    """Tests export of StableDropout module."""

    @unittest.skip("torch 2.0.0 gives `torch.onnx.errors.OnnxExporterError: Module onnx is not installed!`.")
    @require_torch
    @pytest.mark.filterwarnings("ignore:.*Dropout.*:UserWarning:torch.onnx.*")  # torch.onnx is spammy.
    def test_training(self):
        """Tests export of StableDropout in training mode."""
        devnull = open(os.devnull, "wb")
        # drop_prob must be > 0 for the test to be meaningful
        sd = modeling_deberta.StableDropout(0.1)
        # Avoid warnings in training mode
        do_constant_folding = False
        # Dropout is a no-op in inference mode
        training = torch.onnx.TrainingMode.PRESERVE
        input = (torch.randn(2, 2),)

        torch.onnx.export(
            sd,
            input,
            devnull,
            opset_version=12,  # Minimum supported
            do_constant_folding=do_constant_folding,
            training=training,
        )

        # Expected to fail with opset_version < 12
        with self.assertRaises(Exception):
            torch.onnx.export(
                sd,
                input,
                devnull,
                opset_version=11,
                do_constant_folding=do_constant_folding,
                training=training,
            )
