import unittest
from unittest.mock import patch

from transformers.trainer_utils import validate_quantization_for_training


class DummyQuantizationConfig:
    def __init__(self, quant_method="fp8"):
        self.quant_method = quant_method


class DummyQuantizer:
    def __init__(self, is_trainable=False, is_qat_trainable=False):
        self._is_trainable = is_trainable
        self._is_qat_trainable = is_qat_trainable

    @property
    def is_trainable(self):
        return self._is_trainable

    @property
    def is_qat_trainable(self):
        return self._is_qat_trainable

    @property
    def quantization_config(self):
        return DummyQuantizationConfig()


class DummyQuantizedModel:
    def __init__(self, is_trainable=False, hf_peft_config_loaded=False):
        self.is_quantized = True
        self._hf_peft_config_loaded = hf_peft_config_loaded
        self.hf_quantizer = DummyQuantizer(is_trainable=is_trainable)


class ValidateQuantizationForTrainingTest(unittest.TestCase):
    def test_unquantized_model_passes(self):
        model = DummyQuantizedModel()
        model.is_quantized = False
        validate_quantization_for_training(model)

    def test_untrainable_quantized_base_model_raises(self):
        with self.assertRaisesRegex(ValueError, "purely quantized models"):
            validate_quantization_for_training(DummyQuantizedModel())

    def test_qat_flagged_but_untrainable_method_raises(self):
        model = DummyQuantizedModel()
        model.hf_quantizer = DummyQuantizer(is_trainable=False, is_qat_trainable=True)
        with self.assertRaisesRegex(ValueError, "do not support training"):
            validate_quantization_for_training(model)

    def test_trainable_quantized_base_model_without_adapters_raises(self):
        with self.assertRaisesRegex(ValueError, "purely quantized models"):
            validate_quantization_for_training(DummyQuantizedModel(is_trainable=True))

    def test_compiled_quantized_model_raises(self):
        model = DummyQuantizedModel()
        model._orig_mod = object()
        with self.assertRaisesRegex(ValueError, "torch.compile"):
            validate_quantization_for_training(model)

    def test_qat_trainable_model_passes(self):
        model = DummyQuantizedModel(is_trainable=True)
        model.hf_quantizer = DummyQuantizer(is_trainable=True, is_qat_trainable=True)
        validate_quantization_for_training(model)

    def test_native_peft_adapters_passes(self):
        validate_quantization_for_training(DummyQuantizedModel(hf_peft_config_loaded=True))

    def test_peft_wrapped_model_with_trainable_quant_method_passes(self):
        with patch("transformers.trainer_utils._is_peft_model", return_value=True):
            validate_quantization_for_training(DummyQuantizedModel(is_trainable=True))

    def test_peft_wrapped_model_with_untrainable_quant_method_passes(self):
        with patch("transformers.trainer_utils._is_peft_model", return_value=True):
            validate_quantization_for_training(DummyQuantizedModel())
