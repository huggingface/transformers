"""Tests: hard error on total weight mismatch with conversions/quantization (#47405, #47407)."""
import unittest
from unittest.mock import MagicMock, patch

from transformers.core_model_loading import WeightRenaming
from transformers.modeling_utils import LoadStateDictConfig
from transformers.utils.loading_report import LoadStateDictInfo


class TotalMismatchHardErrorTest(unittest.TestCase):
    """A checkpoint matching ~nothing with conversions/quantization must raise."""

    @staticmethod
    def _model_mock(state_dict_len=1):
        """Create a mock model that behaves like PreTrainedModel for _finalize_model_loading."""
        model = MagicMock()
        model.state_dict.return_value = {f"p{i}": None for i in range(state_dict_len)}
        model._adjust_missing_and_unexpected_keys = MagicMock()
        model.mark_tied_weights_as_initialized = MagicMock()
        model._move_missing_keys_from_meta_to_device = MagicMock()
        model._initialize_missing_keys = MagicMock()
        model.tie_weights = MagicMock()
        return model

    @staticmethod
    def _loading_info(missing_count=1):
        return LoadStateDictInfo(
            missing_keys={f"p{i}" for i in range(missing_count)},
            unexpected_keys={"model.llm.some.converted.weight"},
            mismatched_keys=set(),
            error_msgs=[],
            conversion_errors={},
        )

    def test_weight_mapping_raises(self):
        model = self._model_mock()
        cfg = LoadStateDictConfig(
            pretrained_model_name_or_path="dummy/path",
            weight_mapping=[WeightRenaming("a", "b")],
        )
        with self.assertRaises(ValueError):
            model.__class__._finalize_model_loading(model, cfg, self._loading_info())

    def test_hf_quantizer_raises(self):
        model = self._model_mock()
        cfg = LoadStateDictConfig(
            pretrained_model_name_or_path="dummy/path",
            weight_mapping=None,
            hf_quantizer=MagicMock(),
        )
        with self.assertRaises(ValueError):
            model.__class__._finalize_model_loading(model, cfg, self._loading_info())

    def test_no_conversions_no_quantizer_does_not_raise(self):
        model = self._model_mock()
        cfg = LoadStateDictConfig(
            pretrained_model_name_or_path="dummy/path",
            weight_mapping=None,
            hf_quantizer=None,
        )
        # Should not raise
        result = model.__class__._finalize_model_loading(model, cfg, self._loading_info())
        self.assertIsNotNone(result)

    def test_both_raises(self):
        model = self._model_mock()
        cfg = LoadStateDictConfig(
            pretrained_model_name_or_path="dummy/path",
            weight_mapping=[WeightRenaming("a", "b")],
            hf_quantizer=MagicMock(),
        )
        with self.assertRaises(ValueError):
            model.__class__._finalize_model_loading(model, cfg, self._loading_info())

    def test_partial_mismatch_does_not_raise(self):
        """Only 50% mismatch should NOT trigger the error."""
        model = self._model_mock(state_dict_len=100)
        cfg = LoadStateDictConfig(
            pretrained_model_name_or_path="dummy/path",
            weight_mapping=[WeightRenaming("a", "b")],
        )
        info = self._loading_info(missing_count=50)
        # Should not raise (50/100 = 50% < 99%)
        result = model.__class__._finalize_model_loading(model, cfg, info)
        self.assertIsNotNone(result)

    def test_no_unexpected_keys_does_not_raise(self):
        """No unexpected keys means all checkpoint tensors were consumed -> no error."""
        model = self._model_mock()
        cfg = LoadStateDictConfig(
            pretrained_model_name_or_path="dummy/path",
            weight_mapping=[WeightRenaming("a", "b")],
        )
        info = LoadStateDictInfo(
            missing_keys={"p0"},
            unexpected_keys=set(),
            mismatched_keys=set(),
            error_msgs=[],
            conversion_errors={},
        )
        result = model.__class__._finalize_model_loading(model, cfg, info)
        self.assertIsNotNone(result)
