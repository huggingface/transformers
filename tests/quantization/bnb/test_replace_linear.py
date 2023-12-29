import unittest
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Type
from unittest.mock import patch

import bitsandbytes as bnb
from torch import nn
from torch.nn import Linear, Module, ReLU

from transformers import BitsAndBytesConfig
from transformers.integrations.bitsandbytes import replace_with_bnb_linear
from transformers.testing_utils import require_bitsandbytes


class TwoLayerFCModel(Module):
    def __init__(self):
        super().__init__()
        self.fully_connected_1 = Linear(10, 50)
        self.relu = ReLU()
        self.fully_connected_2 = Linear(50, 1)

    def forward(self, x):
        x = self.fully_connected_1(x)
        x = self.relu(x)
        return self.fully_connected_2(x)


class ExtendedTwoLayerFCModel(TwoLayerFCModel):
    def __init__(self):
        super().__init__()
        self.skip_me = Linear(50, 50)  # This layer should be skipped, if in `skip_modules`


##


class NoLinearToSwapModel(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ReLU()
        self.lm_head = nn.Linear(5, 10)  # This layer has to be ignored by default


class OneLinearToSwapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.lm_head = nn.Linear(5, 10)  # This layer has to be ignored by default


class TwoLinearToSwapModel(OneLinearToSwapModel):
    def __init__(self):
        super().__init__()
        self.linear2 = nn.Linear(10, 5)


class NestedModel(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(10, 5)
        self.lm_head = nn.Linear(5, 10)  # This layer has to be ignored by default

        self.submodel0 = NoLinearToSwapModel()
        self.submodel1 = OneLinearToSwapModel()
        self.submodel2 = TwoLinearToSwapModel()


class DoubleNestedModel(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(10, 5)
        self.lm_head = nn.Linear(5, 10)  # This layer has to be ignored by default

        self.submodel_nested_1 = NestedModel()
        self.submodel_nested_2 = NestedModel()


def count_modules_by_names(model: nn.Module, target_names: Optional[List[str]]) -> int:
    """
    Count all submodules within a PyTorch model that have names specified in the target_names list.
    """
    if not target_names:
        return 0

    return sum(1 for name, _ in model.named_modules() if name.split(".")[-1] in target_names)


@dataclass
class TestConfig:
    bnb_cfg: BitsAndBytesConfig
    replacement_linear_class: Type
    shorthand: str
    modules_to_not_convert: Optional[List[str]] = None
    llm_int8_skip_modules: Optional[List[str]] = None


@require_bitsandbytes
class TestReplaceWithBnbLinear(unittest.TestCase):
    def setUp(self):
        self.mock_log_warning_patch = patch("logging.Logger.warning")
        self.mock_log_warning = self.mock_log_warning_patch.start()
        self.warning_checked = False

    def tearDown(self):
        self.mock_log_warning_patch.stop()

        if not self.warning_checked and self.mock_log_warning.call_args_list:
            unexpected_warnings = [call_args[0][0] for call_args in self.mock_log_warning.call_args_list]
            formatted_warnings = "\n   ".join(["Unexpected warning(s) logged:"] + unexpected_warnings)

            self.fail(formatted_warnings)

    def check_warning_logged(self, warning_message):
        self.warning_checked = True

        found = any(warning_message in call_args[0][0] for call_args in self.mock_log_warning.call_args_list)
        self.assertTrue(found, f"Expected warning message '{warning_message}' was not found")

    def verify_replaced_module_count(self, model, replaced_class, expected_count):
        count = sum(1 for module in model.modules() if isinstance(module, replaced_class))
        self.assertEqual(count, expected_count, f"{expected_count} modules should be replaced")

    def common_test_logic(
        self,
        test_cfg: TestConfig,
        dummy_model_class,
        expected_replaced_count,
        expected_warning_message=None,
    ):
        model = dummy_model_class()
        replaced_model = replace_with_bnb_linear(
            model,
            quantization_config=test_cfg.bnb_cfg,
            modules_to_not_convert=test_cfg.modules_to_not_convert,
        )

        self.verify_replaced_module_count(
            replaced_model,
            test_cfg.replacement_linear_class,
            expected_replaced_count,
        )

        for name, module in replaced_model.named_modules():
            if isinstance(module, test_cfg.replacement_linear_class):
                self.assertIsInstance(module, test_cfg.replacement_linear_class)

            elif "lm_head" in name:
                self.assertEqual(type(module), Linear)

        if expected_warning_message:
            self.check_warning_logged(expected_warning_message)

    @staticmethod
    def add_test_case(
        test_cfg,
        *,
        dummy_model_class,
        expected_replaced_count,
        expected_warning_message=None,
    ):
        def test(self):
            self.common_test_logic(
                test_cfg,
                dummy_model_class,
                expected_replaced_count,
                expected_warning_message,
            )

        test.__name__ = f"test_{dummy_model_class.__name__}_{test_cfg.shorthand}"
        setattr(TestReplaceWithBnbLinear, test.__name__, test)

    @staticmethod
    def quantization_test_configs():
        yield TestConfig(
            bnb_cfg=BitsAndBytesConfig(load_in_8bit=True),
            replacement_linear_class=bnb.nn.Linear8bitLt,
            shorthand="8bit",
        )
        yield TestConfig(
            bnb_cfg=BitsAndBytesConfig(load_in_4bit=True),
            replacement_linear_class=bnb.nn.Linear4bit,
            shorthand="4bit",
        )

    @staticmethod
    def scenarios():
        """Generate individual tests for each scenario"""
        for quant_config in TestReplaceWithBnbLinear.quantization_test_configs():
            # Scenario without extra settings
            yield quant_config

            # # Scenario with modules_to_not_convert
            # modules_to_not_convert_config = deepcopy(quant_config)
            # modules_to_not_convert_config.modules_to_not_convert = ["linear1"]
            # yield modules_to_not_convert_config

            # # Scenario with llm_int8_skip_modules
            # llm_int8_skip_modules_config = deepcopy(quant_config)
            # llm_int8_skip_modules_config.llm_int8_skip_modules = ["linear1"]
            # yield llm_int8_skip_modules_config

            # # Scenario with both
            # modules_to_not_convert_config = deepcopy(quant_config)
            # modules_to_not_convert_config.modules_to_not_convert = ["skip_me"]
            # llm_int8_skip_modules_config.llm_int8_skip_modules = ["skip_me"]
            # yield llm_int8_skip_modules_config

    @classmethod
    def generate_tests(cls):
        for test_cfg in cls.scenarios():
            expected_baseline_replacement_counts = {
                NoLinearToSwapModel: 0,
                OneLinearToSwapModel: 1,
                TwoLinearToSwapModel: 2,
                NestedModel: 5,
                DoubleNestedModel: 12,
            }

            for dummy_model_class, initial_expected_count in expected_baseline_replacement_counts.items():
                model_instance = dummy_model_class()

                ignore_count = count_modules_by_names(model_instance, test_cfg.modules_to_not_convert)

                adjusted_expected_count = initial_expected_count - ignore_count

                warning_message = (
                    "no linear modules were found in your model" if dummy_model_class is NoLinearToSwapModel else None
                )

                cls.add_test_case(
                    test_cfg,
                    dummy_model_class=dummy_model_class,
                    expected_replaced_count=adjusted_expected_count,
                    expected_warning_message=warning_message,
                )


TestReplaceWithBnbLinear.generate_tests()

if __name__ == "__main__":
    unittest.main()
