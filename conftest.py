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

# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import doctest
import sys
import warnings
from os.path import abspath, dirname, join

import _pytest
import pytest

from transformers.testing_utils import HfDoctestModule, HfDocTestParser, is_torch_available


NOT_DEVICE_TESTS = {
    "test_tokenization",
    "test_tokenization_mistral_common",
    "test_processing",
    "test_beam_constraints",
    "test_configuration_utils",
    "test_data_collator",
    "test_trainer_callback",
    "test_trainer_utils",
    "test_feature_extraction",
    "test_image_processing",
    "test_image_processor",
    "test_image_transforms",
    "test_optimization",
    "test_retrieval",
    "test_config",
    "test_from_pretrained_no_checkpoint",
    "test_keep_in_fp32_modules",
    "test_gradient_checkpointing_backward_compatibility",
    "test_gradient_checkpointing_enable_disable",
    "test_torch_save_load",
    "test_initialization",
    "test_forward_signature",
    "test_model_get_set_embeddings",
    "test_model_main_input_name",
    "test_correct_missing_keys",
    "test_tie_model_weights",
    "test_can_use_safetensors",
    "test_load_save_without_tied_weights",
    "test_tied_weights_keys",
    "test_model_weights_reload_no_missing_tied_weights",
    "test_mismatched_shapes_have_properly_initialized_weights",
    "test_matched_shapes_have_loaded_weights_when_some_mismatched_shapes_exist",
    "test_model_is_small",
    "test_tf_from_pt_safetensors",
    "test_flax_from_pt_safetensors",
    "ModelTest::test_pipeline_",  # None of the pipeline tests from PipelineTesterMixin (of which XxxModelTest inherits from) are running on device
    "ModelTester::test_pipeline_",
    "/repo_utils/",
    "/utils/",
}

# allow having multiple repository checkouts and not needing to remember to rerun
# `pip install -e '.[dev]'` when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(__file__), "src"))
sys.path.insert(1, git_repo_path)

# silence FutureWarning warnings in tests since often we can't act on them until
# they become normal warnings - i.e. the tests still need to test the current functionality
warnings.simplefilter(action="ignore", category=FutureWarning)


def pytest_configure(config):
    config.addinivalue_line("markers", "is_pipeline_test: mark test to run only when pipelines are tested")
    config.addinivalue_line("markers", "is_staging_test: mark test to run only in the staging environment")
    config.addinivalue_line("markers", "accelerate_tests: mark test that require accelerate")
    config.addinivalue_line("markers", "not_device_test: mark the tests always running on cpu")


def pytest_collection_modifyitems(items):
    for item in items:
        if any(test_name in item.nodeid for test_name in NOT_DEVICE_TESTS):
            item.add_marker(pytest.mark.not_device_test)


def pytest_addoption(parser):
    from transformers.testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from transformers.testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)


def pytest_sessionfinish(session, exitstatus):
    # If no tests are collected, pytest exists with code 5, which makes the CI fail.
    if exitstatus == 5:
        session.exitstatus = 0


# Doctest custom flag to ignore output.
IGNORE_RESULT = doctest.register_optionflag("IGNORE_RESULT")

OutputChecker = doctest.OutputChecker


class CustomOutputChecker(OutputChecker):
    def check_output(self, want, got, optionflags):
        if IGNORE_RESULT & optionflags:
            return True
        return OutputChecker.check_output(self, want, got, optionflags)


doctest.OutputChecker = CustomOutputChecker
_pytest.doctest.DoctestModule = HfDoctestModule
doctest.DocTestParser = HfDocTestParser

if is_torch_available():
    import torch

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    # We set it to `False` for CI. See https://github.com/pytorch/pytorch/issues/157274#issuecomment-3090791615
    torch.backends.cudnn.allow_tf32 = False

    # patched `torch.testing.assert_close` to gather more information.
    original_torch_testing_assert_close = torch.testing.assert_close

    def patched_torch_testing_assert_close(*args, **kwargs):
        try:
            with open("collected.txt", "a") as fp:
                fp.write("")
            original_torch_testing_assert_close(*args, **kwargs)
        except AssertionError as e:

            import os
            full_test_name = os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0]
            test_file, test_class, test_name = full_test_name.split("::")

            import traceback
            stack = traceback.extract_stack()
            target_frame = None
            for frame in stack:
                if test_file in str(frame) and test_name in str(frame):
                    target_frame = frame
                    break

            if target_frame is not None:
                line_number = target_frame.lineno

            if "actual" in kwargs:
                actual = kwargs["actual"]
            else:
                actual = args[0]

            def format(t):
                is_scalar = False
                if t.ndim == 0:
                    t = torch.tensor([t])
                    is_scalar = True

                # `detach` to remove `grad_fn=<...>`, and `to("cpu")` to remove `device='...'`
                t = t.detach().to("cpu")
                t = f"{t}"
                # remove `tensor( ... )` except the content `...`
                t = t.replace("tensor(", "").replace(")", "")
                # sometimes there are extra spaces between `[` and the actual values (for alignment).
                # For example `[[ 0.06, -0.51], [-0.76, -0.49]]`.
                # Let remove such extra spaces
                t = t.replace("[ ", "")
                # Put everything in a single line: replace `\n` by a space ` ` so we still keep `,\n` as `, `
                t = t.replace("\n", " ")
                # Remove repeated spaces
                while "  " in t:
                    t = t.replace("  ", " ")

                # remove leading `[` and `]`
                if is_scalar:
                    t = t[1:-1]

                return t

            # to string
            import json
            actual_str_1 = format(actual)
            actual_str_2 = json.dumps([format(x) for x in actual], indent=4)

            #
            actual_str_2 = actual_str_2.replace('"', '')
            actual_str_2 = actual_str_2.replace("\n]", ",\n]")
            actual_str_2 = actual_str_2.replace("]\n]", "],\n]")

            # tests/models/beit/test_modeling_beit.py::BeitModelIntegrationTest::test_inference_semantic_segmentation
            # tests/models/beit/test_modeling_beit.py:526
            # torch.testing.assert_close(
            # TODO: get the full method body
            # info = f"{full_test_name}\n{test_file}:{line_number}\n\n{target_frame.line}"

            from _pytest._code.source import Source
            with open(test_file) as fp:
                s = fp.read()
                source = Source(s)
                code = '\n'.join(source.getstatement(line_number-1).lines)
            info = f"{full_test_name}\n{test_file}:{line_number}\n\n{code}"

            # Adding the frame that calls this patched method
            caller_frame = stack[-2]
            caller_path = os.path.relpath(caller_frame.filename)
            # info = f"{info}\n\n{caller_path}:{caller_frame.lineno}\n\n{caller_frame.line}"

            from _pytest._code.source import Source
            with open(caller_path) as fp:
                s = fp.read()
                source = Source(s)
                code = '\n'.join(source.getstatement(caller_frame.lineno-1).lines)
            info = f"{info}\n\n{caller_path}:{caller_frame.lineno}\n\n{code}"

            info = f"{info}\n\n{actual_str_1}\n\n{actual_str_2}"
            info = f"{info}\n\n{'=' * 80}\n\n"

            with open("collected.txt", "a") as fp:
                fp.write(info)

            # TODO: what to do here?
            pass

    torch.testing.assert_close = patched_torch_testing_assert_close
