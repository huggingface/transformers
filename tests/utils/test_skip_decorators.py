# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
#
#
#
# this test validates that we can stack skip decorators in groups and whether
# they work correctly with other decorators
#
# since the decorators have already built their decision params (like checking
# env[], we can't mock the env and test each of the combinations), so ideally
# the following 4 should be run. But since we have different CI jobs running
# different configs, all combinations should get covered
#
# RUN_SLOW=1 pytest -rA tests/test_skip_decorators.py
# RUN_SLOW=1 CUDA_VISIBLE_DEVICES="" pytest -rA tests/test_skip_decorators.py
# RUN_SLOW=0 pytest -rA tests/test_skip_decorators.py
# RUN_SLOW=0 CUDA_VISIBLE_DEVICES="" pytest -rA tests/test_skip_decorators.py

import os
import unittest

import pytest
from parameterized import parameterized

from transformers.testing_utils import require_torch, require_torch_gpu, slow, torch_device


# skipping in unittest tests

params = [(1,)]


# test that we can stack our skip decorators with 3rd party decorators
def check_slow():
    run_slow = bool(os.getenv("RUN_SLOW", 0))
    if run_slow:
        assert True
    else:
        assert False, "should have been skipped"


# test that we can stack our skip decorators
def check_slow_torch_cuda():
    run_slow = bool(os.getenv("RUN_SLOW", 0))
    if run_slow and torch_device == "cuda":
        assert True
    else:
        assert False, "should have been skipped"


@require_torch
class SkipTester(unittest.TestCase):
    @slow
    @require_torch_gpu
    def test_2_skips_slow_first(self):
        check_slow_torch_cuda()

    @require_torch_gpu
    @slow
    def test_2_skips_slow_last(self):
        check_slow_torch_cuda()

    # The combination of any skip decorator, followed by parameterized fails to skip the tests
    # 1. @slow manages to correctly skip `test_param_slow_first`
    # 2. but then `parameterized` creates new tests, with a unique name for each parameter groups.
    #    It has no idea that they are to be skipped and so they all run, ignoring @slow
    # Therefore skip decorators must come after `parameterized`
    #
    # @slow
    # @parameterized.expand(params)
    # def test_param_slow_first(self, param=None):
    #     check_slow()

    # This works as expected:
    # 1. `parameterized` creates new tests with unique names
    # 2. each of them gets an opportunity to be skipped
    @parameterized.expand(params)
    @slow
    def test_param_slow_last(self, param=None):
        check_slow()


# skipping in non-unittest tests
# no problem at all here


@slow
@require_torch_gpu
def test_pytest_2_skips_slow_first():
    check_slow_torch_cuda()


@require_torch_gpu
@slow
def test_pytest_2_skips_slow_last():
    check_slow_torch_cuda()


@slow
@pytest.mark.parametrize("param", [1])
def test_pytest_param_slow_first(param):
    check_slow()


@pytest.mark.parametrize("param", [1])
@slow
def test_pytest_param_slow_last(param):
    check_slow()
