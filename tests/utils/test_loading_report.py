# Copyright 2026 The HuggingFace Inc. team.
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

import io
import logging
import unittest
from unittest.mock import patch

from transformers.utils.loading_report import LoadStateDictInfo, log_state_dict_report


class DummyModel:
    pass


class LoadingReportTest(unittest.TestCase):
    def test_create_loading_report_omits_ansi_when_stdout_is_not_tty(self):
        loading_info = LoadStateDictInfo(
            missing_keys={"missing.weight"},
            unexpected_keys={"unexpected.weight"},
            mismatched_keys={("mismatch.weight", (2, 2), (1, 1))},
            error_msgs=[],
            conversion_errors={},
        )

        with patch("sys.stdout.isatty", return_value=False):
            report = loading_info.create_loading_report()

        self.assertIsNotNone(report)
        self.assertNotIn("\x1b[", report)
        self.assertIn("Notes:", report)

    def test_log_state_dict_report_omits_ansi_when_stdout_is_not_tty(self):
        loading_info = LoadStateDictInfo(
            missing_keys={"missing.weight"},
            unexpected_keys={"unexpected.weight"},
            mismatched_keys=set(),
            error_msgs=[],
            conversion_errors={},
        )
        stream = io.StringIO()
        logger = logging.Logger("test_loading_report")
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        logger.handlers = [logging.StreamHandler(stream)]

        with patch("sys.stdout.isatty", return_value=False):
            log_state_dict_report(
                model=DummyModel(),
                pretrained_model_name_or_path="dummy/path",
                ignore_mismatched_sizes=True,
                loading_info=loading_info,
                logger=logger,
            )

        output = stream.getvalue()
        self.assertIn("DummyModel LOAD REPORT from: dummy/path", output)
        self.assertNotIn("\x1b[", output)
