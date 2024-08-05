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

import os
import unittest

from huggingface_hub.utils import are_progress_bars_disabled

import transformers.models.bart.tokenization_bart
from transformers import logging
from transformers.testing_utils import CaptureLogger, mockenv, mockenv_context
from transformers.utils.logging import disable_progress_bar, enable_progress_bar


class HfArgumentParserTest(unittest.TestCase):
    def test_set_level(self):
        logger = logging.get_logger()

        # the current default level is logging.WARNING
        level_origin = logging.get_verbosity()

        logging.set_verbosity_error()
        self.assertEqual(logger.getEffectiveLevel(), logging.get_verbosity())

        logging.set_verbosity_warning()
        self.assertEqual(logger.getEffectiveLevel(), logging.get_verbosity())

        logging.set_verbosity_info()
        self.assertEqual(logger.getEffectiveLevel(), logging.get_verbosity())

        logging.set_verbosity_debug()
        self.assertEqual(logger.getEffectiveLevel(), logging.get_verbosity())

        # restore to the original level
        logging.set_verbosity(level_origin)

    def test_integration(self):
        level_origin = logging.get_verbosity()

        logger = logging.get_logger("transformers.models.bart.tokenization_bart")
        msg = "Testing 1, 2, 3"

        # should be able to log warnings (if default settings weren't overridden by `pytest --log-level-all`)
        if level_origin <= logging.WARNING:
            with CaptureLogger(logger) as cl:
                logger.warning(msg)
            self.assertEqual(cl.out, msg + "\n")

        # this is setting the level for all of `transformers.*` loggers
        logging.set_verbosity_error()

        # should not be able to log warnings
        with CaptureLogger(logger) as cl:
            logger.warning(msg)
        self.assertEqual(cl.out, "")

        # should be able to log warnings again
        logging.set_verbosity_warning()
        with CaptureLogger(logger) as cl:
            logger.warning(msg)
        self.assertEqual(cl.out, msg + "\n")

        # restore to the original level
        logging.set_verbosity(level_origin)

    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        # reset for the env var to take effect, next time some logger call is made
        transformers.utils.logging._reset_library_root_logger()
        # this action activates the env var
        _ = logging.get_logger("transformers.models.bart.tokenization_bart")

        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
        env_level = logging.log_levels[env_level_str]

        current_level = logging.get_verbosity()
        self.assertEqual(
            env_level,
            current_level,
            f"TRANSFORMERS_VERBOSITY={env_level_str}/{env_level}, but internal verbosity is {current_level}",
        )

        # restore to the original level
        os.environ["TRANSFORMERS_VERBOSITY"] = ""
        transformers.utils.logging._reset_library_root_logger()

    @mockenv(TRANSFORMERS_VERBOSITY="super-error")
    def test_env_invalid_override(self):
        # reset for the env var to take effect, next time some logger call is made
        transformers.utils.logging._reset_library_root_logger()
        logger = logging.logging.getLogger()
        with CaptureLogger(logger) as cl:
            # this action activates the env var
            logging.get_logger("transformers.models.bart.tokenization_bart")
        self.assertIn("Unknown option TRANSFORMERS_VERBOSITY=super-error", cl.out)

        # no need to restore as nothing was changed

    def test_advisory_warnings(self):
        # testing `logger.warning_advice()`
        transformers.utils.logging._reset_library_root_logger()

        logger = logging.get_logger("transformers.models.bart.tokenization_bart")
        msg = "Testing 1, 2, 3"

        with mockenv_context(TRANSFORMERS_NO_ADVISORY_WARNINGS="1"):
            # nothing should be logged as env var disables this method
            with CaptureLogger(logger) as cl:
                logger.warning_advice(msg)
            self.assertEqual(cl.out, "")

        with mockenv_context(TRANSFORMERS_NO_ADVISORY_WARNINGS=""):
            # should log normally as TRANSFORMERS_NO_ADVISORY_WARNINGS is unset
            with CaptureLogger(logger) as cl:
                logger.warning_advice(msg)
            self.assertEqual(cl.out, msg + "\n")


def test_set_progress_bar_enabled():
    disable_progress_bar()
    assert are_progress_bars_disabled()

    enable_progress_bar()
    assert not are_progress_bars_disabled()
