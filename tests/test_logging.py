import os
import unittest

import transformers.tokenization_bart
from transformers import logging
from transformers.testing_utils import CaptureLogger, mockenv


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

        logger = logging.get_logger("transformers.tokenization_bart")
        msg = "Testing 1, 2, 3"

        # should be able to log warnings (if default settings weren't overriden by `pytest --log-level-all`)
        if level_origin <= logging.WARNING:
            with CaptureLogger(logger) as cl:
                logger.warn(msg)
            self.assertEqual(cl.out, msg + "\n")

        # this is setting the level for all of `transformers.*` loggers
        logging.set_verbosity_error()

        # should not be able to log warnings
        with CaptureLogger(logger) as cl:
            logger.warn(msg)
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
        _ = logging.get_logger("transformers.tokenization_bart")

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
            logging.get_logger("transformers.tokenization_bart")
        self.assertIn("Unknown option TRANSFORMERS_VERBOSITY=super-error", cl.out)

        # no need to restore as nothing was changed
