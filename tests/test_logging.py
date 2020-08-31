import unittest

from transformers import logging
from transformers.testing_utils import CaptureLogger


class HfArgumentParserTest(unittest.TestCase):
    def test_set_level(self):
        logger = logging.get_logger()

        level_origin = logging.get_verbosity()
        self.assertEqual(level_origin, logging.WARNING)

        logging.set_verbosity_error()
        self.assertEqual(logger.getEffectiveLevel(), logging.get_verbosity())

        logging.set_verbosity_warning()
        self.assertEqual(logger.getEffectiveLevel(), logging.get_verbosity())

        logging.set_verbosity_info()
        self.assertEqual(logger.getEffectiveLevel(), logging.get_verbosity())

        logging.set_verbosity_debug()
        self.assertEqual(logger.getEffectiveLevel(), logging.get_verbosity())

        # restore
        logging.set_verbosity(level_origin)

    def test_integration(self):

        import transformers.tokenization_bart  # noqa

        logger = logging.get_logger("transformers.tokenization_bart")
        msg = "Testing 1, 2, 3"


        # should be able to log warnings (default setting)
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
