import unittest

from transformers import logging


class HfArgumentParserTest(unittest.TestCase):
    def test_set_level(self):
        logger = logging.get_logger()

        logging.set_verbosity_error()
        self.assertEqual(logger.getEffectiveLevel(), logging.get_verbosity())

        logging.set_verbosity_warning()
        self.assertEqual(logger.getEffectiveLevel(), logging.get_verbosity())

        logging.set_verbosity_info()
        self.assertEqual(logger.getEffectiveLevel(), logging.get_verbosity())

        logging.set_verbosity_debug()
        self.assertEqual(logger.getEffectiveLevel(), logging.get_verbosity())
