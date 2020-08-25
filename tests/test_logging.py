import unittest

from transformers import hf_logging

class HfArgumentParserTest(unittest.TestCase):

    def test_set_level(self):
        logger = hf_logging.get_logger()

        hf_logging.set_verbosity_error()
        self.assertEqual(logger.getEffectiveLevel(), hf_logging.get_verbosity())

        hf_logging.set_verbosity_warning()
        self.assertEqual(logger.getEffectiveLevel(), hf_logging.get_verbosity())

        hf_logging.set_verbosity_info()
        self.assertEqual(logger.getEffectiveLevel(), hf_logging.get_verbosity())

        hf_logging.set_verbosity_debug()
        self.assertEqual(logger.getEffectiveLevel(), hf_logging.get_verbosity())
