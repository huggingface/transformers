import unittest

from transformers.testing_utils import Expectation, Expectations


class ExpectationsTest(unittest.TestCase):
    def test_expectations(self):
        expectations = Expectations(
            Expectation.default(1),
            Expectation(2, "cuda", 8),
            Expectation(3, "cuda", 7),
            Expectation(4, "rocm", 9),
            Expectation(5, "rocm"),
            Expectation(6, "cpu"),
        )

        def check(value, type=None, major=None):
            assert expectations.find_expectation(type, major).result == value

        # xpu has no matches so should find default expectation
        check(1, "xpu")
        check(2, "cuda", 8)
        check(3, "cuda", 7)
        check(4, "rocm", 9)
        check(4, "rocm")
        check(2, "cuda", 2)

        with self.assertRaises(ValueError):
            expectations = Expectations()

        expectations = Expectations(Expectation(1, "cuda", 8))
        with self.assertRaises(ValueError):
            expectations.find_expectation("xpu")
