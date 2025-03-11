import unittest

from transformers.testing_utils import Expectation, Expectations, Properties


class ExpectationsTest(unittest.TestCase):
    def test_expectations(self):
        expectations = Expectations(
            Expectation.default(1),
            Expectation(Properties("cuda", 8, 1), 2),
            Expectation(Properties("cuda", 7, 0), 3),
            Expectation(Properties("rocm", 9, 1), 4),
            Expectation(Properties("rocm"), 5),
            Expectation(Properties("cpu"), 6),
        )

        def check(properties: Properties, value):
            assert expectations.find_expectation(properties).result == value

        # xpu has no matches so should find default expectation
        check(Properties("xpu"), 1)
        # major match
        check(Properties("cuda", 8, 0), 2)
        # major match, unaffected by minor
        check(Properties("cuda", 7, 1), 3)
        # full match
        check(Properties("rocm", 9, 1), 4)
        check(Properties("rocm", None, 0), 4)
        check(Properties("cuda", 2), 2)

        with self.assertRaises(ValueError):
            expectations = Expectations()

        expectations = Expectations(Expectation(Properties("cuda", 8, 0), 2))
        with self.assertRaises(ValueError):
            expectations.find_expectation(Properties("xpu"))
