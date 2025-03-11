import unittest

from transformers.testing_utils import Expectation, Expectations, Properties


class ExpectationsTest(unittest.TestCase):
    def test_expectations(self):
        expectations = Expectations(
            Expectation.default(1),
            Expectation(Properties("cuda", 8), 2),
            Expectation(Properties("cuda", 7), 3),
            Expectation(Properties("rocm", 9), 4),
            Expectation(Properties("rocm"), 5),
            Expectation(Properties("cpu"), 6),
        )

        def check(properties: Properties, value):
            assert expectations.find_expectation(properties).result == value

        # xpu has no matches so should find default expectation
        check(Properties("xpu"), 1)
        check(Properties("cuda", 8), 2)
        check(Properties("cuda", 7), 3)
        check(Properties("rocm", 9), 4)
        check(Properties("rocm", None), 4)
        check(Properties("cuda", 2), 2)

        with self.assertRaises(ValueError):
            expectations = Expectations()

        expectations = Expectations(Expectation(Properties("cuda", 8), 2))
        with self.assertRaises(ValueError):
            expectations.find_expectation(Properties("xpu"))
