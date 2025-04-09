import unittest

from transformers.testing_utils import Expectations


class ExpectationsTest(unittest.TestCase):
    def test_expectations(self):
        expectations = Expectations(
            {
                (None, None): 1,
                ("cuda", 8): 2,
                ("cuda", 7): 3,
                ("rocm", 8): 4,
                ("rocm", None): 5,
                ("cpu", None): 6,
                ("xpu", 3): 7,
            }
        )

        def check(value, key):
            assert expectations.find_expectation(key) == value

        # npu has no matches so should find default expectation
        check(1, ("npu", None))
        check(7, ("xpu", 3))
        check(2, ("cuda", 8))
        check(3, ("cuda", 7))
        check(4, ("rocm", 9))
        check(4, ("rocm", None))
        check(2, ("cuda", 2))

        expectations = Expectations({("cuda", 8): 1})
        with self.assertRaises(ValueError):
            expectations.find_expectation(("xpu", None))
