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
            }
        )

        def check(value, key):
            assert expectations.find_expectation(key) == value

        # xpu has no matches so should find default expectation
        check(1, ("xpu", None))
        check(2, ("cuda", 8))
        check(3, ("cuda", 7))
        check(4, ("rocm", 9))
        check(4, ("rocm", None))
        check(2, ("cuda", 2))

        expectations = Expectations({("cuda", 8): 1})
        with self.assertRaises(ValueError):
            expectations.find_expectation(("xpu", None))
