import unittest

from transformers.testing_utils import Expectations


class ExpectationsTest(unittest.TestCase):
    def test_expectations(self):
        # We use the expectations below to make sure the right expectations are found for the right devices.
        # Each value is just a unique ID.
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

        def check(expected_id, device_prop):
            found_id = expectations.find_expectation(device_prop)
            assert found_id == expected_id, f"Expected {expected_id} for {device_prop}, found {found_id}"

        # npu has no matches so should find default expectation
        check(1, ("npu", None, None))
        check(7, ("xpu", 3, None))
        check(2, ("cuda", 8, None))
        check(3, ("cuda", 7, None))
        check(4, ("rocm", 9, None))
        check(4, ("rocm", None, None))
        check(2, ("cuda", 2, None))

        # We also test that if there is no default excpectation and no match is found, a ValueError is raised.
        expectations = Expectations({("cuda", 8): 1})
        with self.assertRaises(ValueError):
            expectations.find_expectation(("xpu", None))
