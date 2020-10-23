import unittest


def fatal_error2(**kwargs):
    assert False, "this is fatal error"


class FailMeAgain(unittest.TestCase):
    def test_error_again(self):
        print("testing again")
        fatal_error2(a=444, b=000)
