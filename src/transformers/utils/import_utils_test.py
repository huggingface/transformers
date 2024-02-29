import unittest
from transformers.src.transformers.utils.import_utils import requires_backends


class TestRequiresBackends(unittest.TestCase):
    def setUp(self):
        self.obj = object()

    def test_requires_backends_with_available_backend(self):
        try:
            requires_backends(self.obj, 'json')
            result = "Success"
        except ImportError as e:
            result = str(e)
        self.assertEqual(result, "Success", "The backend should be available and no ImportError should be raised.")

    def test_requires_backends_with_unavailable_backend(self):
        unavailable_backend = "unavailable_package_1234"
        with self.assertRaises(ImportError) as context:
            requires_backends(self.obj, unavailable_backend)
        self.assertTrue(f"module {self.obj.__class__.__name__} has no attribute {unavailable_backend}" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
