import os
import unittest
from pathlib import Path

from transformers.utils.import_utils import define_import_structure, spread_import_structure


import_structures = Path("import_structures")


def fetch__all__(file_content):
    """
    Returns the content of the __all__ variable in the file content.
    Returns None if not defined, otherwise returns a list of strings.
    """
    lines = file_content.split("\n")
    for line_index in range(len(lines)):
        line = lines[line_index]
        if line.startswith("__all__ = "):
            # __all__ is defined on a single line
            if line.endswith("]"):
                return [obj.strip("\"' ") for obj in line.split("=")[1].strip(" []").split(",")]

            # __all__ is defined on multiple lines
            else:
                _all = []
                for __all__line_index in range(line_index + 1, len(lines)):
                    if lines[__all__line_index].strip() == "]":
                        return _all
                    else:
                        _all.append(lines[__all__line_index].strip("\"', "))


class TestImportStructures(unittest.TestCase):
    base_transformers_path = Path(__file__).parent.parent.parent
    models_path = base_transformers_path / "src" / "transformers" / "models"
    models_import_structure = spread_import_structure(define_import_structure(models_path))

    # TODO: Lysandre
    # See https://app.circleci.com/pipelines/github/huggingface/transformers/104762/workflows/7ba9c6f7-a3b2-44e6-8eaf-749c7b7261f7/jobs/1393260/tests
    @unittest.skip(reason="failing")
    def test_definition(self):
        import_structure = define_import_structure(import_structures)
        import_structure_definition = {
            frozenset(()): {
                "import_structure_raw_register": {"A0", "a0", "A4"},
                "import_structure_register_with_comments": {"B0", "b0"},
            },
            frozenset(("tf", "torch")): {
                "import_structure_raw_register": {"A1", "a1", "A2", "a2", "A3", "a3"},
                "import_structure_register_with_comments": {"B1", "b1", "B2", "b2", "B3", "b3"},
            },
            frozenset(("torch",)): {
                "import_structure_register_with_duplicates": {"C0", "c0", "C1", "c1", "C2", "c2", "C3", "c3"},
            },
        }

        self.assertDictEqual(import_structure, import_structure_definition)

    def test_transformers_specific_model_import(self):
        """
        This test ensures that there is equivalence between what is written down in __all__ and what is
        written down with register().

        It doesn't test the backends attributed to register().
        """
        for architecture in os.listdir(self.models_path):
            if (
                os.path.isfile(self.models_path / architecture)
                or architecture.startswith("_")
                or architecture == "deprecated"
            ):
                continue

            with self.subTest(f"Testing arch {architecture}"):
                import_structure = define_import_structure(self.models_path / architecture)
                backend_agnostic_import_structure = {}
                for requirement, module_object_mapping in import_structure.items():
                    for module, objects in module_object_mapping.items():
                        if module not in backend_agnostic_import_structure:
                            backend_agnostic_import_structure[module] = []

                        backend_agnostic_import_structure[module].extend(objects)

                for module, objects in backend_agnostic_import_structure.items():
                    with open(self.models_path / architecture / f"{module}.py") as f:
                        content = f.read()
                        _all = fetch__all__(content)

                        if _all is None:
                            raise ValueError(f"{module} doesn't have __all__ defined.")

                        error_message = (
                            f"self.models_path / architecture / f'{module}.py doesn't seem to be defined correctly:\n"
                            f"Defined in __all__: {sorted(_all)}\nDefined with register: {sorted(objects)}"
                        )
                        self.assertListEqual(sorted(objects), sorted(_all), msg=error_message)

    # TODO: Lysandre
    # See https://app.circleci.com/pipelines/github/huggingface/transformers/104762/workflows/7ba9c6f7-a3b2-44e6-8eaf-749c7b7261f7/jobs/1393260/tests
    @unittest.skip(reason="failing")
    def test_export_backend_should_be_defined(self):
        with self.assertRaisesRegex(ValueError, "Backend should be defined in the BACKENDS_MAPPING"):
            pass
