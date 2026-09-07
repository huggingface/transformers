import ast
import os
import unittest
from pathlib import Path

import pytest

from transformers.utils.import_utils import (
    BASE_FILE_REQUIREMENTS,
    Backend,
    VersionComparison,
    define_import_structure,
    spread_import_structure,
)


import_structures = Path(__file__).parent / "import_structures"


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
    transformers_path = base_transformers_path / "src" / "transformers"
    models_path = transformers_path / "models"
    models_import_structure = spread_import_structure(define_import_structure(models_path))

    def test_definition(self):
        import_structure = define_import_structure(import_structures)
        valid_frozensets: dict[frozenset | frozenset[str], dict[str, set[str]]] = {
            frozenset(): {
                "import_structure_raw_register": {"A0", "A4", "a0"},
                "import_structure_register_with_comments": {"B0", "b0"},
            },
            frozenset({"random_item_that_should_not_exist"}): {"failing_export": {"A0"}},
            frozenset({"torch"}): {
                "import_structure_raw_register": {"A1", "A2", "A3", "a1", "a2", "a3"},
                "import_structure_register_with_duplicates": {"C0", "C1", "C2", "C3", "c0", "c1", "c2", "c3"},
                "import_structure_register_with_comments": {"B1", "B2", "B3", "b1", "b2", "b3"},
            },
            frozenset({"torch>=2.5"}): {"import_structure_raw_register_with_versions": {"D0", "d0"}},
            frozenset({"torch>2.5"}): {"import_structure_raw_register_with_versions": {"D1", "d1"}},
            frozenset({"torch<=2.5"}): {"import_structure_raw_register_with_versions": {"D2", "d2"}},
            frozenset({"torch<2.5"}): {"import_structure_raw_register_with_versions": {"D3", "d3"}},
            frozenset({"torch==2.5"}): {"import_structure_raw_register_with_versions": {"D4", "d4"}},
            frozenset({"torch!=2.5"}): {"import_structure_raw_register_with_versions": {"D5", "d5"}},
            frozenset({"torch>=2.5", "accelerate<0.20"}): {
                "import_structure_raw_register_with_versions": {"D6", "d6"}
            },
        }

        self.assertEqual(len(import_structure.keys()), len(valid_frozensets.keys()))
        for _frozenset in valid_frozensets:
            self.assertTrue(_frozenset in import_structure)
            self.assertListEqual(
                sorted(import_structure[_frozenset].keys()), sorted(valid_frozensets[_frozenset].keys())
            )
            for module, objects in valid_frozensets[_frozenset].items():
                self.assertTrue(module in import_structure[_frozenset])
                self.assertSetEqual(objects, import_structure[_frozenset][module])

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
                for module_object_mapping in import_structure.values():
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

    def test_auto_image_processor_does_not_require_torchvision(self):
        for backends, modules in self.models_import_structure.items():
            for module, objects in modules.items():
                if "AutoImageProcessor" in objects:
                    self.assertNotIn(
                        "torchvision",
                        backends,
                        f"`{module}` should not require torchvision, got {sorted(backends)}",
                    )

    def test_inferred_torchvision_requirement_matches_the_code(self):
        """
        The rule is textual, so hold it to what the file does. The glob covers the whole package
        because `utils/tests_fetcher.py` infers from `src/transformers/__init__.py`, not `models/`.
        """
        for path in sorted(self.transformers_path.rglob("image_processing_*.py")):
            with self.subTest(path.relative_to(self.transformers_path)):
                content = path.read_text(encoding="utf-8")

                inferred = ()
                for check, requirements in BASE_FILE_REQUIREMENTS.items():
                    if check(path.stem, content):
                        inferred = requirements
                        break

                inherits = any(
                    isinstance(node, ast.ClassDef)
                    and any(
                        getattr(base, "id", getattr(base, "attr", None)) == "TorchvisionBackend" for base in node.bases
                    )
                    for node in ast.parse(content).body
                )

                error_message = f"inherits `TorchvisionBackend`: {inherits}, inferred {sorted(inferred)}"
                self.assertEqual(inherits, "torchvision" in inferred, msg=error_message)

    def test_import_spread(self):
        """
        This test is specifically designed to test that varying levels of depth across import structures are
        respected.

        In this instance, frozensets are at respective depths of 1, 2 and 3, for example:
        - models.{frozensets}
        - models.albert.{frozensets}
        - models.deprecated.transfo_xl.{frozensets}
        """
        initial_import_structure = {
            frozenset(): {"dummy_non_model": {"DummyObject"}},
            "models": {
                frozenset(): {"dummy_config": {"DummyConfig"}},
                "albert": {
                    frozenset(): {"configuration_albert": {"AlbertConfig"}},
                    frozenset({"torch"}): {
                        "modeling_albert": {
                            "AlbertForMaskedLM",
                        }
                    },
                },
                "llama": {
                    frozenset(): {"configuration_llama": {"LlamaConfig"}},
                    frozenset({"torch"}): {
                        "modeling_llama": {
                            "LlamaForCausalLM",
                        }
                    },
                },
                "deprecated": {
                    "transfo_xl": {
                        frozenset({"torch"}): {
                            "modeling_transfo_xl": {
                                "TransfoXLModel",
                            }
                        },
                        frozenset(): {
                            "configuration_transfo_xl": {"TransfoXLConfig"},
                            "tokenization_transfo_xl": {"TransfoXLCorpus", "TransfoXLTokenizer"},
                        },
                    },
                    "deta": {
                        frozenset({"torch"}): {
                            "modeling_deta": {"DetaForObjectDetection", "DetaModel", "DetaPreTrainedModel"}
                        },
                        frozenset(): {"configuration_deta": {"DetaConfig"}},
                        frozenset({"vision"}): {"image_processing_deta": {"DetaImageProcessor"}},
                    },
                },
            },
        }

        ground_truth_spread_import_structure = {
            frozenset(): {
                "dummy_non_model": {"DummyObject"},
                "models.dummy_config": {"DummyConfig"},
                "models.albert.configuration_albert": {"AlbertConfig"},
                "models.llama.configuration_llama": {"LlamaConfig"},
                "models.deprecated.transfo_xl.configuration_transfo_xl": {"TransfoXLConfig"},
                "models.deprecated.transfo_xl.tokenization_transfo_xl": {"TransfoXLCorpus", "TransfoXLTokenizer"},
                "models.deprecated.deta.configuration_deta": {"DetaConfig"},
            },
            frozenset({"torch"}): {
                "models.albert.modeling_albert": {"AlbertForMaskedLM"},
                "models.llama.modeling_llama": {"LlamaForCausalLM"},
                "models.deprecated.transfo_xl.modeling_transfo_xl": {"TransfoXLModel"},
                "models.deprecated.deta.modeling_deta": {"DetaForObjectDetection", "DetaModel", "DetaPreTrainedModel"},
            },
            frozenset({"vision"}): {"models.deprecated.deta.image_processing_deta": {"DetaImageProcessor"}},
        }

        newly_spread_import_structure = spread_import_structure(initial_import_structure)

        self.assertEqual(ground_truth_spread_import_structure, newly_spread_import_structure)


@pytest.mark.parametrize(
    "backend,package_name,version_comparison,version",
    [
        pytest.param(Backend("torch>=2.5 "), "torch", VersionComparison.GREATER_THAN_OR_EQUAL, "2.5"),
        pytest.param(Backend("torchvision==0.19.1"), "torchvision", VersionComparison.EQUAL, "0.19.1"),
    ],
)
def test_backend_specification(
    backend: Backend, package_name: str, version_comparison: VersionComparison, version: str
):
    assert backend.package_name == package_name
    assert VersionComparison.from_string(backend.version_comparison) == version_comparison
    assert backend.version == version
