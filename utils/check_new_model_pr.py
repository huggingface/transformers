#!/usr/bin/env python3
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Validates completeness of new model PRs by checking for required files and structure.

Usage:
    python utils/check_new_model_pr.py --base <base_sha> --head <head_sha> --models <model1> <model2> ...
"""

import argparse
import subprocess
import sys
from pathlib import Path


class ValidationError:
    """Represents a validation error with severity."""

    def __init__(self, model: str, message: str, severity: str = "error"):
        self.model = model
        self.message = message
        self.severity = severity  # "error" or "warning"

    def __str__(self):
        prefix = f"[{self.model}] " if self.model else ""
        return f"::{self.severity}::{prefix}{self.message}"


class NewModelValidator:
    """Validates that new model PRs include all required files and structure."""

    def __init__(self, base_sha: str, head_sha: str, models: list[str]):
        self.base_sha = base_sha
        self.head_sha = head_sha
        self.models = models
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationError] = []

    def get_changed_files(self) -> set[str]:
        """Get list of files changed between base and head."""
        result = subprocess.run(
            ["git", "diff", "--name-only", self.base_sha, self.head_sha],
            capture_output=True,
            text=True,
            check=True,
        )
        return set(result.stdout.strip().split("\n"))

    def add_error(self, model: str, message: str):
        """Add a validation error."""
        self.errors.append(ValidationError(model, message, "error"))

    def add_warning(self, model: str, message: str):
        """Add a validation warning."""
        self.warnings.append(ValidationError(model, message, "warning"))

    def check_toctree(self, changed_files: set[str]):
        """Check that docs toctree was updated."""
        if "docs/source/en/_toctree.yml" not in changed_files:
            self.add_error(
                "",
                "Missing docs/source/en/_toctree.yml update. "
                "New models must be added to the documentation table of contents.",
            )

    def check_auto_mappings(self, changed_files: set[str]):
        """Check that auto mappings were updated."""
        auto_files = [f for f in changed_files if f.startswith("src/transformers/models/auto/")]
        if not auto_files:
            self.add_warning(
                "",
                "No changes detected under src/transformers/models/auto/. "
                "Ensure Auto mappings are updated (configuration_auto.py, modeling_auto.py, "
                "and modality-specific autos like processing_auto.py for multimodal models).",
            )

    def check_model_directory(self, model: str) -> Path | None:
        """Check that model directory exists."""
        model_dir = Path("src/transformers/models") / model
        if not model_dir.exists():
            self.add_error(model, f"Missing directory: {model_dir}")
            return None
        return model_dir

    def check_configuration_file(self, model: str, model_dir: Path):
        """Check for configuration or modular file."""
        config_files = list(model_dir.glob("configuration_*.py"))
        modular_files = list(model_dir.glob("modular_*.py"))

        if not config_files and not modular_files:
            self.add_error(
                model,
                f"Missing configuration_*.py (or modular_*.py) in {model_dir}. "
                "Note: modular_*.py is optional and used in about 40% of models.",
            )

    def check_modeling_file(self, model: str, model_dir: Path):
        """Check for modeling or modular file."""
        modeling_files = list(model_dir.glob("modeling_*.py"))
        modular_files = list(model_dir.glob("modular_*.py"))

        if not modeling_files and not modular_files:
            self.add_error(
                model, f"Missing modeling_*.py (or modular_*.py) in {model_dir}. All models require a modeling file."
            )

    def check_multimodal_files(self, model: str, model_dir: Path, changed_files: set[str]):
        """Check for modality-specific processor files and suggest fast variants."""
        # Check for image processing (vision models)
        image_proc_files = list(model_dir.glob("image_processing_*.py"))
        if image_proc_files:
            # Check if fast variant exists
            fast_image_proc_files = list(model_dir.glob("image_processing_*_fast.py"))
            if not fast_image_proc_files:
                self.add_warning(
                    model,
                    "Found image_processing_*.py but no image_processing_*_fast.py. "
                    "Consider adding an optimized fast image processor for better performance.",
                )

            # Check for processing file (multimodal orchestration)
            processing_files = list(model_dir.glob("processing_*.py"))
            if processing_files:
                # Check if processing_auto.py was updated
                if not any("processing_auto.py" in f for f in changed_files):
                    self.add_warning(
                        model,
                        "Found processing_*.py but processing_auto.py might not be updated. "
                        "Multimodal models with processors must register in processing_auto.py.",
                    )

        # Check for feature extraction (audio models)
        feature_extract_files = list(model_dir.glob("feature_extraction_*.py"))
        if feature_extract_files:
            processing_files = list(model_dir.glob("processing_*.py"))
            if processing_files and not any("processing_auto.py" in f for f in changed_files):
                self.add_warning(
                    model,
                    "Found feature_extraction_*.py with processing_*.py but processing_auto.py might not be updated. "
                    "Audio models with processors must register in processing_auto.py.",
                )

    def check_generation_config(self, model: str, model_dir: Path):
        """Check for generation configuration (generative models)."""
        modeling_files = list(model_dir.glob("modeling_*.py"))
        if modeling_files:
            # Check if any modeling file contains "ForCausalLM" or "Generate"
            for modeling_file in modeling_files:
                content = modeling_file.read_text()
                if "ForCausalLM" in content or "GenerationMixin" in content:
                    gen_config_files = list(model_dir.glob("generation_configuration_*.py"))
                    if not gen_config_files:
                        self.add_warning(
                            model,
                            "This appears to be a generative model (found ForCausalLM or GenerationMixin) "
                            "but no generation_configuration_*.py file was found. "
                            "Consider adding one if the model has special generation defaults.",
                        )
                    break

    def check_tests_directory(self, model: str):
        """Check that tests directory exists with test files."""
        test_dir = Path("tests/models") / model
        if not test_dir.exists():
            self.add_error(model, f"Missing tests directory: {test_dir}")
            return

        test_files = list(test_dir.glob("*.py"))
        if not test_files:
            self.add_error(model, f"Tests directory exists but contains no .py test files: {test_dir}")
            return

        # Check for test_modeling_*.py (always required)
        modeling_test_files = list(test_dir.glob("test_modeling_*.py"))
        if not modeling_test_files:
            self.add_error(model, f"Missing test_modeling_*.py in {test_dir}. This test file is always required.")

    def check_documentation(self, model: str):
        """Check for documentation page."""
        doc_file = Path("docs/source/en/model_doc") / f"{model}.md"
        if not doc_file.exists():
            self.add_error(
                model,
                f"Missing documentation page: {doc_file}. "
                "All models must include a documentation page with architecture overview, "
                "usage examples, and task descriptions.",
            )

    def validate_model(self, model: str, changed_files: set[str]):
        """Run all validation checks for a single model."""
        # Check model directory
        model_dir = self.check_model_directory(model)
        if model_dir is None:
            return

        # Check required files
        self.check_configuration_file(model, model_dir)
        self.check_modeling_file(model, model_dir)

        # Check multimodal and processor files
        self.check_multimodal_files(model, model_dir, changed_files)

        # Check for generation config if applicable
        self.check_generation_config(model, model_dir)

        # Check tests
        self.check_tests_directory(model)

        # Check documentation
        self.check_documentation(model)

    def validate(self) -> bool:
        """Run all validation checks. Returns True if validation passes."""
        if not self.models:
            print("No new models detected in this PR.")
            return True

        print(f"Validating {len(self.models)} new model(s): {', '.join(self.models)}")

        # Get changed files
        changed_files = self.get_changed_files()

        # Global checks
        self.check_toctree(changed_files)
        self.check_auto_mappings(changed_files)

        # Per-model checks
        for model in self.models:
            self.validate_model(model, changed_files)

        # Print all warnings
        for warning in self.warnings:
            print(warning)

        # Print all errors
        for error in self.errors:
            print(error)

        # Summary
        if self.errors:
            print(
                f"\n❌ Validation failed with {len(self.errors)} error(s) and {len(self.warnings)} warning(s). "
                "See above for details."
            )
            return False

        if self.warnings:
            print(f"\n✅ Validation passed with {len(self.warnings)} warning(s).")
        else:
            print("\n✅ All validation checks passed.")

        return True


def main():
    parser = argparse.ArgumentParser(description="Validate new model PR completeness")
    parser.add_argument("--base", required=True, help="Base commit SHA")
    parser.add_argument("--head", required=True, help="Head commit SHA")
    parser.add_argument("--models", nargs="+", required=True, help="List of model names to validate")

    args = parser.parse_args()

    validator = NewModelValidator(args.base, args.head, args.models)
    success = validator.validate()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
