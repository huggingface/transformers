.PHONY: quality style test test-examples

# Check that source code meets quality standards

quality:
	black --check --line-length 119 --target-version py35 examples templates tests src utils
	isort --check-only --recursive examples templates tests src utils
	flake8 examples templates tests src utils

# Format source code automatically

style:
	black --line-length 119 --target-version py35 examples templates tests src utils
	isort --recursive examples templates tests src utils

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run tests for examples

test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/
