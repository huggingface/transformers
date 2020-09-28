.PHONY: quality_checks quality style fixup test test-examples docs

# Check that source code meets quality standards

quality_checks:
	flake8 examples templates tests src utils
	python utils/check_copies.py
	python utils/check_repo.py

quality:
	black --check examples templates tests src utils
	isort --check-only examples templates tests src utils
	${MAKE} quality_checks

# Format source code automatically and check is there are any problems left that need manual fixing

style:
	black examples templates tests src utils
	isort examples templates tests src utils

fixup: style quality_checks

# Make marked copies of snippets of codes conform to the original

fix-copies:
	python utils/check_copies.py --fix_and_overwrite

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run tests for examples

test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/

# Check that docs can build

docs:
	cd docs && make html SPHINXOPTS="-W"
