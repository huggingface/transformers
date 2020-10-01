.PHONY: modified_only_fixup extra_quality_checks quality style fixup fix-copies test test-examples docs


check_dirs := examples templates tests src utils

# get modified files since the branch was made
fork_point_sha := $(shell git merge-base --fork-point master)
joined_dirs    := $(shell echo $(check_dirs) | tr " " "|")
modified_files := $(shell git diff --name-only $(fork_point_sha) | egrep '^($(joined_dirs))')
#$(info modified files are: $(modified_files))

modified_only_fixup:
	@if [ -n "$(modified_files)" ]; then \
		echo "Checking/fixing $(modified_files)"; \
		black $(modified_files); \
		isort $(modified_files); \
		flake8 $(modified_files); \
	else \
		echo "No relevant files were modified"; \
	fi

# Check that source code meets quality standards

extra_quality_checks:
	python utils/check_copies.py
	python utils/check_repo.py

# this target runs checks on all files
quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	${MAKE} extra_quality_checks

# Format source code automatically and check is there are any problems left that need manual fixing

style:
	black $(check_dirs)
	isort $(check_dirs)

# Super fast fix and check target that only works on relevant modified files since the branch was made

fixup: modified_only_fixup extra_quality_checks

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
