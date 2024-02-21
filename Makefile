.PHONY: deps_table_update modified_only_fixup extra_style_checks quality style fixup fix-copies test test-examples

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := examples tests src utils

exclude_folders := examples/research_projects

modified_only_fixup:
	$(eval modified_py_files := $(shell python3 utils/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		ruff check $(modified_py_files) --fix --exclude $(exclude_folders); \
		ruff format $(modified_py_files) --exclude $(exclude_folders);\
	else \
		echo "No library .py files were modified"; \
	fi

# Update src/transformers/dependency_versions_table.py

deps_table_update:
	@python3 setup.py deps_table_update

deps_table_check_updated:
	@md5sum src/transformers/dependency_versions_table.py > md5sum.saved
	@python setup.py deps_table_update
	@md5sum -c --quiet md5sum.saved || (printf "\nError: the version dependency table is outdated.\nPlease run 'make fixup' or 'make style' and commit the changes.\n\n" && exit 1)
	@rm md5sum.saved

# autogenerating code

autogenerate_code: deps_table_update

# Check that the repo is in a good state

repo-consistency:
	python3 utils/check_copies.py
	python3 utils/check_table.py
	python3 utils/check_dummies.py
	python3 utils/check_repo.py
	python3 utils/check_inits.py
	python3 utils/check_config_docstrings.py
	python3 utils/check_config_attributes.py
	python3 utils/check_doctest_list.py
	python3 utils/update_metadata.py --check-only
	python3 utils/check_task_guides.py
	python3 utils/check_docstrings.py
	python3 utils/check_support_list.py

# this target runs checks on all files

quality:
	ruff check $(check_dirs) setup.py conftest.py
	ruff format --check $(check_dirs) setup.py conftest.py
	python3 utils/custom_init_isort.py --check_only
	python3 utils/sort_auto_mappings.py --check_only
	python3 utils/check_doc_toc.py

# Format source code automatically and check is there are any problems left that need manual fixing

extra_style_checks:
	python3 utils/custom_init_isort.py
	python3 utils/sort_auto_mappings.py
	python3 utils/check_doc_toc.py --fix_and_overwrite

# this target runs checks on all files and potentially modifies some of them

style:
	ruff check $(check_dirs) setup.py conftest.py --fix --exclude $(exclude_folders)
	ruff format $(check_dirs) setup.py conftest.py --exclude $(exclude_folders)
	${MAKE} autogenerate_code
	${MAKE} extra_style_checks

# Super fast fix and check target that only works on relevant modified files since the branch was made

fixup: modified_only_fixup extra_style_checks autogenerate_code repo-consistency

# Make marked copies of snippets of codes conform to the original

fix-copies:
	python3 utils/check_copies.py --fix_and_overwrite
	python3 utils/check_table.py --fix_and_overwrite
	python3 utils/check_dummies.py --fix_and_overwrite
	python3 utils/check_doctest_list.py --fix_and_overwrite
	python3 utils/check_task_guides.py --fix_and_overwrite
	python3 utils/check_docstrings.py --fix_and_overwrite

# Run tests for the library

test:
	python3 -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run tests for examples

test-examples:
	python3 -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/

# Run tests for SageMaker DLC release

test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]
	TEST_SAGEMAKER=True python3 -m pytest -n auto  -s -v ./tests/sagemaker


# Release stuff

pre-release:
	python3 utils/release.py

pre-patch:
	python3 utils/release.py --patch

post-release:
	python3 utils/release.py --post_release

post-patch:
	python3 utils/release.py --post_release --patch

build-release:
	rm -rf dist
	rm -rf build
	python3 setup.py bdist_wheel
	python3 setup.py sdist
	python3 utils/check_build.py
