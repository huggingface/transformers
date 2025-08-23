.PHONY: deps_table_update modified_only_fixup extra_style_checks quality style fixup fix-copies test test-examples benchmark

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := examples tests src utils

exclude_folders :=  ""

modified_only_fixup:
	@current_branch=$$(git branch --show-current); \
	if [ "$$current_branch" = "main" ]; then \
		echo "On main branch, running 'style' target instead..."; \
		$(MAKE) style; \
	else \
		modified_py_files=$$(python utils/get_modified_files.py $(check_dirs)); \
		if [ -n "$$modified_py_files" ]; then \
			echo "Checking/fixing files: $${modified_py_files}"; \
			ruff check $${modified_py_files} --fix --exclude $(exclude_folders); \
			ruff format $${modified_py_files} --exclude $(exclude_folders); \
		else \
			echo "No library .py files were modified"; \
		fi; \
	fi

# Update src/transformers/dependency_versions_table.py

deps_table_update:
	@python setup.py deps_table_update

deps_table_check_updated:
	@md5sum src/transformers/dependency_versions_table.py > md5sum.saved
	@python setup.py deps_table_update
	@md5sum -c --quiet md5sum.saved || (printf "\nError: the version dependency table is outdated.\nPlease run 'make fixup' or 'make style' and commit the changes.\n\n" && exit 1)
	@rm md5sum.saved

# autogenerating code

autogenerate_code: deps_table_update

# Check that the repo is in a good state

repo-consistency:
	python utils/check_copies.py
	python utils/check_modular_conversion.py
	python utils/check_dummies.py
	python utils/check_repo.py
	python utils/check_inits.py
	python utils/check_pipeline_typing.py
	python utils/check_config_docstrings.py
	python utils/check_config_attributes.py
	python utils/check_doctest_list.py
	python utils/update_metadata.py --check-only
	python utils/check_docstrings.py

# this target runs checks on all files

quality:
	@python -c "from transformers import *" || (echo '🚨 import failed, this means you introduced unprotected imports! 🚨'; exit 1)
	ruff check $(check_dirs) setup.py conftest.py
	ruff format --check $(check_dirs) setup.py conftest.py
	python utils/sort_auto_mappings.py --check_only
	python utils/check_doc_toc.py
	python utils/check_docstrings.py --check_all


# Format source code automatically and check is there are any problems left that need manual fixing

extra_style_checks:
	python utils/sort_auto_mappings.py
	python utils/check_doc_toc.py --fix_and_overwrite

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
	python utils/check_copies.py --fix_and_overwrite
	python utils/check_modular_conversion.py --fix_and_overwrite
	python utils/check_dummies.py --fix_and_overwrite
	python utils/check_pipeline_typing.py --fix_and_overwrite
	python utils/check_doctest_list.py --fix_and_overwrite
	python utils/check_docstrings.py --fix_and_overwrite

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run tests for examples

test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/

# Run benchmark

benchmark:
	python3 benchmark/benchmark.py --config-dir benchmark/config --config-name generation --commit=diff backend.model=google/gemma-2b backend.cache_implementation=null,static backend.torch_compile=false,true --multirun

# Run tests for SageMaker DLC release

test-sagemaker: # install sagemaker dependencies in advance with pip install .[sagemaker]
	TEST_SAGEMAKER=True python -m pytest -n auto  -s -v ./tests/sagemaker


# Release stuff

pre-release:
	python utils/release.py

pre-patch:
	python utils/release.py --patch

post-release:
	python utils/release.py --post_release

post-patch:
	python utils/release.py --post_release --patch

build-release:
	rm -rf dist
	rm -rf build
	python setup.py bdist_wheel
	python setup.py sdist
	python utils/check_build.py
