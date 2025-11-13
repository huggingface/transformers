.PHONY: deps_table_update modified_only_fixup extra_style_checks quality style fixup fix-copies test test-examples benchmark

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := examples tests src utils scripts benchmark benchmark_v2

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
	python utils/check_init_weights_data.py
	python utils/check_inits.py
	python utils/check_pipeline_typing.py
	python utils/check_config_docstrings.py
	python utils/check_config_attributes.py
	python utils/check_doctest_list.py
	python utils/update_metadata.py --check-only
	python utils/check_docstrings.py
	python utils/add_dates.py

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

# Load checkpoints and verify they are accessible
export XDG_CACHE_HOME=/tmp/cache/
export HF_HOME=/tmp/cache/hf

/tmp/backbone.pth:
	wget -O $@ "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiajEybnhqNWl2cGVtZDhvMDc0ZzF0bTFtIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjMyMjM1NDd9fX1dfQ__&Signature=FkDkgcYMlUa1Oq%7E-emh1yXIIuj89OBh9O8PHcPhTEzw5MHLwggBnLQYGwbPGyLySCz8sfLN2YyXnk47gHHZLITGv%7EkFCGG4cFHN0inQIUqZIwRkghz9QMRqAAIxL1VnajHpGlfmewPhpS8Dawi8V99LOwZ3YQ9GKq3Uif5Re98VXgkL3Qj0KvMXvA%7Ez7w5zh8ZPCW3ggVDsKAf1P-Y66sohOQEwuCQbkycqwwXsXPkw%7EPSw68Ct9dpASlIpXZp-4SFobbTtvbpQ2C6R0E8M7OOkFVH4%7E%7E0W-n-xJpkSDdfrm7B2BWjZ2eDBMI3w4kcexVICP6smbFUZUbIY3uXIV8Q__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1511531990180009"

/tmp/lc.pth:
	wget -O $@ "https://dinov3.llamameta.net/dinov3_vit7b16/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiajEybnhqNWl2cGVtZDhvMDc0ZzF0bTFtIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjMyMjM1NDd9fX1dfQ__&Signature=FkDkgcYMlUa1Oq%7E-emh1yXIIuj89OBh9O8PHcPhTEzw5MHLwggBnLQYGwbPGyLySCz8sfLN2YyXnk47gHHZLITGv%7EkFCGG4cFHN0inQIUqZIwRkghz9QMRqAAIxL1VnajHpGlfmewPhpS8Dawi8V99LOwZ3YQ9GKq3Uif5Re98VXgkL3Qj0KvMXvA%7Ez7w5zh8ZPCW3ggVDsKAf1P-Y66sohOQEwuCQbkycqwwXsXPkw%7EPSw68Ct9dpASlIpXZp-4SFobbTtvbpQ2C6R0E8M7OOkFVH4%7E%7E0W-n-xJpkSDdfrm7B2BWjZ2eDBMI3w4kcexVICP6smbFUZUbIY3uXIV8Q__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1511531990180009"

dinov3/requirements.txt:
	git clone https://github.com/facebookresearch/dinov3.git

get_checkpoints: /tmp/backbone.pth /tmp/lc.pth dinov3/requirements.txt

.venv:
	pip install uv && uv venv && uv pip install -r  dinov3/requirements.txt -e ".[torch]"
	
load_checkpoints: get_checkpoints dinov3/requirements.txt
	uv venv --clear
	uv pip install -r dinov3/requirements.txt
	uv run python test_torchhubload.py

load_hf: .venv get_checkpoints
	uv run python test_load_hf.py