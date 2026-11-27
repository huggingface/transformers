# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

.PHONY: style typing check-code-quality check-repository-consistency check-repo fix-repo test run-mic record-mic


# Checker lists. The two CI jobs (CircleCI runs `make check-code-quality` and
# `make check-repository-consistency` in parallel) own the canonical sets below.
# Local convenience targets `check-repo` and `fix-repo` are *derived* from them,
# so they can never drift out of sync (e.g. silently dropping `auto_mappings`
# from CI, as happened in #45018 → fixed in #45774).

STYLE_CHECKERS := ruff_check, ruff_format, init_isort, sort_auto_mappings, noisy_comments
TYPING_CHECKERS := types, modeling_structure
CODE_QUALITY_CHECKERS := $(TYPING_CHECKERS), $(STYLE_CHECKERS)

REPO_CONSISTENCY_CHECKERS := \
	auto_mappings, \
	imports, \
	import_complexity, \
	copies, \
	modular_conversion, \
	inits, \
	doc_toc, \
	reviewers, \
	modeling_rules_doc, \
	docstrings, \
	dummies, \
	repo, \
	pipeline_typing, \
	config_docstrings, \
	config_attributes, \
	doctest_list, \
	update_metadata, \
	add_dates, \
	deps_table

ALL_CHECKERS := $(CODE_QUALITY_CHECKERS), $(REPO_CONSISTENCY_CHECKERS)


# Runs all linting/formatting scripts, most notably ruff
style:
	@python utils/checkers.py $(STYLE_CHECKERS) --fix

# Runs ty type checker and model structure rules
typing:
	@python utils/checkers.py $(TYPING_CHECKERS)

# Runs typing, ruff linting/formatting, import-order checks and auto-mappings
check-code-quality:
	@python utils/checkers.py $(CODE_QUALITY_CHECKERS)

# Runs a full repository consistency check.
check-repository-consistency:
	@python utils/checkers.py $(REPO_CONSISTENCY_CHECKERS)

# Runs typing and formatting checks + repository consistency check (ignores errors)
check-repo:
	@python utils/checkers.py $(ALL_CHECKERS) --keep-going

# Run all repo checks for which there is an automatic fix, most notably modular conversions
fix-repo:
	@python utils/checkers.py $(ALL_CHECKERS) --fix --keep-going

# Run tests for the library, requires pytest-random-order
test:
	python -m pytest -p random_order -n auto --dist=loadfile -s -v --random-order-bucket=module ./tests/

# Run pinned MIC fixtures against this checkout (MODEL=bert, ALL=1, BASE=origin/main, FLEET=prototypes|control, JOBS=4).
# Opt-in checker (network + model runtimes), deliberately not in any CI checker list above.
run-mic: export MODEL := $(MODEL)
run-mic: export ALL := $(ALL)
run-mic: export BASE := $(BASE)
run-mic: export FLEET := $(FLEET)
run-mic: export JOBS := $(JOBS)
run-mic:
	@python utils/checkers.py mic

# Publish the MIC expected values this PR's contracts job recorded for the cases its gate marks blocked,
# for this PR only (MODEL="gptj codegen", PR=<number>, default: the PR of the current branch). Needs
# write access; same as commenting `record-mic: <models>` on the PR. The values are posted on the PR
# and go live when it merges.
record-mic:
	@test -n "$(MODEL)" || { echo 'usage: make record-mic MODEL="<model> ..." [PR=<number>]'; exit 1; }
	@pr="$(PR)"; test -n "$$pr" || pr=$$(gh pr view --repo huggingface/transformers --json number --jq .number) || exit 1; \
	gh workflow run record-mic.yml --repo huggingface/transformers -f pr="$$pr" -f models="$(MODEL)" && \
	echo "record-mic requested for #$$pr ($(MODEL)); the result is posted on the PR"


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
