<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Checks on a Pull Request

When you open a pull request on 🤗 Transformers, a fair number of checks will be run to make sure the patch you are adding is not breaking anything existing. Those checks are of four types:
- regular tests
- documentation build
- code and documentation style
- general repository consistency

In this document, we will take a stab at explaining what those various checks are and the reason behind them, as well as how to debug them locally if one of them fails on your PR.

Note that, ideally, they require you to have a dev install:

```bash
pip install transformers[dev]
```

or for an editable install:

```bash
pip install -e .[dev]
```

inside the Transformers repo. Since the number of optional dependencies of Transformers has grown a lot, it's possible you don't manage to get all of them. If the dev install fails, make sure to install the Deep Learning framework you are working with (PyTorch, TensorFlow and/or Flax) then do

```bash
pip install transformers[quality]
```

or for an editable install:

```bash
pip install -e .[quality]
```


## Tests

All the jobs that begin with `ci/circleci: run_tests_` run parts of the Transformers testing suite. Each of those jobs focuses on a part of the library in a certain environment: for instance `ci/circleci: run_tests_pipelines_tf` runs the pipelines test in an environment where TensorFlow only is installed.

Note that to avoid running tests when there is no real change in the modules they are testing, only part of the test suite is run each time: a utility is run to determine the differences in the library between before and after the PR (what GitHub shows you in the "Files changes" tab) and picks the tests impacted by that diff. That utility can be run locally with:

```bash
python utils/tests_fetcher.py
```

from the root of the Transformers repo. It will:

1. Check for each file in the diff if the changes are in the code or only in comments or docstrings. Only the files with real code changes are kept.
2. Build an internal map that gives for each file of the source code of the library all the files it recursively impacts. Module A is said to impact module B if module B imports module A. For the recursive impact, we need a chain of modules going from module A to module B in which each module imports the previous one.
3. Apply this map on the files gathered in step 1, which  gives us the list of model files impacted by the PR.
4. Map each of those files to their corresponding test file(s) and get the list of tests to run.

When executing the script locally, you should get the results of step 1, 3 and 4 printed and thus know which tests are run. The script will also create a file named `test_list.txt` which contains the list of tests to run, and you can run them locally with the following command:

```bash
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)
```

Just in case anything slipped through the cracks, the full test suite is also run daily.

## Documentation build

The `build_pr_documentation` job builds and generates a preview of the documentation to make sure everything looks okay once your PR is merged. A bot will add a link to preview the documentation in your PR. Any changes you make to the PR are automatically updated in the preview. If the documentation fails to build, click on **Details** next to the failed job to see where things went wrong. Often, the error is as simple as a missing file in the `toctree`.

If you're interested in building or previewing the documentation locally, take a look at the [`README.md`](https://github.com/huggingface/transformers/tree/main/docs) in the docs folder.

## Code and documentation style

Code formatting is applied to all the source files, the examples and the tests using `black` and `ruff`. We also have a custom tool taking care of the formatting of docstrings and `rst` files (`utils/style_doc.py`), as well as the order of the lazy imports performed in the Transformers `__init__.py` files (`utils/custom_init_isort.py`). All of this can be launched by executing

```bash
make style
```

The CI checks those have been applied inside the `ci/circleci: check_code_quality` check. It also runs `ruff`, that will have a basic look at your code and will complain if it finds an undefined variable, or one that is not used. To run that check locally, use

```bash
make quality
```

This can take a lot of time, so to run the same thing on only the files you modified in the current branch, run

```bash
make fixup
```

This last command will also run all the additional checks for the repository consistency. Let's have a look at them.

## Repository consistency

This regroups all the tests to make sure your PR leaves the repository in a good state, and is performed by the `ci/circleci: check_repository_consistency` check. You can locally run that check by executing the following:

```bash
make repo-consistency
```

This checks that:

- All objects added to the init are documented (performed by `utils/check_repo.py`)
- All `__init__.py` files have the same content in their two sections (performed by `utils/check_inits.py`)
- All code identified as a copy from another module is consistent with the original (performed by `utils/check_copies.py`)
- All configuration classes have at least one valid checkpoint mentioned in their docstrings (performed by `utils/check_config_docstrings.py`)
- All configuration classes only contain attributes that are used in corresponding modeling files (performed by `utils/check_config_attributes.py`)
- The translations of the READMEs and the index of the doc have the same model list as the main README (performed by `utils/check_copies.py`)
- The auto-generated tables in the documentation are up to date (performed by `utils/check_table.py`)
- The library has all objects available even if not all optional dependencies are installed (performed by `utils/check_dummies.py`)
- All docstrings properly document the arguments in the signature of the object (performed by `utils/check_docstrings.py`)

Should this check fail, the first two items require manual fixing, the last four can be fixed automatically for you by running the command

```bash
make fix-copies
```

Additional checks concern PRs that add new models, mainly that:

- All models added are in an Auto-mapping (performed by `utils/check_repo.py`)
<!-- TODO Sylvain, add a check that makes sure the common tests are implemented.-->
- All models are properly tested (performed by `utils/check_repo.py`)

<!-- TODO Sylvain, add the following
- All models are added to the main README, inside the main doc
- All checkpoints used actually exist on the Hub

-->

### Check copies

Since the Transformers library is very opinionated with respect to model code, and each model should fully be implemented in a single file without relying on other models, we have added a mechanism that checks whether a copy of the code of a layer of a given model stays consistent with the original. This way, when there is a bug fix, we can see all other impacted models and choose to trickle down the modification or break the copy.

<Tip>

If a file is a full copy of another file, you should register it in the constant `FULL_COPIES` of `utils/check_copies.py`.

</Tip>

This mechanism relies on comments of the form `# Copied from xxx`. The `xxx` should contain the whole path to the class of function which is being copied below. For instance, `RobertaSelfOutput` is a direct copy of the `BertSelfOutput` class, so you can see [here](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L289) it has a comment:

```py
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
```

Note that instead of applying this to a whole class, you can apply it to the relevant methods that are copied from. For instance [here](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L598) you can see how `RobertaPreTrainedModel._init_weights` is copied from the same method in `BertPreTrainedModel` with the comment:

```py
# Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
```

Sometimes the copy is exactly the same except for names: for instance in `RobertaAttention`, we use `RobertaSelfAttention` instead of `BertSelfAttention` but other than that, the code is exactly the same. This is why `# Copied from` supports simple string replacements with the following syntax: `Copied from xxx with foo->bar`. This means the code is copied with all instances of `foo` being replaced by `bar`. You can see how it used [here](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L304C1-L304C86) in `RobertaAttention` with the comment:

```py
# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
```

Note that there shouldn't be any spaces around the arrow (unless that space is part of the pattern to replace of course).

You can add several patterns separated by a comma. For instance here `CamemberForMaskedLM` is a direct copy of `RobertaForMaskedLM` with two replacements: `Roberta` to `Camembert` and `ROBERTA` to `CAMEMBERT`. You can see [here](https://github.com/huggingface/transformers/blob/15082a9dc6950ecae63a0d3e5060b2fc7f15050a/src/transformers/models/camembert/modeling_camembert.py#L929) this is done with the comment:

```py
# Copied from transformers.models.roberta.modeling_roberta.RobertaForMaskedLM with Roberta->Camembert, ROBERTA->CAMEMBERT
```

If the order matters (because one of the replacements might conflict with a previous one), the replacements are executed from left to right.

<Tip>

If the replacements change the formatting (if you replace a short name by a very long name for instance), the copy is checked after applying the auto-formatter.

</Tip>

Another way when the patterns are just different casings of the same replacement (with an uppercased and a lowercased variants) is just to add the option `all-casing`. [Here](https://github.com/huggingface/transformers/blob/15082a9dc6950ecae63a0d3e5060b2fc7f15050a/src/transformers/models/mobilebert/modeling_mobilebert.py#L1237) is an example in `MobileBertForSequenceClassification` with the comment:

```py
# Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification with Bert->MobileBert all-casing
```

In this case, the code is copied from `BertForSequenceClassification` by replacing:
- `Bert` by `MobileBert` (for instance when using `MobileBertModel` in the init)
- `bert` by `mobilebert` (for instance when defining `self.mobilebert`)
- `BERT` by `MOBILEBERT` (in the constant `MOBILEBERT_INPUTS_DOCSTRING`)
