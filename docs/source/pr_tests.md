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
-->

# Checks on a Pull Request

When you open a pull request on 🤗 Transformers, a fair number of checks will be run to make sure the patch you are adding is not breaking anything existing. Those checks are of four types:
- regular tests
- documentation build
- code and documentation style
- general repository consistency

In this document, we will take a stab at explaining what those various checks are and the reason behind them, as well as how to debug them locally if one of them fails on your PR.

Note that they all require you to have a dev install:

```bash
pip install transformer[dev]
```

or for an editable install:

```bash
pip install -e .[dev]
```

inside the Transformers repo.

## Tests

All the jobs that begin with `ci/circleci: run_tests_` run part of the Transformers testing suite. Each of those job focuses on a part of the library in a certain environment: for instance `ci/circleci: run_tests_pipelines_tf` runs the pipelines test in an environment where TensorFlow only is installed.

Note that to avoid running tests when there is no real change in the modules they are testing, only part of the test suite is run each time: an util is run to determine the differences in the library between before and after the PR (what GitHub shows you in the "Files changes" tab) and picks the tests impacted by that diff. That util can be run locally with:

```bash
python utils/test_fetcher.py
```

from the root of the Transformers repo. It will:

1. Check for each file in the diff if the changes are in the code or only in comments or docstrings. Only the files with real code changes are kept.
2. Build an internal map that gives for each file of the source code of the library all the files it recursively impacts. Module A is said to impact module B if module B imports module A. For the recursive impact, we need a chain of modules going from module A to module B in which each module imports the previous one.
3. Apply this map on the files gathered in step 1 gives us the list of model files impacted by the PR.
4. Map each of those files to their corresponding test file(s) and therefore get the list of tests to run.

When executing the script locally, you should get the results of step 1, 3 and 4 printed and thus know which tests are run. The script will also create a file named `test_list.txt` which contains the list of tests to run, and you can run them locally with the following command:

```bash
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)
```

Just in case anything slipped through the crack, the full test suites are also run daily.

## Documentation build

The job `ci/circleci: build_doc` runs a build of the documentation just to make sure everything will be okay once your PR is merged. If that steps fails, you can inspect it locally by going in the `docs` folder of of the Transformers repo and then typing

```bash
make html
```

Sphinx is not known for its helpful error messages, so you might have to try a few things to really find the source of the error.

## Code and documentation style

Code formatting is applied to all the source files, the examples and the tests using black and isort. We also have a custom tool taking care of the formatting of docstrings and rst files. Both can be launched by executing

```bash
make style
```

The CI checks those have been applied inside the `ci/circleci: check_code_quality` check. It also runs flake8, that will have a basic look at your code and will complain if it finds an undefined variable, or one that is not used. To run that check locally, use

```bash
make quality
```

This can take a lot of time, so to run the same thing on only the scripts you modified, run

```bash
make fixup
```

Those two commands will also run all the additional checks for the repository consistency. Let's have a look at them.

## Repository consistency

