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

# Contribute to 🤗 Transformers

Everyone is welcome to contribute, and we value everybody's contribution. Code
contributions are not the only way to help the community. Answering questions, helping
others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts
about the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply ⭐️ the repository to say thank you.

However you choose to contribute, please be mindful and respect our
[code of conduct](https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md).

**This guide was heavily inspired by the awesome [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md).**

## Ways to contribute

There are several ways you can contribute to 🤗 Transformers:

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Implement new models.
* Contribute to the examples or to the documentation.

If you don't know where to start, there is a special [Good First
Issue](https://github.com/huggingface/transformers/contribute) listing. It will give you a list of
open issues that are beginner-friendly and help you start contributing to open-source. Just comment in the issue that you'd like to work
on it. 

For something slightly more challenging, you can also take a look at the [Good Second Issue](https://github.com/huggingface/transformers/labels/Good%20Second%20Issue) list. In general though, if you feel like you know what you're doing, go for it and we'll help you get there! 🚀

> All contributions are equally valuable to the community. 🥰

## Fixing outstanding issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md/#create-a-pull-request) and open a Pull Request!

## Submitting a bug-related issue or feature request

Do your best to follow these guidelines when submitting a bug-related issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The 🤗 Transformers library is robust and reliable thanks to users who report the problems they encounter.

Before you report an issue, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on GitHub under Issues). Your issue should also be related to bugs in the library itself, and not your code. If you're unsure whether the bug is in your code or the library, please ask on the [forum](https://discuss.huggingface.co/) first. This helps us respond quicker to fixing issues related to the library versus general questions.

Once you've confirmed the bug hasn't already been reported, please include the following information in your issue so we can quickly resolve it:

* Your **OS type and version** and **Python**, **PyTorch** and
  **TensorFlow** versions when applicable.
* A short, self-contained, code snippet that allows us to reproduce the bug in
  less than 30s.
* The *full* traceback if an exception is raised.
* Attach any other additional information, like screenshots, you think may help.

To get the OS and software versions automatically, run the following command:

```bash
transformers-cli env
```

You can also run the same command from the root of the repository:

```bash
python src/transformers/commands/transformers_cli.py env
```

### Do you want a new feature?

If there is a new feature you'd like to see in 🤗 Transformers, please open an issue and describe:

1. What is the *motivation* behind this feature? Is it related to a problem or frustration with the library? Is it a feature related to something you need for a project? Is it something you worked on and think it could benefit the community?

   Whatever it is, we'd love to hear about it!

2. Describe your requested feature in as much detail as possible. The more you can tell us about it, the better we'll be able to help you.
3. Provide a *code snippet* that demonstrates the features usage.
4. If the feature is related to a paper, please include a link.

If your issue is well written we're already 80% of the way there by the time you create it.

We have added [templates](https://github.com/huggingface/transformers/tree/main/templates) to help you get started with your issue.

## Do you want to implement a new model?

New models are constantly released and if you want to implement a new model, please provide the following information

* A short description of the model and link to the paper.
* Link to the implementation if it is open-sourced.
* Link to the model weights if they are available.

If you are willing to contribute the model yourself, let us know so we can help you add it to 🤗 Transformers!

We have added a [detailed guide and templates](https://github.com/huggingface/transformers/tree/main/templates) to help you get started with adding a new model, and we also have a more technical guide for [how to add a model to 🤗 Transformers](https://huggingface.co/docs/transformers/add_new_model).

## Do you want to add documentation?

We're always looking for improvements to the documentation that make it more clear and accurate. Please let us know how the documentation can be improved such as typos and any content that is missing, unclear or inaccurate. We'll be happy to make the changes or help you make a contribution if you're interested!

For more details about how to generate, build, and write the documentation, take a look at the documentation [README](https://github.com/huggingface/transformers/tree/main/docs).

## Create a Pull Request

Before writing any code, we strongly advise you to search through the existing PRs or
issues to make sure nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to contribute to
🤗 Transformers. While `git` is not the easiest tool to use, it has the greatest
manual. Type `git --help` in a shell and enjoy! If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

You'll need **[Python 3.8]((https://github.com/huggingface/transformers/blob/main/setup.py#L426))** or above to contribute to 🤗 Transformers. Follow the steps below to start contributing:

1. Fork the [repository](https://github.com/huggingface/transformers) by
   clicking on the **[Fork](https://github.com/huggingface/transformers/fork)** button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your Github handle>/transformers.git
   cd transformers
   git remote add upstream https://github.com/huggingface/transformers.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 **Do not** work on the `main` branch!

4. Set up a development environment by running the following command in a virtual environment:

   ```bash
   pip install -e ".[dev]"
   ```

   If 🤗 Transformers was already installed in the virtual environment, remove
   it with `pip uninstall transformers` before reinstalling it in editable
   mode with the `-e` flag.
   
   Depending on your OS, and since the number of optional dependencies of Transformers is growing, you might get a
   failure with this command. If that's the case make sure to install the Deep Learning framework you are working with
   (PyTorch, TensorFlow and/or Flax) then do:

   ```bash
   pip install -e ".[quality]"
   ```

   which should be enough for most use cases.

5. Develop the features on your branch.

   As you work on your code, you should make sure the test suite
   passes. Run the tests impacted by your changes like this:

   ```bash
   pytest tests/<TEST_TO_RUN>.py
   ```

   For more information about tests, check out the
   [Testing](https://huggingface.co/docs/transformers/testing) guide.

   🤗 Transformers relies on `black` and `ruff` to format its source code
   consistently. After you make changes, apply automatic style corrections and code verifications
   that can't be automated in one go with:

   ```bash
   make fixup
   ```

   This target is also optimized to only work with files modified by the PR you're working on.

   If you prefer to run the checks one after the other, the following command applies the
   style corrections:

   ```bash
   make style
   ```

   🤗 Transformers also uses `ruff` and a few custom scripts to check for coding mistakes. Quality
   controls are run by the CI, but you can run the same checks with:

   ```bash
   make quality
   ```

   Finally, we have a lot of scripts to make sure we didn't forget to update
   some files when adding a new model. You can run these scripts with:

   ```bash
   make repo-consistency
   ```

   To learn more about those checks and how to fix any issues with them, check out the
   [Checks on a Pull Request](https://huggingface.co/docs/transformers/pr_checks) guide.

   If you're modifying documents under `docs/source` directory, make sure the documentation can still be built. This check will also run in the CI when you open a pull request. To run a local check
   make sure you install the documentation builder:
   
   ```bash
   pip install ".[docs]"
   ```

   Run the following command from the root of the repository:

   ```bash
   doc-builder build transformers docs/source/en --build_dir ~/tmp/test-build
   ```

   This will build the documentation in the `~/tmp/test-build` folder where you can inspect the generated
   Markdown files with your favorite editor. You can also preview the docs on GitHub when you open a pull request.

   Once you're happy with your changes, add changed files with `git add` and
   record your changes locally with `git commit`:

   ```bash
   git add modified_file.py
   git commit
   ```

   Please remember to write [good commit
   messages](https://chris.beams.io/posts/git-commit/) to clearly communicate the changes you made!

   To keep your copy of the code up to date with the original
   repository, rebase your branch on `upstream/branch` *before* you open a pull request or if requested by a maintainer:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Push your changes to your branch:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   If you've already opened a pull request, you'll need to force push with the `--force` flag. Otherwise, if the pull request hasn't been opened yet, you can just push your changes normally.

6. Now you can go to your fork of the repository on GitHub and click on **Pull request** to open a pull request. Make sure you tick off all the boxes in our [checklist](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md/#pull-request-checklist) below. When you're ready, you can send your changes to the project maintainers for review.

7. It's ok if maintainers request changes, it happens to our core contributors
   too! So everyone can see the changes in the pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

### Pull request checklist

☐ The pull request title should summarize your contribution.<br>
☐ If your pull request addresses an issue, please mention the issue number in the pull
request description to make sure they are linked (and people viewing the issue know you
are working on it).<br>
☐ To indicate a work in progress please prefix the title with `[WIP]`. These are
useful to avoid duplicated work, and to differentiate it from PRs ready to be merged.<br>
☐ Make sure existing tests pass.<br>
☐ If adding a new feature, also add tests for it.<br>
   - If you are adding a new model, make sure you use
     `ModelTester.all_model_classes = (MyModel, MyModelWithLMHead,...)` to trigger the common tests.
   - If you are adding new `@slow` tests, make sure they pass using
     `RUN_SLOW=1 python -m pytest tests/models/my_new_model/test_my_new_model.py`.
   - If you are adding a new tokenizer, write tests and make sure
     `RUN_SLOW=1 python -m pytest tests/models/{your_model_name}/test_tokenization_{your_model_name}.py` passes.
   - CircleCI does not run the slow tests, but GitHub Actions does every night!<br>

☐ All public methods must have informative docstrings (see
[`modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
for an example).<br>
☐ Due to the rapidly growing repository, don't add any images, videos and other
non-text files that'll significantly weigh down the repository. Instead, use a Hub
repository such as [`hf-internal-testing`](https://huggingface.co/hf-internal-testing)
to host these files and reference them by URL. We recommend placing documentation
related images in the following repository:
[huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images).
You can open a PR on this dataset repostitory and ask a Hugging Face member to merge it.

For more information about the checks run on a pull request, take a look at our [Checks on a Pull Request](https://huggingface.co/docs/transformers/pr_checks) guide.

### Tests

An extensive test suite is included to test the library behavior and several examples. Library tests can be found in
the [tests](https://github.com/huggingface/transformers/tree/main/tests) folder and examples tests in the
[examples](https://github.com/huggingface/transformers/tree/main/examples) folder.

We like `pytest` and `pytest-xdist` because it's faster. From the root of the
repository, specify a *path to a subfolder or a test file* to run the test.

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
```

Similarly, for the `examples` directory, specify a *path to a subfolder or test file* to run the test. For example, the following command tests the text classification subfolder in the PyTorch `examples` directory:

```bash
pip install -r examples/xxx/requirements.txt  # only needed the first time
python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/text-classification
```

In fact, this is actually how our `make test` and `make test-examples` commands are implemented (not including the `pip install`)!

You can also specify a smaller set of tests in order to test only the feature
you're working on.

By default, slow tests are skipped but you can set the `RUN_SLOW` environment variable to
`yes` to run them. This will download many gigabytes of models so make sure you
have enough disk space, a good internet connection or a lot of patience!

<Tip warning={true}>

Remember to specify a *path to a subfolder or a test file* to run the test. Otherwise, you'll run all the tests in the `tests` or `examples` folder, which will take a very long time!

</Tip>

```bash
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/text-classification
```

Like the slow tests, there are other environment variables available which not enabled by default during testing:
- `RUN_CUSTOM_TOKENIZERS`: Enables tests for custom tokenizers.
- `RUN_PT_FLAX_CROSS_TESTS`: Enables tests for PyTorch + Flax integration.
- `RUN_PT_TF_CROSS_TESTS`: Enables tests for TensorFlow + PyTorch integration.

More environment variables and additional information can be found in the [testing_utils.py](src/transformers/testing_utils.py).

🤗 Transformers uses `pytest` as a test runner only. It doesn't use any
`pytest`-specific features in the test suite itself.

This means `unittest` is fully supported. Here's how to run tests with
`unittest`:

```bash
python -m unittest discover -s tests -t . -v
python -m unittest discover -s examples -t examples -v
```

### Style guide

For documentation strings, 🤗 Transformers follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
Check our [documentation writing guide](https://github.com/huggingface/transformers/tree/main/docs#writing-documentation---specification)
for more information.

### Develop on Windows

On Windows (unless you're working in [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) or WSL), you need to configure git to transform Windows `CRLF` line endings to Linux `LF` line endings:

```bash
git config core.autocrlf input
```

One way to run the `make` command on Windows is with MSYS2:

1. [Download MSYS2](https://www.msys2.org/), and we assume it's installed in `C:\msys64`.
2. Open the command line `C:\msys64\msys2.exe` (it should be available from the **Start** menu).
3. Run in the shell: `pacman -Syu` and install `make` with `pacman -S make`.
4. Add `C:\msys64\usr\bin` to your PATH environment variable.

You can now use `make` from any terminal (Powershell, cmd.exe, etc.)! 🎉

### Sync a forked repository with upstream main (the Hugging Face repository)

When updating the main branch of a forked repository, please follow these steps to avoid pinging the upstream repository which adds reference notes to each upstream PR, and sends unnecessary notifications to the developers involved in these PRs.

1. When possible, avoid syncing with the upstream using a branch and PR on the forked repository. Instead, merge directly into the forked main.
2. If a PR is absolutely necessary, use the following steps after checking out your branch:

```bash
git checkout -b your-branch-for-syncing
git pull --squash --no-commit upstream main
git commit -m '<your message without GitHub references>'
git push --set-upstream origin your-branch-for-syncing
```
