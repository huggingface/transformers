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

# Generating the documentation

To generate the documentation, you first have to build it. Several packages are necessary to build the doc,
you can install them with the following command, at the root of the code repository:

```bash
pip install -e ".[docs]"
```

---
**NOTE**

You only need to generate the documentation to inspect it locally (if you're planning changes and want to
check how they look like before committing for instance). You don't have to commit the built documentation.

---

## Packages installed

Here's an overview of all the packages installed. If you ran the previous command installing all packages from
`requirements.txt`, you do not need to run the following commands.

Building it requires the package `sphinx` that you can
install using:

```bash
pip install -U sphinx
```

You would also need the custom installed [theme](https://github.com/readthedocs/sphinx_rtd_theme) by
[Read The Docs](https://readthedocs.org/). You can install it using the following command:

```bash
pip install sphinx_rtd_theme
```

The third necessary package is the `recommonmark` package to accept Markdown as well as Restructured text:

```bash
pip install recommonmark
```

## Building the documentation

Once you have setup `sphinx`, you can build the documentation by running the following command in the `/docs` folder:

```bash
make html
```

A folder called ``_build/html`` should have been created. You can now open the file ``_build/html/index.html`` in your
browser.

---
**NOTE**

If you are adding/removing elements from the toc-tree or from any structural item, it is recommended to clean the build
directory before rebuilding. Run the following command to clean and build:

```bash
make clean && make html
```

---

It should build the static app that will be available under `/docs/_build/html`

## Adding a new element to the tree (toc-tree)

Accepted files are reStructuredText (.rst) and Markdown (.md). Create a file with its extension and put it
in the source directory. You can then link it to the toc-tree by putting the filename without the extension.

## Preview the documentation in a pull request

Once you have made your pull request, you can check what the documentation will look like after it's merged by
following these steps:

- Look at the checks at the bottom of the conversation page of your PR (you may need to click on "show all checks" to
  expand them).
- Click on "details" next to the `ci/circleci: build_doc` check.
- In the new window, click on the "Artifacts" tab.
- Locate the file "docs/_build/html/index.html" (or any specific page you want to check) and click on it to get a
  preview.

## Writing Documentation - Specification

The `huggingface/transformers` documentation follows the
[Google documentation](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) style. It is
mostly written in ReStructuredText
([Sphinx simple documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html),
[Sourceforge complete documentation](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html)).


### Adding a new tutorial

Adding a new tutorial or section is done in two steps:

- Add a new file under `./source`. This file can either be ReStructuredText (.rst) or Markdown (.md).
- Link that file in `./source/index.rst` on the correct toc-tree.

Make sure to put your new file under the proper section. It's unlikely to go in the first section (*Get Started*), so
depending on the intended targets (beginners, more advanced users or researchers) it should go in section two, three or
four.

### Adding a new model

When adding a new model:

- Create a file `xxx.rst` under `./source/model_doc` (don't hesitate to copy an existing file as template).
- Link that file in `./source/index.rst` on the `model_doc` toc-tree.
- Write a short overview of the model:
    - Overview with paper & authors
    - Paper abstract
    - Tips and tricks and how to use it best
- Add the classes that should be linked in the model. This generally includes the configuration, the tokenizer, and
  every model of that class (the base model, alongside models with additional heads), both in PyTorch and TensorFlow.
  The order is generally:
    - Configuration,
    - Tokenizer
    - PyTorch base model
    - PyTorch head models
    - TensorFlow base model
    - TensorFlow head models

These classes should be added using the RST syntax. Usually as follows:
```
XXXConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XXXConfig
    :members:
```

This will include every public method of the configuration that is documented. If for some reason you wish for a method
not to be displayed in the documentation, you can do so by specifying which methods should be in the docs:

```
XXXTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XXXTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary

```

### Writing source documentation

Values that should be put in `code` should either be surrounded by double backticks: \`\`like so\`\` or be written as
an object using the :obj: syntax: :obj:\`like so\`. Note that argument names and objects like True, None or any strings
should usually be put in `code`.

When mentionning a class, it is recommended to use the :class: syntax as the mentioned class will be automatically
linked by Sphinx: :class:\`~transformers.XXXClass\`

When mentioning a function, it is recommended to use the :func: syntax as the mentioned function will be automatically
linked by Sphinx: :func:\`~transformers.function\`.

When mentioning a method, it is recommended to use the :meth: syntax as the mentioned method will be automatically
linked by Sphinx: :meth:\`~transformers.XXXClass.method\`.

Links should be done as so (note the double underscore at the end): \`text for the link <./local-link-or-global-link#loc>\`__

#### Defining arguments in a method

Arguments should be defined with the `Args:` prefix, followed by a line return and an indentation.
The argument should be followed by its type, with its shape if it is a tensor, and a line return.
Another indentation is necessary before writing the description of the argument.

Here's an example showcasing everything so far:

```
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.AlbertTokenizer`.
            See :meth:`~transformers.PreTrainedTokenizer.encode` and
            :meth:`~transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
```

For optional arguments or arguments with defaults we follow the following syntax: imagine we have a function with the
following signature:

```
def my_function(x: str = None, a: float = 1):
```

then its documentation should look like this:

```
    Args:
        x (:obj:`str`, `optional`):
            This argument controls ...
        a (:obj:`float`, `optional`, defaults to 1):
            This argument is used to ...
```

Note that we always omit the "defaults to :obj:\`None\`" when None is the default for any argument. Also note that even
if the first line describing your argument type and its default gets long, you can't break it on several lines. You can
however write as many lines as you want in the indented description (see the example above with `input_ids`).

#### Writing a multi-line code block

Multi-line code blocks can be useful for displaying examples. They are done like so:

```
Example::

    # first line of code
    # second line
    # etc
```

The `Example` string at the beginning can be replaced by anything as long as there are two semicolons following it.

We follow the [doctest](https://docs.python.org/3/library/doctest.html) syntax for the examples to automatically test
the results stay consistent with the library.

#### Writing a return block

Arguments should be defined with the `Args:` prefix, followed by a line return and an indentation.
The first line should be the type of the return, followed by a line return. No need to indent further for the elements
building the return.

Here's an example for tuple return, comprising several objects:

```
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
```

Here's an example for a single value return:

```
    Returns:
        :obj:`List[int]`: A list of integers in the range [0, 1] --- 1 for a special token, 0 for a sequence token.
```

#### Adding a new section

In ReST section headers are designated as such with the help of a line of underlying characters, e.g.,:

```
Section 1
^^^^^^^^^^^^^^^^^^

Sub-section 1
~~~~~~~~~~~~~~~~~~
```

ReST allows the use of any characters to designate different section levels, as long as they are used consistently within the same document. For details see [sections doc](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections). Because there is no standard different documents often end up using different characters for the same levels which makes it very difficult to know which character to use when creating a new section.

Specifically, if when running `make docs` you get an error like:
```
docs/source/main_classes/trainer.rst:127:Title level inconsistent:
```
you picked an inconsistent character for some of the levels.

But how do you know which characters you must use for an already existing level or when adding a new level?

You can use this helper script:
```
perl -ne '/^(.)\1{100,}/ && do { $h{$1}=++$c if !$h{$1} }; END { %h = reverse %h ; print "$_ $h{$_}\n" for sort keys %h}' docs/source/main_classes/trainer.rst
1 -
2 ~
3 ^
4 =
5 "
```

This tells you which characters have already been assigned for each level.

So using this particular example's output -- if your current section's header uses `=` as its underline character, you now know you're at level 4, and if you want to add a sub-section header you know you want `"` as it'd level 5.

If you needed to add yet another sub-level, then pick a character that is not used already. That is you must pick a character that is not in the output of that script.

Here is the full list of characters that can be used in this context: `= - ` : ' " ~ ^ _ * + # < >`
