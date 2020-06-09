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

Make sure that there is a symlink from the `example` file (in /examples) inside the source folder. Run the following
command to generate it:

```bash
ln -s ../../examples/README.md examples.md
```

Once you have setup `sphinx`, you can build the documentation by running the following command in the `/docs` folder:

```bash
make html
```

A folder called ``_build/html`` should have been created. You can now open the file ``_build/html/index.html`` in your browser. 

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
[Sourceforge complete documentation](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html))

### Adding a new section

A section is a page held in the `Notes` toc-tree on the documentation. Adding a new section is done in two steps:

- Add a new file under `./source`. This file can either be ReStructuredText (.rst) or Markdown (.md).
- Link that file in `./source/index.rst` on the correct toc-tree.

### Adding a new model

When adding a new model:
 
- Create a file `xxx.rst` under `./source/model_doc`. 
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
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XXXConfig
    :members:
```

This will include every public method of the configuration. If for some reason you wish for a method not to be displayed
in the documentation, you can do so by specifying which methods should be in the docs:

```
XXXTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XXXTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary

```

### Writing source documentation

Values that should be put in `code` should either be surrounded by double backticks: \`\`like so\`\` or be written as an object
using the :obj: syntax: :obj:\`like so\`.

When mentionning a class, it is recommended to use the :class: syntax as the mentioned class will be automatically
linked by Sphinx: :class:\`transformers.XXXClass\`

When mentioning a function, it is recommended to use the :func: syntax as the mentioned method will be automatically
linked by Sphinx: :func:\`transformers.XXXClass.method\`

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

            Indices can be obtained using :class:`transformers.AlbertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
```

#### Writing a multi-line code block 

Multi-line code blocks can be useful for displaying examples. They are done like so:

```
Example::

    # first line of code
    # second line
    # etc
```

The `Example` string at the beginning can be replaced by anything as long as there are two semicolons following it.

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
        A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
```
