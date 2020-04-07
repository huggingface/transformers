# Generating the documentation

To generate the documentation, you first have to build it. Several packages are necessary to build the doc,
you can install them with the following command, at the root of the code repository:

```bash
pip install -e ".[docs]"
```

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
