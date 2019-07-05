# Generating the documentation

To generate the documentation, you first have to build it. Building it requires the package `sphinx` that you can 
install using:

```bash
pip install -U sphinx
```

You would also need the custom installed [theme](https://github.com/readthedocs/sphinx_rtd_theme) by 
[Read The Docs](https://readthedocs.org/). You can install it using the following command:

```bash
pip install sphinx_rtd_theme
```

Once you have setup `sphinx`, you can build the documentation by running the following command in the `/docs` folder:

```bash
make html
```

It should build the static app that will be available under `/docs/_build/html`