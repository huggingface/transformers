# Security Policy

## Did you find a bug?

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

## Reporting a Vulnerability
If you've found a security vulnerability in transformers, please do not create an issue on the GitHub issue tracker as it will be visible publicly.

Instead, send an email to security@huggingface.co
