---
name: "\U0001F4DA Migration from pytorch-pretrained-bert or pytorch-transformers"
about: Report a problem when migrating from pytorch-pretrained-bert or pytorch-transformers
  to transformers
title: ''
labels: Migration
assignees: ''

---

# 📚 Migration

## Information

<!-- Important information -->

Model I am using (Bert, XLNet ...):

Language I am using the model on (English, Chinese ...):

The problem arises when using:
* [ ] the official example scripts: (give details below)
* [ ] my own modified scripts: (give details below)

The tasks I am working on is:
* [ ] an official GLUE/SQUaD task: (give the name)
* [ ] my own task or dataset: (give details below)

## Details

<!-- A clear and concise description of the migration issue.
    If you have code snippets, please provide it here as well.
    Important! Use code tags to correctly format your code. See https://help.github.com/en/github/writing-on-github/creating-and-highlighting-code-blocks#syntax-highlighting
    Do not use screenshots, as they are hard to read and (more importantly) don't allow others to copy-and-paste your code.
    -->

## Environment info
<!-- You can run the command `python transformers-cli env` and copy-and-paste its output below.
     Don't forget to fill out the missing fields in that output! -->
 
- `transformers` version:
- Platform:
- Python version:
- PyTorch version (GPU?):
- Tensorflow version (GPU?):
- Using GPU in script?:
- Using distributed or parallel set-up in script?:

<!-- IMPORTANT: which version of the former library do you use? -->
* `pytorch-transformers` or `pytorch-pretrained-bert` version (or branch):


## Checklist

- [ ] I have read the migration guide in the readme.
 ([pytorch-transformers](https://github.com/huggingface/transformers#migrating-from-pytorch-transformers-to-transformers);
  [pytorch-pretrained-bert](https://github.com/huggingface/transformers#migrating-from-pytorch-pretrained-bert-to-transformers))
- [ ] I checked if a related official extension example runs on my machine.
