### Translating the Transformers documentation into your language

As part of our mission to democratize machine learning, we'd love to make the Transformers library available in many more languages! Follow the steps below if you want to help translate the documentation into your language 🙏.

**🗞️ Open an issue**

To get started, navigate to the [Issues](https://github.com/huggingface/transformers/issues) page of this repo and check if anyone else has opened an issue for your language. If not, open a new issue by selecting the "Translation template" from the "New issue" button.

Once an issue exists, post a comment to indicate which chapters you'd like to work on, and we'll add your name to the list.


**🍴 Fork the repository**

First, you'll need to [fork the Transformers repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo). You can do this by clicking on the **Fork** button on the top-right corner of this repo's page.

Once you've forked the repo, you'll want to get the files on your local machine for editing. You can do that by cloning the fork with Git as follows:

```bash
git clone https://github.com/YOUR-USERNAME/transformers.git
```

**📋 Copy-paste the English version with a new language code**

The documentation files are in one leading directory:

- [`docs/source`](https://github.com/huggingface/transformers/tree/main/docs/source): All the documentation materials are organized here by language.

You'll only need to copy the files in the [`docs/source/en`](https://github.com/huggingface/transformers/tree/main/docs/source/en) directory, so first navigate to your fork of the repo and run the following:

```bash
cd ~/path/to/transformers/docs
cp -r source/en source/LANG-ID
```

Here, `LANG-ID` should be one of the ISO 639-1 or ISO 639-2 language codes -- see [here](https://www.loc.gov/standards/iso639-2/php/code_list.php) for a handy table.

**✍️ Start translating**

The fun part comes - translating the text!

The first thing we recommend is translating the part of the `_toctree.yml` file that corresponds to your doc chapter. This file is used to render the table of contents on the website. 

> 🙋 If the `_toctree.yml` file doesn't yet exist for your language, you can create one by copy-pasting from the English version and deleting the sections unrelated to your chapter. Just make sure it exists in the `docs/source/LANG-ID/` directory!

The fields you should add are `local` (with the name of the file containing the translation; e.g. `autoclass_tutorial`), and `title` (with the title of the doc in your language; e.g. `Load pretrained instances with an AutoClass`) -- as a reference, here is the `_toctree.yml` for [English](https://github.com/huggingface/transformers/blob/main/docs/source/en/_toctree.yml):

```yaml
- sections:
  - local: pipeline_tutorial # Do not change this! Use the same name for your .md file
    title: Pipelines for inference # Translate this!
    ...
  title: Tutorials # Translate this!
```

Once you have translated the `_toctree.yml` file, you can start translating the [MDX](https://mdxjs.com/) files associated with your docs chapter.

> 🙋 If you'd like others to help you with the translation, you should [open an issue](https://github.com/huggingface/transformers/issues) and tag @stevhliu.
