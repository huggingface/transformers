# Translating the Transformers documentation into your language

As part of our mission to democratize machine learning, we aim to make the Transformers library available in many more languages! Follow the steps below to help translate the documentation into your language.

## Open an Issue

1. Navigate to the Issues page of this repository.
2. Check if anyone has already opened an issue for your language.
3. If not, create a new issue by selecting the "Translation template" from the "New issue" button.
4. Post a comment indicating which chapters you’d like to work on, and we’ll add your name to the list.

## Fork the Repository

1. First, fork the Transformers repo by clicking the Fork button in the top-right corner.
2. Clone your fork to your local machine for editing with the following command:

    ```bash
    git clone https://github.com/YOUR-USERNAME/transformers.git
    ```
   
   Replace `YOUR-USERNAME` with your GitHub username.

## Copy-paste the English version with a new language code

The documentation files are organized in the following directory:

- **docs/source**: This contains all documentation materials organized by language.

To copy the English version to your new language directory:

1. Navigate to your fork of the repository:

    ```bash
    cd ~/path/to/transformers/docs
    ```

   Replace `~/path/to` with your actual path.

2. Run the following command:

    ```bash
    cp -r source/en source/LANG-ID
    ```

   Replace `LANG-ID` with the appropriate ISO 639-1 or ISO 639-2 language code (see [this table](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) for reference).

## Start translating

Begin translating the text!

1. Start with the `_toctree.yml` file that corresponds to your documentation chapter. This file is essential for rendering the table of contents on the website.

    - If the `_toctree.yml` file doesn’t exist for your language, create one by copying the English version and removing unrelated sections.
    - Ensure it is placed in the `docs/source/LANG-ID/` directory.

    Here’s an example structure for the `_toctree.yml` file:

    ```yaml
    - sections:
      - local: pipeline_tutorial # Keep this name for your .md file
        title: Pipelines for Inference # Translate this
        ...
      title: Tutorials # Translate this
    ```

2. Once you’ve translated the `_toctree.yml`, move on to translating the associated MDX files.

## Collaborate and share

If you'd like assistance with your translation, open an issue and tag `@stevhliu`. Feel free to share resources or glossaries to ensure consistent terminology.
