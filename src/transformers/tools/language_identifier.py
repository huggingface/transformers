from .text_classification import TextClassificationTool


LANGUAGE_IDENTIFIER_DESCRIPTION = (
    "This is a tool that identifies the language of the text passed as input. It takes one input named `text` and "
    "returns the two-letter label of the identified language."
)


class LanguageIdentificationTool(TextClassificationTool):
    """
    Example:

    ```py
    from transformers.tools import LanguageIdentificationTool

    classifier = LanguageIdentificationTool()
    classifier("This is a super nice API!")
    ```
    """

    default_checkpoint = "papluca/xlm-roberta-base-language-detection"
    description = LANGUAGE_IDENTIFIER_DESCRIPTION

    def decode(self, outputs):
        return super().decode(outputs)["label"]
