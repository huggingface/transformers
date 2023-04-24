from .text_classification import TextClassificationTool


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
    description = (
        "This is a tool that identifies the language of the text passed as input. It takes one input named `text` and "
        "returns the two-letter label of the identified language."
    )

    def decode(self, outputs):
        return super().decode(outputs)["label"]
