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
        "identifies the language of the text passed as input. Returns the two-letter label of the identified language."
    )

    def decode(self, outputs):
        return super().decode(outputs)["label"]
