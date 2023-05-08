from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool


class TextSummarizationTool(PipelineTool):
    """
    Example:

    ```py
    from transformers.tools import TextSummarizationTool

    classifier = TextSummarizationTool()
    classifier(long_text)
    ```
    """

    default_checkpoint = "philschmid/bart-large-cnn-samsum"
    description = (
        "This is a tool that summarizes an English text. It takes an input `text` containing the text to summarize, "
        "and returns a summary of the text."
    )
    name = "sumamrizer"
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSeq2SeqLM

    inputs = ["text"]
    outputs = ["text"]

    def encode(self, text):
        return self.pre_processor(text, return_tensors="pt", truncation=True)

    def forward(self, inputs):
        return self.model.generate(**inputs)[0]

    def decode(self, outputs):
        print(outputs)
        return self.pre_processor.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
