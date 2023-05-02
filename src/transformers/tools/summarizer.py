from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool, RemoteTool


TEXT_SUMMARIZATION_CESCRIPTION = "This is a tool that summarizes an English text. It takes an input `text` containing the text to summarize, and returns a summary of the text."


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
    description = TEXT_SUMMARIZATION_CESCRIPTION
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSeq2SeqLM

    def encode(self, raw_inputs):
        return self.pre_processor(raw_inputs, return_tensors="pt", truncation=True)

    def forward(self, inputs):
        return self.model.generate(**inputs)[0]

    def decode(self, outputs):
        return self.pre_processor.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)


class RemoteTextSummarizationTool(RemoteTool):
    """
    Example:

    ```py
    from transformers.tools import RemoteTextClassificationTool

    classifier = RemoteTextClassificationTool()
    classifier("This is a super nice API!", labels=["positive", "negative"])
    ```
    """

    default_checkpoint = "philschmid/bart-large-cnn-samsum"
    description = TEXT_SUMMARIZATION_CESCRIPTION

    def prepare_inputs(self, text):
        return {"inputs": text}

    def extract_outputs(self, outputs):
        return outputs[0]["summary_text"]
