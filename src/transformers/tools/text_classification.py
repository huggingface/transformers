import torch

from ..models.auto import AutoModelForSequenceClassification, AutoTokenizer
from .base import OldRemoteTool, PipelineTool


TEXT_CLASSIFIER_DESCRIPTION = (
    "This is a tool that classifies an English text using provided labels. It takes two inputs: `text`, which should "
    "be the text to classify, and `labels`, which should be the list of labels to use for classification. It returns "
    "the most likely label in the list of provided `labels` for the input text."
)


class TextClassificationTool(PipelineTool):
    """
    Example:

    ```py
    from transformers.tools import TextClassificationTool

    classifier = TextClassificationTool()
    classifier("This is a super nice API!", labels=["positive", "negative"])
    ```
    """

    default_checkpoint = "facebook/bart-large-mnli"
    description = TEXT_CLASSIFIER_DESCRIPTION
    name = "text-classifier"
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSequenceClassification

    def setup(self):
        super().setup()
        config = self.model.config
        self.entailment_id = -1
        for idx, label in config.id2label.items():
            if label.lower().startswith("entail"):
                self.entailment_id = int(idx)
        if self.entailment_id == -1:
            raise ValueError("Could not determine the entailment ID from the model config, please pass it at init.")

    def encode(self, text, labels):
        self._labels = labels
        return self.pre_processor(
            [text] * len(labels),
            [f"This example is {label}" for label in labels],
            return_tensors="pt",
            padding="max_length",
        )

    def decode(self, outputs):
        logits = outputs.logits
        label_id = torch.argmax(logits[:, 2]).item()
        return self._labels[label_id]


class RemoteTextClassificationTool(OldRemoteTool):
    """
    Example:

    ```py
    from transformers.tools import RemoteTextClassificationTool

    classifier = RemoteTextClassificationTool()
    classifier("This is a super nice API!", labels=["positive", "negative"])
    ```
    """

    default_checkpoint = "facebook/bart-large-mnli"
    description = TEXT_CLASSIFIER_DESCRIPTION

    def prepare_inputs(self, text, labels):
        return {"inputs": text, "params": {"candidate_labels": labels}}

    def extract_outputs(self, outputs):
        label = None
        max_score = 0
        for lbl, score in zip(outputs["labels"], outputs["scores"]):
            if score > max_score:
                label = lbl
                max_score = score

        return label
