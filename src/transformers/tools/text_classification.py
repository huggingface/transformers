import torch

from ..models.auto import AutoModelForSequenceClassification, AutoTokenizer
from .base import Tool


class TextClassificationTool(Tool):
    """
    Example:

    ```py
    from transformers.tools import TextClassificationTool

    classifier = TextClassificationTool("distilbert-base-uncased-finetuned-sst-2-english")
    classifier("This is a super nice API!")
    ```
    """

    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSequenceClassification

    description = """
    Text classification tool: it takes text as inputs and labels it. It has {n_labels} available, which are {labels}.
    """

    def __init__(self):
        super().__init__()

        num_labels = self.model.config.num_labels
        labels = list(self.model.config.label2id.keys())

        if len(labels) > 1:
            labels_string = ", ".join(labels[:-1])
            labels_string += f", and {labels[-1]}"
        else:
            raise ValueError("Not enough labels.")

        self.description = self.description.replace("{n_labels}", num_labels).replace("{labels}", labels_string)

    def encode(self, text):
        return self.pre_processor(text, return_tensors="pt")

    def decode(self, outputs):
        logits = outputs.logits
        scores = torch.nn.functional.softmax(logits, dim=-1)
        label_id = torch.argmax(logits[0]).item()
        label = self.model.config.id2label[label_id]
        return {"label": label, "score": scores[0][label_id].item()}
