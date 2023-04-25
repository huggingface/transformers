import torch

from ..models.auto import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from .base import PipelineTool, RemoteTool


class TextClassificationTool(PipelineTool):
    """
    Example:

    ```py
    from transformers.tools import TextClassificationTool

    classifier = TextClassificationTool("distilbert-base-uncased-finetuned-sst-2-english")
    classifier("This is a super nice API!")
    ```
    """

    default_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  # Needs to be updated
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSequenceClassification

    description = (
        "classifies an English text using the following {n_labels} labels: {labels}. It takes a input named `text` "
        "which should be in English and returns a dictionary with two keys named 'label' (the predicted label ) and "
        "'score' (the probability associated to it)."
    )

    def post_init(self):
        if isinstance(self.model, str):
            config = AutoConfig.from_pretrained(self.model)
        else:
            config = self.model.config

        num_labels = config.num_labels
        labels = list(config.label2id.keys())

        if len(labels) > 1:
            labels = [f"'{label}'" for label in labels]
            labels_string = ", ".join(labels[:-1])
            labels_string += f", and {labels[-1]}"
        else:
            raise ValueError("Not enough labels.")

        self.description = self.description.replace("{n_labels}", str(num_labels)).replace("{labels}", labels_string)

    def encode(self, text):
        return self.pre_processor(text, return_tensors="pt")

    def decode(self, outputs):
        logits = outputs.logits
        scores = torch.nn.functional.softmax(logits, dim=-1)
        label_id = torch.argmax(logits[0]).item()
        label = self.model.config.id2label[label_id]
        return {"label": label, "score": scores[0][label_id].item()}


class RemoteTextClassificationTool(RemoteTool):
    """
    Example:

    ```py
    from transformers.tools import RemoteTextClassificationTool

    classifier = RemoteTextClassificationTool("distilbert-base-uncased-finetuned-sst-2-english")
    classifier("This is a super nice API!")
    ```
    """

    default_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  # Needs to be updated
    description = (
        "classifies an English text using the following {n_labels} labels: {labels}. It takes a input named `text` "
        "which should be in English and returns a dictionary with two keys named 'label' (the predicted label ) and "
        "'score' (the probability associated to it)."
    )

    def post_init(self):
        config = AutoConfig.from_pretrained(self.repo_id)
        num_labels = config.num_labels
        labels = list(config.label2id.keys())

        if len(labels) > 1:
            labels = [f"'{label}'" for label in labels]
            labels_string = ", ".join(labels[:-1])
            labels_string += f", and {labels[-1]}"
        else:
            raise ValueError("Not enough labels.")

        self.description = self.description.replace("{n_labels}", str(num_labels)).replace("{labels}", labels_string)

    def extract_outputs(self, outputs):
        label = None
        max_score = 0
        for result in outputs:
            lbl = result["label"]
            score = float(result["score"])
            if score > max_score:
                label = lbl
                max_score = score

        return {"label": label, "score": max_score}
