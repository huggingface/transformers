import torch

from ..models.auto import AutoModelForSequenceClassification, AutoTokenizer
from .base import Pipeline


class TextClassificationPipeline(Pipeline):
    """
    Example:

    ```py
    from transformers.new_pipelines import TextClassificationPipeline

    classifier = TextClassificationPipeline("distilbert-base-uncased-finetuned-sst-2-english")
    classifier("This is a super nice API!")
    ```
    """

    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSequenceClassification

    def encode(self, text):
        return self.pre_processor(text, return_tensors="pt")

    def decode(self, outputs):
        logits = outputs.logits
        scores = torch.nn.functional.softmax(logits, dim=-1)
        label_id = torch.argmax(logits[0]).item()
        label = self.model.config.id2label[label_id]
        return {"label": label, "score": scores[0][label_id].item()}
