import torch

from transformers import Pipeline


class PairClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, second_text=None):
        return self.tokenizer(text, text_pair=second_text, return_tensors="pt")

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        best_class = probabilities.argmax().item()
        label = self.model.config.id2label[best_class]
        score = probabilities.squeeze()[best_class].item()
        logits = logits.squeeze().tolist()
        return {"label": label, "score": score, "logits": logits}
