import warnings
from typing import Dict

import numpy as np

from ..utils import ExplicitEnum, add_end_docstrings, is_tf_available, is_torch_available
from .base import PIPELINE_INIT_ARGS, GenericTensor, Pipeline


if is_tf_available():
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class EntityPairClassificationPipeline(Pipeline):
    """
    Entity Pair classification pipeline using any `ModelForEntityPairClassification`.

    This entity pair classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"entity-pair-classification"` (for classifying the relationship between two tokens).

    The models that this pipeline can use are models that have been fine-tuned on an entity-pair classification task. See
    the up-to-date list of available models on
    """

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        return_tensors = self.framework
        if isinstance(inputs, dict):
            entity_spans = [tuple(x) for x in inputs["entity_spans"]]
            return self.tokenizer(
                text=inputs["text"], entity_spans=entity_spans, return_tensors=return_tensors, **tokenizer_kwargs
            )
        else:
            # This is likely an invalid usage of the pipeline attempting to pass text pairs.
            raise ValueError(
                "The pipeline received invalid inputs, if you are trying to send text alongside entity_spans, you can try to send a"
                ' dictionnary `{"text": "My text", "entity_spans": [[1,2][3,4]] }` in order to perform entity pair classification.'
            )

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits
        probabilities = softmax(logits)
        best_class = np.argmax(probabilities)
        label = self.model.config.id2label[best_class]
        score = probabilities[best_class][0].item()
        return {"label": label, "confidence": score, "probabilities": probabilities, "best_class_id": best_class}
