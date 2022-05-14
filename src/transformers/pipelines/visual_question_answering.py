from typing import List, Union

from transformers.models.auto.modeling_auto import MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING

from ..utils import (
    add_end_docstrings,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
from .base import PIPELINE_INIT_ARGS, Pipeline

print("here")
if is_vision_available():
    print("there")
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class VisualQuestionAnsweringPipeline(Pipeline):
    """
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from PIL import Image

        from ..image_utils import load_image
        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING)

    def _sanitize_parameters(self, top_k=None):
        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        return {}, {}, postprocess_params

    def __call__(
        self,
        images: Union[str, List[str], "Image.Image", List["Image.Image"]],
        question: Union[str, List[str]],
        **kwargs):
        """
        """
        return super().__call__(images, question, **kwargs)

    def preprocess(self, image, question):
        image = load_image(image)
        model_inputs = self.tokenizer(question, return_tensors=self.framework)
        image_features = self.feature_extractor(images=image, return_tensors=self.framework)
        model_inputs.update(image_features)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=5):
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        if self.framework == "pt":
            probs = model_outputs.logits.softmax(-1)[0]
            scores, ids = probs.topk(top_k)
        elif self.framework == "tf":
            probs = stable_softmax(model_outputs.logits, axis=-1)[0]
            topk = tf.math.top_k(probs, k=top_k)
            scores, ids = topk.values.numpy(), topk.indices.numpy()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        scores = scores.tolist()
        ids = ids.tolist()
        return [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
