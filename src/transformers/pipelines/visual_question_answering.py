from typing import Union

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class VisualQuestionAnsweringPipeline(Pipeline):
    """
    Visual Question Answering pipeline using a `AutoModelForVisualQuestionAnswering`. This pipeline is currently only
    available in PyTorch.

    This visual question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifiers: `"visual-question-answering", "vqa"`.

    The models that this pipeline can use are models that have been fine-tuned on a visual question answering task. See
    the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=visual-question-answering).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING)

    def _sanitize_parameters(self, top_k=None, padding=None, truncation=None, **kwargs):
        preprocess_params, postprocess_params = {}, {}
        if padding is not None:
            preprocess_params["padding"] = padding
        if truncation is not None:
            preprocess_params["truncation"] = truncation
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        return preprocess_params, {}, postprocess_params

    def __call__(self, image: Union["Image.Image", str], question: str = None, **kwargs):
        r"""
        Answers open-ended questions about images. The pipeline accepts several types of inputs which are detailed
        below:

        - `pipeline(image=image, question=question)`
        - `pipeline({"image": image, "question": question})`
        - `pipeline([{"image": image, "question": question}])`
        - `pipeline([{"image": image, "question": question}, {"image": image, "question": question}])`

        Args:
            image (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. If given a single image, it can be
                broadcasted to multiple questions.
            question (`str`, `List[str]`):
                The question(s) asked. If given a single question, it can be broadcasted to multiple images.
            top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.
        Return:
            A dictionary or a list of dictionaries containing the result. The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """
        if isinstance(image, (Image.Image, str)) and isinstance(question, str):
            inputs = {"image": image, "question": question}
        else:
            """
            Supports the following format
            - {"image": image, "question": question}
            - [{"image": image, "question": question}]
            - Generator and datasets
            """
            inputs = image
        results = super().__call__(inputs, **kwargs)
        return results

    def preprocess(self, inputs, padding=False, truncation=False):
        image = load_image(inputs["image"])
        model_inputs = self.tokenizer(
            inputs["question"], return_tensors=self.framework, padding=padding, truncation=truncation
        )
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
            probs = model_outputs.logits.sigmoid()[0]
            scores, ids = probs.topk(top_k)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        scores = scores.tolist()
        ids = ids.tolist()
        return [{"score": score, "answer": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
