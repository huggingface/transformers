import types

from transformers.models.auto.modeling_auto import MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING

from ..utils import (
    add_end_docstrings,
    is_torch_available,
    is_vision_available,
    logging,
)
from .base import ArgumentHandler, PIPELINE_INIT_ARGS, Pipeline, Dataset

if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

logger = logging.get_logger(__name__)

class VisualQuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & image) to be mapped to
    an input dict of {"image": ..., "question": ...}    
    """

    def __call__(self, *args, **kwargs):
        # Detect where the actual inputs are
        if args is not None and len(args) > 0:
            if len(args) == 1:
                pipeline_inputs = args[0]
                if isinstance(pipeline_inputs, Dataset) or isinstance(pipeline_inputs, types.GeneratorType):
                    return pipeline_inputs
                elif isinstance(pipeline_inputs, dict):
                    pipeline_inputs = [pipeline_inputs]
                for pipeline_input in pipeline_inputs:
                    if not isinstance(pipeline_input, dict):
                        raise ValueError("Input is expected to be type `List[Dict[str, Union(str, Image.Image)]]`")
                    question = pipeline_input.get("question", None)
                    image = pipeline_input.get("image", None)
                    if not isinstance(question, str):
                        raise ValueError("`question` is a required key in each input dict and expected value type is `str`")
                    if not (isinstance(image, str) or isinstance(image, Image.Image)):
                        raise ValueError("`image` is a required key in each input dict and expected value type is `str` or `Image.Image`")
            else:
                raise ValueError("Please use keyword arguments `question` and `image`")
        elif "question" in kwargs and "image" in kwargs:
            if isinstance(kwargs["question"], list) and (isinstance(kwargs["image"], str) or isinstance(kwargs["image"], Image.Image)):
                pipeline_inputs = [{"question": Q, "image": kwargs["image"]} for Q in kwargs["question"]]
            elif isinstance(kwargs["question"], str) and isinstance(kwargs["image"], list):
                pipeline_inputs = [{"question": kwargs["question"], "image": image} for image in kwargs["image"]]
            elif isinstance(kwargs["question"], list) and isinstance(kwargs["image"], list):
                if len(kwargs["question"]) != len(kwargs["image"]):
                    raise ValueError("Questions and images don't have the same lengths")
                pipeline_inputs = [{"question": Q, "image": C} for Q, C in zip(kwargs["question"], kwargs["image"])]
            elif isinstance(kwargs["question"], str) and (isinstance(kwargs["image"], str) or isinstance(kwargs["image"], Image.Image)):
                pipeline_inputs = [{"question": kwargs["question"], "image": kwargs["image"]}]
            else:
                raise ValueError("Arguments can't be understood")
        else:
            raise ValueError(f"Unknown arguments {kwargs}")
        return pipeline_inputs


@add_end_docstrings(PIPELINE_INIT_ARGS)
class VisualQuestionAnsweringPipeline(Pipeline):
    """
    Visual Question Answering pipeline using a `ModelForVisualQuestionAnswering`. This pipeline is currently only available in
    PyTorch.

    This tabular question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"visual-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a visual question answering task.
    TODO: point to [huggingface.co/models](https://huggingface.co/models?filter=table-question-answering) after the pipeline is added
    """

    def __init__(self, args_parser=VisualQuestionAnsweringArgumentHandler(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args_parser = args_parser
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

    def __call__(self, *args, **kwargs):
        r"""
        Answers open-ended questions about images. The pipeline accepts several types of inputs which are detailed below:

        - `pipeline(image=image, question=question)`
        - `pipeline(image=[image], question=[question])`
        - `pipeline(image=[image, image], question=[question])`
        - `pipeline(image=[image], question=[question, question])`
        - `pipeline(image=[image], question=[question, question])`
        - `pipeline({"image": image, "question": question})`
        - `pipeline([{"image": image, "question": question}])`
        - `pipeline([{"image": image, "question": question}, {"image": image, "question": question}])`

        Args:
            image (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. If given a single image, it can be broadcasted to multiple questions.
            question (`str`, `List[str]`):
                The question(s) asked. If given a single question, it can be broadcasted to multiple images.
            top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.
        Return:
            A dictionary or a list of dictionaries containing result. The dictionaries contain the following keys:

            - **label** (`str`) -- The label identified by the model.
            - **score** (`int`) -- The score attributed by the model for that label.
        """

        pipeline_inputs = self._args_parser(*args, **kwargs)
        results = super().__call__(pipeline_inputs, **kwargs)
        return results

    def preprocess(self, pipeline_input, padding=False, truncation=False):
        image = load_image(pipeline_input["image"])
        model_inputs = self.tokenizer(pipeline_input["question"], return_tensors=self.framework, padding=padding, truncation=truncation)
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
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        scores = scores.tolist()
        ids = ids.tolist()
        return [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
