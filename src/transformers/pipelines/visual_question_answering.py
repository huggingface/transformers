import copy
from typing import Union

from ..utils import ExplicitEnum, add_end_docstrings, is_torch_available, is_vision_available, logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING

logger = logging.get_logger(__name__)


class ModelType(ExplicitEnum):
    CLASSIFIER = "classifier"
    GENERATIVE = "generative"


@add_end_docstrings(PIPELINE_INIT_ARGS)
class VisualQuestionAnsweringPipeline(Pipeline):
    """
    Visual Question Answering pipeline using a `AutoModelForVisualQuestionAnswering`. This pipeline is currently only
    available in PyTorch.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> oracle = pipeline(model="dandelin/vilt-b32-finetuned-vqa")
    >>> image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png"
    >>> oracle(question="What is she wearing ?", image=image_url)
    [{'score': 0.948, 'answer': 'hat'}, {'score': 0.009, 'answer': 'fedora'}, {'score': 0.003, 'answer': 'clothes'}, {'score': 0.003, 'answer': 'sun hat'}, {'score': 0.002, 'answer': 'nothing'}]

    >>> oracle(question="What is she wearing ?", image=image_url, top_k=1)
    [{'score': 0.948, 'answer': 'hat'}]

    >>> oracle(question="Is this a person ?", image=image_url, top_k=1)
    [{'score': 0.993, 'answer': 'yes'}]

    >>> oracle(question="Is this a man ?", image=image_url, top_k=1)
    [{'score': 0.996, 'answer': 'no'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This visual question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifiers: `"visual-question-answering", "vqa"`.

    The models that this pipeline can use are models that have been fine-tuned on a visual question answering task. See
    the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=visual-question-answering).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING)

        if self.model.config.__class__.__name__ in ["GitConfig"]:
            self.model_type = ModelType.GENERATIVE
        elif self.model.config.__class__.__name__ in ["ViltConfig"]:
            self.model_type = ModelType.CLASSIFIER

    def _sanitize_parameters(self, top_k=None, padding=None, truncation=None, **generate_kwargs):
        preprocess_params, forward_params, postprocess_params = {}, {}, {}
        if padding is not None:
            preprocess_params["padding"] = padding
        if truncation is not None:
            preprocess_params["truncation"] = truncation
        if top_k is not None:
            postprocess_params["top_k"] = top_k
            forward_params["top_k"] = top_k
        forward_params.update(generate_kwargs)
        return preprocess_params, forward_params, postprocess_params

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
        image_features = self.image_processor(images=image, return_tensors=self.framework)
        model_inputs.update(image_features)
        return model_inputs

    def _forward(self, model_inputs, top_k=5, **generate_kwargs):
        if self.model_type == ModelType.GENERATIVE:
            generate_kwargs = copy.deepcopy(generate_kwargs)
            generate_kwargs["return_dict_in_generate"] = True
            generate_kwargs["output_scores"] = True
            if "num_beams" in generate_kwargs:
                if top_k > generate_kwargs["num_beams"]:
                    pass  # raise
            elif top_k > 1:
                generate_kwargs["num_beams"] = top_k
            else:
                # activate beam search with two beam to compute scores
                generate_kwargs["num_beams"] = 2
            generate_kwargs["num_return_sequences"] = top_k
            if "max_new_tokens" not in generate_kwargs:
                # defaulting max_new_tokens to 100
                generate_kwargs["max_new_tokens"] = 100
            generate_outputs = self.model.generate(**model_inputs, **generate_kwargs)
            model_outputs = {
                "sequences_scores": generate_outputs.sequences_scores.reshape(
                    (model_inputs["input_ids"].shape[0], top_k)
                ),
                "sequences": generate_outputs.sequences.reshape((model_inputs["input_ids"].shape[0], top_k, -1)),
            }
        elif self.model_type == ModelType.CLASSIFIER:
            model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=5):
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels
        if self.framework == "pt":
            if self.model_type == ModelType.CLASSIFIER:
                probs = model_outputs.logits.sigmoid()[0]
                scores, ids = probs.topk(top_k)
                ids = ids.tolist()
                answers = [self.model.config.id2label[_id] for _id in ids]
            elif self.model_type == ModelType.GENERATIVE:
                scores = model_outputs["sequences_scores"][0]
                decoded_outputs = self.tokenizer.batch_decode(model_outputs["sequences"][0], skip_special_tokens=False)
                answers = [self.postprocess_git_output_single(o) for o in decoded_outputs]
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        scores = scores.tolist()
        return [{"score": score, "answer": answer} for score, answer in zip(scores, answers)]

    def postprocess_git_output_single(self, decoded_output: str) -> str:
        r"""
        Transforms a single output of GIT model into an answer.

        For example: "[CLS] what color is the bus? [SEP] blue [SEP]" returns "blue"
        """
        return decoded_output.split(self.tokenizer.sep_token)[1][1:-1]
