from typing import List, Optional, Union

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING

logger = logging.get_logger(__name__)


def postprocess_sequence_output(model, model_outputs, framework, top_k):
    if top_k > model.config.num_labels:
        top_k = model.config.num_labels

    if framework == "pt":
        probs = model_outputs.logits.sigmoid()[0]
        scores, ids = probs.topk(top_k)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    scores = scores.tolist()
    ids = ids.tolist()
    return [{"score": score, "answer": model.config.id2label[_id]} for score, _id in zip(scores, ids)]


def postprocess_qa_output(model, model_outputs, word_ids, words, framework, top_k):
    # TODO: Test to make sure this works with tensorflow too (or break on the framework)

    # TODO: This is a very poor implementation of start/end (just here for completeness sake).
    # Ideally we can refactor/borrow the implementation in the question answering pipeline.
    results = []
    for i, (s, e) in enumerate(zip(model_outputs.start_logits.argmax(-1), model_outputs.end_logits.argmax(-1))):
        if s > e:
            continue
        else:
            word_start, word_end = word_ids[i][s], word_ids[i][e]
            results.append(
                {
                    "score": 0.5,  # TODO
                    "answer": " ".join(words[word_start : word_end + 1]),
                    "start": word_start,
                    "end": word_end,
                }
            )

    return results


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

    def _sanitize_parameters(self, top_k=None, padding=None, truncation=None, words=None, boxes=None, **kwargs):
        preprocess_params, postprocess_params = {}, {}
        if padding is not None:
            preprocess_params["padding"] = padding
        if truncation is not None:
            preprocess_params["truncation"] = truncation
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        if words is not None or boxes is not None:
            if words is None or boxes is None:
                raise ValueError("Must provide both words and boxes if providing either")
            preprocess_params["words"] = words
            preprocess_params["boxes"] = boxes

        return preprocess_params, {}, postprocess_params

    def __call__(
        self,
        image: [Optional[Union["Image.Image", str]]] = None,
        question: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
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
        if (isinstance(image, (Image.Image, str)) or image is None) and isinstance(question, str):
            inputs = {"question": question}
            if image is not None:
                inputs["image"] = image
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

    def preprocess(self, inputs, padding=False, truncation=False, words=[], boxes=[]):
        image_features = {}
        if "image" in inputs:
            image = load_image(inputs["image"])
            image_features = self.feature_extractor(images=image, return_tensors=self.framework)

        # TODO: If the image is specified, depending on the tokenizer, we should update inputs
        # with the words/bounding boxes.
        #
        # TODO: LayoutLMv1 does not come with a feature extractor that can run OCR, but v2 and v3 do.
        # We could either (a) require users of v1 pass in words (or implement a v1 feature extractor)
        # or (b) build the OCR implementation into this pipeline (and expect users of v2 and v3 to
        # instantiate the pipeline with feature extractors that have OCR disabled).

        if not ("image" in inputs or "words" in inputs):
            raise ValueError("Must provide at least one of an image or words/bounding boxes")

        question = inputs["question"]
        text_pair = None

        extra_tokenizer_params = {}
        if "words" in inputs:
            # TODO: Can we refactor and share (or inherit from) the QuestionAnweringPipeline? I think we want to do
            # the same thing with the tokenizer, except we have a few extra arguments (e.g. the bounding boxes).
            extra_tokenizer_params = {
                "return_token_type_ids": True,
                "return_attention_mask": True,
                "is_split_into_words": True,
            }
            padding = "max_length"
            truncation = True
            question = [question]
            text_pair = inputs["words"]

        encoding = self.tokenizer(
            text=question,
            text_pair=text_pair,
            return_tensors=self.framework,
            padding=padding,
            truncation=truncation,
            **extra_tokenizer_params,
        )

        num_spans = len(encoding["input_ids"])

        if "boxes" in inputs:
            boxes = inputs["boxes"]
            bbox = []
            for batch_index in range(num_spans):
                for i, s, w in zip(
                    encoding.input_ids[batch_index],
                    encoding.sequence_ids(batch_index),
                    encoding.word_ids(batch_index),
                ):
                    if s == 1:
                        bbox.append(boxes[w])
                    elif i == self.tokenizer.sep_token_id:
                        bbox.append([1000] * 4)
                    else:
                        bbox.append([0] * 4)

            if self.framework == "tf":
                # TODO implement
                pass
            elif self.framework == "pt":
                encoding["bbox"] = torch.tensor([bbox])

        # TODO: Handle multiple spans. We'll basically want to duplicate the image features for each span
        # (we can then also remove this assert)
        assert len(image_features) == 0 or list(image_features.items())[0].size(0) == list(encoding.items())[0].size(0)
        encoding.update(image_features)

        # TODO: I think it's cleaner to place the encoding and other context in the dict in separate keys
        # instead of flat at the top level. I'm happy to undo this though if it's not in line with
        # other parts of the code.
        return {
            "encoding": encoding,
            "word_ids": [encoding.word_ids(i) for i in range(len(encoding["input_ids"]))],
            "sequence_ids": [encoding.sequence_ids(i) for i in range(len(encoding["input_ids"]))],
            "inputs": inputs,
        }

    def _forward(self, inputs):
        model_inputs = {k: inputs["encoding"][k] for k in self.tokenizer.model_input_names}
        model_outputs = self.model(**model_inputs)
        return {"outputs": model_outputs, "inputs": inputs}

    def postprocess(self, result, top_k=5):
        model_outputs = result["outputs"]

        # TODO: Is there a better way to do this? I tried using
        # isinstance(model_outputs, SequenceClassifierOutput) but that thinks model_outputs
        # is transformers.utils.generic.ModelOutput
        if "logits" in model_outputs:
            return postprocess_sequence_output(self.model, model_outputs, self.framework, top_k)
        elif "start_logits" in model_outputs and "end_logits" in model_outputs:
            return postprocess_qa_output(
                self.model,
                model_outputs,
                result["inputs"]["word_ids"],
                result["inputs"]["inputs"]["words"],
                self.framework,
                top_k,
            )
        else:
            assert False, "Unknown output format"
