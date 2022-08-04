from typing import List, Optional, Tuple, Union

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING


logger = logging.get_logger(__name__)

# TODO:
#   1. Should we make this a chunk pipeline for consistency with QAPipeline?
#   2. Should we switch padding default to "do_not_pad" and do the same "unsqueeze" trick as the qa pipeline?


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
class DocumentQuestionAnsweringPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING)

    # TODO: Borrow params from QA pipeline probably
    def _sanitize_parameters(
        self,
        padding=None,
        word_boxes=None,  # TODO: Maybe rename to just words
        doc_stride=None,
        max_question_len=None,
        max_seq_len=None,
        top_k=None,
        **kwargs,
    ):
        preprocess_params, postprocess_params = {}, {}
        if padding is not None:
            preprocess_params["padding"] = padding
        if doc_stride is not None:
            preprocess_params["doc_stride"] = doc_stride
        if max_question_len is not None:
            preprocess_params["max_question_len"] = max_question_len
        if max_seq_len is not None:
            preprocess_params["max_seq_len"] = max_seq_len

        if top_k is not None:
            postprocess_params["top_k"] = top_k

        return preprocess_params, {}, postprocess_params

    # TODO: Borrow params from QA pipeline probably
    def __call__(
        self,
        image: Union["Image.Image", str],
        question: Optional[str] = None,
        word_boxes: Tuple[str, List[float]] = None,
        **kwargs,
    ):
        if isinstance(question, str):
            inputs = {"question": question, "image": image, "word_boxes": word_boxes}
        else:
            inputs = image
        return super().__call__(inputs, **kwargs)

    def preprocess(
        self,
        input,
        padding="max_length",
        doc_stride=None,
        max_question_len=64,
        max_seq_len=None,
        word_boxes: Tuple[str, List[float]] = None,
    ):
        # NOTE: This code mirrors the code is question_answering.py
        if max_seq_len is None:
            # TODO: LayoutLM's stride is 512 by default. Is it ok to use that as the min
            # instead of 384?
            max_seq_len = min(self.tokenizer.model_max_length, 512)
        if doc_stride is None:
            doc_stride = min(max_seq_len // 2, 128)

        # TODO: Run OCR on the image if words is None
        # I'll remove this assert once I implement OCR
        assert input["word_boxes"], "This should be fixed and replaced with OCR"

        words = [x[0] for x in input["word_boxes"]]
        boxes = [x[1] for x in input["word_boxes"]]

        encoding = self.tokenizer(
            text=[input["question"]],
            text_pair=words,
            padding=padding,
            truncation="only_second",
            max_length=max_seq_len,
            stride=doc_stride,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            is_split_into_words=True,
            # TODO: We should remove this if we want to remove the default padding
            # and do an unsqueeze like the QA pipeline
            return_tensors=self.framework,
        )

        num_spans = len(encoding["input_ids"])

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
            raise ValueError("Tensorflow preprocessing for DocumentQuestionAnsweringPipeline")
        elif self.framework == "pt":
            encoding["bbox"] = torch.tensor([bbox])

        word_ids = [encoding.word_ids(i) for i in range(num_spans)]

        encoding.pop("overflow_to_sample_mapping", None)
        return {
            **encoding,
            "word_ids": word_ids,
            "words": words,
        }

    def _forward(self, model_inputs):
        word_ids = model_inputs.pop("word_ids", None)
        words = model_inputs.pop("words", None)

        model_outputs = self.model(**model_inputs)

        model_outputs["word_ids"] = word_ids
        model_outputs["words"] = words
        return model_outputs

    def postprocess(self, model_outputs, top_k=5):
        return postprocess_qa_output(
            self.model,
            model_outputs,
            model_outputs["word_ids"],
            model_outputs["words"],
            self.framework,
            top_k,
        )
