from typing import List, Optional, Tuple, Union

from ..utils import add_end_docstrings, is_torch_available, is_vision_available, logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from PIL import Image

    # TODO Will re-introduce when I add images back in
    # from ..image_utils import load_image

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
    # TODO: Update task_summary docs to include an example with document QA and then update the first sentence
    """
    Document Question Answering pipeline using any `AutoModelForDocumentQuestionAnswering`. See the [question answering
    examples](../task_summary#question-answering) for more information.

    This document question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"document-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a document question answering task.
    See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=document-question-answering).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING)

    # TODO: Borrow params from QA pipeline probably
    def _sanitize_parameters(
        self,
        padding=None,
        doc_stride=None,
        max_question_len=None,
        lang: Optional[str] = None,
        tesseract_config: Optional[str] = None,
        max_answer_len=None,
        max_seq_len=None,
        top_k=None,
        handle_impossible_answer=None,
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
        if lang is not None:
            preprocess_params["lang"] = lang
        if tesseract_config is not None:
            preprocess_params["tesseract_config"] = tesseract_config

        if top_k is not None:
            if top_k < 1:
                raise ValueError(f"top_k parameter should be >= 1 (got {top_k})")
            postprocess_params["top_k"] = top_k
        if max_answer_len is not None:
            if max_answer_len < 1:
                raise ValueError(f"max_answer_len parameter should be >= 1 (got {max_answer_len}")
            postprocess_params["max_answer_len"] = max_answer_len
        if handle_impossible_answer is not None:
            postprocess_params["handle_impossible_answer"] = handle_impossible_answer

        return preprocess_params, {}, postprocess_params

    def __call__(
        self,
        image: Union["Image.Image", str],
        question: Optional[str] = None,
        word_boxes: Tuple[str, List[float]] = None,
        **kwargs,
    ):
        # TODO:
        #   - Should we attempt to support a batch of inputs like the question answering pipeline?
        #   - Implement top-k (may come for free when integrating the QA post processor)
        """
        Answer the question(s) given as inputs by using the context(s). The pipeline accepts an image and question, as
        well as an optional list of (word, box) tuples which represent the text in the document. If the `word_boxes`
        are not provided, it will use the Tesseract OCR engine (if available) to extract the words and boxes
        automatically.

        You can invoke the pipeline several ways:

        - `pipeline(image=image, question=question)`
        - `pipeline(image=image, question=question, word_boxes=word_boxes)`
        - `pipeline([{"image": image, "question": question}])`
        - `pipeline([{"image": image, "question": question, "word_boxes": word_boxes}])`

        Args:
            image (`str` or `PIL.Image`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. If given a single image, it can be
                broadcasted to multiple questions.
            question (`str`):
                A question to ask of the document.
            word_boxes (`List[str, Tuple[float, float, float, float]]`, *optional*):
                A list of words and bounding boxes (normalized 0->1000). If you provide this optional input, then the
                pipeline will use these words and boxes instead of running OCR on the image to derive them. This allows
                you to reuse OCR'd results across many invocations of the pipeline without having to re-run it each
                time.
            top_k (`int`, *optional*, defaults to 1):
                The number of answers to return (will be chosen by order of likelihood). Note that we return less than
                top_k answers if there are not enough options available within the context.
            doc_stride (`int`, *optional*, defaults to 128):
                If the words in the document are too long to fit with the question for the model, it will be split in
                several chunks with some overlap. This argument controls the size of that overlap.
            max_answer_len (`int`, *optional*, defaults to 15):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_seq_len (`int`, *optional*, defaults to 384):
                The maximum length of the total sentence (context + question) in tokens of each chunk passed to the
                model. The context will be split in several chunks (using `doc_stride` as overlap) if needed.
            max_question_len (`int`, *optional*, defaults to 64):
                The maximum length of the question after tokenization. It will be truncated if needed.
            handle_impossible_answer (`bool`, *optional*, defaults to `False`):
                Whether or not we accept impossible as an answer.
            lang (`str`, *optional*):
                Language to use while running OCR. Defaults to english.
            tesseract_config (`str`, *optional*):
                Additional flags to pass to tesseract while running OCR.

        Return:
            A `dict` or a list of `dict`: Each result comes as a dictionary with the following keys:

            - **score** (`float`) -- The probability associated to the answer.
            - **start** (`int`) -- The start word index of the answer (in the OCR'd version of the input or provided
              `word_boxes`).
            - **end** (`int`) -- The end word index of the answer (in the OCR'd version of the input or provided
              `word_boxes`).
            - **answer** (`str`) -- The answer to the question.
        """
        if isinstance(question, str):
            inputs = {"question": question, "image": image, "word_boxes": word_boxes}
        else:
            inputs = image
        return super().__call__(inputs, **kwargs)

    # TODO:
    #   1) Use apply_tesseract from layoutlmv3/feature_extraction_layoutlmv3
    def preprocess(
        self,
        input,
        padding="do_not_pad",
        doc_stride=None,
        max_question_len=64,
        max_seq_len=None,
        word_boxes: Tuple[str, List[float]] = None,
    ):
        # NOTE: This code mirrors the code in question answering and will be implemented in a follow up PR
        # to support documents with enough tokens that overflow the model's window
        #        if max_seq_len is None:
        #            # TODO: LayoutLM's stride is 512 by default. Is it ok to use that as the min
        #            # instead of 384 (which the QA model uses)?
        #            max_seq_len = min(self.tokenizer.model_max_length, 512)

        if doc_stride is not None:
            raise ValueError("Unsupported: striding inputs")
            # doc_stride = min(max_seq_len // 2, 128)

        # TODO: Run OCR on the image if words is None
        # I'll remove this assert once I implement OCR
        assert input["word_boxes"], "This should be fixed and replaced with OCR"

        words = [x[0] for x in input["word_boxes"]]
        boxes = [x[1] for x in input["word_boxes"]]

        encoding = self.tokenizer(
            text=input["question"].split(),
            text_pair=words,
            padding=padding,
            max_length=max_seq_len,
            stride=doc_stride,
            return_token_type_ids=True,
            is_split_into_words=True,
            return_tensors=self.framework,
            # TODO: In a future PR, use these feature to handle sequences whose length is longer than
            # the maximum allowed by the model. Currently, the tokenizer will produce a sequence that
            # may be too long for the model to handle.
            # truncation="only_second",
            # return_overflowing_tokens=True,
        )

        # TODO: For now, this should always be num_spans=1 given the flags we've passed in above
        num_spans = len(encoding["input_ids"])

        # For each span, place a bounding box [0,0,0,0] for question and CLS tokens, [1000,1000,1000,1000]
        # for SEP tokens, and the word's bounding box for words in the original document.
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
            raise ValueError("Unsupported: Tensorflow preprocessing for DocumentQuestionAnsweringPipeline")
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
