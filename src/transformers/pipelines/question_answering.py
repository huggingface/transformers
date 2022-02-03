import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from ..data import SquadExample, SquadFeatures, squad_convert_examples_to_features
from ..file_utils import PaddingStrategy, add_end_docstrings, is_tf_available, is_torch_available
from ..modelcard import ModelCard
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, ChunkPipeline


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel

if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING


class QuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped to
    internal [`SquadExample`].

    QuestionAnsweringArgumentHandler manages all the possible to create a [`SquadExample`] from the command-line
    supplied arguments.
    """

    def normalize(self, item):
        if isinstance(item, SquadExample):
            return item
        elif isinstance(item, dict):
            for k in ["question", "context"]:
                if k not in item:
                    raise KeyError("You need to provide a dictionary with keys {question:..., context:...}")
                elif item[k] is None:
                    raise ValueError(f"`{k}` cannot be None")
                elif isinstance(item[k], str) and len(item[k]) == 0:
                    raise ValueError(f"`{k}` cannot be empty")

            return QuestionAnsweringPipeline.create_sample(**item)
        raise ValueError(f"{item} argument needs to be of type (SquadExample, dict)")

    def __call__(self, *args, **kwargs):
        # Detect where the actual inputs are
        if args is not None and len(args) > 0:
            if len(args) == 1:
                inputs = args[0]
            elif len(args) == 2 and {type(el) for el in args} == {str}:
                inputs = [{"question": args[0], "context": args[1]}]
            else:
                inputs = list(args)
        # Generic compatibility with sklearn and Keras
        # Batched data
        elif "X" in kwargs:
            inputs = kwargs["X"]
        elif "data" in kwargs:
            inputs = kwargs["data"]
        elif "question" in kwargs and "context" in kwargs:
            if isinstance(kwargs["question"], list) and isinstance(kwargs["context"], str):
                inputs = [{"question": Q, "context": kwargs["context"]} for Q in kwargs["question"]]
            elif isinstance(kwargs["question"], list) and isinstance(kwargs["context"], list):
                if len(kwargs["question"]) != len(kwargs["context"]):
                    raise ValueError("Questions and contexts don't have the same lengths")

                inputs = [{"question": Q, "context": C} for Q, C in zip(kwargs["question"], kwargs["context"])]
            elif isinstance(kwargs["question"], str) and isinstance(kwargs["context"], str):
                inputs = [{"question": kwargs["question"], "context": kwargs["context"]}]
            else:
                raise ValueError("Arguments can't be understood")
        else:
            raise ValueError(f"Unknown arguments {kwargs}")

        # Normalize inputs
        if isinstance(inputs, dict):
            inputs = [inputs]
        elif isinstance(inputs, Iterable):
            # Copy to avoid overriding arguments
            inputs = [i for i in inputs]
        else:
            raise ValueError(f"Invalid arguments {kwargs}")

        for i, item in enumerate(inputs):
            inputs[i] = self.normalize(item)

        return inputs


@add_end_docstrings(PIPELINE_INIT_ARGS)
class QuestionAnsweringPipeline(ChunkPipeline):
    """
    Question Answering pipeline using any `ModelForQuestionAnswering`. See the [question answering
    examples](../task_summary#question-answering) for more information.

    This question answering pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a question answering task. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=question-answering).
    """

    default_input_names = "question,context"
    handle_impossible_answer = False

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        device: int = -1,
        task: str = "",
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            device=device,
            task=task,
            **kwargs,
        )

        self._args_parser = QuestionAnsweringArgumentHandler()
        self.check_model_type(
            TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING if self.framework == "tf" else MODEL_FOR_QUESTION_ANSWERING_MAPPING
        )

    @staticmethod
    def create_sample(
        question: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Union[SquadExample, List[SquadExample]]:
        """
        QuestionAnsweringPipeline leverages the [`SquadExample`] internally. This helper method encapsulate all the
        logic for converting question(s) and context(s) to [`SquadExample`].

        We currently support extractive question answering.

        Arguments:
            question (`str` or `List[str]`): The question(s) asked.
            context (`str` or `List[str]`): The context(s) in which we will look for the answer.

        Returns:
            One or a list of [`SquadExample`]: The corresponding [`SquadExample`] grouping question and context.
        """
        if isinstance(question, list):
            return [SquadExample(None, q, c, None, None, None) for q, c in zip(question, context)]
        else:
            return SquadExample(None, question, context, None, None, None)

    def _sanitize_parameters(
        self,
        padding=None,
        topk=None,
        top_k=None,
        doc_stride=None,
        max_answer_len=None,
        max_seq_len=None,
        max_question_len=None,
        handle_impossible_answer=None,
        **kwargs
    ):
        # Set defaults values
        preprocess_params = {}
        if padding is not None:
            preprocess_params["padding"] = padding
        if doc_stride is not None:
            preprocess_params["doc_stride"] = doc_stride
        if max_question_len is not None:
            preprocess_params["max_question_len"] = max_question_len

        postprocess_params = {}
        if topk is not None and top_k is None:
            warnings.warn("topk parameter is deprecated, use top_k instead", UserWarning)
            top_k = topk
        if top_k is not None:
            if top_k < 1:
                raise ValueError(f"top_k parameter should be >= 1 (got {top_k})")
            postprocess_params["top_k"] = top_k
        if max_answer_len is not None:
            if max_answer_len < 1:
                raise ValueError(f"max_answer_len parameter should be >= 1 (got {max_answer_len}")
        if max_answer_len is not None:
            postprocess_params["max_answer_len"] = max_answer_len
        if handle_impossible_answer is not None:
            postprocess_params["handle_impossible_answer"] = handle_impossible_answer
        return preprocess_params, {}, postprocess_params

    def __call__(self, *args, **kwargs):
        """
        Answer the question(s) given as inputs by using the context(s).

        Args:
            args ([`SquadExample`] or a list of [`SquadExample`]):
                One or several [`SquadExample`] containing the question and context.
            X ([`SquadExample`] or a list of [`SquadExample`], *optional*):
                One or several [`SquadExample`] containing the question and context (will be treated the same way as if
                passed as the first positional argument).
            data ([`SquadExample`] or a list of [`SquadExample`], *optional*):
                One or several [`SquadExample`] containing the question and context (will be treated the same way as if
                passed as the first positional argument).
            question (`str` or `List[str]`):
                One or several question(s) (must be used in conjunction with the `context` argument).
            context (`str` or `List[str]`):
                One or several context(s) associated with the question(s) (must be used in conjunction with the
                `question` argument).
            topk (`int`, *optional*, defaults to 1):
                The number of answers to return (will be chosen by order of likelihood). Note that we return less than
                topk answers if there are not enough options available within the context.
            doc_stride (`int`, *optional*, defaults to 128):
                If the context is too long to fit with the question for the model, it will be split in several chunks
                with some overlap. This argument controls the size of that overlap.
            max_answer_len (`int`, *optional*, defaults to 15):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_seq_len (`int`, *optional*, defaults to 384):
                The maximum length of the total sentence (context + question) after tokenization. The context will be
                split in several chunks (using `doc_stride`) if needed.
            max_question_len (`int`, *optional*, defaults to 64):
                The maximum length of the question after tokenization. It will be truncated if needed.
            handle_impossible_answer (`bool`, *optional*, defaults to `False`):
                Whether or not we accept impossible as an answer.

        Return:
            A `dict` or a list of `dict`: Each result comes as a dictionary with the following keys:

            - **score** (`float`) -- The probability associated to the answer.
            - **start** (`int`) -- The character start index of the answer (in the tokenized version of the input).
            - **end** (`int`) -- The character end index of the answer (in the tokenized version of the input).
            - **answer** (`str`) -- The answer to the question.
        """

        # Convert inputs to features
        examples = self._args_parser(*args, **kwargs)
        if len(examples) == 1:
            return super().__call__(examples[0], **kwargs)
        return super().__call__(examples, **kwargs)

    def preprocess(self, example, padding="do_not_pad", doc_stride=None, max_question_len=64, max_seq_len=None):

        if max_seq_len is None:
            max_seq_len = min(self.tokenizer.model_max_length, 384)
        if doc_stride is None:
            doc_stride = min(max_seq_len // 2, 128)

        if not self.tokenizer.is_fast:
            features = squad_convert_examples_to_features(
                examples=[example],
                tokenizer=self.tokenizer,
                max_seq_length=max_seq_len,
                doc_stride=doc_stride,
                max_query_length=max_question_len,
                padding_strategy=PaddingStrategy.MAX_LENGTH,
                is_training=False,
                tqdm_enabled=False,
            )
        else:
            # Define the side we want to truncate / pad and the text/pair sorting
            question_first = self.tokenizer.padding_side == "right"

            encoded_inputs = self.tokenizer(
                text=example.question_text if question_first else example.context_text,
                text_pair=example.context_text if question_first else example.question_text,
                padding=padding,
                truncation="only_second" if question_first else "only_first",
                max_length=max_seq_len,
                stride=doc_stride,
                return_tensors="np",
                return_token_type_ids=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            # When the input is too long, it's converted in a batch of inputs with overflowing tokens
            # and a stride of overlap between the inputs. If a batch of inputs is given, a special output
            # "overflow_to_sample_mapping" indicate which member of the encoded batch belong to which original batch sample.
            # Here we tokenize examples one-by-one so we don't need to use "overflow_to_sample_mapping".
            # "num_span" is the number of output samples generated from the overflowing tokens.
            num_spans = len(encoded_inputs["input_ids"])

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # We put 0 on the tokens from the context and 1 everywhere else (question and special tokens)
            p_mask = np.asarray(
                [
                    [tok != 1 if question_first else 0 for tok in encoded_inputs.sequence_ids(span_id)]
                    for span_id in range(num_spans)
                ]
            )

            # keep the cls_token unmasked (some models use it to indicate unanswerable questions)
            if self.tokenizer.cls_token_id is not None:
                cls_index = np.nonzero(encoded_inputs["input_ids"] == self.tokenizer.cls_token_id)
                p_mask[cls_index] = 0

            features = []
            for span_idx in range(num_spans):
                input_ids_span_idx = encoded_inputs["input_ids"][span_idx]
                attention_mask_span_idx = (
                    encoded_inputs["attention_mask"][span_idx] if "attention_mask" in encoded_inputs else None
                )
                token_type_ids_span_idx = (
                    encoded_inputs["token_type_ids"][span_idx] if "token_type_ids" in encoded_inputs else None
                )
                submask = p_mask[span_idx]
                if isinstance(submask, np.ndarray):
                    submask = submask.tolist()
                features.append(
                    SquadFeatures(
                        input_ids=input_ids_span_idx,
                        attention_mask=attention_mask_span_idx,
                        token_type_ids=token_type_ids_span_idx,
                        p_mask=submask,
                        encoding=encoded_inputs[span_idx],
                        # We don't use the rest of the values - and actually
                        # for Fast tokenizer we could totally avoid using SquadFeatures and SquadExample
                        cls_index=None,
                        token_to_orig_map={},
                        example_index=0,
                        unique_id=0,
                        paragraph_len=0,
                        token_is_max_context=0,
                        tokens=[],
                        start_position=0,
                        end_position=0,
                        is_impossible=False,
                        qas_id=None,
                    )
                )

        for i, feature in enumerate(features):
            fw_args = {}
            others = {}
            model_input_names = self.tokenizer.model_input_names + ["p_mask"]

            for k, v in feature.__dict__.items():
                if k in model_input_names:
                    if self.framework == "tf":
                        tensor = tf.constant(v)
                        if tensor.dtype == tf.int64:
                            tensor = tf.cast(tensor, tf.int32)
                        fw_args[k] = tf.expand_dims(tensor, 0)
                    elif self.framework == "pt":
                        tensor = torch.tensor(v)
                        if tensor.dtype == torch.int32:
                            tensor = tensor.long()
                        fw_args[k] = tensor.unsqueeze(0)
                else:
                    others[k] = v

            is_last = i == len(features) - 1
            yield {"example": example, "is_last": is_last, **fw_args, **others}

    def _forward(self, inputs):
        example = inputs["example"]
        model_inputs = {k: inputs[k] for k in self.tokenizer.model_input_names}
        start, end = self.model(**model_inputs)[:2]
        return {"start": start, "end": end, "example": example, **inputs}

    def postprocess(
        self,
        model_outputs,
        top_k=1,
        handle_impossible_answer=False,
        max_answer_len=15,
    ):
        min_null_score = 1000000  # large and positive
        answers = []
        for output in model_outputs:
            start_ = output["start"]
            end_ = output["end"]
            example = output["example"]

            # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
            undesired_tokens = np.abs(np.array(output["p_mask"]) - 1)

            if output.get("attention_mask", None) is not None:
                undesired_tokens = undesired_tokens & output["attention_mask"].numpy()

            # Generate mask
            undesired_tokens_mask = undesired_tokens == 0.0

            # Make sure non-context indexes in the tensor cannot contribute to the softmax
            start_ = np.where(undesired_tokens_mask, -10000.0, start_)
            end_ = np.where(undesired_tokens_mask, -10000.0, end_)

            # Normalize logits and spans to retrieve the answer
            start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
            end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))

            if handle_impossible_answer:
                min_null_score = min(min_null_score, (start_[0, 0] * end_[0, 0]).item())

            # Mask CLS
            start_[0, 0] = end_[0, 0] = 0.0

            starts, ends, scores = self.decode(start_, end_, top_k, max_answer_len, undesired_tokens)
            if not self.tokenizer.is_fast:
                char_to_word = np.array(example.char_to_word_offset)

                # Convert the answer (tokens) back to the original text
                # Score: score from the model
                # Start: Index of the first character of the answer in the context string
                # End: Index of the character following the last character of the answer in the context string
                # Answer: Plain text of the answer
                for s, e, score in zip(starts, ends, scores):
                    token_to_orig_map = output["token_to_orig_map"]
                    answers.append(
                        {
                            "score": score.item(),
                            "start": np.where(char_to_word == token_to_orig_map[s])[0][0].item(),
                            "end": np.where(char_to_word == token_to_orig_map[e])[0][-1].item(),
                            "answer": " ".join(example.doc_tokens[token_to_orig_map[s] : token_to_orig_map[e] + 1]),
                        }
                    )
            else:
                # Convert the answer (tokens) back to the original text
                # Score: score from the model
                # Start: Index of the first character of the answer in the context string
                # End: Index of the character following the last character of the answer in the context string
                # Answer: Plain text of the answer
                question_first = bool(self.tokenizer.padding_side == "right")
                enc = output["encoding"]

                # Encoding was *not* padded, input_ids *might*.
                # It doesn't make a difference unless we're padding on
                # the left hand side, since now we have different offsets
                # everywhere.
                if self.tokenizer.padding_side == "left":
                    offset = (output["input_ids"] == self.tokenizer.pad_token_id).numpy().sum()
                else:
                    offset = 0

                # Sometimes the max probability token is in the middle of a word so:
                # - we start by finding the right word containing the token with `token_to_word`
                # - then we convert this word in a character span with `word_to_chars`
                sequence_index = 1 if question_first else 0
                for s, e, score in zip(starts, ends, scores):
                    s = s - offset
                    e = e - offset
                    try:
                        start_word = enc.token_to_word(s)
                        end_word = enc.token_to_word(e)
                        start_index = enc.word_to_chars(start_word, sequence_index=sequence_index)[0]
                        end_index = enc.word_to_chars(end_word, sequence_index=sequence_index)[1]
                    except Exception:
                        # Some tokenizers don't really handle words. Keep to offsets then.
                        start_index = enc.offsets[s][0]
                        end_index = enc.offsets[e][1]

                    answers.append(
                        {
                            "score": score.item(),
                            "start": start_index,
                            "end": end_index,
                            "answer": example.context_text[start_index:end_index],
                        }
                    )

        if handle_impossible_answer:
            answers.append({"score": min_null_score, "start": 0, "end": 0, "answer": ""})
        answers = sorted(answers, key=lambda x: x["score"], reverse=True)[:top_k]
        if len(answers) == 1:
            return answers[0]
        return answers

    def decode(
        self, start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int, undesired_tokens: np.ndarray
    ) -> Tuple:
        """
        Take the output of any `ModelForQuestionAnswering` and will generate probabilities for each span to be the
        actual answer.

        In addition, it filters out some unwanted/impossible cases like answer len being greater than max_answer_len or
        answer end position being before the starting position. The method supports output the k-best answer through
        the topk argument.

        Args:
            start (`np.ndarray`): Individual start probabilities for each token.
            end (`np.ndarray`): Individual end probabilities for each token.
            topk (`int`): Indicates how many possible answer span(s) to extract from the model output.
            max_answer_len (`int`): Maximum size of the answer to extract from the model's output.
            undesired_tokens (`np.ndarray`): Mask determining tokens that can be part of the answer
        """
        # Ensure we have batch axis
        if start.ndim == 1:
            start = start[None]

        if end.ndim == 1:
            end = end[None]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = np.tril(np.triu(outer), max_answer_len - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if topk == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < topk:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, topk)[0:topk]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        starts, ends = np.unravel_index(idx_sort, candidates.shape)[1:]
        desired_spans = np.isin(starts, undesired_tokens.nonzero()) & np.isin(ends, undesired_tokens.nonzero())
        starts = starts[desired_spans]
        ends = ends[desired_spans]
        scores = candidates[0, starts, ends]

        return starts, ends, scores

    def span_to_answer(self, text: str, start: int, end: int) -> Dict[str, Union[str, int]]:
        """
        When decoding from token probabilities, this method maps token indexes to actual word in the initial context.

        Args:
            text (`str`): The actual context to extract the answer from.
            start (`int`): The answer starting token index.
            end (`int`): The answer end token index.

        Returns:
            Dictionary like `{'answer': str, 'start': int, 'end': int}`
        """
        words = []
        token_idx = char_start_idx = char_end_idx = chars_idx = 0

        for i, word in enumerate(text.split(" ")):
            token = self.tokenizer.tokenize(word)

            # Append words if they are in the span
            if start <= token_idx <= end:
                if token_idx == start:
                    char_start_idx = chars_idx

                if token_idx == end:
                    char_end_idx = chars_idx + len(word)

                words += [word]

            # Stop if we went over the end of the answer
            if token_idx > end:
                break

            # Append the subtokenization length to the running index
            token_idx += len(token)
            chars_idx += len(word) + 1

        # Join text with spaces
        return {
            "answer": " ".join(words),
            "start": max(0, char_start_idx),
            "end": min(len(text), char_end_idx),
        }
