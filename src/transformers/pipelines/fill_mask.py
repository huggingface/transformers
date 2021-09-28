from typing import Dict

import numpy as np

from ..file_utils import add_end_docstrings, is_tf_available, is_torch_available
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, GenericTensor, Pipeline, PipelineException


if is_tf_available():
    import tensorflow as tf


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        top_k (:obj:`int`, defaults to 5):
            The number of predictions to return.
        targets (:obj:`str` or :obj:`List[str]`, `optional`):
            When passed, the model will limit the scores to the passed targets instead of looking up in the whole
            vocab. If the provided targets are not in the model vocab, they will be tokenized and the first resulting
            token will be used (with a warning, and that might be slower).

    """,
)
class FillMaskPipeline(Pipeline):
    """
    Masked language modeling prediction pipeline using any :obj:`ModelWithLMHead`. See the `masked language modeling
    examples <../task_summary.html#masked-language-modeling>`__ for more information.

    This mask filling pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"fill-mask"`.

    The models that this pipeline can use are models that have been trained with a masked language modeling objective,
    which includes the bi-directional models in the library. See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=masked-lm>`__.

    .. note::

        This pipeline only works for inputs with exactly one token masked.
    """

    def get_masked_index(self, input_ids: GenericTensor) -> np.ndarray:
        if self.framework == "tf":
            masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy()
        elif self.framework == "pt":
            masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False)
        else:
            raise ValueError("Unsupported framework")
        return masked_index

    def _ensure_exactly_one_mask_token(self, input_ids: GenericTensor) -> np.ndarray:
        masked_index = self.get_masked_index(input_ids)
        numel = np.prod(masked_index.shape)
        if numel > 1:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                f"More than one mask_token ({self.tokenizer.mask_token}) is not supported",
            )
        elif numel < 1:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                f"No mask_token ({self.tokenizer.mask_token}) found on the input",
            )

    def ensure_exactly_one_mask_token(self, model_inputs: GenericTensor):
        if isinstance(model_inputs, list):
            for model_input in model_inputs:
                self._ensure_exactly_one_mask_token(model_input["input_ids"][0])
        else:
            for input_ids in model_inputs["input_ids"]:
                self._ensure_exactly_one_mask_token(input_ids)

    def preprocess(self, inputs, return_tensors=None, **preprocess_parameters) -> Dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = self.framework
        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors)
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        model_outputs["input_ids"] = model_inputs["input_ids"][0]
        return model_outputs

    def postprocess(self, model_outputs, top_k=5, target_ids=None):
        # Cap top_k if there are targets
        if target_ids is not None and target_ids.shape[0] < top_k:
            top_k = target_ids.shape[0]
        input_ids = model_outputs["input_ids"]
        outputs = model_outputs["logits"]
        result = []

        if self.framework == "tf":
            masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy()

            # Fill mask pipeline supports only one ${mask_token} per sample

            logits = outputs[0, masked_index.item(), :]
            probs = tf.nn.softmax(logits)
            if target_ids is not None:
                probs = tf.gather_nd(probs, tf.reshape(target_ids, (-1, 1)))

            topk = tf.math.top_k(probs, k=top_k)
            values, predictions = topk.values.numpy(), topk.indices.numpy()
        else:
            masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False)
            # Fill mask pipeline supports only one ${mask_token} per sample

            logits = outputs[0, masked_index.item(), :]
            probs = logits.softmax(dim=0)
            if target_ids is not None:
                probs = probs[..., target_ids]

            values, predictions = probs.topk(top_k)

        for v, p in zip(values.tolist(), predictions.tolist()):
            tokens = input_ids.numpy()
            if target_ids is not None:
                p = target_ids[p].tolist()
            tokens[masked_index] = p
            # Filter padding out:
            tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
            result.append(
                {
                    "sequence": self.tokenizer.decode(tokens, skip_special_tokens=True),
                    "score": v,
                    "token": p,
                    "token_str": self.tokenizer.decode(p),
                }
            )
        return result

    def get_target_ids(self, targets, top_k=None):
        if isinstance(targets, str):
            targets = [targets]
        try:
            vocab = self.tokenizer.get_vocab()
        except Exception:
            vocab = {}
        target_ids = []
        for target in targets:
            id_ = vocab.get(target, None)
            if id_ is None:
                input_ids = self.tokenizer(
                    target,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    max_length=1,
                    truncation=True,
                )["input_ids"]
                if len(input_ids) == 0:
                    logger.warning(
                        f"The specified target token `{target}` does not exist in the model vocabulary. "
                        f"We cannot replace it with anything meaningful, ignoring it"
                    )
                    continue
                id_ = input_ids[0]
                # XXX: If users encounter this pass
                # it becomes pretty slow, so let's make sure
                # The warning enables them to fix the input to
                # get faster performance.
                logger.warning(
                    f"The specified target token `{target}` does not exist in the model vocabulary. "
                    f"Replacing with `{self.tokenizer.convert_ids_to_tokens(id_)}`."
                )
            target_ids.append(id_)
        target_ids = list(set(target_ids))
        if len(target_ids) == 0:
            raise ValueError("At least one target must be provided when passed.")
        target_ids = np.array(target_ids)
        return target_ids

    def _sanitize_parameters(self, top_k=None, targets=None):
        postprocess_params = {}

        if targets is not None:
            target_ids = self.get_target_ids(targets, top_k)
            postprocess_params["target_ids"] = target_ids

        if top_k is not None:
            postprocess_params["top_k"] = top_k

        if self.tokenizer.mask_token_id is None:
            raise PipelineException(
                "fill-mask", self.model.base_model_prefix, "The tokenizer does not define a `mask_token`."
            )
        return {}, {}, postprocess_params

    def __call__(self, inputs, *args, **kwargs):
        """
        Fill the masked token in the text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of prompts) with masked tokens.
            targets (:obj:`str` or :obj:`List[str]`, `optional`):
                When passed, the model will limit the scores to the passed targets instead of looking up in the whole
                vocab. If the provided targets are not in the model vocab, they will be tokenized and the first
                resulting token will be used (with a warning, and that might be slower).
            top_k (:obj:`int`, `optional`):
                When passed, overrides the number of predictions to return.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as list of dictionaries with the following keys:

            - **sequence** (:obj:`str`) -- The corresponding input with the mask token prediction.
            - **score** (:obj:`float`) -- The corresponding probability.
            - **token** (:obj:`int`) -- The predicted token id (to replace the masked one).
            - **token** (:obj:`str`) -- The predicted token (to replace the masked one).
        """
        outputs = super().__call__(inputs, **kwargs)
        if isinstance(inputs, list) and len(inputs) == 1:
            return outputs[0]
        return outputs
