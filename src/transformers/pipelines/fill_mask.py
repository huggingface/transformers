from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

from ..file_utils import add_end_docstrings, is_tf_available, is_torch_available
from ..modelcard import ModelCard
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Pipeline, PipelineException


GenericTensor = Union[List["GenericTensor"], "torch.Tensor", "tf.Tensor"]

if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel

if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_MASKED_LM_MAPPING

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_MASKED_LM_MAPPING


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

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        top_k=5,
        targets=None,
        task: str = "",
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=True,
            task=task,
        )

        self.check_model_type(
            TF_MODEL_FOR_MASKED_LM_MAPPING if self.framework == "tf" else MODEL_FOR_MASKED_LM_MAPPING
        )
        self.top_k = top_k
        self.targets = targets
        if self.tokenizer.mask_token_id is None:
            raise PipelineException(
                "fill-mask", self.model.base_model_prefix, "The tokenizer does not define a `mask_token`."
            )

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

    def get_model_inputs(self, inputs, *args, **kwargs) -> Dict:
        if isinstance(inputs, list) and self.tokenizer.pad_token is None:
            model_inputs = []
            for input_ in inputs:
                model_input = self._parse_and_tokenize(input_, padding=False, *args, **kwargs)
                model_inputs.append(model_input)
        else:
            model_inputs = self._parse_and_tokenize(inputs, *args, **kwargs)
        return model_inputs

    def __call__(self, inputs, *args, targets=None, top_k: Optional[int] = None, **kwargs):
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
        model_inputs = self.get_model_inputs(inputs, *args, **kwargs)
        self.ensure_exactly_one_mask_token(model_inputs)
        if isinstance(model_inputs, list):
            outputs = []
            for model_input in model_inputs:
                output = self._forward(model_input, return_tensors=True)
                outputs.append(output)

            batch_size = len(model_inputs)
        else:
            outputs = self._forward(model_inputs, return_tensors=True)
            batch_size = outputs.shape[0] if self.framework == "tf" else outputs.size(0)

        # top_k must be defined
        if top_k is None:
            top_k = self.top_k

        results = []

        if targets is None and self.targets is not None:
            targets = self.targets
        if targets is not None:
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
            # Cap top_k if there are targets
            if top_k > target_ids.shape[0]:
                top_k = target_ids.shape[0]

        for i in range(batch_size):
            if isinstance(model_inputs, list):
                input_ids = model_inputs[i]["input_ids"][0]
            else:
                input_ids = model_inputs["input_ids"][i]
            result = []

            if self.framework == "tf":
                masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy()

                # Fill mask pipeline supports only one ${mask_token} per sample

                if isinstance(outputs, list):
                    logits = outputs[i][0, masked_index.item(), :]
                else:
                    logits = outputs[i, masked_index.item(), :]
                probs = tf.nn.softmax(logits)
                if targets is not None:
                    probs = tf.gather_nd(probs, tf.reshape(target_ids, (-1, 1)))

                topk = tf.math.top_k(probs, k=top_k)
                values, predictions = topk.values.numpy(), topk.indices.numpy()
            else:
                masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False)
                # Fill mask pipeline supports only one ${mask_token} per sample

                if isinstance(outputs, list):
                    logits = outputs[i][0, masked_index.item(), :]
                else:
                    logits = outputs[i, masked_index.item(), :]
                probs = logits.softmax(dim=0)
                if targets is not None:
                    probs = probs[..., target_ids]

                values, predictions = probs.topk(top_k)

            for v, p in zip(values.tolist(), predictions.tolist()):
                tokens = input_ids.numpy()
                if targets is not None:
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

            # Append
            results += [result]

        if len(results) == 1:
            return results[0]
        return results
