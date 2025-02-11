from typing import Dict

from ..utils import add_end_docstrings
from .base import GenericTensor, Pipeline, build_pipeline_init_args


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=True, supports_binary_output=False),
    r"""
        tokenize_kwargs (`dict`, *optional*):
                Additional dictionary of keyword arguments passed along to the tokenizer.
        return_tensors (`bool`, *optional*):
            If `True`, returns a tensor according to the specified framework, otherwise returns a list.""",
)
class FeatureExtractionPipeline(Pipeline):
    """
    Feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base
    transformer, which can be used as features in downstream tasks.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> extractor = pipeline(model="google-bert/bert-base-uncased", task="feature-extraction")
    >>> result = extractor("This is a simple test.", return_tensors=True)
    >>> result.shape  # This is a tensor of shape [1, sequence_length, hidden_dimension] representing the input string.
    torch.Size([1, 8, 768])
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This feature extraction pipeline can currently be loaded from [`pipeline`] using the task identifier:
    `"feature-extraction"`.

    All models may be used for this pipeline. See a list of all models, including community-contributed models on
    [huggingface.co/models](https://huggingface.co/models).
    """

    def _sanitize_parameters(self, truncation=None, tokenize_kwargs=None, return_tensors=None, **kwargs):
        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        if truncation is not None:
            if "truncation" in tokenize_kwargs:
                raise ValueError(
                    "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
                )
            tokenize_kwargs["truncation"] = truncation

        preprocess_params = tokenize_kwargs

        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, **tokenize_kwargs) -> Dict[str, GenericTensor]:
        model_inputs = self.tokenizer(inputs, return_tensors=self.framework, **tokenize_kwargs)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, return_tensors=False):
        # [0] is the first available tensor, logits or last_hidden_state.
        if return_tensors:
            return model_outputs[0]
        if self.framework == "pt":
            return model_outputs[0].tolist()
        elif self.framework == "tf":
            return model_outputs[0].numpy().tolist()

    def __call__(self, *args, **kwargs):
        """
        Extract the features of the input(s).

        Args:
            args (`str` or `List[str]`): One or several texts (or one list of texts) to get the features of.

        Return:
            A nested list of `float`: The features computed by the model.
        """
        return super().__call__(*args, **kwargs)
