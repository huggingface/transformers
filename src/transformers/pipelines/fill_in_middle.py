import enum
import warnings

from ..utils import add_end_docstrings, is_tf_available, is_torch_available
from .base import PIPELINE_INIT_ARGS, Pipeline

# Ensure availability of the required models
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

if is_tf_available():
    import tensorflow as tf
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


@add_end_docstrings(PIPELINE_INIT_ARGS)
class FIMPipeline(Pipeline):
    """
    Fill-in-Middle pipeline using specified models. This pipeline infills text in a specified placeholder
    within the input text.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> fim_generator = pipeline("fill-in-middle", model="codellama/CodeLlama-7b-hf")
    >>> fim_generator("The cat sat on the <CUSTOM_TOKEN>.", infill_token="<CUSTOM_TOKEN>")
    [{'infill_text': "mat"}]
    ```

    The Fill-in-Middle pipeline can be loaded from [`pipeline`] using the task identifier: `"fill-in-middle"`.

    This pipeline uses specific models that are capable of the FIM task.
    """

    # List of models that support FIM
    SUPPORTED_MODELS = [
        "codellama/CodeLlama-7b-hf",
        "codellama/CodeLlama-13b-hf",
        "codellama/CodeLlama-34b-hf",
        "codellama/CodeLlama-7b-Python-hf",
        "codellama/CodeLlama-13b-Python-hf",
        "codellama/CodeLlama-34b-Python-hf",
    ]

    DEFAULT_INFILL_TOKEN = "<FILL_ME>"

    def __init__(self, *args, **kwargs):
        # Check if the provided model is among the supported ones
        model_name = kwargs.get("model", None)
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"The model '{model_name}' is not supported for the Fill-in-Middle task."
            )

        super().__init__(*args, **kwargs)
        self.check_model_type(
            TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        )

    def _parse_infill_token(self, text_inputs, infill_token=None):
        """
        Parses the input text and extracts the prefix and suffix around the infill token.
        """
        if infill_token is None:
            infill_token = self.DEFAULT_INFILL_TOKEN

        if infill_token not in text_inputs:
            raise ValueError(
                f"Infill token {infill_token} not found in the input text."
            )

        prefix, _, suffix = text_inputs.partition(infill_token)
        return prefix, suffix

    def __call__(self, text_inputs, infill_token=None, **kwargs):
        """
        Infills the placeholder token in the input text.

        Args:
            text_inputs (`str`):
                The input text containing the placeholder token to be infilled.
            infill_token (`str`, optional):
                The placeholder token in the text which is to be infilled. Defaults to "<FILL_ME>".

        Return:
            A list of `dict`: Each dictionary contains:
            - **infill_text** (`str`) -- The infilled text.
        """
        prefix, suffix = self._parse_infill_token(text_inputs, infill_token)
        fim_input = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"

        # Tokenize the FIM input
        inputs = self.tokenizer(fim_input, return_tensors=self.framework)

        # Generate the infill text
        model_outputs = self.model.generate(**inputs, **kwargs)
        infill_text = self._extract_infill_text(model_outputs, prefix, suffix)

        return [{"infill_text": infill_text}]

    def _extract_infill_text(self, model_outputs, prefix, suffix):
        """
        Extracts the infill text from the model's output.
        """
        generated_sequence = model_outputs[0]
        generated_text = self.tokenizer.decode(
            generated_sequence, skip_special_tokens=True
        )

        # Extract text after <fim_middle>
        _, _, infill_text = generated_text.partition("<fim_middle>")

        # Remove any residual text after the suffix
        infill_text, _, _ = infill_text.partition(suffix)
        return infill_text.strip()
