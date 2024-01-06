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


# List of models that support FIM
SUPPORTED_MODELS = [
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-13b-hf",
    "codellama/CodeLlama-34b-hf",
    "codellama/CodeLlama-7b-Python-hf",
    "codellama/CodeLlama-13b-Python-hf",
    "codellama/CodeLlama-34b-Python-hf",
]


@add_end_docstrings(PIPELINE_INIT_ARGS)
class FIMPipeline(Pipeline):
    """
    Fill-in-the-middle Language generation pipeline. This pipeline predicts the middle part of a text given it's prefix and suffix
    Example:
    ```python
    >>> from transformers import pipeline
    >>> PROMPT = '''
        def fibonacci(x: int) -> int:
            <FILL_ME>
            return fib(x-1) + fib(x-2)
    '''
    >>> generator = pipeline(model="codellama/CodeLlama-7b-hf")
    >>> generator(PROMPT, do_sample=False)
    [{'generated_text': "\ndef fibonacci(x: int) -> int:\n\tif x == 0:\n\t\treturn 0\n\tif x == 1:\n\t\treturn 1\n\telse:\n\t\treturn fib(x-1) + fib(x-2)\n"}]
    ```
    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
    generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
    text generation parameters in [Text generation strategies](../generation_strategies) and [Text
    generation](text_generation).
    This language generation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"fill-in-middle"`.
    The models that this pipeline can use are models that have been trained with an autoregressive language modeling
    objective, which includes the uni-directional models in the library (e.g. gpt2). See the list of available models
    on [huggingface.co/models](https://huggingface.co/models?filter=text-generation).
    """

    DEFAULT_INFILL_TOKEN = "<FILL_ME>"
    DEFAULT_PREFIX_TOKEN = "<fim_prefix>"
    DEFAULT_SUFFIX_TOKEN = "<fim_suffix>"
    DEFAULT_MIDDLE_TOKEN = "<fim_middle>"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if the provided model is among the supported ones
        model_name = kwargs.get("model", None)
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"The model '{model_name}' is not supported for the Fill-in-Middle task."
            )

        self.check_model_type(
            TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        )
        if "prefix" not in self._preprocess_params:
            # This is very specific. The logic is quite complex and needs to be done
            # as a "default".
            # It also defines both some preprocess_kwargs and generate_kwargs
            # which is why we cannot put them in their respective methods.
            prefix = None
            if self.model.config.prefix is not None:
                prefix = self.model.config.prefix

            if prefix is not None:
                # Recalculate some generate_kwargs linked to prefix.
                preprocess_params, forward_params, _ = self._sanitize_parameters(
                    prefix=prefix, **self._forward_params
                )
                self._preprocess_params = {
                    **self._preprocess_params,
                    **preprocess_params,
                }
                self._forward_params = {**self._forward_params, **forward_params}

    def ensure_infill_token(self, input_ids, infill_token_id):
        """
        Ensures that the input has an infill token in it. If not, throw a value error.
        """
        if infill_token_id not in input_ids["input_ids"]:
            raise ValueError(
                f"The infill token '{self.tokenizer.decode(infill_token_id)}' was not found in the input. Please add it to the input."
            )

    def extract_prefix_suffix(self, input_text, infill_token):
        """
        Extracts the prefix and suffix from the input text.
        """
        prefix, _, suffix = input_text.partition(infill_token)
        return prefix, suffix

    def _sanitize_parameters(
        self,
        infill_token=None,
        return_infilled_sequence=True,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        prefix=None,
        handle_long_generation=None,
        stop_sequence=None,
        add_special_tokens=False,
        **generate_kwargs,
    ):
        preprocess_params = {"add_special_tokens": add_special_tokens}
        preprocess_params["infill_token"] = infill_token or self.DEFAULT_INFILL_TOKEN

        if prefix is not None:
            preprocess_params["prefix"] = prefix
        if prefix:
            prefix_inputs = self.tokenizer(
                prefix,
                padding=False,
                add_special_tokens=add_special_tokens,
                return_tensors=self.framework,
            )
            generate_kwargs["prefix_length"] = prefix_inputs["input_ids"].shape[-1]

        if handle_long_generation is not None:
            if handle_long_generation not in {"hole"}:
                raise ValueError(
                    f"{handle_long_generation} is not a valid value for `handle_long_generation` parameter expected"
                    " [None, 'hole']"
                )
            preprocess_params["handle_long_generation"] = handle_long_generation

        preprocess_params.update(generate_kwargs)
        forward_params = generate_kwargs

        postprocess_params = {}
        postprocess_params["return_infilled_sequence"] = return_infilled_sequence

        if return_full_text is not None and return_type is None:
            if return_text is not None:
                raise ValueError(
                    "`return_text` is mutually exclusive with `return_full_text`"
                )
            if return_tensors is not None:
                raise ValueError(
                    "`return_full_text` is mutually exclusive with `return_tensors`"
                )
            return_type = (
                ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
            )
        if return_tensors is not None and return_type is None:
            if return_text is not None:
                raise ValueError(
                    "`return_text` is mutually exclusive with `return_tensors`"
                )
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params[
                "clean_up_tokenization_spaces"
            ] = clean_up_tokenization_spaces

        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(
                stop_sequence, add_special_tokens=False
            )
            if len(stop_sequence_ids) > 1:
                warnings.warn(
                    "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                    " the stop sequence will be used as the stop sequence string in the interim."
                )
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

        return preprocess_params, forward_params, postprocess_params

    def preprocess(
        self,
        prompt_text,
        infill_token,
        prefix="",
        handle_long_generation=None,
        add_special_tokens=False,
        **generate_kwargs,
    ):
        # Extract prefix and suffix
        prefix, suffix = self.extract_prefix_suffix(prompt_text, infill_token)

        # Assemble the inputs in the PSM (Prefix-Suffix-Middle) format
        prompt_text = (
            self.DEFAULT_PREFIX_TOKEN
            + prefix
            + self.DEFAULT_SUFFIX_TOKEN
            + suffix
            + self.DEFAULT_MIDDLE_TOKEN
        )

        inputs = self.tokenizer(
            prefix + prompt_text,
            padding=False,
            add_special_tokens=add_special_tokens,
            return_tensors=self.framework,
        )
        inputs["prompt_text"] = prompt_text

        # Ensure that infill token is in the input
        self.ensure_infill_token(
            inputs, self.tokenizer(generate_kwargs["infill_token"])
        )

        if handle_long_generation == "hole":
            cur_len = inputs["input_ids"].shape[-1]
            if "max_new_tokens" in generate_kwargs:
                new_tokens = generate_kwargs["max_new_tokens"]
            else:
                new_tokens = (
                    generate_kwargs.get("max_length", self.model.config.max_length)
                    - cur_len
                )
                if new_tokens < 0:
                    raise ValueError("We cannot infer how many new tokens are expected")
            if cur_len + new_tokens > self.tokenizer.model_max_length:
                keep_length = self.tokenizer.model_max_length - new_tokens
                if keep_length <= 0:
                    raise ValueError(
                        "We cannot use `hole` to handle this generation the number of desired tokens exceeds the"
                        " models max length"
                    )

                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][
                        :, -keep_length:
                    ]

        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # Handle empty prompts by setting id and attention mask to None
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            batch_size = 1
        else:
            batch_size = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        prefix_len = generate_kwargs.pop("prefix_length", 0)
        if prefix_len > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = (
                    generate_kwargs.get("max_length") or self.model.config.max_length
                )
                generate_kwargs["max_length"] += prefix_len
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_len

        generated_sequence = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        generated_bs = generated_sequence.shape[0]

        # Reshape the generated sequence depending on the framework
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(
                batch_size, generated_bs // batch_size, *generated_sequence.shape[1:]
            )
        elif self.framework == "tf":
            generated_sequence = tf.reshape(
                generated_sequence,
                (batch_size, generated_bs // batch_size, *generated_sequence.shape[1:]),
            )

        return {
            "generated_sequence": generated_sequence,
            "input_ids": input_ids,
            "prompt_text": prompt_text,
        }

    def __call__(self, text_inputs, **kwargs):
        """
        Infill the middle of a sequence given a sequence with an infill token in it.

        Args:
            args (`str` or `List[str]`):
                One or several prompts (or one list of prompts) to complete.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to return the tensors of predictions (as token indices) in the outputs. If set to
                `True`, the decoded text is not returned.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to return the decoded texts in the outputs.
            return_full_text (`bool`, *optional*, defaults to `True`):
                If set to `False` only added text is returned, otherwise the full text is returned. Only meaningful if
                *return_text* is set to True.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            prefix (`str`, *optional*):
                Prefix added to prompt.
            handle_long_generation (`str`, *optional*):
                By default, this pipelines does not handle long generation (ones that exceed in one form or the other
                the model maximum length). There is no perfect way to adress this (more info
                :https://github.com/huggingface/transformers/issues/14033#issuecomment-948385227). This provides common
                strategies to work around that problem depending on your use case.

                - `None` : default strategy where nothing in particular happens
                - `"hole"`: Truncates left of input, and leaves a gap wide enough to let generation happen (might
                  truncate a lot of the prompt and not suitable when generation exceed the model capacity)

            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Return:
            A list or a list of list of `dict`: Returns one of the following dictionaries (cannot return a combination
            of both `generated_text` and `generated_token_ids`):

            - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
            - **generated_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
              ids of the generated text.
        """
        return super().__call__(text_inputs, **kwargs)

    def postprocess(
        self,
        model_outputs,
        return_infilled_sequence,
        return_type=ReturnType.FULL_TEXT,
        clean_up_tokenization_spaces=True,
    ):
        generated_sequence = model_outputs["generated_sequence"][0].numpy().tolist()
        input_ids = model_outputs["input_ids"]
        prompt_text = model_outputs["prompt_text"]

        records = []

        for sequence in generated_sequence:
            pass
