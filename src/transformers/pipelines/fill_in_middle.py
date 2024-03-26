import enum
import warnings

from ..utils import add_end_docstrings, is_tf_available, is_torch_available
from .base import PIPELINE_INIT_ARGS, Pipeline


# Ensure availability of the required models
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

if is_tf_available():
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


# List of models that support FIM and
SUPPORTED_MODELS = {
    "bigcode/starcoder": ("<fim_prefix>", "<fim_middle>", "<fim_suffix>"),
    "codellama/CodeLlama-7b-hf": ("▁<PRE>", "▁<MID>", "▁<SUF>"),
    "codellama/CodeLlama-13b-hf": ("▁<PRE>", "▁<MID>", "▁<SUF>"),
    "codellama/CodeLlama-34b-hf": ("▁<PRE>", "▁<MID>", "▁<SUF>"),
    "codellama/CodeLlama-7b-Python-hf": ("▁<PRE>", "▁<MID>", "▁<SUF>"),
    "codellama/CodeLlama-13b-Python-hf": ("▁<PRE>", "▁<MID>", "▁<SUF>"),
    "codellama/CodeLlama-34b-Python-hf": ("▁<PRE>", "▁<MID>", "▁<SUF>"),
}


@add_end_docstrings(PIPELINE_INIT_ARGS)
class FimPipeline(Pipeline):
    """
    Fill-in-the-middle Language generation pipeline. This pipeline predicts the middle part of a text given it's prefix and suffix

    Example:
    ```python
    >>> from transformers import pipeline
    >>> PROMPT = '''
        def fib(x: int) -> int:
            <FILL_ME>
            return fib(x-1) + fib(x-2)
    '''
    >>> generator = pipeline(model="codellama/CodeLlama-7b-hf", max_new_tokens=128)
    >>> print(generator(PROMPT, do_sample=False))
    [{'generated_text': "\ndef fib(x: int) -> int:\n\tif x == 0:\n\t\treturn 0\n\tif x == 1:\n\t\treturn 1\n\telse:\n\t\treturn fib(x-1) + fib(x-2)\n"}]
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

    # There are two modes in which infilling is supported: PSM (Prefix-Suffix-Middle) and SPM (Suffix-Prefix-Middle)
    # CodeLlama's HF implementation has a third mode called 'suffix-first' which is added here for total compatibility with CodeLlama family
    # The placement of prefix and suffix along with their respective sentinel tokens depends on the mode
    # More information on this: https://arxiv.org/abs/2207.14255
    DEFAULT_INFILL_MODE = "prefix-suffix-middle"
    DEFAULT_INFILL_TOKEN = "<FILL_ME>"
    ALL_INFILL_MODES = ["prefix-suffix-middle", "suffix-prefix-middle", "suffix-first"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if the provided model is among the supported ones
        if self.model.name_or_path not in SUPPORTED_MODELS:
            raise ValueError(f"The model '{self.model.name_or_path}' is not supported for the Fill-in-Middle task.")

        # Get the Prefix, Suffix and Middle tokens for the corresponding model
        self.PREFIX_TOKEN, self.MIDDLE_TOKEN, self.SUFFIX_TOKEN = SUPPORTED_MODELS[self.model.name_or_path]

        self.check_model_type(
            TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES if self.framework == "tf" else MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
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
                preprocess_params, forward_params, _ = self._sanitize_parameters(prefix=prefix, **self._forward_params)
                self._preprocess_params = {
                    **self._preprocess_params,
                    **preprocess_params,
                }
                self._forward_params = {**self._forward_params, **forward_params}

    def ensure_infill_token(self, prompt_text, infill_token):
        """
        Ensures the prompt text has an infill token in it.
        """
        if infill_token not in prompt_text:
            raise ValueError(f"Infill token '{infill_token}' not in the prompt")

    def extract_prefix_suffix(self, input_text, infill_token):
        """
        Extracts the prefix and suffix from the input text.
        """
        prefix, _, suffix = input_text.partition(infill_token)
        return prefix, suffix

    def _sanitize_parameters(
        self,
        infill_token=None,
        mode=None,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        prefix=None,
        stop_sequence=None,
        add_special_tokens=False,
        **generate_kwargs,
    ):
        preprocess_params = {"add_special_tokens": add_special_tokens}
        preprocess_params["infill_token"] = infill_token or self.DEFAULT_INFILL_TOKEN

        if mode is not None:
            preprocess_params["mode"] = (
                mode.lower() if mode.lower() in self.ALL_INFILL_MODES else self.DEFAULT_INFILL_MODE
            )
        else:
            preprocess_params["mode"] = self.DEFAULT_INFILL_MODE

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

        preprocess_params.update(generate_kwargs)
        forward_params = generate_kwargs

        postprocess_params = {}

        if return_full_text is not None and return_type is None:
            if return_text is not None:
                raise ValueError("`return_text` is mutually exclusive with `return_full_text`")
            if return_tensors is not None:
                raise ValueError("`return_full_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        if return_tensors is not None and return_type is None:
            if return_text is not None:
                raise ValueError("`return_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
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
        mode,
        prefix="",
        add_special_tokens=False,
        **generate_kwargs,
    ):
        # Save the old prompt text since we'll use it at the end to form infilled text
        old_prompt_text = prompt_text

        # Ensure the prompt_text contains the infill token
        self.ensure_infill_token(old_prompt_text, infill_token)

        # Extract prefix and suffix
        input_prefix, input_suffix = self.extract_prefix_suffix(old_prompt_text, infill_token)

        # If the mode is Prefix-Suffix-Middle, arrange the components accordingly
        if mode == "prefix-suffix-middle":
            prompt_text = self.PREFIX_TOKEN + input_prefix + self.SUFFIX_TOKEN + input_suffix + self.MIDDLE_TOKEN
        # Change if the mode is Suffix-Prefix-Middle
        elif mode == "suffix-prefix-middle":
            prompt_text = self.SUFFIX_TOKEN + input_suffix + self.PREFIX_TOKEN + input_prefix + self.MIDDLE_TOKEN
        # CodeLlama's Implementation has a Suffix-First mode which is different than SPM
        elif mode == "suffix-first":
            prompt_text = self.PREFIX_TOKEN + self.SUFFIX_TOKEN + input_suffix + self.MIDDLE_TOKEN + input_prefix

        inputs = self.tokenizer(
            prefix + prompt_text,
            padding=False,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        inputs["prompt_text"] = prompt_text
        inputs["infill_token"] = infill_token
        inputs["old_prompt_text"] = old_prompt_text

        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        infill_token = model_inputs["infill_token"]
        old_prompt_text = model_inputs["old_prompt_text"]

        # Handle empty prompts by setting id and attention mask to None
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None

        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        prefix_len = generate_kwargs.pop("prefix_length", 0)
        if prefix_len > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.model.config.max_length
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

        return {
            "generated_sequence": generated_sequence,
            "old_prompt_text": old_prompt_text,
            "input_ids": input_ids,
            "infill_token": infill_token,
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
        return_type=ReturnType.FULL_TEXT,
        clean_up_tokenization_spaces=True,
    ):
        generated_sequence = model_outputs["generated_sequence"]
        input_ids = model_outputs["input_ids"]
        infill_token = model_outputs["infill_token"]
        prompt = model_outputs["old_prompt_text"]

        records = []

        for sequence in generated_sequence:
            if return_type == ReturnType.TENSORS:
                # If the return type is Tensors, just return the model output, as-is
                record = {"generated_token_ids": sequence}

            elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
                # Decode the text
                filled_text = self.tokenizer.decode(
                    sequence[input_ids.shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )

                # If full text is to be returned, replace the infill token with the generated infilled text
                all_text = filled_text
                if return_type == ReturnType.FULL_TEXT:
                    # First remove all the infill tokens from the text and then add the generated text
                    all_text = prompt.replace(infill_token, all_text)

                record = {"generated_text": all_text}
            records.append(record)

        return records
