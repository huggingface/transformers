import enum

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, TF_MODEL_FOR_CAUSAL_LM_MAPPING

from ..file_utils import add_end_docstrings
from .base import PIPELINE_INIT_ARGS, Pipeline


class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


@add_end_docstrings(PIPELINE_INIT_ARGS)
class TextGenerationPipeline(Pipeline):
    """
    Language generation pipeline using any :obj:`ModelWithLMHead`. This pipeline predicts the words that will follow a
    specified text prompt.

    This language generation pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"text-generation"`.

    The models that this pipeline can use are models that have been trained with an autoregressive language modeling
    objective, which includes the uni-directional models in the library (e.g. gpt2). See the list of available models
    on `huggingface.co/models <https://huggingface.co/models?filter=causal-lm>`__.
    """

    # Prefix text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
    # in https://github.com/rusiaaman/XLNet-gen#methodology
    # and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e

    XL_PREFIX = """
    In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria) are discovered. The
    voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the remainder of the story. 1883 Western
    Siberia, a young Grigori Rasputin is asked by his father and a group of men to perform magic. Rasputin has a vision
    and denounces one of the men as a horse thief. Although his father initially slaps him for making such an
    accusation, Rasputin watches as the man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous, with people, even a bishop,
    begging for his blessing. <eod> </s> <eos>
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(
            TF_MODEL_FOR_CAUSAL_LM_MAPPING if self.framework == "tf" else MODEL_FOR_CAUSAL_LM_MAPPING
        )
        if "prefix" not in self._preprocess_params:
            # This is very specific. The logic is quite complex and needs to be done
            # as a "default".
            # It also defines both some preprocess_kwargs and generate_kwargs
            # which is why we cannot put them in their respective methods.
            prefix = None
            if self.model.config.prefix is not None:
                prefix = self.model.config.prefix
            if prefix is None and self.model.__class__.__name__ in [
                "XLNetLMHeadModel",
                "TransfoXLLMHeadModel",
                "TFXLNetLMHeadModel",
                "TFTransfoXLLMHeadModel",
            ]:
                # For XLNet and TransformerXL we add an article to the prompt to give more state to the model.
                prefix = self.XL_PREFIX
            if prefix is not None:
                # Recalculate some generate_kwargs linked to prefix.
                preprocess_params, forward_params, _ = self._sanitize_parameters(prefix=prefix, **self._forward_params)
                self._preprocess_params = {**self._preprocess_params, **preprocess_params}
                self._forward_params = {**self._forward_params, **forward_params}

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        prefix=None,
        **generate_kwargs
    ):
        preprocess_params = {}
        if prefix is not None:
            preprocess_params["prefix"] = prefix
        if prefix:
            prefix_inputs = self.tokenizer(
                prefix, padding=False, add_special_tokens=False, return_tensors=self.framework
            )
            prefix_length = prefix_inputs["input_ids"].shape[-1]
            if "max_length" in generate_kwargs:
                generate_kwargs["max_length"] += prefix_length
            else:
                generate_kwargs["max_length"] = self.model.config.max_length + prefix_length

            if "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        forward_params = generate_kwargs

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        return preprocess_params, forward_params, postprocess_params

    # overriding _parse_and_tokenize to allow for unusual language-modeling tokenizer arguments
    def _parse_and_tokenize(self, *args, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            kwargs.update({"add_space_before_punct_symbol": True})

        return super()._parse_and_tokenize(*args, **kwargs)

    def __call__(self, text_inputs, **kwargs):
        """
        Complete the prompt(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several prompts (or one list of prompts) to complete.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            return_full_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to :obj:`False` only added text is returned, otherwise the full text is returned Only meaningful
                if `return_text` is set to True.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            prefix (:obj:`str`, `optional`):
                Prefix added to prompt.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **generated_text** (:obj:`str`, present when ``return_text=True``) -- The generated text.
            - **generated_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the generated text.
        """
        return super().__call__(text_inputs, **kwargs)

    def preprocess(self, prompt_text, prefix=""):
        inputs = self.tokenizer(
            prefix + prompt_text, padding=False, add_special_tokens=False, return_tensors=self.framework
        )
        inputs["prompt_text"] = prompt_text
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        prompt_text = model_inputs.pop("prompt_text")
        generated_sequence = self.model.generate(input_ids=input_ids, **generate_kwargs)  # BS x SL
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}

    def postprocess(self, model_outputs, return_type=ReturnType.FULL_TEXT, clean_up_tokenization_spaces=True):
        generated_sequence = model_outputs["generated_sequence"]
        input_ids = model_outputs["input_ids"]
        prompt_text = model_outputs["prompt_text"]
        if self.framework == "pt" and generated_sequence is not None:
            generated_sequence = generated_sequence.cpu()
        generated_sequence = generated_sequence.numpy().tolist()
        if return_type == ReturnType.TENSORS:
            record = {"generated_token_ids": generated_sequence}
        elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
            # Decode text
            record = []
            for sequence in generated_sequence:
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )

                # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    )

                if return_type == ReturnType.FULL_TEXT:
                    all_text = prompt_text + text[prompt_length:]
                else:
                    all_text = text[prompt_length:]

                item = {"generated_text": all_text}
                record.append(item)

        return record
