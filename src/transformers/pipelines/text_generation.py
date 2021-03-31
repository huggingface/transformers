from ..file_utils import add_end_docstrings
from .base import PIPELINE_INIT_ARGS, Pipeline


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

    ALLOWED_MODELS = [
        "XLNetLMHeadModel",
        "TransfoXLLMHeadModel",
        "ReformerModelWithLMHead",
        "GPT2LMHeadModel",
        "GPTNeoForCausalLM",
        "OpenAIGPTLMHeadModel",
        "CTRLLMHeadModel",
        "TFXLNetLMHeadModel",
        "TFTransfoXLLMHeadModel",
        "TFGPT2LMHeadModel",
        "TFOpenAIGPTLMHeadModel",
        "TFCTRLLMHeadModel",
    ]

    def __init__(self, *args, return_full_text=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.check_model_type(self.ALLOWED_MODELS)
        self.return_full_text = return_full_text

    # overriding _parse_and_tokenize to allow for unusual language-modeling tokenizer arguments
    def _parse_and_tokenize(self, *args, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            kwargs.update({"add_space_before_punct_symbol": True})

        return super()._parse_and_tokenize(*args, **kwargs)

    def __call__(
        self,
        text_inputs,
        return_tensors=False,
        return_text=True,
        return_full_text=None,
        clean_up_tokenization_spaces=False,
        prefix=None,
        **generate_kwargs
    ):
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
        prefix = prefix if prefix is not None else self.model.config.prefix
        return_full_text = return_full_text if return_full_text is not None else self.return_full_text

        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]
        results = []
        for prompt_text in text_inputs:
            # Manage correct placement of the tensors
            with self.device_placement():
                if prefix is None and self.model.__class__.__name__ in [
                    "XLNetLMHeadModel",
                    "TransfoXLLMHeadModel",
                    "TFXLNetLMHeadModel",
                    "TFTransfoXLLMHeadModel",
                ]:
                    # For XLNet and TransformerXL we add an article to the prompt to give more state to the model.
                    prefix = self.XL_PREFIX

                if prefix:
                    prefix_inputs = self._parse_and_tokenize(prefix, padding=False, add_special_tokens=False)
                    # This impacts max_length and min_length argument that need adjusting.
                    prefix_length = prefix_inputs["input_ids"].shape[-1]
                    if generate_kwargs.get("max_length", None) is not None:
                        generate_kwargs["max_length"] += prefix_length
                    if generate_kwargs.get("min_length", None) is not None:
                        generate_kwargs["min_length"] += prefix_length

                prefix = prefix or ""
                inputs = self._parse_and_tokenize(prefix + prompt_text, padding=False, add_special_tokens=False)

                # set input_ids to None to allow empty prompt
                if inputs["input_ids"].shape[-1] == 0:
                    inputs["input_ids"] = None
                    inputs["attention_mask"] = None

                if self.framework == "pt" and inputs["input_ids"] is not None:
                    inputs = self.ensure_tensor_on_device(**inputs)

                input_ids = inputs["input_ids"]

                # Ensure that batch size = 1 (batch generation not allowed for now)
                assert (
                    input_ids is None or input_ids.shape[0] == 1
                ), "Batch generation is currently not supported. See https://github.com/huggingface/transformers/issues/3021 for more information."

                output_sequences = self.model.generate(input_ids=input_ids, **generate_kwargs)  # BS x SL

            result = []
            for generated_sequence in output_sequences:
                if self.framework == "pt" and generated_sequence is not None:
                    generated_sequence = generated_sequence.cpu()
                generated_sequence = generated_sequence.numpy().tolist()
                record = {}
                if return_tensors:
                    record["generated_token_ids"] = generated_sequence
                if return_text:
                    # Decode text
                    text = self.tokenizer.decode(
                        generated_sequence,
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

                    if return_full_text:
                        all_text = prompt_text + text[prompt_length:]
                    else:
                        all_text = text[prompt_length:]

                    record["generated_text"] = all_text

                result.append(record)
            results += [result]

        if len(results) == 1:
            return results[0]

        return results
