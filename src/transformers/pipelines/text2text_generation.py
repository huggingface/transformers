from ..file_utils import add_end_docstrings, is_tf_available, is_torch_available
from ..tokenization_utils import TruncationStrategy
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class Text2TextGenerationPipeline(Pipeline):
    """
    Pipeline for text to text generation using seq2seq models.

    This Text2TextGenerationPipeline pipeline can currently be loaded from :func:`~transformers.pipeline` using the
    following task identifier: :obj:`"text2text-generation"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on `huggingface.co/models <https://huggingface.co/models?filter=seq2seq>`__.

    Usage::

        text2text_generator = pipeline("text2text-generation")
        text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
    """

    # Used in the return key of the pipeline.
    return_name = "generated"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.check_model_type(
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
            if self.framework == "tf"
            else MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
        )

    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        """
        Checks wether there might be something wrong with given input with regard to the model.
        """
        return True

    def __call__(
        self,
        *args,
        return_tensors=False,
        return_text=True,
        clean_up_tokenization_spaces=False,
        truncation=TruncationStrategy.DO_NOT_TRUNCATE,
        **generate_kwargs
    ):
        r"""
        Generate the output text(s) using text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                Input text for the encoder.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            truncation (:obj:`TruncationStrategy`, `optional`, defaults to :obj:`TruncationStrategy.DO_NOT_TRUNCATE`):
                The truncation strategy for the tokenization within the pipeline.
                :obj:`TruncationStrategy.DO_NOT_TRUNCATE` (default) will never truncate, but it is sometimes desirable
                to truncate the input to fit the model's max_length instead of throwing an error down the line.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **generated_text** (:obj:`str`, present when ``return_text=True``) -- The generated text.
            - **generated_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the generated text.
        """
        assert return_tensors or return_text, "You must specify return_tensors=True or return_text=True"

        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""
        if isinstance(args[0], list):
            assert (
                self.tokenizer.pad_token_id is not None
            ), "Please make sure that the tokenizer has a pad_token_id when using a batch input"
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            raise ValueError(
                " `args[0]`: {} have the wrong format. The should be either of type `str` or type `list`".format(
                    args[0]
                )
            )

        with self.device_placement():
            inputs = self._parse_and_tokenize(*args, padding=padding, truncation=truncation)

            if self.framework == "pt":
                inputs = self.ensure_tensor_on_device(**inputs)
                input_length = inputs["input_ids"].shape[-1]
            elif self.framework == "tf":
                input_length = tf.shape(inputs["input_ids"])[-1].numpy()

            min_length = generate_kwargs.get("min_length", self.model.config.min_length)
            max_length = generate_kwargs.get("max_length", self.model.config.max_length)
            self.check_inputs(input_length, min_length, max_length)

            generations = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generate_kwargs,
            )
            results = []
            for generation in generations:
                record = {}
                if return_tensors:
                    record[f"{self.return_name}_token_ids"] = generation
                if return_text:
                    record[f"{self.return_name}_text"] = self.tokenizer.decode(
                        generation,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                results.append(record)
            return results


@add_end_docstrings(PIPELINE_INIT_ARGS)
class SummarizationPipeline(Text2TextGenerationPipeline):
    """
    Summarize news articles and other documents.

    This summarizing pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"summarization"`.

    The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is
    currently, '`bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`'. See the up-to-date
    list of available models on `huggingface.co/models <https://huggingface.co/models?filter=summarization>`__.

    Usage::

        # use bart in pytorch
        summarizer = pipeline("summarization")
        summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)

        # use t5 in tf
        summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
        summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)
    """

    # Used in the return key of the pipeline.
    return_name = "summary"

    def __call__(self, *args, **kwargs):
        r"""
        Summarize the text(s) given as inputs.

        Args:
            documents (`str` or :obj:`List[str]`):
                One or several articles (or one list of articles) to summarize.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **summary_text** (:obj:`str`, present when ``return_text=True``) -- The summary of the corresponding
              input.
            - **summary_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``) --
              The token ids of the summary.
        """
        return super().__call__(*args, **kwargs)

    def check_inputs(self, input_length: int, min_length: int, max_length: int) -> bool:
        """
        Checks wether there might be something wrong with given input with regard to the model.
        """
        if input_length < min_length // 2:
            logger.warning(
                "Your min_length is set to {}, but you input_length is only {}. You might consider decreasing min_length manually, e.g. summarizer('...', min_length=10)".format(
                    min_length, input_length
                )
            )

        if input_length < max_length:
            logger.warning(
                "Your max_length is set to {}, but you input_length is only {}. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)".format(
                    max_length, input_length
                )
            )


@add_end_docstrings(PIPELINE_INIT_ARGS)
class TranslationPipeline(Text2TextGenerationPipeline):
    """
    Translates from one language to another.

    This translation pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"translation_xx_to_yy"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=translation>`__.

    Usage::
        en_fr_translator = pipeline("translation_en_to_fr")
        en_fr_translator("How old are you?")
    """

    # Used in the return key of the pipeline.
    return_name = "translation"

    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        if input_length > 0.9 * max_length:
            logger.warning(
                "Your input_length: {} is bigger than 0.9 * max_length: {}. You might consider increasing your max_length manually, e.g. translator('...', max_length=400)".format(
                    input_length, max_length
                )
            )

    def __call__(self, *args, **kwargs):
        r"""
        Translate the text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                Texts to be translated.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **translation_text** (:obj:`str`, present when ``return_text=True``) -- The translation.
            - **translation_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the translation.
        """
        return super().__call__(*args, **kwargs)
