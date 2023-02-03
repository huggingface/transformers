import warnings
from contextlib import contextmanager

from ... import ProcessorMixin
from .feature_extraction_transformer_transducer import TransformerTransducerFeatureExtractor
from .tokenization_transformer_transducer import TransformerTransducerTokenizer


# [NOTE]: copied from Wav2Vec2Processor
class TransformerTransducerProcessor(ProcessorMixin):
    r"""
    Constructs a TransformerTransducer processor which wraps a TransformerTransducer feature extractor and a
    TransformerTransducerTokenizer into a single processor.

    [`TransformerTransducerProcessor`] offers all the functionalities of [`TransformerTransducerFeatureExtractor`] and
    [`PreTrainedTokenizer`]. See the docstring of [`~TransformerTransducerProcessor.__call__`] and
    [`~TransformerTransducerProcessor.decode`] for more information.

    Args:
        feature_extractor (`TransformerTransducerFeatureExtractor`):
            An instance of [`TransformerTransducerFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = "TransformerTransducerFeatureExtractor"
    tokenizer_class = "TransformerTransducerTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except OSError:
            warnings.warn(
                f"Loading a tokenizer inside {cls.__name__} from a config that does not"
                " include a `tokenizer_class` attribute is deprecated and will be "
                "removed in v5. Please add `'tokenizer_class': 'TransformerTransducerTokenizer'`"
                " attribute to either your `config.json` or `tokenizer_config.json` "
                "file to suppress this warning: ",
                FutureWarning,
            )

            feature_extractor = TransformerTransducerFeatureExtractor.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs,
            )
            tokenizer = TransformerTransducerTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs,
            )

            return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to TransformerTransducerFeatureExtractor's
        [`~TransformerTransducerFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~TransformerTransducerProcessor.as_target_processor`] this method forwards all its arguments to
        PreTrainedTokenizer's [`~PreTrainedTokenizer.__call__`]. Please refer to the docstring of the above two methods
        for more information.
        """
        return self.current_processor(*args, **kwargs)

    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to TransformerTransducerFeatureExtractor's
        [`~TransformerTransducerFeatureExtractor.pad`] and returns its output. If used in the context
        [`~TransformerTransducerProcessor.as_target_processor`] this method forwards all its arguments to
        PreTrainedTokenizer's [`~PreTrainedTokenizer.pad`]. Please refer to the docstring of the above two methods for
        more information.
        """
        return self.current_processor.pad(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def mel_compressor(self, *args, **kwargs):
        """"""
        return self.feature_extractor.mel_compressor(*args, **kwargs)

    def log_mel_transform(self, *args, **kwargs):
        """"""
        return self.feature_extractor.log_mel_transform(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        TransformerTransducer.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
