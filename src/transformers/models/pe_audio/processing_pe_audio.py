from ...processing_utils import ProcessorMixin


class PEAudioProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "PEAudioFeatureExtractor"
    tokenizer_class = "AutoTokenizer"


__all__ = ["PEAudioProcessor"]
