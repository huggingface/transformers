from ...processing_utils import ProcessorMixin


class PeAudioProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "PeAudioFeatureExtractor"
    tokenizer_class = "AutoTokenizer"


__all__ = ["PeAudioProcessor"]
