from ...processing_utils import ProcessorMixin


class PerceptionEncoderAudioProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "PerceptionEncoderAudioFeatureExtractor"
    tokenizer_class = "AutoTokenizer"


__all__ = ["PerceptionEncoderAudioProcessor"]
