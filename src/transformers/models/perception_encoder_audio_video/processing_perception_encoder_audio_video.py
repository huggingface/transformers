from ...processing_utils import ProcessorMixin


class PerceptionEncoderAudioVideoProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "video_processor", "tokenizer"]
    feature_extractor_class = "PerceptionEncoderAudioFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "PerceptionEncoderVideoVideoProcessor"


__all__ = ["PerceptionEncoderAudioVideoProcessor"]
