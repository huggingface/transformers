from ...processing_utils import ProcessorMixin


class PeAudioVideoProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "video_processor", "tokenizer"]
    feature_extractor_class = "PeAudioFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "PeVideoVideoProcessor"


__all__ = ["PeAudioVideoProcessor"]
