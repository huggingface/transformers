from ...processing_utils import ProcessorMixin


class PEAudioVideoProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "video_processor", "tokenizer"]
    feature_extractor_class = "PEAudioFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "PEVideoVideoProcessor"


__all__ = ["PEAudioVideoProcessor"]
