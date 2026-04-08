from ...processing_utils import ProcessorMixin


class PeVideoProcessor(ProcessorMixin):
    attributes = ["video_processor", "tokenizer"]
    video_processor_class = "PeVideoVideoProcessor"
    tokenizer_class = "AutoTokenizer"


__all__ = ["PeVideoProcessor"]
