from ...processing_utils import ProcessorMixin


class PEVideoProcessor(ProcessorMixin):
    attributes = ["video_processor", "tokenizer"]
    video_processor_class = "PEVideoVideoProcessor"
    tokenizer_class = "AutoTokenizer"


__all__ = ["PEVideoProcessor"]
