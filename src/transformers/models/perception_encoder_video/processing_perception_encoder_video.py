from ...processing_utils import ProcessorMixin


class PerceptionEncoderVideoProcessor(ProcessorMixin):
    attributes = ["video_processor", "tokenizer"]
    video_processor_class = "PerceptionEncoderVideoVideoProcessor"
    tokenizer_class = "AutoTokenizer"


__all__ = ["PerceptionEncoderVideoProcessor"]
