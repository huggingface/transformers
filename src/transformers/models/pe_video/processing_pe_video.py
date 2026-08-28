from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring


@auto_docstring
class PeVideoProcessor(ProcessorMixin):
    attributes = ["video_processor", "tokenizer"]
    video_processor_class = "PeVideoVideoProcessor"
    tokenizer_class = "AutoTokenizer"


__all__ = ["PeVideoProcessor"]
