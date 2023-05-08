from typing import TYPE_CHECKING

from ..utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "agents": ["Agent", "HfAgent", "OpenAiAgent"],
    "base": ["PipelineTool", "RemoteTool", "Tool", "load_tool"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["document_question_answering"] = ["DocumentQuestionAnsweringTool"]
    _import_structure["generative_question_answering"] = ["GenerativeQuestionAnsweringTool"]
    _import_structure["image_captioning"] = ["ImageCaptioningTool"]
    _import_structure["image_question_answering"] = ["ImageQuestionAnsweringTool"]
    _import_structure["image_segmentation"] = ["ImageSegmentationTool"]
    _import_structure["language_identifier"] = ["LanguageIdentificationTool"]
    _import_structure["speech_to_text"] = ["SpeechToTextTool"]
    _import_structure["text_classification"] = ["TextClassificationTool"]
    _import_structure["text_summarization"] = ["TextSummarizationTool"]
    _import_structure["text_to_speech"] = ["TextToSpeechTool"]
    _import_structure["translation"] = ["TranslationTool"]

if TYPE_CHECKING:
    from .agents import Agent, HfAgent, OpenAiAgent
    from .base import PipelineTool, RemoteTool, Tool, load_tool

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .document_question_answering import DocumentQuestionAnsweringTool
        from .generative_question_answering import GenerativeQuestionAnsweringTool
        from .image_captioning import ImageCaptioningTool
        from .image_question_answering import ImageQuestionAnsweringTool
        from .image_segmentation import ImageSegmentationTool
        from .language_identifier import LanguageIdentificationTool
        from .speech_to_text import SpeechToTextTool
        from .text_classification import TextClassificationTool
        from .text_summarization import TextSummarizationTool
        from .text_to_speech import TextToSpeechTool
        from .translation import TranslationTool
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
