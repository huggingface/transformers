#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from ..utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "agents": ["Agent", "CodeAgent", "ManagedAgent", "ReactAgent", "ReactCodeAgent", "ReactJsonAgent", "Toolbox"],
    "llm_engine": ["HfApiEngine", "TransformersEngine"],
    "monitoring": ["stream_to_gradio"],
    "tools": ["PipelineTool", "Tool", "ToolCollection", "launch_gradio_demo", "load_tool", "tool"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["default_tools"] = ["FinalAnswerTool", "PythonInterpreterTool"]
    _import_structure["document_question_answering"] = ["DocumentQuestionAnsweringTool"]
    _import_structure["image_question_answering"] = ["ImageQuestionAnsweringTool"]
    _import_structure["search"] = ["DuckDuckGoSearchTool", "VisitWebpageTool"]
    _import_structure["speech_to_text"] = ["SpeechToTextTool"]
    _import_structure["text_to_speech"] = ["TextToSpeechTool"]
    _import_structure["translation"] = ["TranslationTool"]

if TYPE_CHECKING:
    from .agents import Agent, CodeAgent, ManagedAgent, ReactAgent, ReactCodeAgent, ReactJsonAgent, Toolbox
    from .llm_engine import HfApiEngine, TransformersEngine
    from .monitoring import stream_to_gradio
    from .tools import PipelineTool, Tool, ToolCollection, launch_gradio_demo, load_tool, tool

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .default_tools import FinalAnswerTool, PythonInterpreterTool
        from .document_question_answering import DocumentQuestionAnsweringTool
        from .image_question_answering import ImageQuestionAnsweringTool
        from .search import DuckDuckGoSearchTool, VisitWebpageTool
        from .speech_to_text import SpeechToTextTool
        from .text_to_speech import TextToSpeechTool
        from .translation import TranslationTool
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
