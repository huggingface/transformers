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
from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool


class TranslationTool(PipelineTool):
    """
    Example:

    ```py
    from transformers.tools import TranslationTool

    translator = TranslationTool()
    translator("This is a super nice API!", src_lang="English", tgt_lang="French")
    ```
    """

    default_checkpoint = "facebook/nllb-200-distilled-600M"
    description = (
        "This is a tool that translates text from a language to another. It takes three inputs: `text`, which should "
        "be the text to translate, `src_lang`, which should be the language of the text to translate and `tgt_lang`, "
        "which should be the language for the desired ouput language. Both `src_lang` and `tgt_lang` are written in "
        "plain English, such as 'Romanian', or 'Albanian'. It returns the text translated in `tgt_lang`."
    )
    name = "translator"
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSeq2SeqLM
    # TODO add all other languages
    lang_to_code = {"French": "fra_Latn", "English": "eng_Latn", "Spanish": "spa_Latn"}

    inputs = ["text", "text", "text"]
    outputs = ["text"]

    def encode(self, text, src_lang, tgt_lang):
        if src_lang not in self.lang_to_code:
            raise ValueError(f"{src_lang} is not a supported language.")
        if tgt_lang not in self.lang_to_code:
            raise ValueError(f"{tgt_lang} is not a supported language.")
        src_lang = self.lang_to_code[src_lang]
        tgt_lang = self.lang_to_code[tgt_lang]
        return self.pre_processor._build_translation_inputs(
            text, return_tensors="pt", src_lang=src_lang, tgt_lang=tgt_lang
        )

    def forward(self, inputs):
        return self.model.generate(**inputs)

    def decode(self, outputs):
        return self.post_processor.decode(outputs[0].tolist(), skip_special_tokens=True)
