from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool


TRANSLATION_DESCRIPTION = (
    "This is a tool that translates text from a language to another. It takes three inputs: `text`, which should be "
    "the text to translate, `src_lang`, which should be the language of the text to translate and `tgt_lang`, which "
    "should be the language for the desired ouput language. Both `src_lang` and `tgt_lang` are written in plain "
    "English, such as 'Romanian', or 'Albanian'. It returns the text translated in `tgt_lang`."
)


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
    description = TRANSLATION_DESCRIPTION
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
