from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool


class TranslationTool(PipelineTool):
    """
    Example:

    ```py
    from transformers.tools import TranslationTool

    translator = TranslationTool("distilbert-base-uncased-finetuned-sst-2-english")
    translator("This is a super nice API!")
    ```
    """

    default_checkpoint = "facebook/nllb-200-distilled-600M"
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSeq2SeqLM

    description = (
        "This is a tool that translates text from {src_lang} to {tgt_lang}. It takes an input named `text` which "
        "should be the text in {src_lang} and returns a dictionary with a single key `'translated_text'` that "
        "contains the translation in {tgt_lang}."
    )

    def __init__(
        self,
        model=None,
        pre_processor=None,
        post_processor=None,
        src_lang=None,
        tgt_lang=None,
        device=None,
        device_map=None,
        model_kwargs=None,
        **hub_kwargs,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        super().__init__(
            model=model,
            pre_processor=pre_processor,
            post_processor=post_processor,
            device=device,
            device_map=device_map,
            model_kwargs=model_kwargs,
            **hub_kwargs,
        )

    def post_init(self):
        codes_to_lang = {"fra_Latn": "French", "eng_Latn": "English"}
        src_lang = codes_to_lang[self.src_lang]
        tgt_lang = codes_to_lang[self.tgt_lang]
        self.description = self.description.replace("{src_lang}", src_lang).replace("{tgt_lang}", tgt_lang)

    def encode(self, text):
        return self.pre_processor._build_translation_inputs(
            text, return_tensors="pt", src_lang=self.src_lang, tgt_lang=self.tgt_lang
        )

    def forward(self, inputs):
        return self.model.generate(**inputs)

    def decode(self, outputs):
        return {"translated_text": self.post_processor.decode(outputs[0].tolist())}
