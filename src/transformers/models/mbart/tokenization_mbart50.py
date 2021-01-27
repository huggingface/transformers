from .tokenization_mbart import MBartTokenizer


FAIRSEQ_LANGUAGE_CODES = [
    "ar_AR",
    "cs_CZ",
    "de_DE",
    "en_XX",
    "es_XX",
    "et_EE",
    "fi_FI",
    "fr_XX",
    "gu_IN",
    "hi_IN",
    "it_IT",
    "ja_XX",
    "kk_KZ",
    "ko_KR",
    "lt_LT",
    "lv_LV",
    "my_MM",
    "ne_NP",
    "nl_XX",
    "ro_RO",
    "ru_RU",
    "si_LK",
    "tr_TR",
    "vi_VN",
    "zh_CN",
    "af_ZA",
    "az_AZ",
    "bn_IN",
    "fa_IR",
    "he_IL",
    "hr_HR",
    "id_ID",
    "ka_GE",
    "km_KH",
    "mk_MK",
    "ml_IN",
    "mn_MN",
    "mr_IN",
    "pl_PL",
    "ps_AF",
    "pt_XX",
    "sv_SE",
    "sw_KE",
    "ta_IN",
    "te_IN",
    "th_TH",
    "tl_XX",
    "uk_UA",
    "ur_PK",
    "xh_ZA",
    "gl_ES",
    "sl_SI",
]


class MBart50Tokenizer(MBartTokenizer):
    """
    Construct an MBART tokenizer for MBART-50 models.

    :class:`~transformers.MBart50Tokenizer` is a subclass of :class:`~transformers.MBartTokenizer`

    The tokenization method is ``<language code> <tokens> <eos>`` for both source and target language documents. Note
    that this is different from :class:`~transformers.MBartTokenizer` where the ``<language code>`` is used as a suffix
    for source language documents.

    Examples::

        >>> from transformers import MBartTokenizer
        >>> tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-50-one-to-many')
        >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
        >>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
        >>> batch: dict = tokenizer.prepare_seq2seq_batch(
        ...     example_english_phrase, src_lang="en_XX", tgt_lang="ro_RO", tgt_texts=expected_translation_romanian, return_tensors="pt"
        ... )

    """

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos]."""
        self.cur_lang_code = self.lang_code_to_id[src_lang]
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[tgt_lang_code] and suffix=[eos]."""
        self.cur_lang_code = self.lang_code_to_id[lang]
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]
