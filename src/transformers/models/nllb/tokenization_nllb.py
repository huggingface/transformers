# Copyright 2022 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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


from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_python import AddedToken, BatchEncoding
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}


FAIRSEQ_LANGUAGE_CODES = ['ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab', 'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt', 'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn', 'ewe_Latn', 'fao_Latn', 'pes_Arab', 'fij_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'knc_Arab', 'knc_Latn', 'kaz_Cyrl', 'kbp_Latn', 'kea_Latn', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 'kon_Latn', 'kor_Hang', 'kmr_Latn', 'lao_Laoo', 'lvs_Latn', 'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Latn', 'mkd_Cyrl', 'plt_Latn', 'mlt_Latn', 'mni_Beng', 'khk_Cyrl', 'mos_Latn', 'mri_Latn', 'zsm_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'gaz_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'pbt_Arab', 'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Beng', 'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn', 'als_Latn', 'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Taml', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tir_Ethi', 'taq_Latn', 'taq_Tfng', 'tpi_Latn', 'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'tzm_Tfng', 'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn', 'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zul_Latn']  # fmt: skip


class NllbTokenizer(TokenizersBackend):
    """
    Construct an NLLB tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>
    <tokens> <eos>` for target language documents.

    Examples:

    ```python
    >>> from transformers import NllbTokenizer

    >>> tokenizer = NllbTokenizer.from_pretrained(
    ...     "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"
    ... )
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
    ```

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values.
        src_lang (`str`, *optional*):
            The language to use as source language for translation.
        tgt_lang (`str`, *optional*):
            The language to use as target language for translation.
        legacy_behaviour (`bool`, *optional*, defaults to `False`):
            Whether to use legacy behaviour (suffix pattern) or new behaviour (prefix pattern).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    prefix_tokens: list[int] = []
    suffix_tokens: list[int] = []

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        src_lang=None,
        tgt_lang=None,
        additional_special_tokens=None,
        legacy_behaviour=False,
        **kwargs,
    ):
        if additional_special_tokens is None:
            additional_special_tokens = kwargs.get("extra_special_tokens", FAIRSEQ_LANGUAGE_CODES)

        mask_token = (
            AddedToken(mask_token, normalized=True, lstrip=True, special=True)
            if isinstance(mask_token, str)
            else mask_token
        )
        self.legacy_behaviour = legacy_behaviour

        if vocab is None:
            vocab = {
                str(bos_token): 0,
                str(pad_token): 1,
                str(eos_token): 2,
                str(unk_token): 3,
            }
        self._vocab = vocab
        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                unk_token=str(unk_token),
                fuse_unk=True,
                byte_fallback=False,
            )
        )

        self._tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Replace(Regex(r"[\n\r\t]"), " "),
                normalizers.NFKC(),
                normalizers.Replace(Regex(r" {2,}"), " "),
            ]
        )

        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)

        # Remove extra_special_tokens from kwargs if present to avoid conflict
        kwargs.pop("extra_special_tokens", None)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            mask_token=mask_token,
            extra_special_tokens=additional_special_tokens,
            legacy_behaviour=legacy_behaviour,
            **kwargs,
        )

        # Build fairseq mappings for backward compatibility
        self.fairseq_offset = 1
        self.fairseq_tokens_to_ids = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "<unk>": 3,
        }
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        self._src_lang = src_lang if src_lang is not None else "eng_Latn"
        self.cur_lang_code = self.convert_tokens_to_ids(self._src_lang)
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: str | None, tgt_lang: str | None, **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def prepare_seq2seq_batch(
        self,
        src_texts: list[str],
        src_lang: str = "eng_Latn",
        tgt_texts: list[str] | None = None,
        tgt_lang: str = "fra_Latn",
        max_length: int | None = None,
        max_target_length: int | None = None,
        padding: str = "longest",
        return_tensors: str | None = None,
        truncation: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        if max_length is None:
            max_length = self.model_max_length

        model_inputs = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )

        if tgt_texts is None:
            return model_inputs

        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length

        # Switch to target mode to set the right special tokens
        self._switch_to_target_mode()
        labels = self(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=truncation,
            **kwargs,
        )
        model_inputs["labels"] = labels["input_ids"]

        # Switch back to input mode
        self._switch_to_input_mode()

        return model_inputs

    def _switch_to_input_mode(self):
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        if self.tgt_lang is None:
            self.tgt_lang = self._src_lang
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting.
        - In legacy mode: No prefix and suffix=[eos, src_lang_code].
        - In default mode: Prefix=[src_lang_code], suffix = [eos]
        """
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)

        if self.legacy_behaviour:
            self.prefix_tokens = []
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.prefix_tokens = [self.cur_lang_code]
            self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target lang setting.
        - In legacy mode: No prefix and suffix=[eos, tgt_lang_code].
        - In default mode: Prefix=[tgt_lang_code], suffix = [eos]
        """
        self.cur_lang_code = self.convert_tokens_to_ids(lang)
        if self.legacy_behaviour:
            self.prefix_tokens = []
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.prefix_tokens = [self.cur_lang_code]
            self.suffix_tokens = [self.eos_token_id]

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )


__all__ = ["NllbTokenizer"]
