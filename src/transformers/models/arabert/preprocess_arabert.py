import html
import logging
import subprocess
import tempfile
from pathlib import Path
import os
import re
from typing import List

# import pyarabic.araby as araby

ACCEPTED_MODELS = [
    "bert-base-arabertv01",
    "bert-base-arabert",
    "bert-base-arabertv02",
    "bert-base-arabertv2",
    "bert-base-arabertv02-twitter",
    "bert-large-arabertv02",
    "bert-large-arabertv2",
    "bert-large-arabertv02-twitter",
    "araelectra-base",
    "araelectra-base-discriminator",
    "araelectra-base-generator",
    "araelectra-base-artydiqa",
    "aragpt2-base",
    "aragpt2-medium",
    "aragpt2-large",
    "aragpt2-mega",
]

SEGMENTED_MODELS = [
    "bert-base-arabert",
    "bert-base-arabertv2",
    "bert-large-arabertv2",
]

SECOND_GEN_MODELS = [
    "bert-base-arabertv02",
    "bert-base-arabertv2",
    "bert-base-arabertv02-twitter",
    "bert-large-arabertv02",
    "bert-large-arabertv2",
    "bert-large-arabertv02-twitter",
    "araelectra-base",
    "araelectra-base-discriminator",
    "araelectra-base-generator",
    "araelectra-base-artydiqa",
    "aragpt2-base",
    "aragpt2-medium",
    "aragpt2-large",
    "aragpt2-mega",
]

TWEET_MODELS = [
    "bert-base-arabertv02-twitter",
    "bert-large-arabertv02-twitter",
]

PREFIX_LIST = [
    "ال",
    "و",
    "ف",
    "ب",
    "ك",
    "ل",
    "لل",
    "\u0627\u0644",
    "\u0648",
    "\u0641",
    "\u0628",
    "\u0643",
    "\u0644",
    "\u0644\u0644",
    "س",
]
SUFFIX_LIST = [
    "ه",
    "ها",
    "ك",
    "ي",
    "هما",
    "كما",
    "نا",
    "كم",
    "هم",
    "هن",
    "كن",
    "ا",
    "ان",
    "ين",
    "ون",
    "وا",
    "ات",
    "ت",
    "ن",
    "ة",
    "\u0647",
    "\u0647\u0627",
    "\u0643",
    "\u064a",
    "\u0647\u0645\u0627",
    "\u0643\u0645\u0627",
    "\u0646\u0627",
    "\u0643\u0645",
    "\u0647\u0645",
    "\u0647\u0646",
    "\u0643\u0646",
    "\u0627",
    "\u0627\u0646",
    "\u064a\u0646",
    "\u0648\u0646",
    "\u0648\u0627",
    "\u0627\u062a",
    "\u062a",
    "\u0646",
    "\u0629",
]


# the never_split list is ussed with the transformers library
_PREFIX_SYMBOLS = [x + "+" for x in PREFIX_LIST]
_SUFFIX_SYMBOLS = ["+" + x for x in SUFFIX_LIST]
_OTHER_TOKENS = ["[رابط]", "[مستخدم]", "[بريد]"]
NEVER_SPLIT_TOKENS = list(set(_PREFIX_SYMBOLS + _SUFFIX_SYMBOLS + _OTHER_TOKENS))

URL_REGEXES = [
    r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
    r"@(https?|ftp)://(-\.)?([^\s/?\.#-]+\.?)+(/[^\s]*)?$@iS",
    r"http[s]?://[a-zA-Z0-9_\-./~\?=%&]+",
    r"www[a-zA-Z0-9_\-?=%&/.~]+",
    r"[a-zA-Z]+\.com",
    r"(?=http)[^\s]+",
    r"(?=www)[^\s]+",
    r"://",
]
USER_MENTION_REGEX = r"@[\w\d]+"
EMAIL_REGEXES = [r"[\w-]+@([\w-]+\.)+[\w-]+", r"\S+@\S+"]
REDUNDANT_PUNCT_PATTERN = (
    r"([!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ【»؛\s+«–…‘]{2,})"
)

REGEX_TATWEEL = r"(\D)\1{2,}"
MULTIPLE_CHAR_PATTERN = re.compile(r"(\D)\1{2,}", re.DOTALL)

REJECTED_CHARS_REGEX = r"[^0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘]"
REJECTED_CHARS_REGEXV2 = r"[^0-9\u0621-\u063A\u0641-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘/]"

REGEX_URL_STEP1 = r"(?=http)[^\s]+"
REGEX_URL_STEP2 = r"(?=www)[^\s]+"
REGEX_URL = r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
REGEX_MENTION = r"@[\w\d]+"
REGEX_EMAIL = r"\S+@\S+"

CHARS_REGEX = r"0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘"
CHARS_REGEXV2 = r"0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘/"

WHITE_SPACED_DOUBLE_QUOTATION_REGEX = r'\"\s+([^"]+)\s+\"'
WHITE_SPACED_SINGLE_QUOTATION_REGEX = r"\'\s+([^']+)\s+\'"
WHITE_SPACED_BACK_QUOTATION_REGEX = r"\`\s+([^`]+)\s+\`"
WHITE_SPACED_EM_DASH = r"\—\s+([^—]+)\s+\—"

LEFT_SPACED_CHARS = r" ([\]!#\$%\),\.:;\?}٪’،؟”؛…»·])"
RIGHT_SPACED_CHARS = r"([\[\(\{“«‘*\~]) "
LEFT_AND_RIGHT_SPACED_CHARS = r" ([\+\-\<\=\>\@\\\^\_\|\–]) "

_HINDI_NUMS = "٠١٢٣٤٥٦٧٨٩"
_ARABIC_NUMS = "0123456789"
HINDI_TO_ARABIC_MAP = str.maketrans(_HINDI_NUMS, _ARABIC_NUMS)


# Diacritics
FATHATAN = u'\u064b'
DAMMATAN = u'\u064c'
KASRATAN = u'\u064d'
FATHA = u'\u064e'
DAMMA = u'\u064f'
KASRA = u'\u0650'
SHADDA = u'\u0651'
SUKUN = u'\u0652'

TATWEEL = u'\u0640'


TASHKEEL = (FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, \
            SUKUN, SHADDA)

def is_tashkeel(archar):
    """Checks if the given ``archar`` Arabic Tashkeel Marks (
        - FATHA, DAMMA, KASRA, SUKUN,
        - SHADDA,
        - FATHATAN, DAMMATAN, KASRATAn)."""
    return archar in TASHKEEL

def is_vocalized(word):
    """Checks if the arabic word is vocalized.
    the word musn't  have any spaces and pounctuations.
    @param word: arabic unicode char
    @type word: unicode
    @return: if the word is vocalized
    @rtype:Boolean
    """
    if word.isalpha():
        return False
    for char in word:
        if is_tashkeel(char):
            break
    else:
        return False
    return True

def strip_tashkeel(text):
    """Strip vowels from a text, include Shadda.
    The striped marks are :
        - FATHA, DAMMA, KASRA
        - SUKUN
        - SHADDA
        - FATHATAN, DAMMATAN, KASRATAN,, , .

    Example:
        >>> text = u"الْعَرَبِيّةُ"
        >>> strip_tashkeel(text)
        العربية

    @param text: arabic text.
    @type text: unicode.
    @return: return a striped text.
    @rtype: unicode.
    """
    if not text:
        return text
    elif is_vocalized(text):
        for char in TASHKEEL:
            text = text.replace(char, '')
    return text

def strip_tatweel(text):
    """
    Strip tatweel from a text and return a result text.

    Example:
        >>> text = u"العـــــربية"
        >>> strip_tatweel(text)
        العربية

    @param text: arabic text.
    @type text: unicode.
    @return: return a striped text.
    @rtype: unicode.

    """
    return text.replace(TATWEEL, '')


class SimpleFarasaSegmenter:
    def __init__(self, logging_level="WARNING"):
        self.task = "segment"
        self.__base_dir = Path(__file__).parent.absolute()
        self.__bin_lib_dir = Path(f"{self.__base_dir}/farasa_bin/lib")
        self.__BASE_CMD = ["java", "-Dfile.encoding=UTF-8", "-jar"]
        self.__segment_cmd = self.__BASE_CMD + [str(self.__bin_lib_dir / "FarasaSegmenterJar.jar")]
        self.__task_proc = None
        self.logger = self._config_logs(logging_level)
        
        self._check_java_version()
        self._check_toolkit_binaries()
        self.__initialize_task_proc()

    def _config_logs(self, logging_level):
        logger = logging.getLogger("simple_farasapy_logger")
        logger.propagate = False
        logger.setLevel(getattr(logging, logging_level.upper()))
        logs_formatter = logging.Formatter("[%(asctime)s - %(name)s - %(levelname)s]: %(message)s")
        if not logger.hasHandlers():
            stream_logger = logging.StreamHandler()
            stream_logger.setFormatter(logs_formatter)
            logger.addHandler(stream_logger)
        return logger

    def _check_java_version(self):
        try:
            version_proc_output = subprocess.check_output(
                ["java", "-version"], stderr=subprocess.STDOUT, encoding="utf8"
            )
            version_pattern = r"\"(\d+(\.\d+){0,1})"
            java_version = float(re.search(version_pattern, version_proc_output).groups()[0])
            if java_version >= 1.7:
                self.logger.debug(f"Your java version is {java_version} which is compatible with Farasa")
            else:
                self.logger.warning("You are using an old version of java. Farasa is compatible with Java 7 and above")
        except subprocess.CalledProcessError as proc_err:
            self.logger.error(f"error occurred: {proc_err}.")
            raise Exception("Please make sure you have installed Java 1.7+ and add it to your PATH.")

    def _check_toolkit_binaries(self):
        if not Path(f"{self.__bin_lib_dir}/FarasaSegmenterJar.jar").is_file():
            raise Exception("FarasaSegmenterJar.jar not found in the expected directory. Please ensure it is correctly placed.")

    def __initialize_task_proc(self):
        self.__task_proc = subprocess.Popen(
            self.__segment_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def segment(self, text: str) -> str:
        return self._do_task(text.strip())

    def _do_task(self, text: str) -> str:
        byted_text = str.encode(text + "\n")
        self.__task_proc.stdin.flush()
        self.__task_proc.stdin.write(byted_text)
        self.__task_proc.stdin.flush()
        
        output = self.__task_proc.stdout.readline().decode("utf8").strip()
        self.__task_proc.stdout.flush()
        return output

class ArabertPreprocessor:
    """
    A Preprocessor class that cleans and preprocesses text for all models in the AraBERT repo.
    It also can unprocess the text ouput of the generated text

    Args:

        model_name (:obj:`str`): model name from the HuggingFace Models page without
        the aubmindlab tag. Will default to a base Arabic preprocessor if model name was not found.
        Current accepted models are:

            - "bert-base-arabertv01"
            - "bert-base-arabert"
            - "bert-base-arabertv02"
            - "bert-base-arabertv2"
            - "bert-base-arabertv02-twitter"
            - "bert-large-arabertv02"
            - "bert-large-arabertv2"
            - "bert-large-arabertv02-twitter"
            - "araelectra-base"
            - "araelectra-base-discriminator"
            - "araelectra-base-generator"
            - "araelectra-base-artydiqa"
            - "aragpt2-base"
            - "aragpt2-medium"
            - "aragpt2-large"
            - "aragpt2-mega"


        remove_html_markup(:obj: `bool`, `optional`, defaults to :obj:`True`): Whether to remove html artfacts,
        should be set to False when preprocessing TyDi QA.

        replace_urls_emails_mentions(:obj:`bool`, `optional`, defaults to :obj:`True`): Whether to replace email urls
        and mentions by special tokens.

        strip_tashkeel(:obj:`bool`, `optional`, defaults to :obj:`True`): remove diacritics (FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA,
        KASRA, SUKUN, SHADDA).

        strip_tatweel(:obj:`bool`, `optional`, defaults to :obj:`True`): remove tatweel '\\u0640'.

        insert_white_spaces(:obj:`bool`, `optional`, defaults to :obj:`True`): insert whitespace before and after all non Arabic digits
        or English digits or Arabic and English Alphabet or the 2 brackets, then inserts whitespace
        between words and numbers or numbers and words.

        remove_non_digit_repetition(:obj:`bool`, `optional`, defaults to :obj:`True`): replace repetition of more than 2 non-digit character with
        2 of this character.

        replace_slash_with_dash(:obj:`bool`, `optional`, defaults to :obj:`None`): Will be automatically set to True in AraBERTv02,
        AraELECTRA and AraGPT2.
        Set to False to force disable, and True to force enable. Replaces the "/"  with "-",
        since "/" is missing from AraBERTv2, AraELECTRA and ARAGPT2 vocabulary.

        map_hindi_numbers_to_arabic(:obj:`bool`, `optional`, defaults to :obj:`None`): Will be automatically set to True in
        AraBERTv02, AraELECTRA and AraGPT2.Set to False to force disable, and True to force enable.
        Replaces hindi numbers with the corresponding Arabic one. ex: "١٩٩٥" --> "1995".
        This is behavior is present by default in AraBERTv1 and v2 (with pre-segmentation),
        and fixes the issue of caused by a bug when inserting white spaces.

        apply_farasa_segmentation(:obj:`bool`, `optional`, defaults to :obj:`None`): Will be automatically set to True in
        AraBERTv2, and AraBERTv1. Set to False to force disable, and True to force enable.

        keep_emojis(:obj:`bool`, `optional`, defaults to :obj:`None`): don't remove emojis while preprocessing.
        Will be automatically set to True in AraBERT trained on tweets.



    Returns:

        ArabertPreprocessor: A preprocessor instance

    Example:

        from preprocess import ArabertPreprocessor

        arabert_prep = ArabertPreprocessor("aubmindlab/bert-base-arabertv2")

        arabert_prep.preprocess("SOME ARABIC TEXT")
    """

    def __init__(
        self,
        model_name: str,
        remove_html_markup: bool = True,
        replace_urls_emails_mentions: bool = True,
        strip_tashkeel: bool = True,
        strip_tatweel: bool = True,
        insert_white_spaces: bool = True,
        remove_non_digit_repetition: bool = True,
        keep_emojis: bool = None,
        replace_slash_with_dash: bool = None,
        map_hindi_numbers_to_arabic: bool = None,
        apply_farasa_segmentation: bool = None,
    ):

        model_name = model_name.replace("aubmindlab/", "").replace("wissamantoun/", "")

        if model_name not in ACCEPTED_MODELS:
            logging.warning(
                """Model provided is not in the accepted model list. Preprocessor will default to a base Arabic preprocessor"""
            )
            self.model_name = "bert-base-arabertv02"
        else:
            self.model_name = model_name

        if apply_farasa_segmentation is None:
            if self.model_name in SEGMENTED_MODELS:
                self.apply_farasa_segmentation = True
            else:
                self.apply_farasa_segmentation = False
        else:
            if (
                apply_farasa_segmentation == False
                and self.model_name in SEGMENTED_MODELS
            ):
                logging.warning(
                    "The selected model_name requires Farasa pre-segmentation, but apply_farasa_segmentation was set to False!"
                )

            self.apply_farasa_segmentation = apply_farasa_segmentation

        if self.apply_farasa_segmentation:
            try:
                # from farasa.segmenter import FarasaSegmenter

                self.farasa_segmenter = SimpleFarasaSegmenter()
            except ModuleNotFoundError:
                logging.error(
                    "farasapy is not installed, you want be able to process text for AraBERTv1 and v2. Install it using: pip install farasapy"
                )

        if keep_emojis is None:
            if self.model_name in TWEET_MODELS:
                self.keep_emojis = True
            else:
                self.keep_emojis = False
        else:
            if keep_emojis == False and self.model_name in TWEET_MODELS:
                logging.warning(
                    "The selected model_name is trained on emojis, but keep_emojis was set to False!"
                )
            self.keep_emojis = keep_emojis

        if self.keep_emojis:
            import emoji

            self.emoji = emoji
            if self.apply_farasa_segmentation:
                logging.warning(
                    "Keeping tweets with Farasa Segmentation is 10 times slower"
                )
            emoji_regex = "".join(list(self.emoji.UNICODE_EMOJI["en"].keys()))
            self.REJECTED_CHARS_REGEX = "[^%s%s]" % (
                CHARS_REGEX if self.model_name in SECOND_GEN_MODELS else CHARS_REGEXV2,
                emoji_regex,
            )
        else:
            self.REJECTED_CHARS_REGEX = (
                REJECTED_CHARS_REGEX
                if self.model_name in SECOND_GEN_MODELS
                else REJECTED_CHARS_REGEXV2
            )

        self.remove_html_markup = remove_html_markup
        self.replace_urls_emails_mentions = replace_urls_emails_mentions
        self.strip_tashkeel = strip_tashkeel
        self.strip_tatweel = strip_tatweel
        self.insert_white_spaces = insert_white_spaces
        self.remove_non_digit_repetition = remove_non_digit_repetition

        if replace_slash_with_dash is None:
            if self.model_name in SECOND_GEN_MODELS:
                self.replace_slash_with_dash = True
            else:
                self.replace_slash_with_dash = False
        else:
            self.replace_slash_with_dash = replace_slash_with_dash

        if map_hindi_numbers_to_arabic is None:
            if self.model_name in SECOND_GEN_MODELS:
                self.map_hindi_numbers_to_arabic = True
            else:
                self.map_hindi_numbers_to_arabic = False
        else:
            self.map_hindi_numbers_to_arabic = map_hindi_numbers_to_arabic

    def preprocess(self, text: str) -> str:
        """
        Preprocess takes an input text line an applies the same preprocessing used in AraBERT
                            pretraining, or according to settings

        Args:

            text (:obj:`str`): inout text string

        Returns:

            string: A preprocessed string depending on which model was selected
        """
        if (
            self.model_name == "bert-base-arabert"
            or self.model_name == "bert-base-arabertv01"
        ):
            return self._preprocess_v1(
                text,
                do_farasa_tokenization=self.apply_farasa_segmentation,
            )

        if self.model_name in SECOND_GEN_MODELS:
            return self._preprocess_v2(text)

        return self._preprocess_v3(text)

    def unpreprocess(self, text: str, desegment: bool = True) -> str:
        """Re-formats the text to a classic format where punctuations, brackets, parenthesis are not seperated by whitespaces.
        The objective is to make the generated text of any model appear natural and not preprocessed.

        Args:
            text (:obj:`str`): input text to be un-preprocessed
            desegment (:obj:`bool`, optional): [whether or not to remove farasa pre-segmentation before]..

        Returns:
            str: The unpreprocessed (and possibly Farasa-desegmented) text.
        """

        if self.apply_farasa_segmentation and desegment:
            text = self.desegment(text)

        # removes the spaces around quotation marks ex: i " ate " an apple --> i "ate" an apple
        # https://stackoverflow.com/a/53436792/5381220
        text = re.sub(WHITE_SPACED_DOUBLE_QUOTATION_REGEX, '"' + r"\1" + '"', text)
        text = re.sub(WHITE_SPACED_SINGLE_QUOTATION_REGEX, "'" + r"\1" + "'", text)
        text = re.sub(WHITE_SPACED_BACK_QUOTATION_REGEX, "\`" + r"\1" + "\`", text)
        text = re.sub(WHITE_SPACED_EM_DASH, "\—" + r"\1" + "\—", text)

        # during generation, sometimes the models don't put a space after the dot, this handles it
        text = text.replace(".", " . ")
        text = " ".join(text.split())

        # handle decimals
        text = re.sub(r"(\d+) \. (\d+)", r"\1.\2", text)
        text = re.sub(r"(\d+) \, (\d+)", r"\1,\2", text)

        text = re.sub(LEFT_AND_RIGHT_SPACED_CHARS, r"\1", text)
        text = re.sub(LEFT_SPACED_CHARS, r"\1", text)
        text = re.sub(RIGHT_SPACED_CHARS, r"\1", text)

        return text

    def desegment(self, text: str) -> str:
        """
        Use this function if sentence tokenization was done using
        `from arabert.preprocess_arabert import preprocess` with Farasa enabled
        AraBERT segmentation using Farasa adds a space after the '+' for prefixes,
        and after before the '+' for suffixes

        Example:
        >>> desegment('ال+ دراس +ات')
        الدراسات
        """
        text = text.replace("+ ", "+")
        text = text.replace(" +", "+")
        text = " ".join([self._desegmentword(word) for word in text.split(" ")])
        return text

    def _desegmentword(self, orig_word: str) -> str:
        """
        Word segmentor that takes a Farasa Segmented Word and removes the '+' signs

        Example:
        >>> _desegmentword("ال+يومي+ة")
        اليومية
        """
        word = orig_word.replace("ل+ال+", "لل")
        if "ال+ال" not in orig_word:
            word = word.replace("ل+ال", "لل")
        word = word.replace("+", "")
        word = word.replace("للل", "لل")
        return word

    def _preprocess_v3(self, text: str) -> str:
        text = str(text)
        text = html.unescape(text)
        if self.strip_tashkeel:
            text = strip_tashkeel(text)
        if self.strip_tatweel:
            text = strip_tatweel(text)

        if self.replace_urls_emails_mentions:
            # replace all possible URLs
            for reg in URL_REGEXES:
                text = re.sub(reg, " [رابط] ", text)
            # REplace Emails with [بريد]
            for reg in EMAIL_REGEXES:
                text = re.sub(reg, " [بريد] ", text)
            # replace mentions with [مستخدم]
            text = re.sub(USER_MENTION_REGEX, " [مستخدم] ", text)

        if self.remove_html_markup:
            # remove html line breaks
            text = re.sub("<br />", " ", text)
            # remove html markup
            text = re.sub("</?[^>]+>", " ", text)

        if self.map_hindi_numbers_to_arabic:
            text = text.translate(HINDI_TO_ARABIC_MAP)

        # remove repeated characters >2
        if self.remove_non_digit_repetition:
            text = self._remove_non_digit_repetition(text)

        # insert whitespace before and after all non Arabic digits or English Digits and Alphabet and the 2 brackets
        if self.insert_white_spaces:
            text = re.sub(
                "([^0-9\u0621-\u063A\u0641-\u064A\u0660-\u0669a-zA-Z ])",
                r" \1 ",
                text,
            )

            # re-fix brackets
            text = text.replace("[ رابط ]", "[رابط]")
            text = text.replace("[ بريد ]", "[بريد]")
            text = text.replace("[ مستخدم ]", "[مستخدم]")

            # insert whitespace between words and numbers or numbers and words
            text = re.sub(
                "(\d+)([\u0621-\u063A\u0641-\u064A\u066A-\u066C\u0654-\u0655]+)",
                r" \1 \2 ",
                text,
            )
            text = re.sub(
                "([\u0621-\u063A\u0641-\u064A\u066A-\u066C\u0654-\u0655]+)(\d+)",
                r" \1 \2 ",
                text,
            )

        # remove unwanted characters
        text = re.sub(self.REJECTED_CHARS_REGEX, " ", text)

        # remove extra spaces
        text = " ".join(text.replace("\uFE0F", "").split())

        

        if self.apply_farasa_segmentation:
            if self.keep_emojis:
                new_text = []
                for word in text.split():
                    if word in list(self.emoji.UNICODE_EMOJI["en"].keys()):
                        new_text.append(word)
                    else:
                        new_text.append(self.farasa_segmenter.segment(word))
                text = " ".join(new_text)
            else:
                text = self.farasa_segmenter.segment(text)
            return self._farasa_segment(text)

        # ALl the other models dont require Farasa Segmentation
        return text

    def _preprocess_v2(self, text: str) -> str:
        text = str(text)
        text = html.unescape(text)
        if self.strip_tashkeel:
            text = strip_tashkeel(text)
        if self.strip_tatweel:
            text = strip_tatweel(text)

        if self.replace_urls_emails_mentions:
            # replace all possible URLs
            for reg in URL_REGEXES:
                text = re.sub(reg, " [رابط] ", text)
            # REplace Emails with [بريد]
            for reg in EMAIL_REGEXES:
                text = re.sub(reg, " [بريد] ", text)
            # replace mentions with [مستخدم]
            text = re.sub(USER_MENTION_REGEX, " [مستخدم] ", text)

        if self.remove_html_markup:
            # remove html line breaks
            text = re.sub("<br />", " ", text)
            # remove html markup
            text = re.sub("</?[^>]+>", " ", text)

        if self.map_hindi_numbers_to_arabic:
            text = text.translate(HINDI_TO_ARABIC_MAP)

        # remove repeated characters >2
        if self.remove_non_digit_repetition:
            text = self._remove_non_digit_repetition(text)

        # insert whitespace before and after all non Arabic digits or English Digits and Alphabet and the 2 brackets
        if self.insert_white_spaces:
            text = re.sub(
                "([^0-9\u0621-\u063A\u0641-\u064A\u0660-\u0669a-zA-Z\[\]])",
                r" \1 ",
                text,
            )

            # insert whitespace between words and numbers or numbers and words
            text = re.sub(
                "(\d+)([\u0621-\u063A\u0641-\u064A\u0660-\u066C]+)", r" \1 \2 ", text
            )
            text = re.sub(
                "([\u0621-\u063A\u0641-\u064A\u0660-\u066C]+)(\d+)", r" \1 \2 ", text
            )

        if self.replace_slash_with_dash:
            text = text.replace("/", "-")

        # remove unwanted characters
        text = re.sub(self.REJECTED_CHARS_REGEX, " ", text)

        # remove extra spaces
        text = " ".join(text.replace("\uFE0F", "").split())

        if (
            self.model_name == "bert-base-arabertv2"
            or self.model_name == "bert-large-arabertv2"
        ):
            if self.keep_emojis:
                new_text = []
                for word in text.split():
                    if word in list(self.emoji.UNICODE_EMOJI["en"].keys()):
                        new_text.append(word)
                    else:
                        new_text.append(self.farasa_segmenter.segment(word))
                text = " ".join(new_text)
            else:
                text = self.farasa_segmenter.segment(text)
            return self._farasa_segment(text)

        # ALl the other models dont require Farasa Segmentation
        return text

    def _preprocess_v1(self, text: str, do_farasa_tokenization: bool) -> str:
        """
        AraBERTv1 preprocessing Function
        """
        text = str(text)
        if self.strip_tashkeel:
            text = strip_tashkeel(text)

        text = re.sub(r"\d+\/[ء-ي]+\/\d+\]", "", text)
        text = re.sub("ـ", "", text)
        text = re.sub("[«»]", ' " ', text)

        if self.replace_urls_emails_mentions:
            # replace the [رابط] token with space if you want to clean links
            text = re.sub(REGEX_URL_STEP1, "[رابط]", text)
            text = re.sub(REGEX_URL_STEP2, "[رابط]", text)
            text = re.sub(REGEX_URL, "[رابط]", text)
            text = re.sub(REGEX_EMAIL, "[بريد]", text)
            text = re.sub(REGEX_MENTION, "[مستخدم]", text)
        text = re.sub("…", r"\.", text).strip()
        text = self._remove_redundant_punct(text)

        if self.replace_urls_emails_mentions:
            text = re.sub(r"\[ رابط \]|\[ رابط\]|\[رابط \]", " [رابط] ", text)
            text = re.sub(r"\[ بريد \]|\[ بريد\]|\[بريد \]", " [بريد] ", text)
            text = re.sub(r"\[ مستخدم \]|\[ مستخدم\]|\[مستخدم \]", " [مستخدم] ", text)

        if self.remove_non_digit_repetition:
            text = self._remove_non_digit_repetition(text)

        if self.insert_white_spaces:
            text = re.sub(
                "([^0-9\u0621-\u063A\u0641-\u0669\u0671-\u0673a-zA-Z\[\]])",
                r" \1 ",
                text,
            )
        if do_farasa_tokenization:
            text = self._tokenize_arabic_words_farasa(text)

        text = " ".join(text.split())

        return text

    def _farasa_segment(self, text: str) -> str:
        line_farasa = text.split()
        segmented_line = []
        for index, word in enumerate(line_farasa):
            if word in ["[", "]"]:
                continue
            if word in ["رابط", "بريد", "مستخدم"] and line_farasa[index - 1] in [
                "[",
                "]",
            ]:
                segmented_line.append("[" + word + "]")
                continue
            if "+" not in word:
                segmented_line.append(word)
                continue
            segmented_word = self._split_farasa_output(word)
            segmented_line.extend(segmented_word)

        return " ".join(segmented_line)

    def _split_farasa_output(self, word: str) -> str:
        segmented_word = []
        temp_token = ""
        for i, c in enumerate(word):
            if c == "+":
                # if the token is KAF, it could be a suffix or prefix
                if temp_token == "ك":
                    # if we are at the second token, then KAF is surely a prefix
                    if i == 1:
                        segmented_word.append(temp_token + "+")
                        temp_token = ""
                    # If the KAF token is between 2 tokens
                    elif word[i - 2] == "+":
                        # if the previous token is prefix, then this KAF must be a prefix
                        if segmented_word[-1][-1] == "+":
                            segmented_word.append(temp_token + "+")
                            temp_token = ""
                        # else it is a suffix, this KAF could not be a second suffix
                        else:
                            segmented_word.append("+" + temp_token)
                            temp_token = ""
                    # if Kaf is at the end, this is handled with the statement after the loop
                elif temp_token in PREFIX_LIST:
                    segmented_word.append(temp_token + "+")
                    temp_token = ""
                elif temp_token in SUFFIX_LIST:
                    segmented_word.append("+" + temp_token)
                    temp_token = ""
                else:
                    segmented_word.append(temp_token)
                    temp_token = ""
                continue
            temp_token += c
        if temp_token != "":
            if temp_token in SUFFIX_LIST:
                segmented_word.append("+" + temp_token)
            else:
                segmented_word.append(temp_token)
        return segmented_word

    def _tokenize_arabic_words_farasa(self, line_input: str) -> str:

        if self.keep_emojis:
            # insert whitespace before and after all non Arabic digits or English Digits and Alphabet and the 2 brackets
            line_farasa = []
            for word in line_input.split():
                if word in list(self.emoji.UNICODE_EMOJI["en"].keys()):
                    line_farasa.append(word)
                else:
                    line_farasa.append(self.farasa_segmenter.segment(word))
        else:
            line_farasa = self.farasa_segmenter.segment(line_input).split()

        segmented_line = []
        for index, word in enumerate(line_farasa):
            if word in ["[", "]"]:
                continue
            if word in ["رابط", "بريد", "مستخدم"] and line_farasa[index - 1] in [
                "[",
                "]",
            ]:
                segmented_line.append("[" + word + "]")
                continue
            segmented_word = []
            for token in word.split("+"):
                if token in PREFIX_LIST:
                    segmented_word.append(token + "+")
                elif token in SUFFIX_LIST:
                    segmented_word.append("+" + token)
                else:
                    segmented_word.append(token)
            segmented_line.extend(segmented_word)
        return " ".join(segmented_line)

    def _remove_non_digit_repetition(self, text: str) -> str:
        """
        :param text:  the input text to remove elongation
        :return: delongated text
        """
        # loop over the number of times the regex matched the text
        # OLD
        # for index_ in range(len(re.findall(REGEX_TATWEEL, text))):
        #     elongation = re.search(REGEX_TATWEEL, text)
        #     if elongation:
        #         elongation_pattern = elongation.group()
        #         elongation_replacement = elongation_pattern[0]
        #         elongation_pattern = re.escape(elongation_pattern)
        #         text = re.sub(
        #             elongation_pattern, elongation_replacement, text, flags=re.MULTILINE
        #         )
        #     else:
        #         break

        # New
        text = MULTIPLE_CHAR_PATTERN.sub(r"\1\1", text)
        return text

    def _remove_redundant_punct(self, text: str) -> str:
        text_ = text
        result = re.search(REDUNDANT_PUNCT_PATTERN, text)
        dif = 0
        while result:
            sub = result.group()
            sub = sorted(set(sub), key=sub.index)
            sub = " " + "".join(list(sub)) + " "
            text = "".join(
                (text[: result.span()[0] + dif], sub, text[result.span()[1] + dif :])
            )
            text_ = "".join(
                (text_[: result.span()[0]], text_[result.span()[1] :])
            ).strip()
            dif = abs(len(text) - len(text_))
            result = re.search(REDUNDANT_PUNCT_PATTERN, text_)
        text = re.sub(r"\s+", " ", text)
        return text.strip()