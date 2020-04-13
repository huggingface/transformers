from .tokenization_utils import PreTrainedTokenizer
import sentencepiece
from mosestokenizer import MosesTokenizer, MosesDetokenizer, MosesSentenceSplitter, MosesPunctuationNormalizer

from .apply_bpe import BPE
from .tokenization_xlm_roberta import XLMRobertaTokenizer
from typing import Dict, List, Tuple, Optional
from torch import Tensor

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "marian/en-de": "https://s3.amazonaws.com/models.huggingface.co/bert/marian/en-de/source.spm"
    }}
VOCAB_NAME = 'source.spm'
import yaml

class MarianSPTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"source_spm": VOCAB_NAME,  'target_spm': 'target.spm',
                         'vocab': 'opus.spm32k-spm32k.vocab.yml',
                         'tokenizer_config_file': 'tokenizer_config.json',
                         #'source_bpe': 'source.bpe',
                         #'target_bpe': 'target_bpe'
                         }

    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = {m: 512 for m in PRETRAINED_VOCAB_FILES_MAP}
    # TODO(SS): model_input_names = ["attention_mask"]


    def __init__(self, vocab=None, source_bpe=None, target_bpe=None, source_spm=None, target_spm=None, source_lang=None,
                 target_lang=None, unk_token='<unk>', eos_token='</s>', **kwargs):

        super().__init__(
            #bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            #sep_token=sep_token,
            #cls_token=cls_token,
            #pad_token=pad_token,
            #mask_token=mask_token,
            #**kwargs,
        )
        self.vocab: dict = yaml.load(open(vocab), Loader=yaml.BaseLoader)
        self.unk_token_id = self.vocab[self.unk_token]
        self.eos_token_id =  self.vocab[self.eos_token]

        self.bpe_source = None
        self.bpe_target = None
        self.spm_source = None
        self.spm_target = None
        self.tokenizer = None
        self.detokenizer = None
        self.sentences = []
        # load BPE model for pre-processing
        if source_bpe:
            BPEcodes = open(source_bpe, 'r', encoding="utf-8")
            self.bpe_source = BPE(BPEcodes)
            self.tokenizer = MosesTokenizer(source_lang)

        if target_bpe:
            BPEcodes = open(target_bpe, 'r', encoding="utf-8")
            self.bpe_target = BPE(BPEcodes)
            self.detokenizer = MosesDetokenizer(target_lang)

        # load SentencePiece model for pre-processing
        if source_spm:
            self.spm_source = sentencepiece.SentencePieceProcessor()
            self.spm_source.Load(source_spm)
        if target_spm:
            self.spm_target = sentencepiece.SentencePieceProcessor()
            self.spm_target.Load(target_spm)

        # pre- and post-processing tools
        self.sentence_splitter = MosesSentenceSplitter(source_lang)
        self.normalizer = MosesPunctuationNormalizer(source_lang)

    @property
    def has_bpe(self):
        return self.bpe_source is not None


    def split_and_segment(self, source_text: str) -> List[str]:
        sentSource: list = self.sentence_splitter([self.normalizer(source_text)])
        sentences = []
        for s in sentSource:
            if self.has_bpe:
                tokens = ' '.join(self.tokenizer(s))
                sentences.append(self.bpe_source.process_line(tokens))
            else:
                sentences.append(' '.join(self.spm_source.EncodeAsPieces(s)))
        return sentences

    def postprocess(self, sentences: List[str]) -> List[str]:
        processed = []
        for index, s in enumerate(sentences):
            received = s.strip().split(' ||| ')
            r = received[0]
            # undo segmentation
            if self.spm_target:
                translated = self.spm_target.DecodePieces(r.split(' '))
            elif self.bpe_source:
                translated = self.detokenizer(r.replace('@@ ', '').split().split())
            else:
                raise NotImplementedError('dont expect to hit this')
                translated = r.replace(' ','').replace('▁',' ').strip()
            processed.append(translated)
        return processed

    def _append_special_tokens_and_truncate(self, raw_text: str, max_length: int,) -> List[int]:
        tokenized_text: str = self.split_and_segment(raw_text)
        ids: list = self.convert_tokens_to_ids(tokenized_text)[:max_length]
        return ids + [self.eos_token_id]

    def prepare_translation_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, Tensor]:
        """
        Arguments:
            src_texts: list of src language texts
            src_lang: default en_XX (english)
            tgt_texts: list of tgt language texts
            tgt_lang: default ro_RO (romanian)
            max_length: (None) defer to config (1024 for mbart-large-en-ro)
            pad_to_max_length: (bool)

        Returns:
            dict with keys input_ids, attention_mask, decoder_input_ids, each value is a torch.Tensor.
        """
        if max_length is None:
            max_length = self.max_len
        encoder_ids: list = [self._append_special_tokens_and_truncate(t, src_lang, max_length - 2) for t in src_texts]
        encoder_inputs = self.batch_encode_plus(
            encoder_ids,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
        )

        if tgt_texts is not None:
            decoder_ids = [self._append_special_tokens_and_truncate(t, tgt_lang, max_length - 2) for t in tgt_texts]
            decoder_inputs = self.batch_encode_plus(
                decoder_ids,
                add_special_tokens=False,
                return_tensors=return_tensors,
                max_length=max_length,
                pad_to_max_length=pad_to_max_length,
            )
        else:
            decoder_inputs = {}
        return {
            "input_ids": encoder_inputs["input_ids"],
            "attention_mask": encoder_inputs["attention_mask"],
            "decoder_input_ids": decoder_inputs.get("input_ids", None),
        }
