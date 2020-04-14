from .tokenization_utils import PreTrainedTokenizer
import sentencepiece
from mosestokenizer import MosesTokenizer, MosesDetokenizer, MosesSentenceSplitter, MosesPunctuationNormalizer

from .apply_bpe import BPE
from .tokenization_xlm_roberta import XLMRobertaTokenizer
from typing import Dict, List, Tuple, Optional
from torch import Tensor

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab": {
        #"marian/en-de": "https://s3.amazonaws.com/models.huggingface.co/bert/marian/en-de/source.spm"
    }
}
import yaml

def load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.BaseLoader)

class MarianSPTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"source_spm": 'source.spm',  'target_spm': 'target.spm',
                         'vocab': 'opus.spm32k-spm32k.vocab.yml',
                         'tokenizer_config_file': 'tokenizer_config.json',
                         #'source_bpe': 'source.bpe',
                         #'target_bpe': 'target_bpe'
                         }

    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = {m: 512 for m in PRETRAINED_VOCAB_FILES_MAP}

    def __init__(self, vocab=None, source_spm=None, target_spm=None, source_lang=None,
                 target_lang=None, unk_token='<unk>', eos_token='</s>',
                 pad_token='<pad>',
                 max_len=512,

                 **kwargs):

        super().__init__(
            #bos_token=bos_token,
            max_len=max_len,
            eos_token=eos_token,
            unk_token=unk_token,
            #sep_token=sep_token,
            #cls_token=cls_token,
            pad_token=pad_token,
            #mask_token=mask_token,
            #**kwargs,
        )
        self.encoder = {k: int(v) for k, v in load_yaml(vocab).items()}
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.source_lang = source_lang
        self.target_lang = target_lang

        # load SentencePiece model for pre-processing

        self.spm_source = sentencepiece.SentencePieceProcessor()
        self.spm_source.Load(source_spm)

        self.spm_target = sentencepiece.SentencePieceProcessor()
        self.spm_target.Load(target_spm)

        # pre- and post-processing tools
        #self.sentence_splitter = MosesSentenceSplitter(source_lang)  # TODO(SS): this would require lots of book-keeping.
        self.normalizer = MosesPunctuationNormalizer(source_lang)

    @property
    def has_bpe(self):
        return self.bpe_source is not None

    def _convert_token_to_id(self, token):
        return self.encoder[token]

    def _convert_id_to_token(self, index: int):
        """Converts an index (integer) in a token (str) using the encoder."""
        return self.decoder.get(index, self.unk_token)

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

    def _append_special_tokens_and_truncate(self, tokens: str, max_length: int, ) -> List[int]:
        ids: list = self.convert_tokens_to_ids(tokens)[:max_length]
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
        src_texts = [self.spm_source.encode_as_pieces(self.normalizer(t)) for t in src_texts]
        if tgt_texts is not None:
            tgt_texts = [self.spm_target.encode_as_pieces(self.normalizer(t)) for t in tgt_texts]
        encoder_ids: list = [self._append_special_tokens_and_truncate(t,  max_length - 1) for t in src_texts]
        encoder_inputs = self.batch_encode_plus(
            encoder_ids,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
        )
        model_inputs = {
            "input_ids": encoder_inputs["input_ids"],
            "attention_mask": encoder_inputs["attention_mask"],
        }
        if tgt_texts is None:
            return model_inputs


        decoder_ids = [self._append_special_tokens_and_truncate(t, max_length - 1) for t in tgt_texts]
        decoder_inputs = self.batch_encode_plus(
            decoder_ids,
            add_special_tokens=False,
            return_tensors=return_tensors,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
        )

        model_inputs["decoder_input_ids"] =  decoder_inputs["input_ids"]
        #model_inputs["decoder_attention_mask"] = decoder_inputs["decoder_attention_mask"]
        return model_inputs

