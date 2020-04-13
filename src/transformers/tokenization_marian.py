from .tokenization_utils import PreTrainedTokenizer
import sentencepiece
from mosestokenizer import MosesTokenizer, MosesDetokenizer, MosesSentenceSplitter, MosesPunctuationNormalizer

from .apply_bpe import BPE
from .tokenization_xlm_roberta import XLMRobertaTokenizer

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
        self.vocab = yaml.load(open(vocab), Loader=yaml.BaseLoader)

        self.bpe_source = None
        self.bpe_target = None
        self.sp_processor_source = None
        self.sp_processor_target = None
        self.tokenizer = None
        self.detokenizer = None
        self.sentences = []
        # load BPE model for pre-processing
        if source_bpe:
            # print("load BPE codes from " + source_bpe, flush=True)
            BPEcodes = open(source_bpe, 'r', encoding="utf-8")
            self.bpe_source = BPE(BPEcodes)

            self.tokenizer = MosesTokenizer(source_lang)

            self.detokenizer = MosesDetokenizer(target_lang)
        if target_bpe:
            # print("load BPE codes from " + target_bpe, flush=True)
            BPEcodes = open(target_bpe, 'r', encoding="utf-8")
            self.bpe_target = BPE(BPEcodes)

        # load SentencePiece model for pre-processing
        if source_spm:
            # print("load sentence piece model from " + source_spm, flush=True)
            self.sp_processor_source = sentencepiece.SentencePieceProcessor()
            self.sp_processor_source.Load(source_spm)
        if target_spm:
            # print("load sentence piece model from " + target_spm, flush=True)
            self.sp_processor_target = sentencepiece.SentencePieceProcessor()
            self.sp_processor_target.Load(target_spm)

        # pre- and post-processing tools
        self.sentence_splitter = MosesSentenceSplitter(source_lang)
        self.normalizer = MosesPunctuationNormalizer(source_lang)

    def preprocess(self, srctxt):
        sentSource = self.sentence_splitter([self.normalizer(srctxt)])
        self.sentences=[]
        for s in sentSource:
            if self.tokenizer:
                # print('raw sentence: ' + s, flush=True)
                tokenized = ' '.join(self.tokenizer(s))
                # print('tokenized sentence: ' + tokenized, flush=True)
                segmented = self.bpe_source.process_line(tokenized)
            elif self.sp_processor_source:
                print('raw sentence: ' + s, flush=True)
                segmented = ' '.join(self.sp_processor_source.EncodeAsPieces(s))
                # print(segmented, flush=True)
            self.sentences.append(segmented)
        return self.sentences

    def postprocess(self, sentences):
        sentTranslated = []
        for index, s in enumerate(sentences):
            received = s.strip().split(' ||| ')
            # print(received, flush=True)

            # undo segmentation
            if self.bpe_source:
                translated = received[0].replace('@@ ','')
            elif self.sp_processor_target:
                translated = self.sp_processor_target.DecodePieces(received[0].split(' '))
            else:
                translated = received[0].replace(' ','').replace('▁',' ').strip()

            # self.parse_alignments(index, received)

            if self.detokenizer:
                detokenized = self.detokenizer(translated.split())
            else:
                detokenized = translated

            sentTranslated.append(detokenized)
        return sentTranslated
