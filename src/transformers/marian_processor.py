
from .apply_bpe import BPE
from mosestokenizer import MosesSentenceSplitter, MosesPunctuationNormalizer, MosesTokenizer, MosesDetokenizer
import sentencepiece

class ContentProcessor():
    def __init__(self,  srclang,
            targetlang, sourcebpe=None, targetbpe=None,sourcespm=None,targetspm=None):
        self.bpe_source = None
        self.bpe_target = None
        self.sp_processor_source = None
        self.sp_processor_target = None
        self.tokenizer = None
        self.detokenizer = None
        self.sentences=[]
        # load BPE model for pre-processing
        if sourcebpe:
            # print("load BPE codes from " + sourcebpe, flush=True)
            BPEcodes = open(sourcebpe, 'r', encoding="utf-8")
            self.bpe_source = BPE(BPEcodes)

            self.tokenizer = MosesTokenizer(srclang)
            self.detokenizer = MosesDetokenizer(targetlang)
        if targetbpe:
            # print("load BPE codes from " + targetbpe, flush=True)
            BPEcodes = open(targetbpe, 'r', encoding="utf-8")
            self.bpe_target = BPE(BPEcodes)

        # load SentencePiece model for pre-processing
        if sourcespm:
            # print("load sentence piece model from " + sourcespm, flush=True)
            self.sp_processor_source = sentencepiece.SentencePieceProcessor()
            self.sp_processor_source.Load(sourcespm)
        if targetspm:
            # print("load sentence piece model from " + targetspm, flush=True)
            self.sp_processor_target = sentencepiece.SentencePieceProcessor()
            self.sp_processor_target.Load(targetspm)

        # pre- and post-processing tools
        self.sentence_splitter = MosesSentenceSplitter(srclang)
        self.normalizer = MosesPunctuationNormalizer(srclang)

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

