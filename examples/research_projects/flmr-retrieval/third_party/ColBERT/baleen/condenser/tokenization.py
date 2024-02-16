import torch

from transformers import ElectraTokenizerFast

class AnswerAwareTokenizer():
    def __init__(self, total_maxlen, bert_model='google/electra-base-discriminator'):
        self.total_maxlen = total_maxlen

        self.tok = ElectraTokenizerFast.from_pretrained(bert_model)

    def process(self, questions, passages, all_answers=None, mask=None):
        return TokenizationObject(self, questions, passages, all_answers, mask)

    def tensorize(self, questions, passages):
        query_lengths = self.tok(questions, padding='longest', return_tensors='pt').attention_mask.sum(-1)

        encoding = self.tok(questions, passages, padding='longest', truncation='longest_first',
                            return_tensors='pt', max_length=self.total_maxlen, add_special_tokens=True)

        return encoding, query_lengths

    def get_all_candidates(self, encoding, index):
        offsets, endpositions = self.all_word_positions(encoding, index)

        candidates = [(offset, endpos)
                      for idx, offset in enumerate(offsets)
                      for endpos in endpositions[idx:idx+10]]

        return candidates

    def all_word_positions(self, encoding, index):
        words = encoding.word_ids(index)
        offsets = [position
                   for position, (last_word_number, current_word_number) in enumerate(zip([-1] + words, words))
                   if last_word_number != current_word_number]

        endpositions = offsets[1:] + [len(words)]

        return offsets, endpositions

    def characters_to_tokens(self, text, answers, encoding, index, offset, endpos):
        # print(text, answers, encoding, index, offset, endpos)
        # endpos = endpos - 1

        for offset_ in range(offset, len(text)+1):
            tokens_offset = encoding.char_to_token(index, offset_)
            # print(f'tokens_offset = {tokens_offset}')
            if tokens_offset is not None:
                break

        for endpos_ in range(endpos, len(text)+1):
            tokens_endpos = encoding.char_to_token(index, endpos_)
            # print(f'tokens_endpos = {tokens_endpos}')
            if tokens_endpos is not None:
                break

        # None on whitespace!
        assert tokens_offset is not None, (text, answers, offset)
        # assert tokens_endpos is not None, (text, answers, endpos)
        tokens_endpos = tokens_endpos if tokens_endpos is not None else len(encoding.tokens(index))

        return tokens_offset, tokens_endpos

    def tokens_to_answer(self, encoding, index, text, tokens_offset, tokens_endpos):
        # print(encoding, index, text, tokens_offset, tokens_endpos, len(encoding.tokens(index)))

        char_offset = encoding.word_to_chars(index, encoding.token_to_word(index, tokens_offset)).start

        try:
            char_next_offset = encoding.word_to_chars(index, encoding.token_to_word(index, tokens_endpos)).start
            char_endpos = char_next_offset
        except:
            char_endpos = encoding.word_to_chars(index, encoding.token_to_word(index, tokens_endpos-1)).end

        assert char_offset is not None
        assert char_endpos is not None

        return text[char_offset:char_endpos].strip()


class TokenizationObject():
    def __init__(self, tokenizer: AnswerAwareTokenizer, questions, passages, answers=None, mask=None):
        assert type(questions) is list and type(passages) is list
        assert len(questions) in [1, len(passages)]

        if mask is None:
            mask = [True for _ in passages]

        self.mask = mask

        self.tok = tokenizer
        self.questions = questions if len(questions) == len(passages) else questions * len(passages)
        self.passages = passages
        self.answers = answers

        self.encoding, self.query_lengths = self._encode()
        self.passages_only_encoding, self.candidates, self.candidates_list = self._candidize()

        if answers is not None:
            self.gold_candidates = self.answers  # self._answerize()

    def _encode(self):
        return self.tok.tensorize(self.questions, self.passages)

    def _candidize(self):
        encoding = self.tok.tok(self.passages, add_special_tokens=False)

        all_candidates = [self.tok.get_all_candidates(encoding, index) for index in range(len(self.passages))]

        bsize, maxcands = len(self.passages), max(map(len, all_candidates))
        all_candidates = [cands + [(-1, -1)] * (maxcands - len(cands)) for cands in all_candidates]

        candidates = torch.tensor(all_candidates)
        assert candidates.size() == (bsize, maxcands, 2), (candidates.size(), (bsize, maxcands, 2), (self.questions, self.passages))

        candidates = candidates + self.query_lengths.unsqueeze(-1).unsqueeze(-1)

        return encoding, candidates, all_candidates
