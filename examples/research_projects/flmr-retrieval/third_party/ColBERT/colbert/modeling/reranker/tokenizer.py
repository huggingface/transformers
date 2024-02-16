from transformers import AutoTokenizer

class RerankerTokenizer():
    def __init__(self, total_maxlen, base):
        self.total_maxlen = total_maxlen
        self.tok = AutoTokenizer.from_pretrained(base)

    def tensorize(self, questions, passages):
        assert type(questions) in [list, tuple], type(questions)
        assert type(passages) in [list, tuple], type(passages)

        encoding = self.tok(questions, passages, padding='longest', truncation='longest_first',
                            return_tensors='pt', max_length=self.total_maxlen, add_special_tokens=True)

        return encoding
