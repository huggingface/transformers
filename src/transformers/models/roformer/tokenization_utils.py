from typing import List

from tokenizers import NormalizedString, PreTokenizedString, normalizers


class JiebaPreTokenizer:
    def __init__(self, vocab) -> None:
        self.vocab = vocab
        self.normalizers = normalizers.BertNormalizer(
            clean_text=False,
            handle_chinese_chars=True,
            strip_accents=False,
            lowercase=False,
        )
        try:
            import jieba
        except ImportError:
            raise ImportError(
                "You need to install jieba to use RoFormerTokenizer."
                "See https://pypi.org/project/jieba/ for installation."
            )
        self.jieba = jieba

    def jieba_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        splits = []
        for token in self.jieba.cut(str(normalized_string), HMM=False):
            if token in self.vocab:
                splits.append(NormalizedString(token))
            else:
                new_normalized_string = self.normalizers.normalize_str(token)
                new_normalized_string_list = new_normalized_string.split()
                if len(new_normalized_string_list) == 1:
                    splits.append(NormalizedString(new_normalized_string))
                else:
                    for new_normalized_string in new_normalized_string_list:
                        if new_normalized_string:
                            splits.append(NormalizedString(new_normalized_string))

        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.jieba_split)
