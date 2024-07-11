from transformers import PreTrainedTokenizerFast
import regex as re
import json
import base64
import os


class GLMTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(
            self,
            vocab_file,
            merges_file,
            tokenizer_file=None,
            **kwargs
    ):
        # Ensure the vocab_file and merges_file are passed to the base class constructor
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            **kwargs
        )
        self.vocab_file = vocab_file

        # Load mergeable ranks from the vocab file
        self.mergeable_ranks = {}
        with open(vocab_file, 'rb') as file:
            data = json.load(file)
            for key, value in data.items():
                self.mergeable_ranks[base64.b64decode(key.encode("utf-8")).decode("utf-8")] = value

        self.decoder = {rank: token for token, rank in self.mergeable_ranks.items()}
        self.n_words = len(self.decoder)

    @property
    def vocab_size(self):
        return self.n_words

    def get_vocab(self):
        """Returns vocab as a dict"""
        return {self._convert_id_to_token(i): i for i in range(self.vocab_size)}

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        vocab_file_path = os.path.join(save_directory,
                                       (filename_prefix + "-" if filename_prefix else "") + "vocab.json")
        merges_file_path = os.path.join(save_directory,
                                        (filename_prefix + "-" if filename_prefix else "") + "merges.txt")
        with open(vocab_file_path, 'w', encoding='utf-8') as f:
            json.dump({base64.b64encode(token.encode("utf-8")).decode("utf-8"): rank for token, rank in
                       self.mergeable_ranks.items()}, f, ensure_ascii=False)
        with open(merges_file_path, 'w', encoding='utf-8') as f:
            f.write("some merges data")

        return (vocab_file_path, merges_file_path)
