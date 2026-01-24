from transformers.models.rag.tokenization_rag import RagTokenizer


class DummyTokenizer:
    def __init__(self, encode_output):
        self._encode_output = encode_output

    def __call__(self, *args, **kwargs):
        return {"input_ids": self._encode_output}

    def encode(self, *args, **kwargs):
        return self._encode_output

    def convert_tokens_to_ids(self, token):
        return 42 if token == "<patch>" else 0


def test_rag_tokenizer_encode_forwards_to_current_tokenizer():
    q = DummyTokenizer([1, 2, 3])
    g = DummyTokenizer([4, 5])
    tok = RagTokenizer(question_encoder=q, generator=g)

    # default: question_encoder
    assert tok.encode("hi") == [1, 2, 3]

    tok._switch_to_target_mode()
    try:
        # target: generator
        assert tok.encode("hi") == [4, 5]
    finally:
        tok._switch_to_input_mode()

    # back to input: question_encoder
    assert tok.encode("hi") == [1, 2, 3]


def test_rag_tokenizer_patch_token_and_id_follow_current_tokenizer():
    q = DummyTokenizer([1])
    g = DummyTokenizer([2])
    tok = RagTokenizer(question_encoder=q, generator=g)

    # default: question_encoder
    tok.patch_token = "<patch>"
    assert q.patch_token == "<patch>"
    assert tok.patch_token == "<patch>"
    assert tok.patch_token_id == 42

    tok.patch_token_id = 7
    assert q.patch_token_id == 7
    assert tok.patch_token_id == 7

    tok._switch_to_target_mode()
    try:
        # target: generator
        tok.patch_token = "<patch>"
        assert g.patch_token == "<patch>"
        assert tok.patch_token == "<patch>"
        assert tok.patch_token_id == 42

        tok.patch_token_id = 9
        assert g.patch_token_id == 9
        assert tok.patch_token_id == 9
    finally:
        tok._switch_to_input_mode()

    # back to input: should still see question_encoder's values
    assert tok.patch_token_id == 7
