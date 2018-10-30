"""
Prepare input data for Google's BERT Model.

Contains some functions from tensor2tensor library: https://github.com/tensorflow/tensor2tensor

"""
from typing import NamedTuple, List, Union, Tuple

TokenizedSentence = List[str]
TokenizedInput = Union[Tuple[TokenizedSentence, TokenizedSentence], TokenizedSentence]

class DataProcessor():
    def __init__(self, vocab_path):
        self.encoder_file_path = encoder_file_path
        self.token_indexer = json.load(open(vocab_path))


    def tokenize(text):
    """Encode a unicode string as a list of tokens.

    Args:
        text: a unicode string
    Returns:
        a list of tokens as Unicode strings
    """
    if not text:
        return []
    ret = []
    token_start = 0
    # Classify each character in the input string
    is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
    for pos in range(1, len(text)):
        if is_alnum[pos] != is_alnum[pos - 1]:
        token = text[token_start:pos]
        if token != u" " or token_start == 0:
            ret.append(token)
        token_start = pos
    final_token = text[token_start:]
    ret.append(final_token)
    return ret


    def detokenize(tokens):
    """Decode a list of tokens to a unicode string.

    Args:
        tokens: a list of Unicode strings
    Returns:
        a unicode string
    """
    token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
    ret = []
    for i, token in enumerate(tokens):
        if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
        ret.append(u" ")
        ret.append(token)
    return "".join(ret)

    def encode(input_sentences: List[TokenizedInput]) -> np.array:
        """ Prepare a torch.Tensor of inputs for BERT model from a string.

        Args:
            input_sentences: list of
                - pairs of tokenized sentences (sentence_A, sentence_B) or
                - tokenized sentences (will be considered as sentence_A only)

        Return:
            Numpy array of formated inputs for BERT model
        """
        batch_size = sum(min(len(x), n_perso_permute) for x in X1)
        input_mask = np.zeros((n_batch, n_cands, n_ctx), dtype=np.float32)
        input_array = np.zeros((n_batch, n_cands, n_ctx, 3), dtype=np.int32)
        i = 0
        for tokenized_input in input_sentences:
            x1j, lxj, lperso, lhisto, dialog_embed = format_transformer_input(x1, x2, xcand_j, text_encoder,
                                                                                dialog_embed_mode, max_len=max_len,
                                                                                add_start_stop=True)
            lmj = len(xcand_j[:max_len]) + 1
            xmb[i, j, :lxj, 0] = x1j
            if dialog_embed_mode == 1 or dialog_embed_mode == 2:
                xmb[i, j, :lxj, 2] = dialog_embed
            mmb[i, j, :lxj] = 1
            if fix_lm_index: # Take one before so we don't predict from classify token...
                mmb_eval[i, j, (lxj-lmj-1):lxj-1] = 1 # This one only mask the response so we get the perplexity on the response only
            else:
                mmb_eval[i, j, (lxj-lmj):lxj] = 1 # This one only mask the response so we get the perplexity on the response only
            xmb[i, j, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
            i += 1
        return input_array, input_mask