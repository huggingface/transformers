# coding=utf-8
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np


# even when tokens themselves have whitespace, for most tasks we want our offsets to not include the whitespace
def whitespace_reduce(offsets, text):
    for ti in range(offsets.shape[0]):
        nstart = offsets[ti, 0]
        nend = offsets[ti, 1]
        while nstart < nend and text[nstart].isspace():
            nstart += 1
        while nstart < nend and text[nend - 1].isspace():
            nend -= 1
        if nstart < nend:
            # assert text[offsets[ti, 0]:offsets[ti, 1]].strip() == text[nstart:nend]
            offsets[ti, 0] = nstart
            offsets[ti, 1] = nend


UNMATCHABLE_TOKENS = [u'�']  # consider putting in tokenization_utils, so subclasses can override


def match_back_by_length(tlens, offsets, tstart, tend):
    # split the chunk, dividing the chunk's span proportionally among the tokens
    if tend - tstart <= 1:
        return
    # when this is called, all offsets[tstart:tend] are the same (the same chunk)
    text_end = offsets[tstart, 1]
    prev_end = offsets[tstart, 0]
    tok_len_remaining = sum(tlens[tstart:tend]) + 0.1
    for ci in range(tstart, tend):
        text_remaining = text_end - prev_end
        token_len = tlens[ci] + 0.1/(tend-tstart)
        offsets[ci, 0] = prev_end
        if ci < tend - 1:
            scale = text_remaining / tok_len_remaining
            leni = int(round(scale * token_len))
            offsets[ci, 1] = min(text_end, offsets[ci, 0] + leni)
            prev_end = offsets[ci, 1]
        tok_len_remaining -= token_len


def match_back_by_text(tokens, text, tlens, offsets, tstart, tend):
    match_start = tstart
    match_end = tend - 1
    txt_start = offsets[match_start, 0]
    txt_end = offsets[match_end, 1]
    orig_txt_start = txt_start
    orig_txt_end = txt_end
    # try to find the token strings in the original text
    text = text.lower()  # any other length preserving normalizing?
    while match_start <= match_end and txt_start < txt_end:
        findndx = text.find(tokens[match_start].lower(), txt_start, txt_end)
        pre_skip = findndx - txt_start
        rfindndx = text.rfind(tokens[match_end].lower(), txt_start, txt_end)
        post_skip = txt_end - rfindndx - len(tokens[match_end])
        # do we skip more of the text by matching the first token or the last token?
        # we want to greedily make good matches
        if findndx != -1 and (rfindndx == -1 or pre_skip <= post_skip):
            offsets[match_start, 0] = findndx
            offsets[match_start, 1] = offsets[match_start, 0] + len(tokens[match_start])
            txt_start = offsets[match_start, 1]
            match_start += 1
        elif rfindndx != -1:
            offsets[match_end, 0] = rfindndx
            offsets[match_end, 1] = offsets[match_end, 0] + len(tokens[match_end])
            txt_end = offsets[match_end, 0]
            match_end -= 1
        else:
            break
    # we matched everything. good job!
    if match_start > match_end:
        return
    # we messed up, hand it all to the match_by_length
    if txt_start > txt_end or sum(tlens[match_start:match_end+1]) > txt_end - txt_start:
        txt_start = orig_txt_start
        txt_end = orig_txt_end
        match_start = tstart
        match_end = tend-1
    # anything leftover we match by length
    for leftover in range(match_start, match_end + 1):
        offsets[leftover, 0] = txt_start
        offsets[leftover, 1] = txt_end
    match_back_by_length(tlens, offsets, match_start, match_end + 1)
    # DEBUG: show what we came up with
    # print('*' * 10)
    # print(f'{text[offsets[tstart, 0]:offsets[tend-1, 1]]}')
    # for ti in range(tstart, tend):
    #    print(f'"{tokens[ti]}" "{text[offsets[ti, 0]:offsets[ti, 1]]}"')
    # print('*' * 10)


# when a chunk becomes multiple tokens, we find what spans of the chunk each token corresponds to
def multitoken_chunk_offsets(tokens, text, tlens, offsets, tstart, tend):
    if tend - tstart <= 1:
        return
    if offsets[tstart, 1] - offsets[tstart, 0] == sum(tlens[tstart:tend]):
        # the sum of token length is the chunk length, just chop it up
        match_back_by_length(tlens, offsets, tstart, tend)
    else:
        # try matching the token text from the start and end, then punt on the middle
        match_back_by_text(tokens, text, tlens, offsets, tstart, tend)


# handle chunks that have multiple tokens
def tokens_in_chunks(tokens, text, offsets):
    assert len(tokens) == offsets.shape[0]
    same_chunk_start = 0
    tlens = [len(t) if t not in UNMATCHABLE_TOKENS else 0 for t in tokens]
    for i in range(1, len(tokens)):
        if not np.array_equal(offsets[same_chunk_start, :], offsets[i, :]):
            multitoken_chunk_offsets(tokens, text, tlens, offsets, same_chunk_start, i)
            same_chunk_start = i
    multitoken_chunk_offsets(tokens, text, tlens, offsets, same_chunk_start, len(tokens))


def finalize_token_offsets(detokenized_tokens, text, offsets):
    """
    detokenized_tokens are tokens that can usually be matched by to the text (best effort).
    The text is just the original text, before any cleaning by the tokenizer.
    The offsets are currently chunk offsets, multiple tokens have the same offsets.
    We will give the tokens non-overlapping offsets that match the text as closely as possible.
    :param detokenized_tokens: tokens from the tokenizer, with artifacts removed to match the original text better
    :param text: the original text, before our preprocessing got its hands on it
    :param offsets: len(tokens) x 2, giving start and end for each token,
                    initially this is the start and end of the chunk
    :return: the numpy array of offsets is modified
    """
    # adjust token offsets to not include leading/trailing whitespace
    whitespace_reduce(offsets, text)
    # find offsets for sub-chunk tokens
    tokens_in_chunks(detokenized_tokens, text, offsets)

    # asserts on non-overlapping spans with start <= end
    for i in range(offsets.shape[0]):
        assert offsets[i, 0] <= offsets[i, 1]
        if i > 0:
            assert offsets[i-1, 1] <= offsets[i, 0]
