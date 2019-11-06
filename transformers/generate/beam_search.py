# coding=utf-8
# MIT License

# Copyright (c) 2017-Present OpenNMT

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Use Beam Search to generate sequences using encoder-decoder models.
"""
import torch
from torch import nn

import logging


logger = logging.getLogger(__name__)


class BeamSearch(nn.Module):
    def __init__(
        self,
        model,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        batch_size,
        beam_size,
        min_length,
        max_length,
        alpha=0,
        block_repeating_trigrams=True,
        device=torch.device("cpu"),
    ):
        r"""
        Inputs:
            **model**: instance of ``transformers.PreTrainedEncoderDecoder``
                The pretrained encoder-decoder model that will be used to generate the sequences.
            **batch_size**: (`optional`) int
                Batch size of the inputs. The value is set automatically when calling `forward`.
            **beam_size**: int
                Number of beams that are used for each element on the batch.
            **min_length**: int
                Minimum number of steps performed by the beam search before terminating.
            **max_length**: int
                Maximum number of steps performed by the beam search. Any beam that has not finished
                will return its current solution with the highest probability. The sequence that is
                returned has a length of max_length-1 to account for the end token that is subsequently added.
            **alpha**: float
                Parameter of the length penalty. Read the documentation of the `_length_penalty` method for mode details.
            **block_repeating_trigrams**: bool
                Whether to block sequences that have repeating 3-grams.
        """
        super(BeamSearch, self).__init__()
        self.model = model
        self.device = device

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.batch_size = batch_size
        self.beam_size = beam_size
        self.min_length = min_length
        self.max_length = max_length

        self.block_repeating_trigram = block_repeating_trigrams
        self.apply_length_penalty = False if alpha == 0 else True
        self.alpha = alpha

        self._init_beam_state(batch_size)

    def __len__(self):
        try:
            return self.growing_beams.size(1)
        except NameError:
            return 0

    def _init_beam_state(self, batch_size):
        """ (re-)Initialize the state of the beams. """
        self.hypotheses = [[] for _ in range(batch_size)]
        self.batch_offset = torch.arange(batch_size, dtype=torch.long, device=self.device)
        self.beam_offset = torch.arange(
            0,
            batch_size * self.beam_size,
            step=self.beam_size,
            dtype=torch.long,
            device=self.device,
        )
        self.growing_beams = torch.full(
            (batch_size * self.beam_size, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        self.topk_log_probabilities = torch.tensor(
            [0.0] + [float("-inf")] * (self.beam_size - 1),
            dtype=torch.float,
            device=self.device,
        ).repeat(batch_size)
        self.results = {
            "predictions": [[] for _ in range(batch_size)],
            "scores": [[] for _ in range(batch_size)],
        }
        self._step = 0
        self.is_done = False

    def forward(self, encoder_input_ids, **model_kwargs):
        """ Generate a sequence using Beam Search. """
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.
        kwargs_common = {
            argument: value
            for argument, value in model_kwargs.items()
            if not argument.startswith("encoder_") and not argument.startswith("decoder_")
        }
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder = kwargs_common.copy()
        kwargs_encoder.update(
            {
                argument[len("encoder_") :]: value
                for argument, value in model_kwargs.items()
                if argument.startswith("encoder_")
            }
        )
        kwargs_decoder.update(
            {
                argument[len("decoder_") :]: value
                for argument, value in model_kwargs.items()
                if argument.startswith("decoder_")
            }
        )

        # forward pass on the encoder
        encoder_outputs = self.model.encoder(encoder_input_ids, **kwargs_encoder)
        encoder_hidden_states = encoder_outputs[0]
        kwargs_decoder["encoder_hidden_states"] = tile(
            encoder_hidden_states, self.beam_size, dim=0
        )
        kwargs_decoder["encoder_attention_mask"] = tile(
            kwargs_encoder["attention_mask"], self.beam_size, dim=0
        )

        # grow the beam by generating sequences in an autoregressive way
        batch_size, block_size = encoder_input_ids.size()
        self._init_beam_state(batch_size)
        for step in range(self.max_length):
            # Add padding tokens
            decoder_input = torch.full(
                (self.growing_beams.size(0), block_size),
                self.pad_token_id,
                dtype=torch.long,
                device=self.growing_beams.device,
            )
            decoder_input[:, : self.growing_beams.size(1)] = self.growing_beams

            # compute decoder_attention_mask
            decoder_mask = torch.ones_like(decoder_input)
            idx_pad_tokens = decoder_input == self.pad_token_id
            decoder_mask[idx_pad_tokens] = 0
            kwargs_decoder["attention_mask"] = decoder_mask

            outputs = self.model.decoder(decoder_input, **kwargs_decoder)
            last_token_scores = outputs[0][:, -1, :].squeeze(1)
            log_probabilities = torch.nn.functional.log_softmax(last_token_scores, dim=0)
            surviving_beams_rows = self.grow(log_probabilities)
            if self.is_done:
                break

            kwargs_decoder["encoder_hidden_states"] = kwargs_decoder[
                "encoder_hidden_states"
            ].index_select(0, surviving_beams_rows)
            kwargs_decoder["encoder_attention_mask"] = kwargs_decoder[
                "encoder_attention_mask"
            ].index_select(0, surviving_beams_rows)

        return self.results

    def grow(self, log_probabilities):
        """ Grow the beams by one step. """
        self._step += 1

        # The number of beams changes as some beams finish so we define _B
        vocab_size = log_probabilities.size(-1)
        _B = log_probabilities.size(0) // self.beam_size

        # Multiply each beam probability with the probability of the
        # next token (conditioned on the words in the beam).
        log_probabilities += self.topk_log_probabilities.view(-1, 1)

        self._enforce_min_length(log_probabilities)
        if self.block_repeating_trigram:
            self._remove_beams_with_repeating_trigrams(log_probabilities, _B)

        # Find the `beam_size` (previous_beam + token) combinations with
        # the highest score
        self.topk_log_probabilities, topk_ids = torch.topk(
            log_probabilities.view(_B, self.beam_size * vocab_size), self.beam_size, dim=1
        )

        # Apply the length penalty. The +1 accounts for the [EOS] token
        # that will be added if the beam ends.
        topk_scores = self.topk_log_probabilities
        if self.apply_length_penalty:
            topk_scores /= self._length_penalty()

        # Retrieve the corresponding respective beam and token id
        # topk_token_ids[i] will be added to topk_beam_ids[i]
        topk_beam_ids = topk_ids.div(vocab_size)
        topk_token_ids = topk_ids.fmod(vocab_size)

        # Retrieve the row index of the surviving beams in the original
        # view of the log_probabilities tensor
        surviving_beams_per_batch = topk_beam_ids + self.beam_offset[:_B].view(-1, 1)
        surviving_beams_rows = surviving_beams_per_batch.view(-1)

        # Append the last predictions
        self.growing_beams = torch.cat(
            [
                self.growing_beams.index_select(0, surviving_beams_rows),
                topk_token_ids.view(-1, 1),
            ],
            1,
        )

        # Check if any of the beam searches has ended during this
        # growth step. Also if top beam (most probable) has ended
        # for one element of the batch.
        is_finished = topk_token_ids.eq(self.eos_token_id)
        self._enforce_max_length(is_finished)
        if is_finished.any():
            non_finished = self._cut_finished(is_finished, topk_scores)
            self.batch_offset = self.batch_offset.index_select(0, non_finished)
            surviving_beams_per_batch = surviving_beams_per_batch.index_select(
                0, non_finished
            )
            self.topk_log_probabilities = self.topk_log_probabilities.index_select(
                0, non_finished
            )

            surviving_beams_rows = surviving_beams_per_batch.view(-1)
            self.growing_beams = self.growing_beams.index_select(0, surviving_beams_rows)

        return surviving_beams_rows

    def _cut_finished(self, is_finished, topk_scores):
        """ Save the finished searches and cut the correponding sequences off
        the beams. """
        is_top_beam_finished = is_finished[:, 0].eq(True)

        # Save the finished searches
        predictions = self.growing_beams.view(
            -1, self.beam_size, self.growing_beams.size(1)
        )
        for i in range(is_finished.size(0)):
            if is_top_beam_finished[i]:
                is_finished[i].fill_(1)
            finished_hyp = is_finished[i].nonzero().view(-1)

            # Store the finished beams as a (score, prediction) hypothesis.
            b = self.batch_offset[i]
            for j in finished_hyp:
                self.hypotheses[b].append((topk_scores[i, j], predictions[i, j, :]))

            # If the batch reached the end, save the best hypotheses
            # in terms of length-penalized score.
            if is_top_beam_finished[i]:
                best_score, best_prediction = max(self.hypotheses[b], key=lambda x: x[0])
                self.results["scores"][b].append(best_score)
                self.results["predictions"][b].append(best_prediction)

        non_finished = is_top_beam_finished.eq(False).nonzero().view(-1)
        if len(non_finished) == 0:
            self.is_done = True

        return non_finished

    def _remove_beams_with_repeating_trigrams(self, log_probabilities, _B):
        if self._step + 1 > 3:  # [BOS] does not count
            for i in range(_B * self.beam_size):
                tokens = self.growing_beams[i]
                trigrams = [
                    (tokens[j - 1], tokens[j], tokens[j + 1])
                    for j in range(1, len(self) - 1)
                ]
                last_trigram = tuple(trigrams[-1])
                if last_trigram in trigrams[:-1]:
                    log_probabilities[i] = -1e20

    def _enforce_min_length(self, log_probabilities):
        if self._step < self.min_length:
            log_probabilities[:, self.eos_token_id] = -1e20

    def _enforce_max_length(self, is_finished):
        # +1 because we will need to add an [EOS] token
        if self._step + 1 == self.max_length:
            is_finished.fill_(1)

    def _length_penalty(self):
        """ The calculation of the length penalty follows that of [1].

        [1] Wu, Yonghui, et al. "Google's neural machine translation system:
        Bridging the gap between human and machine translation." arXiv preprint
        arXiv:1609.08144 (2016).
        """
        return ((5.0 + (self._step + 1)) / 6.0) ** self.alpha


def tile(x, count, dim=0):
    """
    Tiles `x` along dimension `dim` `count` times.

    Example:
        >> ex = torch.tensor([1,2],[3,4])
        >> tile(ex, 2, 0)
        torch.Tensor([[1,2],[1,2],[3,4],[3,4]])
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def fit_to_block_size(sequence, block_size, pad_token_id):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter we append padding tokens to the right.
    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        return torch.cat(
            (sequence, torch.tensor([pad_token_id] * (block_size - len(sequence)))), dim=0
        )


def build_lm_labels(sequence, pad_token_id):
    """ Padding token, encoded as 0, are represented by the value -1 so they
    are not taken into account in the loss computation. """
    padded = sequence.clone()
    padded[padded == pad_token_id] = -1
    return padded


def build_mask(sequence, pad_token_id):
    """ Builds the mask. The attention mechanism will only attend to positions
    with value 1. """
    mask = torch.ones_like(sequence)
    idx_pad_tokens = sequence == pad_token_id
    mask[idx_pad_tokens] = 0
    return mask
