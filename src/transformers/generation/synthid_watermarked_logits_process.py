# coding=utf-8
# Copyright 2020 Google DeepMind
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import torch

from .logits_process import LogitsProcessor


class SynthIDTextWatermarkState:
    """SynthID watermarking state."""

    def __init__(
        self,
        batch_size: int,
        ngram_len: int,
        context_history_size: int,
        device: torch.device,
    ):
        """Initializes the state.

        Args:
            batch_size: Batch size.
            ngram_len: Ngram length.
            context_history_size: Size of the tensor to keep track of seen contexts.
            device: Device to use.
        """
        self.context = torch.zeros(
            (batch_size, ngram_len - 1),
            dtype=torch.int64,
            device=device,
        )
        self.context_history = torch.zeros(
            (batch_size, context_history_size),
            dtype=torch.int64,
            device=device,
        )
        self.num_calls = 0


class SynthIDTextWatermarkLogitsProcessor(LogitsProcessor):
    """TODO: Provide information here once all the code bits are in."""

    def __init__(
        self,
        ngram_len: int,
        keys: List[int],
        sampling_table_size: int,
        sampling_table_seed: int,
        context_history_size: int,
        device: torch.device,
        skip_first_ngram_calls: bool = False,
        debug_mode: bool = False,
    ):
        """Initializes the logits processor.

        Args:
            ngram_len: Ngram length.
            keys: A sequence of watermarking keys, one for each depth.
            sampling_table_size: Size of the sampling table.
            sampling_table_seed: Random seed to generate the sampling table.
            context_history_size: Size of the tensor to keep track of seen contexts.
            device: Device to use.
            skip_first_ngram_calls: Whether to skip first ngram calls.
            debug_mode: Logits are modified to uniform one got before watermarking
            modification is applied. This is to test the implementation.
        """
        self.ngram_len = ngram_len
        self.keys = torch.tensor(keys, device=device)

        generator = torch.Generator(device=device).manual_seed(sampling_table_seed)
        self.sampling_table = torch.randint(
            low=0,
            high=2,
            size=(sampling_table_size,),
            generator=generator,
            device=device,
        )
        self.context_history_size = context_history_size
        self.device = device
        self.state = None
        self.skip_first_ngram_calls = skip_first_ngram_calls
        self.debug_mode = debug_mode

    def _init_state(self, batch_size: int):
        """Initializes the state."""
        self.state = SynthIDTextWatermarkState(
            batch_size=batch_size,
            ngram_len=self.ngram_len,
            context_history_size=self.context_history_size,
            device=self.device,
        )

    def update_scores(
        self,
        scores: torch.FloatTensor,
        g_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Updates scores using the g values.

        We assume that the scores are in the log space.
        Args:
            scores: Scores (batch_size, vocab_size).
            g_values: G valus (batch_size, vocab_size, depth).

        Returns:
            Updated scores (batch_size, vocab_size).
        """
        _, _, depth = g_values.shape
        device = scores.device

        probs = torch.softmax(scores, dim=1)

        for i in range(depth):
            g_values_at_depth = g_values[:, :, i]
            g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
            probs = probs * (1 + g_values_at_depth - g_mass_at_depth)

        log_probs = torch.log(probs)
        log_probs = torch.where(
            torch.isfinite(log_probs), log_probs, torch.tensor(-1e12, device=device)
        )
        return log_probs
    

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """Calls the logits processor statefully.

        This function computes top_k internally and returns the indices mapping
        from top_k scores to dense scores.

        Args:
            input_ids: Input token ids (batch_size, inputs_len).
            scores: Scores (batch_size, vocab_size).

        Returns:
            Tuple of
                Watermarked updated scores (batch_size, top_k)
                Top k indices (batch_size, top_k).
                original scores for perplexity calculations (batch_size, top_k)
        """
        self._check_input_ids_shape(input_ids)
        batch_size, vocab_size = scores.shape

        if self.debug_mode:
            scores = torch.ones_like(scores)

        # TODO: Compute top_k and then process the scores.
        # Currently indices is just a arange to compute watermarking on the desnse logits.
        all_indices = torch.stack([
            torch.arange(vocab_size, device=self.device)
            for _ in range(batch_size)
        ])

        if self.state is None:
            # Initialize watermarking state if it does not exist.
            self._init_state(batch_size)
        else:
            # Append last input id (which is the input id added in last call) to the
            # previous context so we have the context to be used for current
            # watermarking. 
            self.state.context = torch.concat(
                (self.state.context, input_ids[:, -1:]),
                dim=1,
            )
            self.state.context = self.state.context[:, 1:]

        assert self.state is not None

        self.state.num_calls += 1

        # Don't watermark the first ngram_len - 1 tokens if set.
        if self.skip_first_ngram_calls and self.state.num_calls < self.ngram_len:
            return scores

        # 2. Generate random keys for each ngram key combination.
        ngram_keys, hash_result_with_just_context = self._compute_keys(
            self.state.context, all_indices
        )
        # ngram_keys shape [batch_size, top_k, depth]

        # 3. Sample g values.
        g_values = self.sample_g_values(ngram_keys)
        # g_values shape [batch_size, top_k, depth]

        # 4. Modify scores.
        updated_scores = self.update_scores(scores, g_values)
        # updated scores shape [batch_size, top_k]

        # 5. Check if the current watermarking context was previously used, if
        # yes skip watermarking.
        hash_result_with_just_context = hash_result_with_just_context[:, None]
        is_repeated_context = (
            self.state.context_history == hash_result_with_just_context
        ).any(
            dim=1,
            keepdim=True,
        )
        self.state.context_history = torch.concat(
            (hash_result_with_just_context, self.state.context_history),
            dim=1,
        )[:, :-1]

        updated_watermarked_scores = torch.where(
            is_repeated_context,
            input=scores,
            other=updated_scores,
        )
        return updated_watermarked_scores
    
    def accumulate_hash(
        self,
        current_hash: torch.LongTensor,
        data: torch.LongTensor,
        multiplier: int = 6364136223846793005,
        increment: int = 1,
    ) -> torch.LongTensor:
        """Accumulate hash of data on current hash.

        Method uses adapted linear congruential generator with newlib/musl parameters.

        This function has following property -
        f(x, data[T]) = f(f(x, data[:T - 1]), data[T])

        This function expects current_hash.shape and data.shape[:-1] to
        match/broadcastable.

        Args:
            current_hash: (shape,)
            data: (shape, tensor_len)
            multiplier: (int) multiplier of linear congruential generator
            increment: (int) increment of linear congruential generator

        Returns:
            upadted hash (shape,)
        """
        for i in range(data.shape[-1]):
            current_hash = torch.add(current_hash, data[..., i])
            current_hash = torch.mul(current_hash, multiplier)
            current_hash = torch.add(current_hash, increment)
        return current_hash

    def compute_ngram_keys(
        self,
        ngrams: torch.LongTensor,
    ) -> torch.LongTensor:
        """Computes random keys for each ngram and depth.

        Args:
            ngrams: Ngrams (batch_size, num_ngrams, ngram_len).

        Returns:
            ngram keys (batch_size, num_ngrams, depth).
        """
        if len(ngrams.shape) != 3:
            raise ValueError(
                "Ngrams should be of shape (batch_size, num_ngrams, ngram_len), but"
                f" is {ngrams.shape}"
            )
        if ngrams.shape[2] != self.ngram_len:
            raise ValueError(
                "Ngrams should be of shape (batch_size, num_ngrams, ngram_len),"
                f" where ngram_len is {self.ngram_len}, but is {ngrams.shape}"
            )
        batch_size, _, _ = ngrams.shape

        hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
        # hash_result shape [batch_size,]
        # ngrams shape [batch_size, num_ngrams, ngram_len]
        hash_result = torch.vmap(
            self.accumulate_hash, in_dims=(None, 1), out_dims=1
        )(hash_result, ngrams)
        # hash_result shape [batch_size, num_ngrams]

        keys = self.keys[None, None, :, None]
        # hash_result shape [batch_size, num_ngrams]
        # keys shape [1, 1, depth, 1]
        hash_result = torch.vmap(
            self.accumulate_hash, in_dims=(None, 2), out_dims=2
        )(hash_result, keys)
        # hash_result shape [batch_size, num_ngrams, depth]

        return hash_result

    def _compute_keys(
        self,
        n_minus_1_grams: torch.LongTensor,
        indices: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Computes random keys for each ngram and depth.

        Args:
            n_minus_1_grams: Ngrams (batch_size, ngram_len - 1).
            indices: indices of the continuations (batch_size, num_indices)

        Returns:
            Ngram keys (batch_size, num_indices, depth).
        """
        batch_size, _ = n_minus_1_grams.shape

        hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
        # First hash n_minus_1 gram, for each batch entry we have a single
        # n_minus_1 gram context.
        # hash_result shape [batch_size]
        # n_minus_1_gram shape [batch_size, ngram_len - 1]
        hash_result_with_just_context = self.accumulate_hash(
            hash_result, n_minus_1_grams
        )
        # hash_result shape [batch_size,]
        # Indices is of shape [batch_size, num_indices], so we make it
        # [batch_size, num_indices, 1] so we can vmap over num_indices dim.
        hash_result = torch.vmap(
            self.accumulate_hash, in_dims=(None, 1), out_dims=1
        )(hash_result_with_just_context, indices[:, :, None])
        # hash_result shape [batch_size, num_indices]
        # Basically we have a hash for each batch entry and each indices
        # Now we add watermarking keys to this hash.
        # keys are of shape [depth,]
        # We add batch, num_indices and data dimension to this making it
        # [1, 1, depth, 1].
        # So we can vmap over the depth dimension for compute_hash
        keys = self.keys[None, None, :, None]
        hash_result = torch.vmap(
            self.accumulate_hash, in_dims=(None, 2), out_dims=2
        )(hash_result, keys)
        # hash_result shape should be [batch_size, num_indices, depth]
        return hash_result, hash_result_with_just_context

    def sample_g_values(self, ngram_keys: torch.LongTensor) -> torch.LongTensor:
        """Samples g values from Bernoulli distribution.

        It is not possible to pass random keys in a vectorized way in torch. Instead
        we pre-compute a random sampling table, and use apply modulo table size to
        map from ngram keys (int64) to g values.

        Args:
            ngram_keys: Random keys (batch_size, num_ngrams, depth).

        Returns:
            G values (batch_size, num_ngrams, depth).
        """
        (sampling_table_size,) = self.sampling_table.shape
        sampling_table = self.sampling_table.reshape((1, 1, sampling_table_size))
        ngram_keys = ngram_keys % sampling_table_size
        return torch.take_along_dim(sampling_table, indices=ngram_keys, dim=2)

    def _check_input_ids_shape(self, input_ids: torch.LongTensor):
        """Checks the shape of input ids."""
        if len(input_ids.shape) != 2:
            raise ValueError(
                "Input ids should be of shape (batch_size, input_len), but is"
                f" {input_ids.shape}"
            )

    def compute_g_values(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Computes g values for each ngram from the given sequence of tokens.

        Args:
            input_ids: Input token ids (batch_size, input_len).

        Returns:
            G values (batch_size, input_len - (ngram_len - 1), depth).
        """
        self._check_input_ids_shape(input_ids)
        ngrams = input_ids.unfold(dimension=1, size=self.ngram_len, step=1)
        ngram_keys = self.compute_ngram_keys(ngrams)
        return self.sample_g_values(ngram_keys)

    def compute_context_repetition_mask(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Computes repetition mask.

        0 and 1 stand for repeated and not repeated context n-1 grams respectively.

        Args:
            input_ids: Input token ids (batch_size, input_len).

        Returns:
            Repetitions mask (batch_size, input_len - (ngram_len - 1)).
        """
        self._check_input_ids_shape(input_ids)
        batch_size, _ = input_ids.shape
        state = SynthIDTextWatermarkState(
            batch_size=batch_size,
            ngram_len=self.ngram_len,
            context_history_size=self.context_history_size,
            device=self.device,
        )
        contexts = input_ids[:, :-1].unfold(
            dimension=1,
            size=self.ngram_len - 1,
            step=1,
        )
        _, num_contexts, _ = contexts.shape

        are_repeated_contexts = []
        for i in range(num_contexts):
            context = contexts[:, i, :]
            hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
            context_hash = self.accumulate_hash(hash_result, context)[
                :, None
            ]
            is_repeated_context = (state.context_history == context_hash).any(
                dim=1,
                keepdim=True,
            )
            are_repeated_contexts.append(is_repeated_context)
            state.context_history = torch.concat(
                (context_hash, state.context_history),
                dim=1,
            )[:, :-1]
        are_repeated_contexts = torch.concat(are_repeated_contexts, dim=1)

        return torch.logical_not(are_repeated_contexts)

    def compute_eos_token_mask(
        self,
        input_ids: torch.LongTensor,
        eos_token_id: int,
    ) -> torch.LongTensor:
        """Computes repetitions mask.

        1 stands for ngrams that don't contain EOS tokens and vice versa.

        Args:
            input_ids: Input token ids (batch_size, input_len).
            eos_token_id: EOS token ID.

        Returns:
            EOS token mask (batch_size, input_len).
        """
        self._check_input_ids_shape(input_ids)
        noneos_masks = []
        all_eos_equated = input_ids == eos_token_id
        for eos_equated in all_eos_equated:
            nonzero_idx = torch.nonzero(eos_equated)
            noneos_mask = torch.ones_like(eos_equated)
            if nonzero_idx.shape[0] != 0:
                noneos_mask[nonzero_idx[0][0] :] = 0
            noneos_masks.append(noneos_mask)
        return torch.stack(noneos_masks, dim=0)
    
    def expected_mean_g_value(
        self,
        vocab_size: int,
        coinflip_prob: float = 0.5,
    ) -> float:
        """Compute expected mean g-value after watermarking, assuming uniform LM dist.

        This is the theoretical expected value for single-layer watermarking.

        Args:
            vocab_size: The size of the vocabulary.
            coinflip_prob: Probability of 1 in boolean prf.

        Returns:
            The expected mean g-value for watermarked text.
        """
        return coinflip_prob + coinflip_prob * (1 - coinflip_prob) * (
            1 - (1 / vocab_size)
        )
