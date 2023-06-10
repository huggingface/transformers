from typing import Optional, List

import numpy as np


class AudiocraftPatternProvider:
    def __init__(self, num_codebooks: int = 4, delays: Optional[List[int]] = None):
        """
        Args:
            num_codebooks (`int`, *optional*, defaults to 4):
                Number of codebooks.
            delays (`List[int]`, *optional*):
                Delay for each of the codebooks. If delays are not defined, each codebook is delayed by 1 compared to
                the previous one.
        """

        if delays is None:
            delays = list(range(num_codebooks))
        else:
            if sorted(delays) != delays:
                raise ValueError("Delays should increase with codebooks. Ensure delays are monotonically increasing")
            if len(delays) != num_codebooks:
                raise ValueError(f"Require one delay for each codebook, got {len(delays)} delays and {num_codebooks} codebooks")

        self.delays = delays
        self.num_codebooks = num_codebooks

        self.max_delay = max(delays)

    def get_delay_pattern(self, seq_len: int):
        # first timestep has no delay (empty list)
        pattern = [[]]
        for time in range(0, seq_len + self.max_delay):
            timestep = []
            for codebook, delay in enumerate(self.delays):
                offset_time = time - delay
                if offset_time >= 0:
                    timestep.append([offset_time, codebook])
            pattern.append(timestep)
        return pattern

    def build_pattern_sequence(self, gen_codes: np.ndarray, special_token_id: int, keep_only_valid_steps: bool = False):
        """Build sequence corresponding to the pattern from generated codebook array gen_codes.

        Args:
            gen_codes (`np.ndarray` of shape `(batch_size, num_codebooks, seq_length)`): Input array of multi-codebooks
                sequence.
            special_token_id (`int`): Special token used to fill non-pattern coordinates in the new sequence.
            keep_only_valid_steps (bool): Build a sequence from the pattern up to valid (= fully defined) steps.
                Steps that are beyond valid steps will be replaced by the special_token_id.
        Returns:
            values (`np.ndarray` of shape `(batch_size, num_codebooks, seq_length)`): Interleaved sequence matching the delay pattern.
            mask (`np.ndarray` of shape `(num_codebooks, seq_length)`): Mask corresponding to indexes that matches valid indexes in the delay pattern.
        """
        bsz, num_codebooks, seq_len = gen_codes.shape

        if num_codebooks != self.num_codebooks:
            raise ValueError(f"Expected {self.num_codebooks} codebooks, got {num_codebooks}")

        gen_codes = np.reshape(gen_codes, (bsz, -1))
        # append the special token id as the last index of our flattened generated codebook array
        gen_codes = np.concatenate([gen_codes, np.full(gen_codes[:, :1], fill_value=special_token_id)], axis=1)

        delay_pattern = self.get_delay_pattern(seq_len)

        if keep_only_valid_steps:
            # discard the delay tokens appended at the end of our pattern
            valid_step = len(delay_pattern) - self.max_delay
            delay_pattern = delay_pattern[:valid_step]

        indices = np.ones((num_codebooks, len(delay_pattern)), dtype=int) * num_codebooks * seq_len
        mask = np.zeros((num_codebooks, len(delay_pattern)), dtype=bool)
        # fill indexes with last sequence step value that will correspond to the special token id (last token index)
        # iterate over the pattern and fill scattered indexes and mask
        for s, sequence_coords in enumerate(delay_pattern):
            for coords in sequence_coords:
                offset_time, codebook = coords
                if offset_time < seq_len:
                    indices[codebook, s] = offset_time + codebook * seq_len
                    mask[codebook, s] = True

        values = gen_codes[:, indices.flatten()]
        values = np.reshape(values, (bsz, num_codebooks, mask.shape[-1]))
        return values, mask


    # TODO(SG): check correctness of this function
    def revert_pattern_sequences(self, gen_codes: np.ndarray, special_token_id: int, keep_only_valid_steps: bool = False, is_model_output: bool = False):
        """Revert a sequence built from the pattern back to the original multi-codebook sequence without interleaving.
        The sequence is reverted using up to timesteps if specified, and non-pattern coordinates
        are filled with the special token.

        Args:
            gen_codes (`np.ndarray` of shape `(batch_size, num_codebooks, seq_length)`): Interleaved sequence array
            for the multi-codebook ids obtained from the pattern generator.
            special_token_id (`int`): Special token used to fill non-pattern coordinates in the new sequence.
            keep_only_valid_steps (bool): Build a sequence from the pattern up to valid (= fully defined) steps.
                Steps that are beyond valid steps will be replaced by the special_token_id.
            is_model_output (bool): Whether to keep the sequence item corresponding to initial special token or not.
        Returns:
            values (`np.ndarray` of shape `(batch_size, num_codebooks, seq_length)`): Interleaved sequence matching the delay pattern.
            mask (`np.ndarray` of shape `(num_codebooks, seq_length)`): Mask corresponding to indexes that matches valid indexes in the delay pattern.
        """
        bsz, num_codebooks, seq_len = gen_codes.shape

        if num_codebooks != self.num_codebooks:
            raise ValueError(f"Expected {self.num_codebooks} codebooks, got {num_codebooks}")

        gen_codes = np.reshape(gen_codes, (bsz, -1))
        # append the special token id as the last index of our flattened generated codebook array
        gen_codes = np.concatenate([gen_codes, np.full(gen_codes[:, :1], fill_value=special_token_id)], axis=1)

        delay_pattern = self.get_delay_pattern(seq_len)

        valid_step = len(delay_pattern) - self.max_delay
        if keep_only_valid_steps:
            # discard the delay tokens appended at the end of our pattern
            delay_pattern = delay_pattern[:valid_step]

        if is_model_output:
            # discard the first special token
            delay_pattern = delay_pattern[1:]

        indices = np.zeros((num_codebooks, len(delay_pattern)), dtype=int)
        mask = np.zeros((num_codebooks, len(delay_pattern)), dtype=bool)
        # fill indexes with last sequence step value that will correspond to the special token id (last token index)
        # iterate over the pattern and fill scattered indexes and mask
        for s, sequence_coords in enumerate(delay_pattern):
            if s < seq_len:
                for coords in sequence_coords:
                    offset_time, codebook = coords
                    if offset_time < valid_step:
                        indices[codebook, offset_time] = s + codebook * seq_len
                        mask[codebook, offset_time] = True

        values = gen_codes[:, indices.flatten()]
        values = np.reshape(values, (bsz, num_codebooks, mask.shape[-1]))
        return values, mask

    def get_first_step_at_timestep(self, seq_len: int, codebook: Optional[int] = None):
        """Get the first timestep in the pattern layout that corresponds to the specified time index seq_len
        and optionally to a given codebook.
        """
        if codebook is not None and codebook > self.num_codebooks:
            raise ValueError(f"Provided number of codebooks {codebook} is greater than the pattern's number of codebooks {self.num_codebooks}")

        coords = []
        for step, seq_codes in enumerate(self.get_delay_pattern(seq_len)):
            for code in seq_codes:
                offset_time, codebook_id = code
                if offset_time == seq_len and (codebook is None or codebook_id == codebook):
                    coords.append(step)

        if len(coords) > 0:
            return coords[0]
        else:
            raise ValueError(f"Timestep {seq_len} not present in pattern")

