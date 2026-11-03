# Copyright 2026 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""Kerdock / Delsarte-Goethals codebook construction for Granite Switch.

Granite Switch routes adapters by writing an *address* into a small associative
memory and later reading it back out of an attention score. Addresses are
encoded as near-orthogonal unit vectors drawn from a Kerdock or
Delsarte-Goethals code, so that a dot product against the wrong address is
bounded by the code's mutual coherence rather than being arbitrary.

The codes are built as Z_4-linear codes over the Galois ring GR(4, m-1), then
pushed to binary via the Gray map, following Hammons et al., "The
Z_4-linearity of Kerdock, Preparata, Goethals, and related codes", IEEE Trans.
Inform. Theory 40(2), 1994.

Four configurations are supported (antipodal-free, i.e. the Z_4 constant term
is restricted to {0, 1}; see [`KerdockDGCodeGenerator`]):

| code    | m | N = 2^m | capacity | coherence |
|---------|---|---------|----------|-----------|
| Kerdock | 6 |      64 |    2,048 |     1/8   |
| DG(6,1) | 6 |      64 |   65,536 |     1/4   |
| Kerdock | 8 |     256 |   32,768 |     1/16  |
| DG(8,1) | 8 |     256 |  4.2M    |     1/8   |

Everything in this module runs once, on the host, during model construction.
`KerdockDGCodeGenerator.precompute_codebook` materialises the full
`[capacity, N]` codebook, which the model registers as a persistent buffer.
"""

import math

import numpy as np
import torch

from ...utils import logging


logger = logging.get_logger(__name__)


class GaloisRing:
    """Galois ring GR(4, m-1) = Z_4[X] / (h(X)).

    `h(X)` is a Hensel lift to Z_4 of a primitive polynomial over F_2, so that
    `h` divides `X^n - 1` with `n = 2^(m-1) - 1`. Polynomials are represented
    as little-endian numpy coefficient vectors, `coeffs[i]` being the
    coefficient of `X^i`.

    Args:
        m (`int`):
            Even integer >= 4. The binary code length is `N = 2^m`.
    """

    def __init__(self, m: int):
        if m % 2 != 0 or m < 4:
            raise ValueError(f"m must be even and >= 4, got {m}")

        self.m = m
        self.deg = m - 1  # Degree of the Galois ring extension.
        self.n = (1 << self.deg) - 1  # Order of the Teichmuller unit group.
        self.N = 1 << m  # Binary code length, after the Gray map.

        self.h2_coeffs = self._get_primitive_polynomial_f2(self.deg)
        self.h_coeffs = self._hensel_lift(self.h2_coeffs)
        self.teichmuller_set = self._build_teichmuller_set()
        self.trace_table_z4 = self._build_trace_table_z4()

        # F_2 trace tables are only needed by DG(m,r) correction terms, and
        # only for one power of xi per level, so they are built on demand.
        self._trace_table_f2_cache: dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------- setup

    def _get_primitive_polynomial_f2(self, deg: int) -> np.ndarray:
        """Return a primitive polynomial of degree `deg` over F_2.

        The returned array is `[a_0, a_1, ..., a_deg]` with `a_deg == 1`.
        """
        # sum_i a_i X^i encoded as the integer sum_i a_i 2^i.
        primitive_polys = {
            3: 0b1011,  # X^3 + X + 1
            5: 0b100101,  # X^5 + X^2 + 1
            7: 0b10001001,  # X^7 + X^3 + 1
            9: 0b1000010001,  # X^9 + X^4 + 1
        }
        if deg not in primitive_polys:
            raise NotImplementedError(f"No primitive polynomial tabulated for degree {deg}")

        poly_int = primitive_polys[deg]
        coeffs = np.zeros(deg + 1, dtype=np.uint8)
        for i in range(deg + 1):
            coeffs[i] = (poly_int >> i) & 1
        return coeffs

    def _hensel_lift(self, h2: np.ndarray) -> np.ndarray:
        """Lift a primitive polynomial from F_2 to Z_4.

        Returns `h` in Z_4[X] with `h == h2 (mod 2)` and `h | X^n - 1` in
        Z_4[X]. Lifts for the degrees this module needs are tabulated; any
        other degree falls back to the natural lift, which is checked and
        warned about rather than trusted.
        """
        known_lifts: dict[int, dict[tuple[int, ...], np.ndarray]] = {
            # X^3 + X + 1          ->  3 + X + 2X^2 + X^3
            3: {(1, 1, 0, 1): np.array([3, 1, 2, 1], dtype=np.uint8)},
            # X^5 + X^2 + 1        ->  3 + 2X + 3X^2 + X^5
            5: {(1, 0, 1, 0, 0, 1): np.array([3, 2, 3, 0, 0, 1], dtype=np.uint8)},
            # X^7 + X^3 + 1        ->  3 + X^3 + 2X^5 + X^7
            7: {(1, 0, 0, 1, 0, 0, 0, 1): np.array([3, 0, 0, 1, 0, 2, 0, 1], dtype=np.uint8)},
        }

        h2_tuple = tuple(int(c) for c in h2)
        deg = len(h2) - 1
        if deg in known_lifts and h2_tuple in known_lifts[deg]:
            return known_lifts[deg][h2_tuple]

        h_tilde = h2.copy().astype(np.uint8)

        # Verify X^n == 1 (mod h_tilde) by binary exponentiation.
        x_power = np.array([1], dtype=np.uint8)
        base = np.array([0, 1], dtype=np.uint8)  # X
        exponent = self.n
        while exponent > 0:
            if exponent & 1:
                x_power = self._poly_mod_z4(self._poly_mult_z4(x_power, base), h_tilde)
            base = self._poly_mod_z4(self._poly_mult_z4(base, base), h_tilde)
            exponent >>= 1

        if not (len(x_power) == 1 and x_power[0] == 1):
            logger.warning(
                f"Natural lift does not satisfy X^{self.n} == 1 (mod h); got {x_power}. The resulting code may "
                "have suboptimal coherence."
            )
        return h_tilde

    def _build_teichmuller_set(self) -> list[np.ndarray]:
        """Build the Teichmuller set T = {0, 1, z, z^2, ..., z^(n-1)}.

        Here `z = X mod h(X)`. `len(T) == 2^(m-1)`.
        """
        teichmuller = [
            np.array([0], dtype=np.uint8),
            np.array([1], dtype=np.uint8),
        ]
        zeta = np.array([0, 1], dtype=np.uint8)  # X
        current = zeta.copy()
        for i in range(2, self.n + 1):
            teichmuller.append(current.copy())
            if i < self.n:
                current = self._poly_mod_z4(self._poly_mult_z4(current, zeta), self.h_coeffs)
        return teichmuller

    def _build_trace_table_z4(self) -> np.ndarray:
        """Tabulate `tr(X^i * t)` for every `i < deg` and every `t` in T.

        Shape `[deg, len(T)]`. This turns the Z_4 trace into a matmul:
        `tr(b * t) = sum_i b[i] * table[i, t] (mod 4)`.
        """
        t_size = len(self.teichmuller_set)
        trace_table = np.zeros((self.deg, t_size), dtype=np.uint8)
        for i in range(self.deg):
            x_power = np.zeros(i + 1, dtype=np.uint8)
            x_power[i] = 1  # X^i
            for j, xi in enumerate(self.teichmuller_set):
                product = self._poly_mod_z4(self._poly_mult_z4(x_power, xi), self.h_coeffs)
                trace_table[i, j] = self.trace_z4(product)
        return trace_table

    def trace_table_f2(self, power: int) -> np.ndarray:
        """Tabulate `Tr(g_bar * t_bar^power)` over F_2, shape `[len(T), len(T)]`.

        Indexed `[gamma_index, xi_index]`, both over the Teichmuller set reduced
        mod 2. Used by the DG(m,r) correction terms. Results are cached.
        """
        cached = self._trace_table_f2_cache.get(power)
        if cached is not None:
            return cached

        t_size = len(self.teichmuller_set)
        reduced = [self.reduce_mod_2(t) for t in self.teichmuller_set]
        # t_bar^power depends only on `power` and the xi index, so hoist it out
        # of the gamma loop.
        xi_powers = [self.poly_power_f2(t_bar, power) for t_bar in reduced]

        table = np.zeros((t_size, t_size), dtype=np.uint8)
        for gamma_idx, gamma_bar in enumerate(reduced):
            for xi_idx, xi_power in enumerate(xi_powers):
                product = self._poly_mod_f2(self._poly_mult_f2(gamma_bar, xi_power), self.h2_coeffs)
                table[gamma_idx, xi_idx] = self.trace_f2(product)

        self._trace_table_f2_cache[power] = table
        return table

    # ------------------------------------------------- polynomials over F_2

    def _poly_add_f2(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = np.zeros(max(len(a), len(b)), dtype=np.uint8)
        result[: len(a)] = a
        result[: len(b)] ^= b
        return result

    def _poly_mult_f2(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = np.zeros(len(a) + len(b) - 1, dtype=np.uint8)
        for i in range(len(a)):
            if a[i]:
                for j in range(len(b)):
                    result[i + j] ^= (a[i] * b[j]) & 1
        return result

    def _poly_mod_f2(self, a: np.ndarray, modulus: np.ndarray) -> np.ndarray:
        a = a.copy()
        mod_deg = len(modulus) - 1
        while np.count_nonzero(a) > 0 and len(a) > mod_deg:
            if a[-1] == 0:
                a = a[:-1]
                continue
            shift = len(a) - len(modulus)
            for i in range(len(modulus)):
                a[shift + i] ^= modulus[i]
            a = a[:-1]
        return a if len(a) > 0 else np.array([0], dtype=np.uint8)

    def poly_power_f2(self, a: np.ndarray, exp: int) -> np.ndarray:
        """Compute `a^exp` in F_2[X] / (h2(X)) by binary exponentiation."""
        if exp == 0:
            return np.array([1], dtype=np.uint8)

        result = np.array([1], dtype=np.uint8)
        base = a.copy()
        while exp > 0:
            if exp & 1:
                result = self._poly_mod_f2(self._poly_mult_f2(result, base), self.h2_coeffs)
            base = self._poly_mod_f2(self._poly_mult_f2(base, base), self.h2_coeffs)
            exp >>= 1
        return result

    def reduce_mod_2(self, a: np.ndarray) -> np.ndarray:
        """Project GR(4, m-1) onto F_{2^(m-1)} by reducing coefficients mod 2."""
        return a % 2

    def trace_f2(self, a_bar: np.ndarray) -> int:
        """Field trace `Tr: F_{2^(m-1)} -> F_2`, i.e. `sum_i a^(2^i)`."""
        result = np.zeros(max(1, len(a_bar)), dtype=np.uint8)
        current = a_bar.copy()
        for _ in range(self.deg):
            result = self._poly_add_f2(result, current)
            current = self._poly_mod_f2(self._poly_mult_f2(current, current), self.h2_coeffs)
        return int(result[0] % 2)

    # ------------------------------------------------- polynomials over Z_4

    def _poly_add_z4(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = np.zeros(max(len(a), len(b)), dtype=np.uint8)
        result[: len(a)] = a
        result[: len(b)] = (result[: len(b)] + b) % 4
        return result

    def _poly_mult_z4(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = np.zeros(len(a) + len(b) - 1, dtype=np.uint8)
        for i in range(len(a)):
            if a[i]:
                for j in range(len(b)):
                    result[i + j] = (result[i + j] + a[i] * b[j]) % 4
        return result

    def _poly_mod_z4(self, a: np.ndarray, modulus: np.ndarray) -> np.ndarray:
        a = a.copy()
        mod_deg = len(modulus) - 1

        while len(a) > 1 and a[-1] == 0:
            a = a[:-1]

        while len(a) > mod_deg:
            lead = a[-1]
            if lead == 0:
                a = a[:-1]
                continue
            # `modulus` is monic, so the quotient term is just `lead * X^shift`.
            shift = len(a) - len(modulus)
            for i in range(len(modulus)):
                # int() avoids uint8 overflow in the intermediate product.
                a[shift + i] = (int(a[shift + i]) - int(lead) * int(modulus[i])) % 4
            a = a[:-1]
            while len(a) > 1 and a[-1] == 0:
                a = a[:-1]

        return a if len(a) > 0 else np.array([0], dtype=np.uint8)

    def frobenius(self, a: np.ndarray) -> np.ndarray:
        """Frobenius automorphism `a -> a^2` in the Galois ring."""
        return self._poly_mod_z4(self._poly_mult_z4(a, a), self.h_coeffs)

    def trace_z4(self, a: np.ndarray) -> int:
        """Galois ring trace `tr(a) = sum_{i<deg} frobenius^i(a)` in Z_4."""
        result = np.zeros(max(1, len(a)), dtype=np.uint8)
        current = a.copy()
        for _ in range(self.deg):
            result = self._poly_add_z4(result, current)
            current = self.frobenius(current)
        # The trace lands in Z_4, which is the constant term.
        return int(result[0] % 4)


class KerdockDGCodeGenerator:
    """Build the Kerdock / DG(m,1) codebook used to address Switch memory.

    A Kerdock codeword over Z_4 is `c_{a,b}(xi) = a + tr(b * xi)` evaluated at
    every `xi` in the Teichmuller set; DG(m,1) adds the half-strength
    correction `2 * Tr(gamma_bar * xi_bar^3)`. The Gray map then takes each Z_4
    symbol to two bits, giving a length-`N` binary word that is finally mapped
    to `+-1` and normalised.

    `a` is restricted to `{0, 1}` rather than all of Z_4: `c_{a,b}` and
    `c_{a+2,b}` are exact antipodes after the Gray map, so admitting both would
    put a pair of addresses at coherence 1 and make them indistinguishable.
    This halves the nominal capacity.

    Args:
        m (`int`):
            Either 6 or 8. Fixes the vector dimension `N = 2^m`.
        code_type (`str`, *optional*, defaults to `"kerdock"`):
            `"kerdock"` for K(m) = DG(m,0), `"dg1"` for DG(m,1). Typed `str`
            rather than a `Literal` because the only caller reads it from a
            checkpoint's JSON config; the value is validated below.
    """

    def __init__(self, m: int, code_type: str = "kerdock"):
        if m not in (6, 8):
            raise ValueError(f"m must be 6 or 8, got {m}")
        if code_type not in ("kerdock", "dg1"):
            raise ValueError(f"code_type must be 'kerdock' or 'dg1', got {code_type}")

        self.m = m
        self.code_type = code_type
        self.N = 2**m

        self.gr = GaloisRing(m)

        if code_type == "kerdock":
            self.capacity = 2 ** (2 * m - 1)  # 2 * 4^(m-1)
            self.coherence = 1.0 / math.sqrt(self.N)
        else:
            self.capacity = 2 ** (3 * m - 2)  # 2 * 4^(m-1) * 2^(m-1)
            self.coherence = 2.0 / math.sqrt(self.N)

    def precompute_codebook(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Materialise every code vector as a `[capacity, N]` tensor.

        The model registers the result as a persistent buffer so that address
        lookup is an index into a tensor, which `torch.compile` can trace.

        Raises:
            ValueError: if the codebook would exceed 128 MB, which DG(8,1) does
                at 4.2M vectors of dimension 256. Such a configuration needs an
                on-the-fly generator instead of a materialised table.
        """
        size_bytes = self.capacity * self.N * 4  # float32
        max_bytes = 128 * 1024 * 1024
        if size_bytes > max_bytes:
            raise ValueError(
                f"Codebook too large to precompute: {self.capacity} x {self.N} = "
                f"{size_bytes / 1024**2:.0f} MiB exceeds the {max_bytes // 1024**2} MiB limit. Use a smaller m or "
                "the 'kerdock' code type."
            )
        addresses = torch.arange(self.capacity, dtype=torch.long)
        return self._generate_code_vectors(addresses, dtype=dtype)

    def _generate_code_vectors(self, addresses: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Generate unit-norm code vectors for a batch of integer addresses.

        Args:
            addresses (`torch.Tensor`):
                `[B]` int64 addresses in `[0, capacity)`.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Output dtype.

        Returns:
            `torch.Tensor`: `[B, N]` unit-norm code vectors on `addresses.device`.
        """
        device = addresses.device
        batch = addresses.shape[0]
        t_size = len(self.gr.teichmuller_set)
        four_to_deg = 4**self.gr.deg

        trace_table_z4 = torch.from_numpy(self.gr.trace_table_z4).to(device=device, dtype=torch.float32)

        if self.code_type == "kerdock":
            kerdock_addrs = addresses
            gamma1_indices = None
        else:
            kerdock_size = 2 * four_to_deg  # a in {0, 1} only
            gamma1_indices = addresses // kerdock_size
            kerdock_addrs = addresses % kerdock_size

        # a in {0, 1}; b as a little-endian base-4 coefficient vector.
        a = kerdock_addrs // four_to_deg
        b_indices = kerdock_addrs % four_to_deg
        b_coeffs = torch.zeros((batch, self.gr.deg), device=device, dtype=torch.float32)
        for i in range(self.gr.deg):
            b_coeffs[:, i] = (b_indices % 4).to(torch.float32)
            b_indices = b_indices // 4

        # tr(b * xi) for every xi, as [B, deg] @ [deg, t_size]. Both operands
        # are integers in [0, 3] and deg <= 7, so the exact sum is at most 63
        # and float32 accumulation is exact.
        traces = torch.matmul(b_coeffs, trace_table_z4)
        z4_codewords = (a.unsqueeze(1) + traces.round().long()) % 4  # [B, t_size]

        if gamma1_indices is not None:
            # DG(m,1) correction: 2 * Tr(gamma_bar_1 * xi_bar^3).
            trace_table_f2 = torch.from_numpy(self.gr.trace_table_f2(3)).to(device=device, dtype=torch.long)
            corrections = 2 * trace_table_f2[gamma1_indices, :]  # [B, t_size]
            z4_codewords = (z4_codewords + corrections) % 4

        # Gray map Z_4 -> F_2^2: 0 -> (0,0), 1 -> (0,1), 2 -> (1,1), 3 -> (1,0).
        gray_map = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]], device=device, dtype=torch.float32)
        bits = gray_map[z4_codewords].reshape(batch, 2 * t_size)  # [B, N]

        # 0 -> +1, 1 -> -1, then normalise. Every word has the same norm
        # sqrt(N), but divide by the measured norm so the output is unit-norm
        # regardless of dtype rounding.
        signed = 1.0 - 2.0 * bits
        code_vectors = signed / torch.linalg.norm(signed, dim=1, keepdim=True)
        return code_vectors.to(dtype=dtype)


def recover_count_from_signal(counting_signal: torch.Tensor, capacity: int) -> torch.Tensor:
    """Invert the `1 / (1 + n)` counting signal back into the integer `n`.

    Granite Switch learns how many control tokens precede a position by reading
    a single attention score that equals `1 / (1 + n)`. Inversion is always
    done in float32: in bfloat16 the ULP is 0.5 over `[64, 128)`, so
    round-half-to-even turns neighbouring counts into the same value. float32
    inverts exactly up to roughly 8.4M.

    Every operation is a tensor op, so this stays on device and is traceable by
    `torch.compile`.

    Args:
        counting_signal (`torch.Tensor`):
            Attention score of shape `[...]`.
        capacity (`int`):
            Codebook capacity; counts are clamped into `[0, capacity)`.

    Returns:
        `torch.Tensor`: int64 tensor of counts, same shape as `counting_signal`.
    """
    count = 1.0 / counting_signal.float() - 1.0
    return torch.clamp(torch.round(count).long(), 0, capacity - 1)
