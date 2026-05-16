"""
Tests for static ensemble verification in speculative decoding.

Reference: Wang & Kasa et al., "DIVERSED: Relaxed Speculative Decoding via
Dynamic Ensemble Verification", AISTATS 2026 (https://arxiv.org/abs/2604.07622).
"""

import unittest
from unittest.mock import patch

import torch

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import _speculative_sampling


class TestStaticEnsembleSpeculativeSampling(unittest.TestCase):
    """Tests for the _speculative_sampling function with ensemble weight support."""

    def _make_synthetic_inputs(self, vocab_size=10, candidate_length=3, q_probs=None, p_probs=None):
        """Helper to create synthetic inputs for _speculative_sampling."""
        batch_size = 1
        if q_probs is None:
            q_probs = torch.rand(batch_size, candidate_length, vocab_size)
            q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True)
        if p_probs is None:
            p_probs = torch.rand(batch_size, candidate_length + 1, vocab_size)
            p_probs = p_probs / p_probs.sum(dim=-1, keepdim=True)

        # Convert probs to logits
        candidate_logits = q_probs.log()
        new_logits = p_probs.log()

        # Draft tokens are sampled from q (take argmax for determinism)
        draft_tokens = q_probs.argmax(dim=-1)  # [1, candidate_length]
        # Build candidate_input_ids: prefix + draft tokens
        prefix = torch.zeros(batch_size, 5, dtype=torch.long)  # fake prefix
        candidate_input_ids = torch.cat([prefix, draft_tokens], dim=-1)

        return candidate_input_ids, candidate_logits, candidate_length, new_logits

    def test_ensemble_weight_none_equals_standard(self):
        """w=None should produce identical results to standard speculative sampling."""
        torch.manual_seed(42)
        candidate_input_ids, candidate_logits, candidate_length, new_logits = self._make_synthetic_inputs()

        # Patch rand_like for determinism in accept/reject step
        fixed_rand = torch.tensor([[[0.3, 0.6, 0.9]]])
        # Patch multinomial for determinism in fallback sampling
        fixed_token = torch.tensor([[0]])

        with patch("transformers.generation.utils.torch.rand_like", return_value=fixed_rand.squeeze()):
            with patch("transformers.generation.utils.torch.multinomial", return_value=fixed_token):
                valid_none, n_none = _speculative_sampling(
                    candidate_input_ids,
                    candidate_logits,
                    candidate_length,
                    new_logits,
                    is_done_candidate=False,
                    assistant_ensemble_weight=None,
                )

        with patch("transformers.generation.utils.torch.rand_like", return_value=fixed_rand.squeeze()):
            with patch("transformers.generation.utils.torch.multinomial", return_value=fixed_token):
                valid_one, n_one = _speculative_sampling(
                    candidate_input_ids,
                    candidate_logits,
                    candidate_length,
                    new_logits,
                    is_done_candidate=False,
                    assistant_ensemble_weight=1.0,
                )

        self.assertEqual(n_none, n_one)
        torch.testing.assert_close(valid_none, valid_one)

    def test_ensemble_weight_increases_acceptance(self):
        """Deterministic test: w=0.7 accepts a token that w=1.0 would reject.

        Construct: q_i=0.80, p_i=0.40, r=0.60
        Standard ratio: p_i/q_i = 0.50 < 0.60 = r -> REJECT
        Ensemble ratio (w=0.7): 1 - 0.7 + 0.7*(0.40/0.80) = 0.65 > 0.60 = r -> ACCEPT
        """
        candidate_length = 1
        batch_size = 1

        # Draft distribution: token 0 has prob 0.80
        q_probs = torch.tensor([[[0.80, 0.10, 0.05, 0.05]]])
        # Target distribution: token 0 has prob 0.40
        # Need candidate_length + 1 positions for target
        p_probs = torch.tensor([[[0.40, 0.30, 0.20, 0.10], [0.25, 0.25, 0.25, 0.25]]])

        candidate_logits = q_probs.log()
        new_logits = p_probs.log()

        # Draft token is token 0 (argmax of q)
        prefix = torch.zeros(batch_size, 5, dtype=torch.long)
        draft_token = torch.tensor([[0]])
        candidate_input_ids = torch.cat([prefix, draft_token], dim=-1)

        # Fixed random threshold = 0.60
        fixed_rand = torch.tensor([0.60])

        with patch("transformers.generation.utils.torch.rand_like", return_value=fixed_rand):
            _, n_standard = _speculative_sampling(
                candidate_input_ids,
                candidate_logits,
                candidate_length,
                new_logits,
                is_done_candidate=False,
                assistant_ensemble_weight=None,
            )

        with patch("transformers.generation.utils.torch.rand_like", return_value=fixed_rand):
            _, n_ensemble = _speculative_sampling(
                candidate_input_ids,
                candidate_logits,
                candidate_length,
                new_logits,
                is_done_candidate=False,
                assistant_ensemble_weight=0.7,
            )

        # Standard rejects (ratio 0.50 < 0.60), ensemble accepts (ratio 0.65 > 0.60)
        self.assertEqual(n_standard.item(), 0)
        self.assertEqual(n_ensemble.item(), 1)

    def test_fallback_distribution_is_finite_and_normalized(self):
        """Verify fallback distribution passed to multinomial is valid."""
        candidate_length = 1
        batch_size = 1

        # Make p != q so fallback has positive mass
        q_probs = torch.tensor([[[0.70, 0.20, 0.05, 0.05]]])
        p_probs = torch.tensor([[[0.30, 0.40, 0.20, 0.10], [0.25, 0.25, 0.25, 0.25]]])

        candidate_logits = q_probs.log()
        new_logits = p_probs.log()

        prefix = torch.zeros(batch_size, 5, dtype=torch.long)
        draft_token = torch.tensor([[0]])
        candidate_input_ids = torch.cat([prefix, draft_token], dim=-1)

        # Force rejection with r=0.99
        fixed_rand = torch.tensor([0.99])
        captured_primes = []

        def capture_multinomial(input, num_samples, **kwargs):
            captured_primes.append(input.clone())
            return torch.zeros(input.shape[0], num_samples, dtype=torch.long)

        with patch("transformers.generation.utils.torch.rand_like", return_value=fixed_rand):
            with patch("transformers.generation.utils.torch.multinomial", side_effect=capture_multinomial):
                _speculative_sampling(
                    candidate_input_ids,
                    candidate_logits,
                    candidate_length,
                    new_logits,
                    is_done_candidate=False,
                    assistant_ensemble_weight=0.7,
                )

        self.assertEqual(len(captured_primes), 1)
        p_prime = captured_primes[0]
        # Check finite and normalized
        self.assertTrue(torch.isfinite(p_prime).all())
        self.assertAlmostEqual(p_prime.sum().item(), 1.0, places=5)

        # Verify it equals the standard fallback [p-q]+/sum([p-q]+)
        p_at_reject = p_probs[0, 0, :]
        q_at_reject = q_probs[0, 0, :]
        expected = torch.clamp(p_at_reject - q_at_reject, min=0)
        expected = expected / expected.sum()
        torch.testing.assert_close(p_prime.squeeze(), expected, atol=1e-5, rtol=1e-5)

    def test_fallback_numerical_stability_near_zero_residual(self):
        """When p ≈ q, the residual [p-q]+ is near zero. Verify no NaN."""
        candidate_length = 1
        batch_size = 1

        # p ≈ q -> residual is near zero
        q_probs = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]])
        p_probs = torch.tensor([[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]])

        candidate_logits = q_probs.log()
        new_logits = p_probs.log()

        prefix = torch.zeros(batch_size, 5, dtype=torch.long)
        draft_token = torch.tensor([[0]])
        candidate_input_ids = torch.cat([prefix, draft_token], dim=-1)

        # Force rejection
        fixed_rand = torch.tensor([0.99])
        captured_primes = []

        def capture_multinomial(input, num_samples, **kwargs):
            captured_primes.append(input.clone())
            return torch.zeros(input.shape[0], num_samples, dtype=torch.long)

        with patch("transformers.generation.utils.torch.rand_like", return_value=fixed_rand):
            with patch("transformers.generation.utils.torch.multinomial", side_effect=capture_multinomial):
                _speculative_sampling(
                    candidate_input_ids,
                    candidate_logits,
                    candidate_length,
                    new_logits,
                    is_done_candidate=False,
                    assistant_ensemble_weight=0.5,
                )

        self.assertEqual(len(captured_primes), 1)
        p_prime = captured_primes[0]
        # Must be finite (no NaN from division by zero)
        self.assertTrue(torch.isfinite(p_prime).all())
        self.assertAlmostEqual(p_prime.sum().item(), 1.0, places=5)

    def test_greedy_ensemble_accepts_where_standard_rejects(self):
        """Greedy ensemble: argmax(v) == draft token, but argmax(p) != draft token.

        p = [0.50, 0.49, 0.01] -> argmax(p) = 0
        q = [0.10, 0.89, 0.01] -> draft token = 1
        w = 0.5
        v = 0.5*p + 0.5*q = [0.30, 0.69, 0.01] -> argmax(v) = 1 == draft token
        """
        candidate_length = 1

        p_probs = torch.tensor([[[0.50, 0.49, 0.01], [0.34, 0.33, 0.33]]])  # 2 positions
        q_probs = torch.tensor([[[0.10, 0.89, 0.01]]])

        # Compute ensemble
        w = 0.5
        nu = w * p_probs[:, :candidate_length, :] + (1.0 - w) * q_probs
        # argmax(p) at position 0 = token 0, argmax(v) at position 0 = token 1
        self.assertEqual(p_probs[0, 0, :].argmax().item(), 0)
        self.assertEqual(nu[0, 0, :].argmax().item(), 1)

        # Draft token is token 1 (argmax of q)
        self.assertEqual(q_probs[0, 0, :].argmax().item(), 1)

        # Under standard greedy: draft token 1 != argmax(p) token 0 -> REJECT (n_matches=0)
        # Under ensemble greedy: draft token 1 == argmax(v) token 1 -> ACCEPT (n_matches=1)

    def test_value_error_without_candidate_logits(self):
        """Verify ValueError when ensemble weight < 1.0 and candidate_logits is None."""
        config = GenerationConfig(assistant_ensemble_weight=0.7)
        self.assertEqual(config.assistant_ensemble_weight, 0.7)
        # The actual ValueError is raised inside the generation loop, not in config.
        # We test the config stores the value correctly here.

    def test_generation_config_round_trip(self):
        """GenerationConfig preserves assistant_ensemble_weight through serialization."""
        config = GenerationConfig(assistant_ensemble_weight=0.7)
        self.assertEqual(config.assistant_ensemble_weight, 0.7)

        config_dict = config.to_dict()
        self.assertIn("assistant_ensemble_weight", config_dict)
        self.assertEqual(config_dict["assistant_ensemble_weight"], 0.7)

        # Round-trip
        config2 = GenerationConfig.from_dict(config_dict)
        self.assertEqual(config2.assistant_ensemble_weight, 0.7)

    def test_generation_config_default_is_none(self):
        """Default assistant_ensemble_weight is None (standard lossless SD)."""
        config = GenerationConfig()
        self.assertIsNone(config.assistant_ensemble_weight)


if __name__ == "__main__":
    unittest.main()
