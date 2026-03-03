"""
Generate TDT loss reference fixtures using NeMo's TDTLossPytorch.

Usage (requires NeMo installed, no CUDA needed):
    python tests/models/parakeet/generate_tdt_loss_fixtures.py

Outputs:
    tests/fixtures/parakeet/expected_tdt_loss.json

The fixture contains deterministic inputs and expected loss values
computed by NeMo's TDTLossPytorch. Our tdt_loss implementation is
tested against these values in test_modeling_parakeet.py::TDTLossTest.
"""

import json
import os

import torch


def make_test_inputs():
    torch.manual_seed(42)
    batch_size, max_t, max_u, vocab_size, num_durations = 2, 8, 4, 5, 5
    blank = vocab_size

    combined_logits = torch.randn(batch_size, max_t, max_u + 1, vocab_size + 1 + num_durations)
    targets = torch.randint(0, vocab_size, (batch_size, max_u))
    logit_lengths = torch.tensor([max_t, max_t - 1])
    target_lengths = torch.tensor([max_u, max_u - 1])

    return {
        "combined_logits": combined_logits,
        "token_logits": combined_logits[..., : vocab_size + 1],
        "duration_logits": combined_logits[..., vocab_size + 1 :],
        "targets": targets,
        "logit_lengths": logit_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
        "durations": [0, 1, 2, 3, 4],
    }


def _patched_compute_forward_prob(self, acts, duration_acts, labels, act_lens, label_lens):
    """NeMo's compute_forward_prob with .cuda() replaced by device-aware allocation.

    This is identical to NeMo's TDTLossPytorch.compute_forward_prob except
    `log_alpha = log_alpha.cuda()` is replaced with `device=acts.device`, and
    `torch.Tensor([-1000.0]).cuda()[0]` is replaced with `torch.tensor(-1000.0, device=acts.device)`.
    The loss math is unchanged.
    """
    B, T, U, _ = acts.shape
    log_alpha = torch.zeros(B, T, U, device=acts.device)

    for b in range(B):
        for t in range(T):
            for u in range(U):
                if u == 0:
                    if t == 0:
                        log_alpha[b, t, u] = 0.0
                    else:
                        log_alpha[b, t, u] = -1000.0
                        for n, l in enumerate(self.durations):
                            if t - l >= 0 and l > 0:
                                tmp = (
                                    log_alpha[b, t - l, u]
                                    + acts[b, t - l, u, self.blank]
                                    + duration_acts[b, t - l, u, n]
                                )
                                log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])
                else:
                    log_alpha[b, t, u] = -1000.0
                    for n, l in enumerate(self.durations):
                        if t - l >= 0:
                            if l > 0:
                                tmp = (
                                    log_alpha[b, t - l, u]
                                    + acts[b, t - l, u, self.blank]
                                    + duration_acts[b, t - l, u, n]
                                )
                                log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])
                            tmp = (
                                log_alpha[b, t - l, u - 1]
                                + acts[b, t - l, u - 1, labels[b, u - 1]]
                                + duration_acts[b, t - l, u - 1, n]
                            )
                            log_alpha[b, t, u] = self.logsumexp(tmp, 1.0 * log_alpha[b, t, u])

    log_probs = []
    for b in range(B):
        tt = torch.tensor(-1000.0, device=acts.device)
        for n, l in enumerate(self.durations):
            if act_lens[b] - l >= 0 and l > 0:
                bb = (
                    log_alpha[b, act_lens[b] - l, label_lens[b]]
                    + acts[b, act_lens[b] - l, label_lens[b], self.blank]
                    + duration_acts[b, act_lens[b] - l, label_lens[b], n]
                )
                tt = self.logsumexp(bb, 1.0 * tt)
        log_probs.append(tt)

    return torch.stack(log_probs), log_alpha


def compute_nemo_reference(inputs):
    """Run NeMo's TDTLossPytorch.

    On CPU, monkey-patches compute_forward_prob to avoid NeMo's hardcoded .cuda().
    On CUDA, runs NeMo unmodified.
    """
    import nemo.collections.asr.losses.rnnt_pytorch as rnnt_mod

    need_patch = not torch.cuda.is_available()
    orig = None
    if need_patch:
        print("No CUDA available — patching NeMo's compute_forward_prob for CPU (math unchanged)")
        orig = rnnt_mod.TDTLossPytorch.compute_forward_prob
        rnnt_mod.TDTLossPytorch.compute_forward_prob = _patched_compute_forward_prob

    results = {}
    for reduction in ["sum", "mean"]:
        loss_fn = rnnt_mod.TDTLossPytorch(
            blank=inputs["blank"],
            durations=inputs["durations"],
            reduction=reduction,
            sigma=0.0,
        )
        loss = loss_fn(
            acts=inputs["combined_logits"],
            labels=inputs["targets"],
            act_lens=inputs["logit_lengths"],
            label_lens=inputs["target_lengths"],
        )
        results[reduction] = loss.item()
        print(f"NeMo TDT loss (reduction={reduction}): {loss.item():.10f}")

    if orig is not None:
        rnnt_mod.TDTLossPytorch.compute_forward_prob = orig

    return results


def main():
    inputs = make_test_inputs()
    nemo_results = compute_nemo_reference(inputs)

    fixture = {
        "_comment": "Generated by generate_tdt_loss_fixtures.py using NeMo's TDTLossPytorch. "
        "Inputs use torch.manual_seed(42), batch=2, T=8, U=4, vocab=5, durations=[0,1,2,3,4].",
        "seed": 42,
        "batch_size": 2,
        "max_t": 8,
        "max_u": 4,
        "vocab_size": 5,
        "durations": [0, 1, 2, 3, 4],
        "targets": inputs["targets"].tolist(),
        "logit_lengths": inputs["logit_lengths"].tolist(),
        "target_lengths": inputs["target_lengths"].tolist(),
        "expected_loss_sum": nemo_results["sum"],
        "expected_loss_mean": nemo_results["mean"],
    }

    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "parakeet", "expected_tdt_loss.json")
    output_path = os.path.normpath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2)

    print(f"\nFixture written to {output_path}")


if __name__ == "__main__":
    main()
